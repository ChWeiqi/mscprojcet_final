#include "postgres.h"
#include "gpu_accelerator.h"

#include "access/heapam.h"
#include "catalog/pg_class_d.h"
#include "executor/nodeCustom.h"
#include "executor/nodeNestloop.h"
#include "executor/executor.h"
#include "executor/execdebug.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"
#include "nodes/pathnodes.h"
#include "nodes/plannodes.h"
#include "nodes/extensible.h"
#include "nodes/pg_list.h"
#include "utils/guc.h"
#include "utils/spccache.h"

/*
 * Paths parameterized by the parent can be considered to be parameterized by
 * any of its child.
 */
#define PATH_PARAM_BY_PARENT(path, rel)	\
	((path)->param_info && bms_overlap(PATH_REQ_OUTER(path),	\
									   (rel)->top_parent_relids))
#define PATH_PARAM_BY_REL_SELF(path, rel)  \
	((path)->param_info && bms_overlap(PATH_REQ_OUTER(path), (rel)->relids))

#define PATH_PARAM_BY_REL(path, rel)	\
	(PATH_PARAM_BY_REL_SELF(path, rel) || PATH_PARAM_BY_PARENT(path, rel))

/* Static variables */
static set_join_pathlist_hook_type set_join_pathlist_next = NULL;

static CustomPathMethods gpujoin_path_methods;
static CustomScanMethods gpujoin_plan_methods;
static CustomExecMethods gpujoin_exec_methods;

typedef struct GpuCustomJoinPath
{
    CustomPath *cpath;
    Index outer_relid;
    Path *outer_path; /* outer path (always one) */
    Plan *outer_plan;
    Path *inner_path;
    Plan *inner_plan;
    JoinType jointype;
    List *qual_clauses;
    List *joinrestrictinfo

} GpuCustomJoinPath;

typedef struct GpuCustomJoinPlanInfo
{

} GpuCustomJoinPlanInfo;

typedef struct GpuCustomJoinState
{
    CustomScanState css;
    NestLoopState *nls;

} GpuCustomJoinState;

/* Function declarations */
static void SetGpuJoinPath(PlannerInfo *root,
                           RelOptInfo *joinrel,
                           RelOptInfo *outerrel,
                           RelOptInfo *innerrel,
                           JoinType jointype,
                           JoinPathExtraData *extra);
void try_gpu_nestloop_path(PlannerInfo *root,
                           RelOptInfo *joinrel,
                           Path *outer_path,
                           Path *inner_path,
                           List *pathkeys,
                           JoinType jointype,
                           JoinPathExtraData *extra);
CustomPath *create_gpu_nestloop_path(PlannerInfo *root,
                                     RelOptInfo *joinrel,
                                     JoinType jointype,
                                     JoinCostWorkspace *workspace,
                                     JoinPathExtraData *extra,
                                     Path *outer_path,
                                     Path *inner_path,
                                     List *restrict_clauses,
                                     List *pathkeys,
                                     Relids required_outer);
Node *CreateGpuJoinState(CustomScan *cscan);
Plan *PlanGpuJoinPath(PlannerInfo *root,
                      RelOptInfo *rel,
                      CustomPath *best_path,
                      List *tlist,
                      List *clauses,
                      List *custom_plans);
void BeginGpuJoin(CustomScanState *node, EState *estate, int eflags);
TupleTableSlot *ExecGpuJoin(CustomScanState *node);
void EndGpuJoin(CustomScanState *node);

void _init_gpu_join_hook(void)
{
    // elog(INFO, "init join_pathlist ...");
    set_join_pathlist_next = set_join_pathlist_hook;
    set_join_pathlist_hook = SetGpuJoinPath;

    /* setup path methods */
    memset(&gpujoin_path_methods, 0, sizeof(gpujoin_path_methods));
    gpujoin_path_methods.CustomName = "GPUJoin";
    gpujoin_path_methods.PlanCustomPath = PlanGpuJoinPath;

    /* setup plan methods */
    memset(&gpujoin_plan_methods, 0, sizeof(gpujoin_plan_methods));
    gpujoin_plan_methods.CustomName = "GPUJoin";
    gpujoin_plan_methods.CreateCustomScanState = CreateGpuJoinState;
    RegisterCustomScanMethods(&gpujoin_plan_methods);

    /* setup exec methods */
    memset(&gpujoin_exec_methods, 0, sizeof(gpujoin_exec_methods));
    gpujoin_exec_methods.CustomName = "GPUJoin";
    gpujoin_exec_methods.BeginCustomScan = BeginGpuJoin;
    gpujoin_exec_methods.ExecCustomScan = ExecGpuJoin;
    gpujoin_exec_methods.EndCustomScan = EndGpuJoin;
}

static void SetGpuJoinPath(PlannerInfo *root,
                           RelOptInfo *joinrel,
                           RelOptInfo *outerrel,
                           RelOptInfo *innerrel,
                           JoinType jointype,
                           JoinPathExtraData *extra)
{
    bool mergejoin_allowed = true;
    Path *inner_cheapest_total = innerrel->cheapest_total_path;
    Path *matpath = NULL;
    ListCell *lc1;

    /* Only consider nested loop join now */

    /* Nestloop only supports inner, left, semi, and anti joins.
     * We only focus on inner and left joins here, so filter unsupported join here
     */
    if (jointype != JOIN_INNER &&
        jointype != JOIN_LEFT)
        return;

    /*
     * If inner_cheapest_total is parameterized by the outer rel, ignore it;
     * we will consider it below as a member of cheapest_parameterized_paths,
     * but the other possibilities considered in this routine aren't usable.
     */
    if (PATH_PARAM_BY_REL(inner_cheapest_total, outerrel))
        inner_cheapest_total = NULL;

    /*
     * Consider materializing the cheapest inner path, unless
     * enable_material is off or the path in question materializes its
     * output anyway.
     */
    if (enable_material && inner_cheapest_total != NULL &&
        !ExecMaterializesOutput(inner_cheapest_total->pathtype))
        matpath = (Path *)
            create_material_path(innerrel, inner_cheapest_total);

    /* For each outer, scan the inner */
    foreach (lc1, outerrel->pathlist)
    {
        Path *outerpath = (Path *)lfirst(lc1);
        List *merge_pathkeys;

        /*
         * We cannot use an outer path that is parameterized by the inner rel.
         */
        if (PATH_PARAM_BY_REL(outerpath, innerrel))
            continue;

        /*
         * Even in a nestloop, the result will have outer's sort order,
         * so we get outer pathkeys here.
         */
        merge_pathkeys = build_join_pathkeys(root, joinrel, jointype,
                                             outerpath->pathkeys);

        /*
         * Consider nestloop joins using this outer path and various
         * available paths for the inner relation.  We consider the
         * cheapest-total paths for each available parameterization of the
         * inner relation, including the unparameterized case.
         */
        ListCell *lc2;
        foreach (lc2, innerrel->cheapest_parameterized_paths)
        {
            Path *innerpath = (Path *)lfirst(lc2);
            Path *mpath;
            try_gpu_nestloop_path(root,
                                  joinrel,
                                  outerpath,
                                  innerpath,
                                  merge_pathkeys,
                                  jointype,
                                  extra);
        }
    }
}

void try_gpu_nestloop_path(PlannerInfo *root,
                           RelOptInfo *joinrel,
                           Path *outer_path,
                           Path *inner_path,
                           List *pathkeys,
                           JoinType jointype,
                           JoinPathExtraData *extra)
{
    Relids required_outer;
    JoinCostWorkspace workspace;
    RelOptInfo *innerrel = inner_path->parent;
    RelOptInfo *outerrel = outer_path->parent;
    Relids innerrelids;
    Relids outerrelids;
    Relids inner_paramrels = PATH_REQ_OUTER(inner_path);
    Relids outer_paramrels = PATH_REQ_OUTER(outer_path);

    /*
     * If we are forming an outer join at this join, it's nonsensical to use
     * an input path that uses the outer join as part of its parameterization.
     * (This can happen despite our join order restrictions, since those apply
     * to what is in an input relation not what its parameters are.)
     */
    if (extra->sjinfo->ojrelid != 0 &&
        (bms_is_member(extra->sjinfo->ojrelid, inner_paramrels) ||
         bms_is_member(extra->sjinfo->ojrelid, outer_paramrels)))
        return;

    /*
     * Paths are parameterized by top-level parents, so run parameterization
     * tests on the parent relids.
     */
    if (innerrel->top_parent_relids)
        innerrelids = innerrel->top_parent_relids;
    else
        innerrelids = innerrel->relids;

    if (outerrel->top_parent_relids)
        outerrelids = outerrel->top_parent_relids;
    else
        outerrelids = outerrel->relids;

    /*
     * Check to see if proposed path is still parameterized, and reject if the
     * parameterization wouldn't be sensible --- unless allow_star_schema_join
     * says to allow it anyway.  Also, we must reject if have_dangerous_phv
     * doesn't like the look of it, which could only happen if the nestloop is
     * still parameterized.
     */
    required_outer = calc_nestloop_required_outer(outerrelids, outer_paramrels,
                                                  innerrelids, inner_paramrels);
    // if (required_outer &&
    //     ((!bms_overlap(required_outer, extra->param_source_rels) &&
    //       !allow_star_schema_join(root, outerrelids, inner_paramrels)) ||
    //      have_dangerous_phv(root, outerrelids, inner_paramrels)))
    // {
    //     /* Waste no memory when we reject a path here */
    //     bms_free(required_outer);
    //     return;
    // }

    /* If we got past that, we shouldn't have any unsafe outer-join refs */
    // Assert(!have_unsafe_outer_join_ref(root, outerrelids, inner_paramrels));

    /*
     * The origin path calculate a cheap lower bound on the path's cost here,
     * then use add_path_precheck() to see if the path is clearly going to be
     * dominated by some existing path for the joinrel.
     * Theoretically, the GPU-based path will always pass the check. So in
     * order to simplify the process, do not precheck here.
     */

    /*
     * If the inner path is parameterized, it is parameterized by the
     * topmost parent of the outer rel, not the outer rel itself.  Fix
     * that.
     */
    if (PATH_PARAM_BY_PARENT(inner_path, outer_path->parent))
    {
        inner_path = reparameterize_path_by_child(root, inner_path,
                                                  outer_path->parent);

        /*
         * If we could not translate the path, we can't create nest loop
         * path.
         */
        if (!inner_path)
        {
            bms_free(required_outer);
            return;
        }
    }

    add_path(joinrel, (Path *)
                          create_gpu_nestloop_path(root,
                                                   joinrel,
                                                   jointype,
                                                   &workspace,
                                                   extra,
                                                   outer_path,
                                                   inner_path,
                                                   extra->restrictlist,
                                                   pathkeys,
                                                   required_outer));
}

CustomPath *create_gpu_nestloop_path(PlannerInfo *root,
                                     RelOptInfo *joinrel,
                                     JoinType jointype,
                                     JoinCostWorkspace *workspace,
                                     JoinPathExtraData *extra,
                                     Path *outer_path,
                                     Path *inner_path,
                                     List *restrict_clauses,
                                     List *pathkeys,
                                     Relids required_outer)
{
    // CustomPath *cpath = makeNode(CustomPath);
    // NestPath *nestpath = makeNode(NestPath);
    GpuCustomJoinPath *gpath;
    Relids inner_req_outer = PATH_REQ_OUTER(inner_path);
    ParamPathInfo *param_info;

    /*
     * If the inner path is parameterized by the outer, we must drop any
     * restrict_clauses that are due to be moved into the inner path.  We have
     * to do this now, rather than postpone the work till createplan time,
     * because the restrict_clauses list can affect the size and cost
     * estimates for this path.  We detect such clauses by checking for serial
     * number match to clauses already enforced in the inner path.
     */
    if (bms_overlap(inner_req_outer, outer_path->parent->relids))
    {
        Bitmapset *enforced_serials = get_param_path_clause_serials(inner_path);
        List *jclauses = NIL;
        ListCell *lc;

        foreach (lc, restrict_clauses)
        {
            RestrictInfo *rinfo = (RestrictInfo *)lfirst(lc);

            if (!bms_is_member(rinfo->rinfo_serial, enforced_serials))
                jclauses = lappend(jclauses, rinfo);
        }
        restrict_clauses = jclauses;
    }

    param_info = get_joinrel_parampathinfo(root,
                                           joinrel,
                                           outer_path,
                                           inner_path,
                                           extra->sjinfo,
                                           required_outer,
                                           &restrict_clauses);

    // NodeSetTag(cpath, T_CustomPath);
    // cpath->path.pathtype = T_CustomScan;
    // cpath->path.parent = joinrel;
    // cpath->path.pathtarget = joinrel->reltarget;
    // cpath->path.param_info = param_info;
    // /* parallel parameters */
    // cpath->path.parallel_aware = false;
    // cpath->path.parallel_safe = joinrel->consider_parallel &&
    //                             outer_path->parallel_safe && inner_path->parallel_safe;
    // cpath->path.parallel_workers = outer_path->parallel_workers;

    // /* the pathkeys follows the outer's pathkeys */
    // cpath->path.pathkeys = pathkeys;

    // cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    // cpath->custom_paths = inner_path;
    // cpath->methods = &gpujoin_path_methods;

    // /* Save NestPath Info in custom private */
    // nestpath->jpath.path.pathtype = T_NestLoop;
    // nestpath->jpath.path.parent = joinrel;
    // nestpath->jpath.path.pathtarget = joinrel->reltarget;
    // nestpath->jpath.path.param_info = param_info;
    // nestpath->jpath.path.parallel_aware = false;
    // nestpath->jpath.path.parallel_safe = joinrel->consider_parallel &&
    //                                      outer_path->parallel_safe && inner_path->parallel_safe;
    // /* This is a foolish way to estimate parallel_workers, but for now... */
    // nestpath->jpath.path.parallel_workers = outer_path->parallel_workers;
    // nestpath->jpath.path.pathkeys = pathkeys;
    // nestpath->jpath.jointype = jointype;
    // nestpath->jpath.inner_unique = extra->inner_unique;
    // nestpath->jpath.outerjoinpath = outer_path;
    // nestpath->jpath.innerjoinpath = inner_path;
    // nestpath->jpath.joinrestrictinfo = restrict_clauses;

    // cpath->custom_private = list_make1(nestpath);

    gpath = palloc0(sizeof(GpuCustomJoinPath));
    NodeSetTag(gpath, T_CustomPath);
    gpath->cpath->path.pathtype = T_CustomScan;
    gpath->cpath->path.parent = joinrel;
    gpath->cpath->path.pathtarget = joinrel->reltarget;
    gpath->cpath->path.param_info = param_info;
    /* parallel parameters */
    gpath->cpath->path.parallel_aware = false;
    gpath->cpath->path.parallel_safe = joinrel->consider_parallel &&
                                       outer_path->parallel_safe && inner_path->parallel_safe;
    gpath->cpath->path.parallel_workers = outer_path->parallel_workers;
    gpath->cpath->path.pathkeys = pathkeys;
    gpath->cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    gpath->cpath->custom_paths = inner_path;
    gpath->cpath->methods = &gpujoin_path_methods;
    gpath->jointype = jointype;
    gpath->inner_path = inner_path;
    gpath->outer_path = outer_path;
    gpath->joinrestrictinfo = restrict_clauses;
    gpath->outer_relid = outer_path->parent->relid;

    /* Consider GPU cost in this nestloop here */
    // gpu_cost_nestloop(cpath, root, outer_path, inner_path, restrict_clauses, extra);
    gpu_cost_nestloop(&gpath->cpath, root, outer_path, inner_path, restrict_clauses, extra);

    return &gpath->cpath;
}

/* Convert a path to a plan */
Plan *PlanGpuJoinPath(PlannerInfo *root,
                      RelOptInfo *rel,
                      CustomPath *best_path,
                      List *tlist,
                      List *clauses,
                      List *custom_plans)
{
    GpuCustomJoinPath *gpath = (GpuCustomJoinPath *)best_path;
    CustomScan *cscan = makeNode(CustomScan);
    Index outer_relid = gpath->outer_relid;
    CustomPath *cpath = gpath->cpath;

    cscan = makeNode(CustomScan);
    /* Cost data has been copied in create_customscan_plan, no need to do here */

    cscan->scan.plan.targetlist = tlist;
    cscan->scan.plan.qual = NIL;
    cscan->scan.scanrelid = 0;

    cscan->flags = best_path->flags;
    cscan->methods = &gpujoin_plan_methods;
    /* custom_plans has been processed in create_customscan_plan */
    cscan->custom_scan_tlist = tlist;

    return &cscan->scan.plan;
}

Node *CreateGpuJoinState(CustomScan *cscan)
{
    // GpuCustomJoinState *gcjs = palloc0(sizeof(GpuCustomJoinState));

    // NodeSetTag(gcjs, T_CustomScanState);

    // gcjs->css.flags = cscan->flags;
    // gcjs->css.methods = &gpujoin_exec_methods;

    // return (Node *)gcjs;
    CustomScanState *css = palloc0(sizeof(CustomScanState));
    css->slotOps = &TTSOpsHeapTuple;
    css->methods = &gpujoin_exec_methods;
    css->flags = cscan->flags;

    return (Node *)css;
}

void BeginGpuJoin(CustomScanState *node, EState *estate, int eflags)
{
    elog(INFO, "begin gpujoin");
}

TupleTableSlot *ExecGpuJoin(CustomScanState *node)
{
    GpuCustomJoinState *gcsj = (GpuCustomJoinState *)node;
    NestLoop *nl;
    NestLoopState *nls = gcsj->nls;
    PlanState *innerPlan;
    PlanState *outerPlan;
    TupleTableSlot *outerTupleSlot;
    TupleTableSlot *innerTupleSlot;
    ExprState *joinqual;
    ExprState *otherqual;
    ExprContext *econtext;
    ListCell *lc;

    CHECK_FOR_INTERRUPTS();

    /*
     * get information from the node
     */
    ENL1_printf("getting info from node");

    nl = (NestLoop *)node->ss.ps.plan;
    joinqual = node->ss.ps.qual;
    otherqual = node->ss.ps.plan->qual;
    outerPlan = outerPlanState(node);
    innerPlan = innerPlanState(node);
    econtext = node->ss.ps.ps_ExprContext;

    /*
     * Reset per-tuple memory context to free any expression evaluation
     * storage allocated in the previous tuple cycle.
     */
    ResetExprContext(econtext);

    /*
     * Ok, everything is setup for the join so now loop until we return a
     * qualifying join tuple.
     */
    ENL1_printf("entering main loop");

    for (;;)
    {
        /*
         * If we don't have an outer tuple, get the next one and reset the
         * inner scan.
         */
        if (nls->nl_NeedNewOuter)
        {
            ENL1_printf("getting new outer tuple");
            outerTupleSlot = ExecProcNode(outerPlan);

            /*
             * if there are no more outer tuples, then the join is complete..
             */
            if (TupIsNull(outerTupleSlot))
            {
                ENL1_printf("no outer tuple, ending join");
                return NULL;
            }

            ENL1_printf("saving new outer tuple information");
            econtext->ecxt_outertuple = outerTupleSlot;
            nls->nl_NeedNewOuter = false;
            nls->nl_MatchedOuter = false;

            /*
             * fetch the values of any outer Vars that must be passed to the
             * inner scan, and store them in the appropriate PARAM_EXEC slots.
             */
            foreach (lc, nl->nestParams)
            {
                NestLoopParam *nlp = (NestLoopParam *)lfirst(lc);
                int paramno = nlp->paramno;
                ParamExecData *prm;

                prm = &(econtext->ecxt_param_exec_vals[paramno]);
                /* Param value should be an OUTER_VAR var */
                Assert(IsA(nlp->paramval, Var));
                Assert(nlp->paramval->varno == OUTER_VAR);
                Assert(nlp->paramval->varattno > 0);
                prm->value = slot_getattr(outerTupleSlot,
                                          nlp->paramval->varattno,
                                          &(prm->isnull));
                /* Flag parameter value as changed */
                innerPlan->chgParam = bms_add_member(innerPlan->chgParam,
                                                     paramno);
            }

            /*
             * now rescan the inner plan
             */
            ENL1_printf("rescanning inner plan");
            ExecReScan(innerPlan);
        }

        /*
         * we have an outerTuple, try to get the next inner tuple.
         */
        ENL1_printf("getting new inner tuple");

        innerTupleSlot = ExecProcNode(innerPlan);
        econtext->ecxt_innertuple = innerTupleSlot;

        if (TupIsNull(innerTupleSlot))
        {
            ENL1_printf("no inner tuple, need new outer tuple");

            nls->nl_NeedNewOuter = true;

            if (!nls->nl_MatchedOuter &&
                (nls->js.jointype == JOIN_LEFT ||
                 nls->js.jointype == JOIN_ANTI))
            {
                /*
                 * We are doing an outer join and there were no join matches
                 * for this outer tuple.  Generate a fake join tuple with
                 * nulls for the inner tuple, and return it if it passes the
                 * non-join quals.
                 */
                econtext->ecxt_innertuple = nls->nl_NullInnerTupleSlot;

                ENL1_printf("testing qualification for outer-join tuple");

                if (otherqual == NULL || ExecQual(otherqual, econtext))
                {
                    /*
                     * qualification was satisfied so we project and return
                     * the slot containing the result tuple using
                     * ExecProject().
                     */
                    ENL1_printf("qualification succeeded, projecting tuple");

                    return ExecProject(nls->js.ps.ps_ProjInfo);
                }
                else
                    InstrCountFiltered2(nls, 1);
            }

            /*
             * Otherwise just return to top of loop for a new outer tuple.
             */
            continue;
        }

        /*
         * at this point we have a new pair of inner and outer tuples so we
         * test the inner and outer tuples to see if they satisfy the node's
         * qualification.
         *
         * Only the joinquals determine MatchedOuter status, but all quals
         * must pass to actually return the tuple.
         */
        ENL1_printf("testing qualification");

        if (ExecQual(joinqual, econtext))
        {
            nls->nl_MatchedOuter = true;

            /* In an antijoin, we never return a matched tuple */
            if (nls->js.jointype == JOIN_ANTI)
            {
                nls->nl_NeedNewOuter = true;
                continue; /* return to top of loop */
            }

            /*
             * If we only need to join to the first matching inner tuple, then
             * consider returning this one, but after that continue with next
             * outer tuple.
             */
            if (nls->js.single_match)
                nls->nl_NeedNewOuter = true;

            if (otherqual == NULL || ExecQual(otherqual, econtext))
            {
                /*
                 * qualification was satisfied so we project and return the
                 * slot containing the result tuple using ExecProject().
                 */
                ENL1_printf("qualification succeeded, projecting tuple");

                return ExecProject(nls->js.ps.ps_ProjInfo);
            }
            else
                InstrCountFiltered2(node, 1);
        }
        else
            InstrCountFiltered1(node, 1);

        /*
         * Tuple fails qual, so free per-tuple memory and try again.
         */
        ResetExprContext(econtext);

        ENL1_printf("qualification failed, looping");
    }
}

void EndGpuJoin(CustomScanState *node)
{
    GpuCustomJoinState *gcjs = (GpuCustomJoinState *)node;

    if (gcjs->css.ss.ss_currentScanDesc)
        heap_endscan(gcjs->css.ss.ss_currentScanDesc);
    elog(INFO, "end gpuscan");
}