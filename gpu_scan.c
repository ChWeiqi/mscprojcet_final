#include "c.h"
#include "postgres.h"
#include "gpu_accelerator.h"

#include "access/heapam.h"
#include "catalog/pg_class_d.h"
#include "executor/nodeCustom.h"
#include "executor/executor.h"
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

/* Static variables */
static set_rel_pathlist_hook_type set_rel_pathlist_next = NULL;
static CustomPathMethods gpuscan_path_methods;
static CustomScanMethods gpuscan_plan_methods;
static CustomExecMethods gpuscan_exec_methods;

typedef struct GpuCustomScanState
{
    CustomScanState css;

} GpuCustomScanState;

/* Function declarations */

/* Hook to get control in set_rel_pathlist() */
static void SetGpuScanPath(PlannerInfo *root, RelOptInfo *rel, Index rti,
                           RangeTblEntry *rte);

/* Custom scan path callback */
static Plan *PlanGpuScanPath(PlannerInfo *root, RelOptInfo *rel,
                             CustomPath *best_path, List *tlist,
                             List *clauses, List *custom_plans);

/* Custom scan plan callback */
static Node *CreateGpuScanState(CustomScan *cscan);

/* Custom scan plan execution callbacks */
static void BeginGpuscan(CustomScanState *node, EState *estate, int eflags);
static TupleTableSlot *ExecGpuScan(CustomScanState *node);
static void EndGpuScan(CustomScanState *node);
static TupleTableSlot *ExecGpuScanFetch(ScanState *node,
                                        ExecScanAccessMtd accessMtd,
                                        ExecScanRecheckMtd recheckMtd);

void _init_rel_pathlist_hook(void)
{
    // elog(INFO, "init rel_pathlist_hook ...");
    set_rel_pathlist_next = set_rel_pathlist_hook;
    set_rel_pathlist_hook = SetGpuScanPath;

    /* setup path methods */
    memset(&gpuscan_path_methods, 0, sizeof(gpuscan_path_methods));
    gpuscan_path_methods.CustomName = "GPUScan";
    gpuscan_path_methods.PlanCustomPath = PlanGpuScanPath;

    /* setup plan methods */
    memset(&gpuscan_plan_methods, 0, sizeof(gpuscan_plan_methods));
    gpuscan_plan_methods.CustomName = "GPUScan";
    gpuscan_plan_methods.CreateCustomScanState = CreateGpuScanState;
    RegisterCustomScanMethods(&gpuscan_plan_methods);

    /* setup exec methods */
    memset(&gpuscan_exec_methods, 0, sizeof(gpuscan_exec_methods));
    gpuscan_exec_methods.CustomName = "GPUScan";
    gpuscan_exec_methods.BeginCustomScan = BeginGpuscan;
    gpuscan_exec_methods.ExecCustomScan = ExecGpuScan;
    gpuscan_exec_methods.EndCustomScan = EndGpuScan;
}

static void
SetGpuScanPath(PlannerInfo *root, RelOptInfo *baserel, Index rti,
               RangeTblEntry *rte)
{
    elog(INFO, "set gpu scan path");

    /* If the relation has relation been proven empty, nothing need to do */
    if (is_dummy_rel(baserel))
    {
        return;
    }

    /* Create GPU accelerated scan path */
    if (baserel->rtekind != RTE_RELATION)
    {
        return;
    }
    if (rte->relkind != RELKIND_RELATION)
    {
        return;
    }

    Relids required_outer;
    CustomPath *cpath = makeNode(CustomPath);
    int parallel_workers;

    required_outer = baserel->lateral_relids;

    cpath->path.type = T_CustomPath;
    cpath->path.pathtype = T_CustomScan;
    cpath->path.parent = baserel;
    cpath->path.pathtarget = baserel->reltarget;
    cpath->path.param_info = get_baserel_parampathinfo(root, baserel, required_outer);

    // parallel_workers = compute_parallel_worker(rel, rel->pages, -1,
    //                                            max_parallel_workers_per_gather);
    parallel_workers = 0;
    cpath->path.parallel_aware = (parallel_workers > 0);
    cpath->path.parallel_safe = baserel->consider_parallel;
    cpath->path.parallel_workers = parallel_workers;

    gpu_cost_seqscan(cpath, root, baserel, cpath->path.param_info);

    cpath->path.pathkeys = NIL; /* unsorted results */

    cpath->flags = CUSTOMPATH_SUPPORT_PROJECTION;
    cpath->custom_paths = NIL;
    cpath->methods = &gpuscan_path_methods;

    add_path(baserel, cpath);
}

static Plan *
PlanGpuScanPath(PlannerInfo *root, RelOptInfo *rel, CustomPath *best_path,
                List *tlist, List *clauses, List *custom_plans)
{
    CustomScan *cscan = makeNode(CustomScan);
    Index scan_relid = best_path->path.parent->relid;

    /* it should be a base rel... */
    Assert(scan_relid > 0);
    // Assert(best_path->parent->rtekind == RTE_RELATION);

    /* Reduce RestrictInfo list to bare expressions; ignore pseudoconstants */
    clauses = extract_actual_clauses(clauses, false);

    /* set scanrelid */
    cscan->scan.scanrelid = rel->relid;
    /* set targetlist as is  */
    cscan->scan.plan.targetlist = tlist;
    /* reduce RestrictInfo list to bare expressions */
    cscan->scan.plan.qual = clauses;

    cscan->flags = best_path->flags;

    cscan->custom_plans = NIL;

    // cscan->custom_exprs =

    // cscan->custom_private =

    // cscan->custom_scan_tlist =

    // cscan->custom_relids =

    cscan->methods = &gpuscan_plan_methods;

    return &cscan->scan.plan;
}

static Node *
CreateGpuScanState(CustomScan *cscan)
{
    GpuCustomScanState *gcss = palloc0(sizeof(GpuCustomScanState));

    Assert(cscan->methods == &gpuscan_plan_methods);

    NodeSetTag(gcss, T_CustomScanState);

    gcss->css.flags = cscan->flags;
    gcss->css.methods = &gpuscan_exec_methods;
    gcss->css.slotOps = &TTSOpsBufferHeapTuple;;

    return (Node *)&gcss->css;
}

static void BeginGpuscan(CustomScanState *node, EState *estate, int eflags)
{
    elog(INFO, "begin gpuscan");
    GpuCustomScanState *gcss = (GpuCustomScanState *)node;
    CustomScanState css = gcss->css;

    /* ReInit to use table_slot_callbacks here */
    // ExecInitScanTupleSlot(estate, &css.ss,
    //                       RelationGetDescr(css.ss.ss_currentRelation),
    //                       table_slot_callbacks(css.ss.ss_currentRelation));

}

TupleTableSlot *
GpuSeqNext(CustomScanState *node)
{
    HeapTuple tuple;
    TableScanDesc scandesc;
    EState *estate;
    ScanDirection direction;
    TupleTableSlot *slot;

    /*
     * get information from the estate and scan state
     */
    scandesc = node->ss.ss_currentScanDesc;
    estate = node->ss.ps.state;
    direction = estate->es_direction;
    slot = node->ss.ss_ScanTupleSlot;

    if (scandesc == NULL)
    {
        /*
         * We reach here if the scan is not parallel, or if we're serially
         * executing a scan that was planned to be parallel.
         */
        scandesc = table_beginscan(node->ss.ss_currentRelation,
                                   estate->es_snapshot,
                                   0, NULL);
        node->ss.ss_currentScanDesc = scandesc;
    }

    /*
     * get the next tuple from the table
     */
    if (table_scan_getnextslot(scandesc, direction, slot))
        return slot;
    return NULL;
}

bool GpuSeqRecheck(CustomScanState *node, TupleTableSlot *slot)
{
    /*
     * Note that unlike IndexScan, SeqScan never use keys in heap_beginscan
     * (and this is very bad) - so, here we do not check are keys ok or not.
     */
    return true;
}

/*
 * ExecScanFetch -- check interrupts & fetch next potential tuple
 *
 * This routine is concerned with substituting a test tuple if we are
 * inside an EvalPlanQual recheck.  If we aren't, just execute
 * the access method's next-tuple routine.
 */
static TupleTableSlot *
ExecGpuScanFetch(ScanState *node,
                 ExecScanAccessMtd accessMtd,
                 ExecScanRecheckMtd recheckMtd)
{
    EState *estate = node->ps.state;

    CHECK_FOR_INTERRUPTS();

    if (estate->es_epq_active != NULL)
    {
        EPQState *epqstate = estate->es_epq_active;

        /*
         * We are inside an EvalPlanQual recheck.  Return the test tuple if
         * one is available, after rechecking any access-method-specific
         * conditions.
         */
        Index scanrelid = ((Scan *)node->ps.plan)->scanrelid;

        if (scanrelid == 0)
        {
            /*
             * This is a ForeignScan or CustomScan which has pushed down a
             * join to the remote side.  The recheck method is responsible not
             * only for rechecking the scan/join quals but also for storing
             * the correct tuple in the slot.
             */

            TupleTableSlot *slot = node->ss_ScanTupleSlot;

            if (!(*recheckMtd)(node, slot))
                ExecClearTuple(slot); /* would not be returned by scan */
            return slot;
        }
        else if (epqstate->relsubs_done[scanrelid - 1])
        {
            /*
             * Return empty slot, as either there is no EPQ tuple for this rel
             * or we already returned it.
             */

            TupleTableSlot *slot = node->ss_ScanTupleSlot;

            return ExecClearTuple(slot);
        }
        else if (epqstate->relsubs_slot[scanrelid - 1] != NULL)
        {
            /*
             * Return replacement tuple provided by the EPQ caller.
             */

            TupleTableSlot *slot = epqstate->relsubs_slot[scanrelid - 1];

            Assert(epqstate->relsubs_rowmark[scanrelid - 1] == NULL);

            /* Mark to remember that we shouldn't return it again */
            epqstate->relsubs_done[scanrelid - 1] = true;

            /* Return empty slot if we haven't got a test tuple */
            if (TupIsNull(slot))
                return NULL;

            /* Check if it meets the access-method conditions */
            if (!(*recheckMtd)(node, slot))
                return ExecClearTuple(slot); /* would not be returned by
                                              * scan */
            return slot;
        }
        else if (epqstate->relsubs_rowmark[scanrelid - 1] != NULL)
        {
            /*
             * Fetch and return replacement tuple using a non-locking rowmark.
             */

            TupleTableSlot *slot = node->ss_ScanTupleSlot;

            /* Mark to remember that we shouldn't return more */
            epqstate->relsubs_done[scanrelid - 1] = true;

            if (!EvalPlanQualFetchRowMark(epqstate, scanrelid, slot))
                return NULL;

            /* Return empty slot if we haven't got a test tuple */
            if (TupIsNull(slot))
                return NULL;

            /* Check if it meets the access-method conditions */
            if (!(*recheckMtd)(node, slot))
                return ExecClearTuple(slot); /* would not be returned by
                                              * scan */
            return slot;
        }
    }

    /*
     * Run the node-type-specific access method function to get the next tuple
     */
    return (*accessMtd)(node);
}

static TupleTableSlot *ExecGpuScan(CustomScanState *node)
{
    elog(INFO, "exec gpuscan");
    return ExecGpuScanFetch(&node->ss, (ExecScanAccessMtd)GpuSeqNext, (ExecScanRecheckMtd)GpuSeqRecheck);
}

static void EndGpuScan(CustomScanState *node)
{
    GpuCustomScanState *gcss = (GpuCustomScanState *)node;

    if (gcss->css.ss.ss_currentScanDesc)
        heap_endscan(gcss->css.ss.ss_currentScanDesc);
    elog(INFO, "end gpuscan");
}
