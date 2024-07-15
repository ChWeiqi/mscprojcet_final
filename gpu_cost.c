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

/* Default gpu cost setting*/
double gpu_operator_cost = DEFAULT_CPU_OPERATOR_COST / 16;
double gpu_tuple_cost = DEFAULT_CPU_TUPLE_COST;
double gpu_startup_cost = 100 * DEFAULT_SEQ_PAGE_COST;

double
get_gpu_operator_ratio(void)
{
    if (cpu_operator_cost > 0.0)
    {
        return gpu_operator_cost / cpu_operator_cost;
    }
    return gpu_operator_cost == 0.0 ? 1.0 : disable_cost;
}

void gpu_cost_seqscan(CustomPath *cpath, PlannerInfo *root,
                      RelOptInfo *baserel, ParamPathInfo *param_info)
{
    Path *path = &cpath->path;
    double gpu_ratio = get_gpu_operator_ratio();
    double gpu_tuple_cost;
    double spc_seq_page_cost;
    double spc_rand_page_cost;

    Cost startup_cost = 0;
    Cost gpu_run_cost;
    Cost disk_run_cost;
    QualCost qpqual_cost;
    Cost gpu_per_tuple;
    double ntuples = baserel->tuples;

    /* Should only be applied to base relations */
    Assert(baserel->relid > 0);
    Assert(baserel->rtekind == RTE_RELATION);

    /* Mark the path with the correct row estimate */
    if (param_info)
        path->rows = param_info->ppi_rows;
    else
        path->rows = baserel->rows;

    /* Fetch estimated page cost for tablespace containing table */
    get_tablespace_page_costs(baserel->reltablespace,
                              NULL,
                              &spc_seq_page_cost);

    /* Disk cost */
    disk_run_cost = spc_seq_page_cost * baserel->pages;

    /* Compute evaluation costs of a baserel's restriction quals, plus any
     * movable join quals that have been pushed down to the scan.
     * Same logic in get_restriction_qual_cost(root, baserel, param_info, &qpqual_cost);
     */
    if (param_info)
    {
        /* Include costs of pushed-down clauses */
        cost_qual_eval(&qpqual_cost, param_info->ppi_clauses, root);

        qpqual_cost.startup += baserel->baserestrictcost.startup;
        qpqual_cost.per_tuple += baserel->baserestrictcost.per_tuple;
    }
    else
        qpqual_cost = baserel->baserestrictcost;

    /* Restriction quals cost */
    startup_cost += qpqual_cost.startup;
    gpu_per_tuple = qpqual_cost.per_tuple * gpu_ratio + gpu_tuple_cost;
    ntuples = baserel->tuples;
    gpu_run_cost = gpu_per_tuple * ntuples;

    /* Projection columns cost */
    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += path->pathtarget->cost.startup;
    gpu_run_cost += path->pathtarget->cost.per_tuple * path->rows;

    /* Adjust costing for parallelism, if used. Not applicable for sequential scan */
    // if (path->parallel_workers > 0)
    // {
    //     double parallel_divisor = get_parallel_divisor(path);

    //     /* The CPU cost is divided among all the workers. */
    //     gpu_run_cost /= parallel_divisor;

    //     /*
    //      * It may be possible to amortize some of the I/O cost, but probably
    //      * not very much, because most operating systems already do aggressive
    //      * prefetching.  For now, we assume that the disk run cost can't be
    //      * amortized at all.
    //      */

    //     /*
    //      * In the case of a parallel plan, the row count needs to represent
    //      * the number of tuples processed per worker.
    //      */
    //     path->rows = clamp_row_est(path->rows / parallel_divisor);
    // }

    path->startup_cost = startup_cost;
    path->total_cost = startup_cost + gpu_run_cost + disk_run_cost;
}

void gpu_cost_nestloop(CustomPath *cpath, PlannerInfo *root,
                       Path *outer_path, Path *inner_path,
                       List *restrict_clauses, JoinPathExtraData *extra)
{
    Cost startup_cost = 0;
    Cost run_cost = 0;
    double outer_path_rows = outer_path->rows;
    double inner_path_rows = inner_path->rows;
    Cost inner_rescan_start_cost;
    Cost inner_rescan_total_cost;
    Cost inner_run_cost;
    Cost inner_rescan_run_cost;
    Cost gpu_per_tuple;
    double gpu_ratio = gpu_operator_cost / cpu_operator_cost;
    QualCost restrict_qual_cost;
    double ntuples;

    /* Protect some assumptions below that rowcounts aren't zero */
    if (outer_path_rows <= 0)
        outer_path_rows = 1;
    if (inner_path_rows <= 0)
        inner_path_rows = 1;

    /* Mark the path with the correct row estimate */
    if (cpath->path.param_info)
        cpath->path.rows = cpath->path.param_info->ppi_rows;
    else
        cpath->path.rows = cpath->path.parent->rows;

    /*
     * For partial paths, scale row estimate.
     * Not consider parallel execution here for now.
     */
    // if (cpath->path.parallel_workers > 0)
    // {
    //     double parallel_divisor = get_parallel_divisor(&cpath->path);

    //     cpath->path.rows =
    //         clamp_row_est(cpath->path.rows / parallel_divisor);
    // }

    if (!enable_nestloop)
        startup_cost += disable_cost;

    /* estimate costs to rescan the inner relation */
    // cost_rescan(root, inner_path,
    //             &inner_rescan_start_cost,
    //             &inner_rescan_total_cost);

    /*
     * startup_cost = outer's startup_cost + inner's startup_cost
     * run_cost = outer's run_cost + inner's run cost
     *            + (outer's rows(estimated) - 1) * inner rescan total_cost
     */
    startup_cost += outer_path->startup_cost + inner_path->startup_cost;
    run_cost += outer_path->total_cost - outer_path->startup_cost;
    run_cost += inner_run_cost;
    if (outer_path_rows > 1)
    {
        run_cost += (outer_path_rows - 1) * inner_rescan_start_cost;
        run_cost += (outer_path_rows - 1) * inner_rescan_run_cost;
    }

    /* Compute number of tuples processed (not number emitted!) */
    ntuples = outer_path_rows * inner_path_rows;
    /* CPU & GPU costs */
    cost_qual_eval(&restrict_qual_cost, restrict_clauses, root);
    startup_cost += restrict_qual_cost.startup;

    /* Cost to preload inner heap tuples by CPU */
    gpu_per_tuple = gpu_tuple_cost + restrict_qual_cost.per_tuple * gpu_ratio;
    run_cost += gpu_per_tuple * ntuples;

    /* tlist eval costs are paid per output row, not per tuple scanned */
    startup_cost += cpath->path.pathtarget->cost.startup;
    run_cost += cpath->path.pathtarget->cost.per_tuple * cpath->path.rows;

    cpath->path.startup_cost = startup_cost;
    cpath->path.total_cost = startup_cost + run_cost;
}
