#include "postgres.h"

#include "fmgr.h"
#include "miscadmin.h"
#include "executor/nodeCustom.h"
#include "optimizer/cost.h"


/* Global variables */
// extern bool enable_gpuscan;

/* Function declarations */
extern void _init_rel_pathlist_hook(void);
extern void _init_executorRun_hook(void);

extern TupleTableSlot *GpuSeqNext(CustomScanState *node);
extern bool GpuSeqRecheck(CustomScanState *node, TupleTableSlot *slot);

extern void gpu_cost_nestloop(CustomPath *cpath,
                       PlannerInfo *root,
                       Path *outer_path,
                       Path *inner_path,
                       List *restrict_clauses,
                       JoinPathExtraData *extra);
extern void gpu_cost_seqscan(CustomPath *cpath, PlannerInfo *root,
                      RelOptInfo *baserel, ParamPathInfo *param_info);