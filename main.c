#include "postgres.h"

#include "gpu_accelerator.h"
#include "utils/guc.h"

PG_MODULE_MAGIC;

/* Function declarations */
void _PG_init(void);

bool enable_gpuscan;

void _PG_init(void)
{
    DefineCustomBoolVariable(
        "enable_gpuscan", "Enables the use of GPU accelerated seq-scan", NULL,
        &enable_gpuscan, true, PGC_USERSET, GUC_NOT_IN_SAMPLE, NULL, NULL, NULL);

    // elog(INFO, "Loading gpu_scan extension ...");
    _init_rel_pathlist_hook();
    _init_executorRun_hook();
    _init_gpu_join_hook();
}
