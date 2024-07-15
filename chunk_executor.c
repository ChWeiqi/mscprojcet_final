#include "postgres.h"
#include "gpu_accelerator.h"

#include "access/relscan.h"
#include "access/tableam.h"
#include "executor/executor.h"
#include "executor/execdebug.h"
#include "executor/tuptable.h"
#include "nodes/execnodes.h"
#include "nodes/pathnodes.h"

static uint chunk_size = 1024; // Define the chunk size
static ExecutorRun_hook_type PreviousExecutorRunHook = NULL;
typedef struct TupleTableSlotChunk
{
	TupleTableSlot **tuples;
	int length;
} TupleTableSlotChunk;

/* Function declaration */
static bool can_execute_with_gpu(QueryDesc *queryDesc);
void gpu_ExecutorRun(QueryDesc *queryDesc,
					 ScanDirection direction,
					 uint64 count,
					 bool execute_once);
static void ExecutePlanGPU(EState *estate,
						   PlanState *planstate,
						   bool use_parallel_mode,
						   CmdType operation,
						   bool sendTuples,
						   uint64 numberTuples,
						   ScanDirection direction,
						   DestReceiver *dest,
						   bool execute_once);
static TupleTableSlotChunk *ExecProcNodeGPU(PlanState *node, EState *estate);
static TupleTableSlotChunk *ExecGpuScanChunk(PlanState *node, EState *estate);
// static TupleTableSlot *ExecGpuScanFetch(ScanState *node,
// 										ExecScanAccessMtd accessMtd,
// 										ExecScanRecheckMtd recheckMtd);

void _init_executorRun_hook(void)
{
	elog(INFO, "init executorRun_hook ...");
	PreviousExecutorRunHook = ExecutorRun_hook;
	ExecutorRun_hook = gpu_ExecutorRun;
}

// void _PG_fini(void)
// {
// 	ExecutorRun_hook = PreviousExecutorRunHook;
// }

void gpu_ExecutorRun(QueryDesc *queryDesc,
					 ScanDirection direction,
					 uint64 count,
					 bool execute_once)
{
	EState *estate;
	CmdType operation;
	DestReceiver *dest;
	bool sendTuples;
	MemoryContext oldcontext;

	/* sanity checks */
	Assert(queryDesc != NULL);

	estate = queryDesc->estate;

	Assert(estate != NULL);
	Assert(!(estate->es_top_eflags & EXEC_FLAG_EXPLAIN_ONLY));

	/*
	 * Switch into per-query memory context
	 */
	oldcontext = MemoryContextSwitchTo(estate->es_query_cxt);

	/* Allow instrumentation of Executor overall runtime */
	if (queryDesc->totaltime)
		InstrStartNode(queryDesc->totaltime);

	/*
	 * extract information from the query descriptor and the query feature.
	 */
	operation = queryDesc->operation;
	dest = queryDesc->dest;

	/*
	 * startup tuple receiver, if we will be emitting tuples
	 */
	estate->es_processed = 0;

	sendTuples = (operation == CMD_SELECT ||
				  queryDesc->plannedstmt->hasReturning);

	if (sendTuples)
		dest->rStartup(dest, operation, queryDesc->tupDesc);

	/*
	 * run plan
	 */
	if (!ScanDirectionIsNoMovement(direction))
	{
		if (execute_once && queryDesc->already_executed)
			elog(ERROR, "can't re-execute query flagged for single execution");
		queryDesc->already_executed = true;

		ExecutePlanGPU(estate,
					   queryDesc->planstate,
					   queryDesc->plannedstmt->parallelModeNeeded,
					   operation,
					   sendTuples,
					   count,
					   direction,
					   dest,
					   execute_once);
	}

	/*
	 * Update es_total_processed to keep track of the number of tuples
	 * processed across multiple ExecutorRun() calls.
	 */
	estate->es_total_processed += estate->es_processed;

	/*
	 * shutdown tuple receiver, if we started it
	 */
	if (sendTuples)
		dest->rShutdown(dest);

	if (queryDesc->totaltime)
		InstrStopNode(queryDesc->totaltime, estate->es_processed);

	MemoryContextSwitchTo(oldcontext);
}

/* ----------------------------------------------------------------
 *		ExecutePlanGPU
 *
 *		Processes the query plan using GPU acceleration until we have retrieved
 *		'numberTuples' tuples, moving in the specified direction.
 *
 *		Runs to completion if numberTuples is 0
 * ----------------------------------------------------------------
 */
static void
ExecutePlanGPU(EState *estate,
			   PlanState *planstate,
			   bool use_parallel_mode,
			   CmdType operation,
			   bool sendTuples,
			   uint64 numberTuples,
			   ScanDirection direction,
			   DestReceiver *dest,
			   bool execute_once)
{
	uint64 current_tuple_count;
	TupleTableSlotChunk *chunk;
	TupleTableSlot *slot;
	bool endLoop = false;
	int i;

	/* initialize local variables */
	current_tuple_count = 0;

	/*
	 * Set the direction.
	 */
	estate->es_direction = direction;

	/*
	 * If the plan might potentially be executed multiple times, we must force
	 * it to run without parallelism, because we might exit early.
	 */
	if (!execute_once)
		use_parallel_mode = false;

	estate->es_use_parallel_mode = use_parallel_mode;
	if (use_parallel_mode)
		EnterParallelMode();

	/*
	 * Only for the custom scan, we use chunk load mode
	 */
	if (nodeTag(planstate) == T_CustomScanState)
	{
		elog(INFO, "use chunk load execution");
		/*
		 * Loop until we've processed the proper number of tuples from the plan.
		 */
		for (;;)
		{
			/* Reset the per-output-tuple-chunk exprcontext */
			ResetPerTupleExprContext(estate);

			// TODO: if limit < chunk_size, edit chunk_size dynamicly
			// Execute the plan and obtain a chunk of tuples
			chunk = ExecProcNodeGPU(planstate, estate);

			for (i = 0; i < chunk_size; i++)
			{
				slot = chunk->tuples[i];

				/*
				 * if the tuple is null, then we assume there is nothing more to
				 * process so we just end the loop...
				 */
				if (TupIsNull(slot))
				{
					endLoop = true;
					break;
				}

				/*
				 * If we have a junk filter, then project a new tuple with the junk
				 * removed.
				 *
				 * Store this new "clean" tuple in the junkfilter's resultSlot.
				 * (Formerly, we stored it back over the "dirty" tuple, which is WRONG
				 * because that tuple slot has the wrong descriptor.)
				 */
				if (estate->es_junkFilter != NULL)
					slot = ExecFilterJunk(estate->es_junkFilter, slot);

				/*
				 * If we are supposed to send the tuple somewhere, do so. (In
				 * practice, this is probably always the case at this point.)
				 */
				if (sendTuples)
				{
					/*
					 * If we are not able to send the tuple, we assume the destination
					 * has closed and no more tuples can be sent. If that's the case,
					 * end the loop.
					 */
					if (!dest->receiveSlot(slot, dest))
					{
						endLoop = true;
						break;
					}
				}

				/*
				 * Count tuples processed, if this is a SELECT.  (For other operation
				 * types, the ModifyTable plan node must count the appropriate
				 * events.)
				 */
				if (operation == CMD_SELECT)
					(estate->es_processed)++;

				/*
				 * check our tuple count.. if we've processed the proper number then
				 * quit, else loop again and process more tuples.  Zero numberTuples
				 * means no limit.
				 */
				current_tuple_count++;
				if (numberTuples && numberTuples == current_tuple_count)
				{
					endLoop = true;
					break;
				}
			}

			/*
			 * If endLoop is true here, it means:
			 * 1. There is no more tuples to process, or
			 * 2. The destination has closed and no more tuples can be sent, or
			 * 3. We have processed the proper numver of tuples.
			 * So we end the whole loop here.
			 */
			if (endLoop)
			{
				break;
			}
		}
	}
	else
	{
		for (;;)
		{
			/* Reset the per-output-tuple exprcontext */
			ResetPerTupleExprContext(estate);

			/*
			 * Execute the plan and obtain a tuple
			 */
			slot = ExecProcNode(planstate);

			/*
			 * if the tuple is null, then we assume there is nothing more to
			 * process so we just end the loop...
			 */
			if (TupIsNull(slot))
				break;

			/*
			 * If we have a junk filter, then project a new tuple with the junk
			 * removed.
			 *
			 * Store this new "clean" tuple in the junkfilter's resultSlot.
			 * (Formerly, we stored it back over the "dirty" tuple, which is WRONG
			 * because that tuple slot has the wrong descriptor.)
			 */
			if (estate->es_junkFilter != NULL)
				slot = ExecFilterJunk(estate->es_junkFilter, slot);

			/*
			 * If we are supposed to send the tuple somewhere, do so. (In
			 * practice, this is probably always the case at this point.)
			 */
			if (sendTuples)
			{
				/*
				 * If we are not able to send the tuple, we assume the destination
				 * has closed and no more tuples can be sent. If that's the case,
				 * end the loop.
				 */
				if (!dest->receiveSlot(slot, dest))
					break;
			}

			/*
			 * Count tuples processed, if this is a SELECT.  (For other operation
			 * types, the ModifyTable plan node must count the appropriate
			 * events.)
			 */
			if (operation == CMD_SELECT)
				(estate->es_processed)++;

			/*
			 * check our tuple count.. if we've processed the proper number then
			 * quit, else loop again and process more tuples.  Zero numberTuples
			 * means no limit.
			 */
			current_tuple_count++;
			if (numberTuples && numberTuples == current_tuple_count)
				break;
		}
	}

	/*
	 * If we know we won't need to back up, we can release resources at this
	 * point.
	 */
	if (!(estate->es_top_eflags & EXEC_FLAG_BACKWARD))
		ExecShutdownNode(planstate);

	if (use_parallel_mode)
		ExitParallelMode();
}

static TupleTableSlotChunk *ExecProcNodeGPU(PlanState *node, EState *estate)
{
	if (node->chgParam != NULL) /* something changed? */
		ExecReScan(node);		/* let ReScan handle this */

	TupleTableSlotChunk *chunk;
	// CustomScanState *css = castNode(CustomScanState, node);

	/* Only support custom scan */
	if (nodeTag(node) == T_CustomScanState)
	{
		return ExecGpuScanChunk(node, estate);
	}
	else
	{
		elog(ERROR, "Only support SeqScan for now");
		return chunk;
	}
}

// TupleTableSlotChunk *ExecGpuScanChunk(ScanState *node, EState *estate)
TupleTableSlotChunk *ExecGpuScanChunk(PlanState *node, EState *estate)
{
	ExprContext *econtext;
	ExprState *qual;
	ProjectionInfo *projInfo;
	uint64 current_tuple_count;
	TupleTableSlotChunk *chunk = palloc0(sizeof(TupleTableSlotChunk));
	TupleTableSlot **slots = palloc0(chunk_size * sizeof(TupleTableSlot *));
	CustomScanState *css = castNode(CustomScanState, node);

	// Initialize local variable
	current_tuple_count = 0;

	/*
	 * Fetch data from node
	 */
	qual = css->ss.ps.qual;
	projInfo = css->ss.ps.ps_ProjInfo;
	econtext = css->ss.ps.ps_ExprContext;

	for (;;)
	{
		TupleTableSlot *slot;
		/*
		 * If we have neither a qual to check nor a projection to do, just skip
		 * all the overhead and return the raw scan tuple.
		 */
		if (!qual && !projInfo)
		{
			ResetExprContext(econtext);
			// slot = ExecGpuScanFetch(node, (ExecScanAccessMtd)GpuSeqNext, (ExecScanRecheckMtd)GpuSeqRecheck);
			slot = css->ss.ps.ExecProcNode(node);
			if (TupIsNull(slot))
			{
				break;
			}
		}
		else
		{
			/* interrupt checks are in ExecScanFetch */

			/*
			 * Reset per-tuple memory context to free any expression evaluation
			 * storage allocated in the previous tuple cycle.
			 */
			ResetExprContext(econtext);
			slot = css->ss.ps.ExecProcNode(node);

			/*
			 * if the slot returned by the accessMtd contains NULL, then it means
			 * there is nothing more to scan so we just return an empty slot,
			 * being careful to use the projection result slot so it has correct
			 * tupleDesc.
			 */
			if (TupIsNull(slot))
			{
				if (projInfo)
				{
					slot = ExecClearTuple(projInfo->pi_state.resultslot);
				}
				break;
			}
			else
			{
				/*
				 * place the current tuple into the expr context
				 */
				econtext->ecxt_scantuple = slot;

				/*
				 * check that the current tuple satisfies the qual-clause
				 *
				 * check for non-null qual here to avoid a function call to ExecQual()
				 * when the qual is null ... saves only a few cycles, but they add up
				 * ...
				 */
				if (qual == NULL || ExecQual(qual, econtext))
				{
					/*
					 * Found a satisfactory scan tuple.
					 */
					if (projInfo)
					{
						/*
						 * Form a projection tuple, store it in the result tuple slot
						 * and return it.
						 */
						slot = ExecProject(projInfo);
					}
				}
				else
				{
					InstrCountFiltered1(node, 1);
					continue;
				}
				/*
				 * Tuple fails qual, so free per-tuple memory and try again.
				 */
				ResetExprContext(econtext);
			}
		}
		// Allocate for each slot in chunk
		slots[current_tuple_count] = ExecAllocTableSlot(&estate->es_tupleTable,
														slot->tts_tupleDescriptor,
														slot->tts_ops);
		// Copy the result slot into slots in chunk
		// Cannot use origin slot here because the pointer will be reused in next loop
		ExecCopySlot(slots[current_tuple_count], slot);

		current_tuple_count++;
		if (current_tuple_count == chunk_size)
		{
			break;
		}
	}

	chunk->tuples = slots;
	chunk->length = current_tuple_count;

	return chunk;
}


