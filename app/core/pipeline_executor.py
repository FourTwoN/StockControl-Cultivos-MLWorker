"""Pipeline executor for DSL primitives.

Interprets Chain, Group, and Chord structures and executes
steps with proper parallelization using asyncio.gather.
"""

import asyncio
import time
from typing import Any

from app.core.pipeline_dsl import Chain, Chord, Group, PipelineElement, StepSignature
from app.core.processing_context import ProcessingContext
from app.core.step_registry import StepRegistry
from app.infra.logging import get_logger


logger = get_logger(__name__)


class PipelineExecutor:
    """Executes pipeline DSL structures.

    Traverses the pipeline tree and executes steps according
    to their composition (chain, group, chord).
    """

    async def execute(
        self,
        pipeline: PipelineElement,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute a pipeline element.

        Dispatches to appropriate handler based on element type.

        Args:
            pipeline: Any pipeline element (Step, Chain, Group, Chord)
            ctx: Initial processing context

        Returns:
            Final context after pipeline execution
        """
        if isinstance(pipeline, Chain):
            return await self._execute_chain(pipeline, ctx)
        elif isinstance(pipeline, Group):
            return await self._execute_group(pipeline, ctx)
        elif isinstance(pipeline, Chord):
            return await self._execute_chord(pipeline, ctx)
        elif isinstance(pipeline, StepSignature):
            return await self._execute_step(pipeline, ctx)
        else:
            raise TypeError(f"Unknown pipeline element type: {type(pipeline)}")

    async def _execute_chain(
        self,
        chain: Chain,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute steps sequentially.

        Each step receives the context from the previous step.
        """
        chain_start = time.perf_counter()
        step_count = len(chain.steps)

        logger.info(
            "Pipeline chain started",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            total_steps=step_count,
        )

        for idx, element in enumerate(chain.steps, 1):
            step_name = self._get_element_name(element)
            logger.info(
                f"Step {idx}/{step_count} starting",
                tenant_id=ctx.tenant_id,
                image_id=ctx.image_id,
                step=step_name,
                step_number=idx,
            )
            ctx = await self.execute(element, ctx)

        chain_duration_ms = int((time.perf_counter() - chain_start) * 1000)
        logger.info(
            "Pipeline chain completed",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            total_steps=step_count,
            duration_ms=chain_duration_ms,
        )

        return ctx

    def _get_element_name(self, element: PipelineElement) -> str:
        """Get descriptive name for a pipeline element."""
        if isinstance(element, StepSignature):
            return element.name
        elif isinstance(element, Chain):
            return f"chain({len(element.steps)} steps)"
        elif isinstance(element, Group):
            return f"group({len(element.steps)} branches)"
        elif isinstance(element, Chord):
            return f"chord({len(element.group.steps)} branches)"
        return type(element).__name__

    async def _execute_step(
        self,
        sig: StepSignature,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute a single step.

        Injects step kwargs into context.step_config before execution.
        """
        step_start = time.perf_counter()

        logger.debug(
            "Step execution starting",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            step=sig.name,
            kwargs=sig.kwargs,
        )

        try:
            # Get step instance from registry
            step_instance = StepRegistry.get(sig.name)

            # Inject step-specific config
            if sig.kwargs:
                ctx = ctx.with_step_config(sig.kwargs)

            # Execute step
            result = await step_instance.execute(ctx)

            step_duration_ms = int((time.perf_counter() - step_start) * 1000)
            logger.info(
                "Step completed successfully",
                tenant_id=ctx.tenant_id,
                image_id=ctx.image_id,
                step=sig.name,
                duration_ms=step_duration_ms,
            )

            return result

        except Exception as e:
            step_duration_ms = int((time.perf_counter() - step_start) * 1000)
            logger.error(
                "Step execution FAILED",
                tenant_id=ctx.tenant_id,
                image_id=ctx.image_id,
                step=sig.name,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=step_duration_ms,
                exc_info=True,
            )
            raise

    async def _execute_group(
        self,
        grp: Group,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute steps in parallel with asyncio.gather.

        All branches receive the same input context.
        Results are merged into a single output context.
        """
        group_start = time.perf_counter()
        branch_names = [self._get_element_name(e) for e in grp.steps]

        logger.info(
            "Parallel group started",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            branch_count=len(grp.steps),
            branches=branch_names,
        )

        # Create tasks for parallel execution
        tasks = [self.execute(element, ctx) for element in grp.steps]

        # Execute all branches in parallel
        results: list[ProcessingContext] = await asyncio.gather(*tasks)

        # Merge results back into original context
        merged = self._merge_contexts(ctx, results)

        group_duration_ms = int((time.perf_counter() - group_start) * 1000)

        # Log summary of parallel execution
        total_detections = len(merged.raw_detections)
        logger.info(
            "Parallel group completed",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            branch_count=len(grp.steps),
            total_detections_merged=total_detections,
            duration_ms=group_duration_ms,
        )

        return merged

    async def _execute_chord(
        self,
        chrd: Chord,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute group in parallel, then run callback.

        The callback receives the merged context from all branches.
        """
        chord_start = time.perf_counter()
        callback_name = chrd.callback.name if chrd.callback else "none"

        logger.info(
            "Chord execution started (parallel + callback)",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            branch_count=len(chrd.group.steps),
            callback=callback_name,
        )

        # Execute group first
        ctx = await self._execute_group(chrd.group, ctx)

        # Execute callback if present
        if chrd.callback:
            logger.info(
                "Chord callback starting",
                tenant_id=ctx.tenant_id,
                image_id=ctx.image_id,
                callback=callback_name,
                detections_to_aggregate=len(ctx.raw_detections),
            )
            ctx = await self._execute_step(chrd.callback, ctx)

        chord_duration_ms = int((time.perf_counter() - chord_start) * 1000)
        logger.info(
            "Chord completed",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            callback=callback_name,
            duration_ms=chord_duration_ms,
        )

        return ctx

    def _merge_contexts(
        self,
        original: ProcessingContext,
        results: list[ProcessingContext],
    ) -> ProcessingContext:
        """Merge results from parallel branches.

        Combines detections, classifications, and results from all branches.
        Segments are preserved from original (not modified in parallel branches).

        Args:
            original: Context before parallel execution
            results: Contexts from each parallel branch

        Returns:
            Merged context with combined results
        """
        # Collect all detections from branches
        all_detections: list[dict[str, Any]] = []
        all_classifications: list[dict[str, Any]] = []
        merged_results: dict[str, Any] = {}

        branch_summaries = []
        for idx, result_ctx in enumerate(results):
            det_count = len(result_ctx.raw_detections)
            all_detections.extend(result_ctx.raw_detections)
            all_classifications.extend(result_ctx.raw_classifications)
            merged_results.update(result_ctx.results)
            branch_summaries.append(f"branch_{idx}={det_count}")

        logger.debug(
            "Merging parallel branch results",
            tenant_id=original.tenant_id,
            image_id=original.image_id,
            branch_count=len(results),
            detections_per_branch=", ".join(branch_summaries),
            total_detections=len(all_detections),
        )

        # Build merged context preserving segments from original
        return ProcessingContext(
            tenant_id=original.tenant_id,
            image_id=original.image_id,
            session_id=original.session_id,
            image_path=original.image_path,
            config=original.config,
            raw_segments=original.raw_segments,
            raw_detections=all_detections,
            raw_classifications=all_classifications,
            results={**original.results, **merged_results},
            step_config={},  # Clear step config after merge
            segment_crops=original.segment_crops,
        )
