"""Pipeline executor for DSL primitives.

Interprets Chain, Group, and Chord structures and executes
steps with proper parallelization using asyncio.gather.
"""

import asyncio
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
        logger.debug("Executing chain", steps=len(chain.steps))

        for element in chain.steps:
            ctx = await self.execute(element, ctx)

        return ctx

    async def _execute_step(
        self,
        sig: StepSignature,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute a single step.

        Injects step kwargs into context.step_config before execution.
        """
        logger.debug("Executing step", step=sig.name, kwargs=sig.kwargs)

        # Get step instance from registry
        step_instance = StepRegistry.get(sig.name)

        # Inject step-specific config
        if sig.kwargs:
            ctx = ctx.with_step_config(sig.kwargs)

        # Execute step
        result = await step_instance.execute(ctx)

        return result

    async def _execute_group(
        self,
        grp: Group,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute steps in parallel with asyncio.gather.

        All branches receive the same input context.
        Results are merged into a single output context.
        """
        logger.debug("Executing group", branches=len(grp.steps))

        # Create tasks for parallel execution
        tasks = [self.execute(element, ctx) for element in grp.steps]

        # Execute all branches in parallel
        results: list[ProcessingContext] = await asyncio.gather(*tasks)

        # Merge results back into original context
        merged = self._merge_contexts(ctx, results)

        return merged

    async def _execute_chord(
        self,
        chrd: Chord,
        ctx: ProcessingContext,
    ) -> ProcessingContext:
        """Execute group in parallel, then run callback.

        The callback receives the merged context from all branches.
        """
        logger.debug(
            "Executing chord",
            branches=len(chrd.group.steps),
            callback=chrd.callback.name if chrd.callback else None,
        )

        # Execute group first
        ctx = await self._execute_group(chrd.group, ctx)

        # Execute callback if present
        if chrd.callback:
            ctx = await self._execute_step(chrd.callback, ctx)

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

        for result_ctx in results:
            all_detections.extend(result_ctx.raw_detections)
            all_classifications.extend(result_ctx.raw_classifications)
            merged_results.update(result_ctx.results)

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
