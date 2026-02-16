"""Segmentation step wrapper.

Wraps SegmentationProcessor to implement PipelineStep interface.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SegmentationStep(PipelineStep):
    """Segmentation step that extracts segments from images.

    Wraps SegmentationProcessor and converts results to dicts for context.
    """

    @property
    def name(self) -> str:
        """Return step name.

        Returns:
            "segmentation"
        """
        return "segmentation"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute segmentation and return context with segments.

        Args:
            ctx: Current processing context

        Returns:
            New context with raw_segments populated

        Raises:
            RuntimeError: If segmentation fails
        """
        logger.info(
            "Running segmentation",
            image_id=ctx.image_id,
            image_path=ctx.image_path.name,
        )

        try:
            # Get segmentation processor from registry
            processor = get_processor_registry().get("segmentation")

            # Run segmentation
            segment_results = await processor.process(ctx.image_path)

            # Convert SegmentResult objects to dicts
            segment_dicts: list[dict[str, Any]] = [
                seg.to_dict() for seg in segment_results
            ]

            logger.info(
                "Segmentation completed",
                image_id=ctx.image_id,
                segments_found=len(segment_dicts),
            )

            # Return new context with segments
            return ctx.with_segments(segment_dicts)

        except Exception as e:
            logger.error(
                "Segmentation step failed",
                image_id=ctx.image_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Segmentation step failed: {e}") from e
