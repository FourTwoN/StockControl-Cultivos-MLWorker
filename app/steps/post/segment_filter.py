"""Segment filtering post-processor.

Filters segments based on tenant-specific rules.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SegmentFilterStep(PipelineStep):
    """Filters segments based on configuration rules.

    Supports filtering strategies like keeping only the largest
    segment of specific types while preserving others.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this step.

        Returns:
            String identifier "segment_filter"
        """
        return "segment_filter"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute segment filtering.

        Args:
            ctx: Current processing context with raw_segments

        Returns:
            New context with filtered segments
        """
        filter_type = ctx.config.get("segment_filter_type")

        if not filter_type:
            logger.debug("No segment filter type specified, skipping filter")
            return ctx

        if filter_type == "largest_claro":
            filtered_segments = self._filter_largest_claro(ctx.raw_segments)
            logger.info(
                "Filtered segments using largest_claro",
                original_count=len(ctx.raw_segments),
                filtered_count=len(filtered_segments),
            )
            return ctx.with_segments(filtered_segments)

        logger.warning(
            "Unknown segment filter type",
            filter_type=filter_type,
        )
        return ctx

    def _filter_largest_claro(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter to keep only largest segmento/claro-cajon.

        Keeps the largest segment of type 'segmento' or 'claro-cajon',
        and preserves all other segment types unchanged.

        Args:
            segments: List of segment dictionaries

        Returns:
            Filtered list of segments
        """
        target_types = {"segmento", "claro-cajon"}

        # Separate target segments from others
        target_segments = [s for s in segments if s.get("type") in target_types]
        other_segments = [s for s in segments if s.get("type") not in target_types]

        if not target_segments:
            return segments

        # Find largest target segment by area
        largest = max(target_segments, key=lambda s: s.get("area", 0))

        # Return largest target + all other types
        return [largest, *other_segments]
