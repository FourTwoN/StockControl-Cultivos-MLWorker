"""Aggregate detections from parallel branches.

This step is used as a chord callback to merge and post-process
detections from parallel detection branches.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)


class AggregateDetectionsStep(PipelineStep):
    """Aggregates detections from parallel detection branches.

    Called after chord(group(...)) to merge detection results.
    Can perform additional post-processing like NMS deduplication.
    """

    @property
    def name(self) -> str:
        """Return step name.

        Returns:
            "aggregate_detections"
        """
        return "aggregate_detections"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Aggregate and post-process detections.

        The context already has merged detections from PipelineExecutor.
        This step adds statistics and can perform additional processing.

        Args:
            ctx: Context with merged detections from parallel branches

        Returns:
            Context with aggregation results
        """
        detections = ctx.raw_detections

        # Group detections by source segment
        by_segment: dict[int, list[dict[str, Any]]] = {}
        for det in detections:
            segment_idx = det.get("source_segment_idx", -1)
            if segment_idx not in by_segment:
                by_segment[segment_idx] = []
            by_segment[segment_idx].append(det)

        # Compute statistics
        stats = {
            "total_detections": len(detections),
            "detections_by_segment": {
                str(idx): len(dets) for idx, dets in by_segment.items()
            },
            "segments_with_detections": len(by_segment),
        }

        logger.info(
            "Aggregated detections",
            image_id=ctx.image_id,
            total=stats["total_detections"],
            segments=stats["segments_with_detections"],
        )

        return ctx.with_results({"aggregation": stats})
