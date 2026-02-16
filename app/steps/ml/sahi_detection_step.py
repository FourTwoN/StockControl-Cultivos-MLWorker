"""SAHI detection step wrapper.

Wraps SAHIDetectorProcessor to implement PipelineStep interface.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SAHIDetectionStep(PipelineStep):
    """SAHI detection step that detects plants using tiling.

    Wraps SAHIDetectorProcessor and converts results to dicts for context.
    Reads SAHI configuration from context.config.
    """

    @property
    def name(self) -> str:
        """Return step name.

        Returns:
            "sahi_detection"
        """
        return "sahi_detection"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute SAHI detection and return context with detections.

        Args:
            ctx: Current processing context with SAHI config

        Returns:
            New context with raw_detections populated

        Raises:
            RuntimeError: If SAHI detection fails
        """
        # Read SAHI config from context (with defaults)
        slice_height = ctx.config.get("sahi_slice_height", 512)
        slice_width = ctx.config.get("sahi_slice_width", 512)
        overlap_ratio = ctx.config.get("sahi_overlap_ratio", 0.25)

        logger.info(
            "Running SAHI detection",
            image_id=ctx.image_id,
            image_path=ctx.image_path.name,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
        )

        try:
            # Get SAHI detector processor from registry
            processor = get_processor_registry().get("sahi_detection")

            # Run SAHI detection with config
            detection_results = await processor.process(
                ctx.image_path,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=overlap_ratio,
            )

            # Convert DetectionResult objects to dicts
            detection_dicts: list[dict[str, Any]] = [
                det.to_dict() for det in detection_results
            ]

            logger.info(
                "SAHI detection completed",
                image_id=ctx.image_id,
                detections_found=len(detection_dicts),
            )

            # Return new context with detections
            return ctx.with_detections(detection_dicts)

        except Exception as e:
            logger.error(
                "SAHI detection step failed",
                image_id=ctx.image_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"SAHI detection step failed: {e}") from e
