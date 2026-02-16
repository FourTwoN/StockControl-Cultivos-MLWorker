"""Detection step wrapper.

Wraps DetectorProcessor to implement PipelineStep interface.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


class DetectionStep(PipelineStep):
    """Detection step that detects plants in images.

    Wraps DetectorProcessor and converts results to dicts for context.
    """

    @property
    def name(self) -> str:
        """Return step name.

        Returns:
            "detection"
        """
        return "detection"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute detection and return context with detections.

        Args:
            ctx: Current processing context

        Returns:
            New context with raw_detections populated

        Raises:
            RuntimeError: If detection fails
        """
        logger.info(
            "Running detection",
            image_id=ctx.image_id,
            image_path=ctx.image_path.name,
        )

        try:
            # Get detection processor from registry
            processor = get_processor_registry().get("detection")

            # Run detection
            detection_results = await processor.process(ctx.image_path)

            # Convert DetectionResult objects to dicts
            detection_dicts: list[dict[str, Any]] = [
                det.to_dict() for det in detection_results
            ]

            logger.info(
                "Detection completed",
                image_id=ctx.image_id,
                detections_found=len(detection_dicts),
            )

            # Return new context with detections
            return ctx.with_detections(detection_dicts)

        except Exception as e:
            logger.error(
                "Detection step failed",
                image_id=ctx.image_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Detection step failed: {e}") from e
