"""Detection step wrapper.

Wraps DetectorProcessor to implement PipelineStep interface.
Supports segment-aware detection when segment_type is specified in step_config.
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

        If segment_type is specified in step_config, only processes
        segments of that type and transforms coordinates to full image.

        Args:
            ctx: Current processing context

        Returns:
            New context with raw_detections populated

        Raises:
            RuntimeError: If detection fails
        """
        segment_type = ctx.step_config.get("segment_type")

        if segment_type:
            return await self._execute_segment_aware(ctx, segment_type)
        return await self._execute_full_image(ctx)

    async def _execute_full_image(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute detection on full image."""
        logger.info(
            "Running detection on full image",
            image_id=ctx.image_id,
            image_path=ctx.image_path.name,
        )

        try:
            processor = get_processor_registry().get("detection")
            detection_results = await processor.process(ctx.image_path)

            detection_dicts: list[dict[str, Any]] = [
                det.to_dict() for det in detection_results
            ]

            logger.info(
                "Detection completed",
                image_id=ctx.image_id,
                detections_found=len(detection_dicts),
            )

            return ctx.with_detections(detection_dicts)

        except Exception as e:
            logger.error(
                "Detection step failed",
                image_id=ctx.image_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Detection step failed: {e}") from e

    async def _execute_segment_aware(
        self,
        ctx: ProcessingContext,
        segment_type: str,
    ) -> ProcessingContext:
        """Execute detection on segments of specific type.

        Filters segments by class_name, processes each crop,
        and transforms coordinates back to full image space.
        """
        # Filter segments by type
        target_segments = [
            s for s in ctx.raw_segments if s.get("class_name") == segment_type
        ]

        if not target_segments:
            logger.info(
                "No segments of type found",
                segment_type=segment_type,
                image_id=ctx.image_id,
            )
            return ctx.with_detections([])

        logger.info(
            "Running segment-aware detection",
            image_id=ctx.image_id,
            segment_type=segment_type,
            segment_count=len(target_segments),
        )

        processor = get_processor_registry().get("detection")
        all_detections: list[dict[str, Any]] = []

        for segment in target_segments:
            segment_idx = segment.get("segment_idx")
            crop_path = ctx.segment_crops.get(segment_idx)

            if not crop_path or not crop_path.exists():
                logger.warning(
                    "No crop found for segment",
                    segment_idx=segment_idx,
                    segment_type=segment_type,
                )
                continue

            try:
                # Run detection on crop
                results = await processor.process(crop_path)

                # Transform coordinates to full image
                bbox = segment.get("bbox", [0, 0, 0, 0])
                offset_x, offset_y = bbox[0], bbox[1]

                for det in results:
                    det_dict = det.to_dict()
                    det_dict = self._transform_to_full_image(
                        det_dict, offset_x, offset_y, segment_idx
                    )
                    all_detections.append(det_dict)

            except Exception as e:
                logger.error(
                    "Detection failed for segment",
                    segment_idx=segment_idx,
                    error=str(e),
                )
                continue

        logger.info(
            "Segment-aware detection completed",
            image_id=ctx.image_id,
            segment_type=segment_type,
            total_detections=len(all_detections),
        )

        return ctx.with_detections(all_detections)

    def _transform_to_full_image(
        self,
        detection: dict[str, Any],
        offset_x: float,
        offset_y: float,
        segment_idx: int,
    ) -> dict[str, Any]:
        """Transform detection coordinates from crop to full image.

        Args:
            detection: Detection dict with bbox in crop coordinates
            offset_x: X offset of crop in full image
            offset_y: Y offset of crop in full image
            segment_idx: Source segment index

        Returns:
            Detection dict with transformed coordinates
        """
        result = {**detection, "source_segment_idx": segment_idx}

        if "bbox" in detection:
            bbox = detection["bbox"]
            result["bbox"] = [
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2] + offset_x,
                bbox[3] + offset_y,
            ]

        return result
