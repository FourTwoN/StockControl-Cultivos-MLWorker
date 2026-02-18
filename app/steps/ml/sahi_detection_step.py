"""SAHI detection step wrapper.

Wraps SAHIDetectorProcessor to implement PipelineStep interface.
Supports segment-aware detection when segment_type is specified in step_config.
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

        If segment_type is specified in step_config, only processes
        segments of that type and transforms coordinates to full image.

        Args:
            ctx: Current processing context with SAHI config

        Returns:
            New context with raw_detections populated

        Raises:
            RuntimeError: If SAHI detection fails
        """
        segment_type = ctx.step_config.get("segment_type")

        if segment_type:
            return await self._execute_segment_aware(ctx, segment_type)
        return await self._execute_full_image(ctx)

    def _get_sahi_config(self, ctx: ProcessingContext) -> tuple[int, int, float]:
        """Get SAHI configuration from context."""
        slice_height = ctx.config.get("sahi_slice_height", 512)
        slice_width = ctx.config.get("sahi_slice_width", 512)
        overlap_ratio = ctx.config.get("sahi_overlap_ratio", 0.25)
        return slice_height, slice_width, overlap_ratio

    async def _execute_full_image(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute SAHI detection on full image."""
        slice_height, slice_width, overlap_ratio = self._get_sahi_config(ctx)

        logger.info(
            "Running SAHI detection on full image",
            image_id=ctx.image_id,
            image_path=ctx.image_path.name,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
        )

        try:
            processor = get_processor_registry().get("sahi_detection")
            detection_results = await processor.process(
                ctx.image_path,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=overlap_ratio,
            )

            detection_dicts: list[dict[str, Any]] = [
                det.to_dict() for det in detection_results
            ]

            logger.info(
                "SAHI detection completed",
                image_id=ctx.image_id,
                detections_found=len(detection_dicts),
            )

            return ctx.with_detections(detection_dicts)

        except Exception as e:
            logger.error(
                "SAHI detection step failed",
                image_id=ctx.image_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"SAHI detection step failed: {e}") from e

    async def _execute_segment_aware(
        self,
        ctx: ProcessingContext,
        segment_type: str,
    ) -> ProcessingContext:
        """Execute SAHI detection on segments of specific type.

        Filters segments by class_name, processes each crop with tiling,
        and transforms coordinates back to full image space.
        """
        # Filter segments by type
        target_segments = [
            s for s in ctx.raw_segments if s.get("class_name") == segment_type
        ]

        if not target_segments:
            logger.info(
                "No segments of type found for SAHI",
                segment_type=segment_type,
                image_id=ctx.image_id,
            )
            return ctx.with_detections([])

        slice_height, slice_width, overlap_ratio = self._get_sahi_config(ctx)

        logger.info(
            "Running segment-aware SAHI detection",
            image_id=ctx.image_id,
            segment_type=segment_type,
            segment_count=len(target_segments),
            slice_height=slice_height,
            slice_width=slice_width,
        )

        processor = get_processor_registry().get("sahi_detection")
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
                # Run SAHI detection on crop
                results = await processor.process(
                    crop_path,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_ratio=overlap_ratio,
                )

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
                    "SAHI detection failed for segment",
                    segment_idx=segment_idx,
                    error=str(e),
                )
                continue

        logger.info(
            "Segment-aware SAHI detection completed",
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
