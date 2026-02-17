"""Segment filtering post-processor.

Filters segments based on tenant-specific rules and generates
cropped images for downstream detection steps.
"""

import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

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
        """Execute segment filtering and generate crops.

        Args:
            ctx: Current processing context with raw_segments

        Returns:
            New context with filtered segments and crop paths
        """
        filter_type = ctx.config.get("segment_filter_type")

        if not filter_type:
            logger.debug("No segment filter type specified, skipping filter")
            # Still generate crops for detection steps
            return await self._generate_crops(ctx)

        if filter_type == "largest_claro":
            filtered_segments = self._filter_largest_claro(ctx.raw_segments)
            logger.info(
                "Filtered segments using largest_claro",
                original_count=len(ctx.raw_segments),
                filtered_count=len(filtered_segments),
            )
            ctx = ctx.with_segments(filtered_segments)
            # Generate crops for filtered segments
            return await self._generate_crops(ctx)

        logger.warning(
            "Unknown segment filter type",
            filter_type=filter_type,
        )
        return await self._generate_crops(ctx)

    async def _generate_crops(self, ctx: ProcessingContext) -> ProcessingContext:
        """Generate cropped images for each segment.

        Crops are saved to temp files and paths stored in context.

        Args:
            ctx: Context with segments to crop

        Returns:
            Context with segment_crops populated
        """
        if not ctx.raw_segments:
            return ctx

        # Load original image
        image = cv2.imread(str(ctx.image_path))
        if image is None:
            logger.warning("Could not load image for cropping", path=ctx.image_path)
            return ctx

        crops: dict[int, Path] = {}

        for segment in ctx.raw_segments:
            segment_idx = segment.get("segment_idx")
            if segment_idx is None:
                continue

            crop_path = self._crop_segment(image, segment, ctx.session_id)
            if crop_path:
                crops[segment_idx] = crop_path
                logger.debug(
                    "Generated crop",
                    segment_idx=segment_idx,
                    class_name=segment.get("class_name"),
                    crop_path=str(crop_path),
                )

        logger.info("Generated segment crops", count=len(crops))
        return ctx.with_segment_crops(crops)

    def _crop_segment(
        self,
        image: np.ndarray,
        segment: dict[str, Any],
        session_id: str,
    ) -> Path | None:
        """Crop a segment from the image using its bounding box.

        Args:
            image: Original image as numpy array
            segment: Segment dict with bbox
            session_id: Session ID for temp file naming

        Returns:
            Path to cropped image file, or None if failed
        """
        bbox = segment.get("bbox")
        if not bbox or len(bbox) != 4:
            return None

        segment_idx = segment.get("segment_idx", 0)

        # Extract bbox coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]

        # Ensure bounds are within image
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning("Invalid crop bounds", segment_idx=segment_idx, bbox=bbox)
            return None

        # Crop the region
        crop = image[y1:y2, x1:x2]

        # Save to temp file
        crop_path = Path(tempfile.gettempdir()) / f"{session_id}_crop_{segment_idx}.jpg"
        cv2.imwrite(str(crop_path), crop)

        return crop_path

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
