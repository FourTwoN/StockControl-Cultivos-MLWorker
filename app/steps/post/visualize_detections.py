"""Visualization post-processor.

Draws detections on the original image and saves to storage.
"""

from pathlib import Path
from typing import Any

import cv2

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)

# Colors for different size categories (BGR format)
SIZE_COLORS = {
    1: (255, 0, 0),    # SIZE_S - Blue
    2: (0, 255, 0),    # SIZE_M - Green
    3: (0, 165, 255),  # SIZE_L - Orange
    4: (0, 0, 255),    # SIZE_XL - Red
}

DEFAULT_COLOR = (0, 255, 0)  # Green


class VisualizeDetectionsStep(PipelineStep):
    """Draws detection bounding boxes on the original image.

    Handles coordinate transformation from crop-relative to image-absolute
    coordinates using segment bounding boxes.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this step.

        Returns:
            String identifier "visualize_detections"
        """
        return "visualize_detections"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Draw detections on image and save.

        Args:
            ctx: Current processing context with raw_detections, raw_segments, and sizes

        Returns:
            Context with visualization_path added to results
        """
        detections = ctx.raw_detections
        segments = ctx.raw_segments
        sizes = ctx.results.get("sizes", {})

        if not detections:
            logger.debug("No detections to visualize")
            return ctx

        # Load original image
        image = cv2.imread(str(ctx.image_path))
        if image is None:
            logger.warning("Could not load image for visualization", path=ctx.image_path)
            return ctx

        # Build segment offset lookup (segment_idx -> (offset_x, offset_y))
        segment_offsets = self._build_segment_offsets(segments)

        # Draw each detection
        drawn_count = 0
        for idx, detection in enumerate(detections):
            bbox = self._get_absolute_bbox(detection, segment_offsets)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            size_id = sizes.get(idx, 2)
            color = SIZE_COLORS.get(size_id, DEFAULT_COLOR)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            drawn_count += 1

        # Draw summary
        summary = f"Detections: {drawn_count}"
        cv2.putText(
            image,
            summary,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

        # Save annotated image
        output_dir = Path(ctx.config.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ctx.session_id}_detections.jpg"

        cv2.imwrite(str(output_path), image)

        logger.info(
            "Saved visualization",
            output_path=str(output_path),
            detection_count=drawn_count,
        )

        return ctx.with_results({"visualization_path": str(output_path)})

    def _build_segment_offsets(
        self, segments: list[dict[str, Any]]
    ) -> dict[int, tuple[float, float]]:
        """Build lookup of segment index to crop offset.

        The offset is the top-left corner of the segment's bounding box,
        which needs to be added to crop-relative detection coordinates.

        Args:
            segments: List of segment dictionaries with bbox and segment_idx

        Returns:
            Dictionary mapping segment_idx to (offset_x, offset_y)
        """
        offsets: dict[int, tuple[float, float]] = {}

        for segment in segments:
            segment_idx = segment.get("segment_idx")
            bbox = segment.get("bbox", [])

            if segment_idx is not None and len(bbox) >= 2:
                # bbox is [x1, y1, x2, y2], offset is top-left corner
                offsets[segment_idx] = (bbox[0], bbox[1])

        return offsets

    def _get_absolute_bbox(
        self,
        detection: dict[str, Any],
        segment_offsets: dict[int, tuple[float, float]],
    ) -> tuple[int, int, int, int] | None:
        """Convert detection to absolute image coordinates.

        Detections use center_x_px, center_y_px, width_px, height_px
        relative to their source segment crop. This method converts
        to absolute x1, y1, x2, y2 coordinates.

        Args:
            detection: Detection dictionary
            segment_offsets: Lookup of segment_idx to (offset_x, offset_y)

        Returns:
            Tuple of (x1, y1, x2, y2) in absolute image coordinates, or None
        """
        # Get detection dimensions (crop-relative)
        center_x = detection.get("center_x_px")
        center_y = detection.get("center_y_px")
        width = detection.get("width_px")
        height = detection.get("height_px")

        if None in (center_x, center_y, width, height):
            # Try bbox format as fallback
            bbox = detection.get("bbox", [])
            if len(bbox) == 4:
                return tuple(int(round(c)) for c in bbox)
            return None

        # Calculate bbox from center and dimensions
        half_w = width / 2
        half_h = height / 2
        x1 = center_x - half_w
        y1 = center_y - half_h
        x2 = center_x + half_w
        y2 = center_y + half_h

        # Apply segment offset if available
        source_segment_idx = detection.get("source_segment_idx")
        if source_segment_idx is not None and source_segment_idx in segment_offsets:
            offset_x, offset_y = segment_offsets[source_segment_idx]
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y

        return (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
