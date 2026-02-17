"""Visualization post-processor.

Draws detections on the original image and saves to storage.
"""

from pathlib import Path

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

    Saves the annotated image to the output directory configured
    in the processing context.
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
            ctx: Current processing context with raw_detections and sizes

        Returns:
            Context with visualization_path added to results
        """
        detections = ctx.raw_detections
        sizes = ctx.results.get("sizes", {})

        if not detections:
            logger.debug("No detections to visualize")
            return ctx

        # Load original image
        image = cv2.imread(str(ctx.image_path))
        if image is None:
            logger.warning("Could not load image for visualization", path=ctx.image_path)
            return ctx

        # Draw each detection
        for idx, detection in enumerate(detections):
            bbox = detection.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
            size_id = sizes.get(idx, 2)
            color = SIZE_COLORS.get(size_id, DEFAULT_COLOR)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw size label
            label = f"S{size_id}"
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 4),
                (x1 + text_width + 4, y1),
                color,
                -1,
            )

            # Text
            cv2.putText(
                image,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        # Draw summary
        summary = f"Detections: {len(detections)}"
        cv2.putText(
            image,
            summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Save annotated image
        output_dir = Path(ctx.config.get("output_dir", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{ctx.session_id}_detections.jpg"

        cv2.imwrite(str(output_path), image)

        logger.info(
            "Saved visualization",
            output_path=str(output_path),
            detection_count=len(detections),
        )

        return ctx.with_results({"visualization_path": str(output_path)})
