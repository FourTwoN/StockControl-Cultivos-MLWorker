"""Size calculation post-processor.

Calculates plant sizes using z-score based classification.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)

# Size constants
SIZE_S = 1
SIZE_M = 2
SIZE_L = 3
SIZE_XL = 4


class SizeCalculatorStep(PipelineStep):
    """Calculates plant sizes based on detection dimensions.

    Uses z-score normalization to classify plants into size categories
    (S, M, L, XL) based on their bounding box areas.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this step.

        Returns:
            String identifier "size_calculator"
        """
        return "size_calculator"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Calculate sizes for all detections.

        Args:
            ctx: Current processing context with raw_detections

        Returns:
            New context with sizes added to results
        """
        detections = ctx.raw_detections

        if not detections:
            logger.debug("No detections to size")
            return ctx.with_results({"sizes": {}})

        sizes = self._calculate_sizes(detections)

        logger.info(
            "Calculated sizes for detections",
            detection_count=len(detections),
            size_distribution={
                f"SIZE_{size_name}": sum(1 for s in sizes.values() if s == size_id)
                for size_name, size_id in [
                    ("S", SIZE_S),
                    ("M", SIZE_M),
                    ("L", SIZE_L),
                    ("XL", SIZE_XL),
                ]
            },
        )

        return ctx.with_results({"sizes": sizes})

    def _calculate_sizes(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[int, int]:
        """Calculate size for each detection using z-scores.

        Args:
            detections: List of detection dictionaries with bbox

        Returns:
            Dictionary mapping detection index to size ID
        """
        if len(detections) == 1:
            # Single detection gets medium size
            return {0: SIZE_M}

        # Calculate normalized areas
        areas = []
        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            areas.append(area)

        # Calculate z-scores
        mean_area = sum(areas) / len(areas)
        variance = sum((a - mean_area) ** 2 for a in areas) / len(areas)
        std_dev = variance**0.5

        if std_dev == 0:
            # All same size - assign medium
            return dict.fromkeys(range(len(detections)), SIZE_M)

        z_scores = [(area - mean_area) / std_dev for area in areas]

        # Classify based on z-score thresholds (Demeter original values)
        # z < -2.0: S (very small)
        # -1.0 <= z <= 1.0: M (modal/medium)
        # z > 2.0: XL (very large)
        # L fills the gap between M and XL
        sizes = {}
        for idx, z_score in enumerate(z_scores):
            if z_score < -2.0:
                sizes[idx] = SIZE_S
            elif z_score <= 1.0:
                sizes[idx] = SIZE_M
            elif z_score <= 2.0:
                sizes[idx] = SIZE_L
            else:
                sizes[idx] = SIZE_XL

        return sizes
