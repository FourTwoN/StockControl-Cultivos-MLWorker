"""Segmentation Processor - YOLO segmentation for region extraction.

Extracts segments/regions from images using YOLO segmentation model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.infra.logging import get_logger
from app.ml.model_cache import ModelCache
from app.processors.base_processor import BaseProcessor

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
else:
    Results = Any

logger = get_logger(__name__)


@dataclass(frozen=True)
class SegmentResult:
    """Single segmentation result.

    Attributes:
        segment_idx: Segment index
        class_name: Segment class name
        confidence: Segmentation confidence
        bbox: Bounding box (x1, y1, x2, y2) in pixels
        mask_rle: Run-length encoded mask (optional)
        area_px: Segment area in pixels
    """

    segment_idx: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    area_px: float
    mask_rle: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "segment_idx": self.segment_idx,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "area_px": self.area_px,
            "mask_rle": self.mask_rle,
        }


class SegmentationProcessor(BaseProcessor[list[SegmentResult]]):
    """YOLO segmentation processor for region extraction.

    Uses YOLO segmentation model to identify and extract regions of interest.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        worker_id: int = 0,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialize segmentation processor.

        Args:
            model_path: Path to YOLO segmentation model
            worker_id: GPU worker ID
            confidence_threshold: Minimum confidence score
        """
        super().__init__(model_path, worker_id, confidence_threshold)
        self._model: Any = None
        self._worker_id_cached: int | None = None

    async def process(
        self,
        image_path: str | Path,
        classes: list[str] | None = None,
    ) -> list[SegmentResult]:
        """Segment image to extract regions.

        Args:
            image_path: Path to image file
            classes: Optional list of class names to filter results

        Returns:
            List of SegmentResult objects

        Raises:
            FileNotFoundError: If image_path doesn't exist
            RuntimeError: If segmentation fails
        """
        # Validate image path
        image_path = self._validate_image_path(image_path)

        # Get model from singleton (lazy load)
        if self._model is None or self._worker_id_cached != self.worker_id:
            logger.info("Loading segmentation model", worker_id=self.worker_id)
            self._model = ModelCache.get_model("segment", self.worker_id)
            self._worker_id_cached = self.worker_id
            logger.info("Segmentation model loaded")

        # Run YOLO segmentation
        try:
            logger.debug(
                "Running segmentation",
                image=image_path.name,
                confidence_threshold=self.confidence_threshold,
            )

            results = self._model.predict(
                source=str(image_path),
                conf=self.confidence_threshold,
                verbose=False,
                device=None,
            )

            # Parse results
            segments = self._parse_results(results[0], classes)

            logger.info(
                "Segmentation complete",
                image=image_path.name,
                segments=len(segments),
            )

            return segments

        except Exception as e:
            logger.error(
                "Segmentation failed",
                image=image_path.name,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Segmentation failed: {e}") from e

    def _parse_results(
        self,
        result: "Results",
        classes: list[str] | None = None,
    ) -> list[SegmentResult]:
        """Parse YOLO segmentation results.

        Args:
            result: YOLO Results object
            classes: Optional list of class names to filter results

        Returns:
            List of SegmentResult objects
        """
        segments: list[SegmentResult] = []

        # Check for detections
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug("No segments found")
            return segments

        # Check for masks
        has_masks = result.masks is not None

        for idx, (box, cls, conf) in enumerate(
            zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf, strict=False)
        ):
            # Get class name
            class_id = int(cls.item())
            class_name = result.names[class_id]

            # Filter by classes if specified
            if classes and class_name not in classes:
                continue

            # Extract bbox
            x1, y1, x2, y2 = (float(coord) for coord in box.tolist())
            bbox = (x1, y1, x2, y2)

            # Calculate area from bbox (or mask if available)
            area = (x2 - x1) * (y2 - y1)

            segments.append(
                SegmentResult(
                    segment_idx=idx,
                    class_name=class_name,
                    confidence=float(conf.item()),
                    bbox=bbox,
                    area_px=area,
                    mask_rle=None,  # TODO: Add RLE encoding if masks needed
                )
            )

        # Sort by area descending (largest segments first)
        segments.sort(key=lambda s: s.area_px, reverse=True)

        return segments
