"""Segmentation Processor - YOLO segmentation for region extraction.

Extracts segments/regions from images using YOLO segmentation model.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        bbox: Bounding box (x1, y1, x2, y2) in normalized coords
        area_px: Segment area in pixels
        polygon: List of (x, y) points in normalized coords (optional)
        mask_rle: Run-length encoded mask (optional)
    """

    segment_idx: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    area_px: float
    polygon: list[tuple[float, float]] | None = None
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
        tenant_id: str,
        worker_id: int = 0,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialize segmentation processor.

        Args:
            tenant_id: Tenant identifier for model loading
            worker_id: GPU worker ID
            confidence_threshold: Minimum confidence score
        """
        super().__init__(tenant_id, worker_id, confidence_threshold)
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
        inference_start = time.perf_counter()

        # Validate image path
        image_path = self._validate_image_path(image_path)

        # Get model from singleton (lazy load)
        if self._model is None or self._worker_id_cached != self.worker_id:
            load_start = time.perf_counter()
            logger.info(
                "Loading segmentation model from cache",
                worker_id=self.worker_id,
                model_type="segment",
            )
            self._model = ModelCache.get_model(
                tenant_id=self.tenant_id,
                model_type="segment",
                worker_id=self.worker_id,
            )
            self._worker_id_cached = self.worker_id

            is_onnx = getattr(self._model, "_mlworker_is_onnx", False)
            device = getattr(self._model, "_mlworker_device", "unknown")
            load_duration_ms = int((time.perf_counter() - load_start) * 1000)

            logger.info(
                "Segmentation model ready",
                worker_id=self.worker_id,
                is_onnx=is_onnx,
                device=device,
                model_classes=list(self._model.names.values()) if hasattr(self._model, "names") else [],
                load_duration_ms=load_duration_ms,
            )

        # Run YOLO segmentation
        try:
            logger.info(
                "Running segmentation inference",
                image=image_path.name,
                image_size_bytes=image_path.stat().st_size if image_path.exists() else 0,
                confidence_threshold=self.confidence_threshold,
            )

            # For ONNX models, pass device to predict()
            device = getattr(self._model, "_mlworker_device", None)

            predict_start = time.perf_counter()
            results = self._model.predict(
                source=str(image_path),
                conf=self.confidence_threshold,
                verbose=False,
                device=device,
            )
            predict_duration_ms = int((time.perf_counter() - predict_start) * 1000)

            # Parse results
            segments = self._parse_results(results[0], classes)

            total_duration_ms = int((time.perf_counter() - inference_start) * 1000)

            # Log detailed segment info
            segment_summary = [
                f"{s.class_name}(idx={s.segment_idx}, conf={s.confidence:.2f}, area={s.area_px:.0f})"
                for s in segments
            ]

            logger.info(
                "Segmentation inference completed",
                image=image_path.name,
                segments_found=len(segments),
                segment_classes=[s.class_name for s in segments],
                inference_ms=predict_duration_ms,
                total_ms=total_duration_ms,
            )

            if segments:
                logger.debug(
                    "Segment details",
                    segments=segment_summary,
                )

            return segments

        except Exception as e:
            total_duration_ms = int((time.perf_counter() - inference_start) * 1000)
            logger.error(
                "Segmentation inference FAILED",
                image=image_path.name,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=total_duration_ms,
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
