"""Detector Processor - Standard YOLO detection for plant detection.

Ported from DemeterAI-back with adaptations for async Cloud Tasks processing.
Runs direct YOLO detection (without SAHI tiling).
"""

from dataclasses import dataclass, field
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
class DetectionResult:
    """Single plant detection result.

    Attributes:
        center_x_px: Center X coordinate in absolute pixels
        center_y_px: Center Y coordinate in absolute pixels
        width_px: Bounding box width in pixels
        height_px: Bounding box height in pixels
        confidence: Detection confidence score (0.0-1.0)
        class_name: YOLO class name (e.g., "plant")
        segment_idx: Segment index this detection belongs to (set by coordinator)
        image_id: Image UUID this detection belongs to (set by coordinator)
    """

    center_x_px: float
    center_y_px: float
    width_px: float
    height_px: float
    confidence: float
    class_name: str
    segment_idx: int = -1
    image_id: str = ""

    def __post_init__(self) -> None:
        """Validate detection fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError(
                f"width_px and height_px must be positive, got width={self.width_px}, "
                f"height={self.height_px}"
            )

        if self.center_x_px < 0 or self.center_y_px < 0:
            raise ValueError(
                f"center coordinates must be non-negative, got "
                f"center_x={self.center_x_px}, center_y={self.center_y_px}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "center_x_px": self.center_x_px,
            "center_y_px": self.center_y_px,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "segment_idx": self.segment_idx,
            "image_id": self.image_id,
        }


class DetectorProcessor(BaseProcessor[list[DetectionResult]]):
    """Standard YOLO detector for plant detection.

    Runs direct YOLO detection without tiling.

    Architecture:
        - Uses ModelCache for model persistence
        - Lazy model loading on first process() call
        - Thread-safe via ModelCache synchronization
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        worker_id: int = 0,
        confidence_threshold: float = 0.80,
    ) -> None:
        """Initialize detector processor.

        Args:
            model_path: Path to YOLO detection model (optional - uses default from ModelCache)
            worker_id: GPU worker ID (0, 1, 2, ...)
            confidence_threshold: Minimum confidence score (0.0-1.0)
        """
        super().__init__(model_path, worker_id, confidence_threshold)
        self._model: Any = None
        self._worker_id_cached: int | None = None

    async def process(
        self,
        image_path: str | Path,
        segment: Any = None,
        classes: list[str] | None = None,
    ) -> list[DetectionResult]:
        """Detect plants in image using standard YOLO.

        Args:
            image_path: Path to image file
            segment: Optional segment info for context
            classes: Optional list of class names to filter results

        Returns:
            List of DetectionResult objects, sorted by confidence descending

        Raises:
            FileNotFoundError: If image_path doesn't exist
            RuntimeError: If YOLO detection fails
        """
        # Validate image path
        image_path = self._validate_image_path(image_path)

        # Get model from singleton (lazy load)
        if self._model is None or self._worker_id_cached != self.worker_id:
            logger.info("Loading detection model", worker_id=self.worker_id)
            self._model = ModelCache.get_model("detect", self.worker_id)
            self._worker_id_cached = self.worker_id
            logger.info("Detection model loaded")

        # Run YOLO detection
        try:
            logger.debug(
                "Running detection",
                image=image_path.name,
                confidence_threshold=self.confidence_threshold,
            )

            # For ONNX models, pass device to predict()
            device = getattr(self._model, "_mlworker_device", None)

            results = self._model.predict(
                source=str(image_path),
                conf=self.confidence_threshold,
                verbose=False,
                device=device,
            )

            # Parse YOLO results
            detections = self._parse_yolo_results(results[0], classes)

            logger.info(
                "Detection complete",
                image=image_path.name,
                detections=len(detections),
                confidence_threshold=self.confidence_threshold,
            )

            return detections

        except Exception as e:
            logger.error(
                "Detection failed",
                image=image_path.name,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Detection failed: {e}") from e

    def _parse_yolo_results(
        self,
        result: "Results",
        classes: list[str] | None = None,
    ) -> list[DetectionResult]:
        """Parse YOLO Results object into DetectionResult objects.

        Args:
            result: YOLO Results object from model.predict()
            classes: Optional list of class names to filter results

        Returns:
            List of DetectionResult objects, sorted by confidence descending
        """
        detections: list[DetectionResult] = []

        # Check if any detections found
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug("No detections in image")
            return detections

        # Extract detections
        for box, cls, conf in zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf,
            strict=False,
        ):
            # Get class name
            class_id = int(cls.item())
            class_name = result.names[class_id]

            # Filter by classes if specified
            if classes and class_name not in classes:
                continue

            # Extract bbox coordinates
            x1, y1, x2, y2 = (float(coord) for coord in box.tolist())

            # Calculate center and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1

            detections.append(
                DetectionResult(
                    center_x_px=center_x,
                    center_y_px=center_y,
                    width_px=width,
                    height_px=height,
                    confidence=float(conf.item()),
                    class_name=class_name,
                )
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug("Parsed detections", count=len(detections))

        return detections
