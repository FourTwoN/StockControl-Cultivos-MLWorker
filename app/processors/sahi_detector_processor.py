"""SAHI Detector Processor - Tiled detection for large segments.

This processor runs SAHI (Slicing Aided Hyper Inference) tiled detection on
large segments. It slices the image into tiles, detects on each tile, and
intelligently merges results.

Critical for large segments where standard YOLO fails to detect small plants.

Migrated from Demeter Backend.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import torch
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    torch = None
    AutoDetectionModel = None
    get_sliced_prediction = None

try:
    from PIL import Image
except ImportError:
    Image = None

from app.infra.logging import get_logger
from app.ml.model_cache import ModelCache
from app.processors.base_processor import BaseProcessor
from app.processors.detector_processor import DetectionResult

if TYPE_CHECKING:
    from sahi.prediction import PredictionResult
else:
    PredictionResult = Any

logger = get_logger(__name__)


class SAHIDetectorProcessor(BaseProcessor[list[DetectionResult]]):
    """SAHI tiled detector for large segments.

    Uses SAHI library to:
    1. Slice large segments into 512x512 tiles with 25% overlap
    2. Run YOLO detection on each tile
    3. Intelligently merge results with GREEDYNMM algorithm

    Critical Innovation:
        Traditional YOLO fails on large images (downscaling loses small plants).
        SAHI solves this with optimal tile size + intelligent merging.

    Architecture:
        - Uses ModelCache for model persistence
        - Lazy model loading on first process() call
        - Thread-safe via ModelCache synchronization
    """

    def __init__(
        self,
        tenant_id: str,
        worker_id: int = 0,
        confidence_threshold: float = 0.80,
    ) -> None:
        """Initialize SAHI detector processor.

        Args:
            tenant_id: Tenant identifier for model loading
            worker_id: GPU worker ID (0, 1, 2, ...)
            confidence_threshold: Minimum confidence score (0.0-1.0)
        """
        super().__init__(tenant_id, worker_id, confidence_threshold)
        self._model: Any = None
        self._worker_id_cached: int | None = None

    async def process(
        self,
        image_path: str | Path,
        segment: Any = None,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_ratio: float = 0.25,
        **kwargs: Any,
    ) -> list[DetectionResult]:
        """Detect plants in segment using SAHI tiling.

        Args:
            image_path: Path to segment crop image
            segment: Optional SegmentResult for logging/context
            slice_height: Tile height in pixels (default 512)
            slice_width: Tile width in pixels (default 512)
            overlap_ratio: Overlap percentage (0.0-1.0, default 0.25)
            **kwargs: Additional arguments (ignored)

        Returns:
            List of DetectionResult objects in original image coordinates,
            sorted by confidence descending

        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If image dimensions invalid
            RuntimeError: If SAHI detection fails
        """
        image_path = self._validate_image_path(image_path)

        if Image is None:
            raise RuntimeError("PIL (Pillow) is required for image processing")

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            raise ValueError(f"Failed to read image {image_path}: {e}") from e

        if img_width <= 0 or img_height <= 0:
            raise ValueError(
                f"Invalid image dimensions: {img_width}x{img_height}. Image may be corrupted."
            )

        if self._model is None or self._worker_id_cached != self.worker_id:
            logger.info(
                "Loading detection model",
                worker_id=self.worker_id,
            )
            self._model = ModelCache.get_model(
                tenant_id=self.tenant_id,
                model_type="detect",
                worker_id=self.worker_id,
            )
            self._worker_id_cached = self.worker_id
            logger.info("Detection model loaded successfully")

        if torch and torch.cuda.is_available():
            device = f"cuda:{self.worker_id}"
            logger.debug("Using GPU device", device=device)
        else:
            device = "cpu"
            logger.debug("Using CPU device (GPU not available)")

        if AutoDetectionModel is None:
            raise RuntimeError("SAHI library is required for tiled detection")

        detector = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model=self._model,
            confidence_threshold=self.confidence_threshold,
            device=device,
        )

        start_time = time.time()

        try:
            if get_sliced_prediction is None:
                raise RuntimeError("SAHI library is required for sliced prediction")

            result = get_sliced_prediction(
                str(image_path),
                detector,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                postprocess_type="GREEDYNMM",
                postprocess_match_threshold=0.5,
                auto_slice_resolution=False,
                perform_standard_pred=False,
                postprocess_class_agnostic=False,
                verbose=0,
            )

            elapsed = time.time() - start_time

            detections = self._parse_sahi_results(result)

            logger.info(
                "SAHI detection completed",
                image=image_path.name,
                dimensions=f"{img_width}x{img_height}",
                detections=len(detections),
                elapsed_s=f"{elapsed:.2f}",
                confidence_threshold=self.confidence_threshold,
            )

            return detections

        except Exception as e:
            logger.error(
                "SAHI detection failed",
                image=image_path.name,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"SAHI detection failed: {e}") from e

    def _parse_sahi_results(self, sahi_result: "PredictionResult") -> list[DetectionResult]:
        """Parse SAHI prediction results into DetectionResult objects.

        Args:
            sahi_result: SAHI PredictionResult object from get_sliced_prediction()

        Returns:
            List of DetectionResult objects, sorted by confidence descending
        """
        detections: list[DetectionResult] = []

        for obj_pred in sahi_result.object_prediction_list:
            bbox = obj_pred.bbox

            center_x = bbox.minx + (bbox.maxx - bbox.minx) / 2
            center_y = bbox.miny + (bbox.maxy - bbox.miny) / 2
            width = bbox.maxx - bbox.minx
            height = bbox.maxy - bbox.miny

            confidence = float(obj_pred.score.value)
            class_name = str(obj_pred.category.name)

            detections.append(
                DetectionResult(
                    center_x_px=float(center_x),
                    center_y_px=float(center_y),
                    width_px=float(width),
                    height_px=float(height),
                    confidence=confidence,
                    class_name=class_name,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)

        logger.debug(
            "Parsed SAHI results",
            detections=len(detections),
        )

        return detections
