"""ModelCache singleton for YOLO model management.

Ported from DemeterAI-back with adaptations for GCS model loading.

Provides:
- Singleton pattern for model caching
- Thread-safe model loading
- GPU/CPU device assignment
- GCS model download support
- Memory cleanup utilities
"""

import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

from app.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for optional dependencies
try:
    import torch
    from ultralytics import YOLO

    TORCH_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    logger.warning("torch/ultralytics not available - ML features disabled")

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    storage = None  # type: ignore
    GCS_AVAILABLE = False

ModelType = Literal["segment", "detect", "classify"]


class ModelCache:
    """Singleton cache for YOLO models (per worker, per model type)."""

    _instances: dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()
    _local_model_dir: Path | None = None

    @classmethod
    def get_model(cls, model_type: ModelType, worker_id: int = 0) -> Any:
        """Get or create cached model instance.

        Args:
            model_type: "segment", "detect", or "classify"
            worker_id: GPU worker ID (0, 1, 2, etc.)

        Returns:
            Cached YOLO model instance

        Raises:
            ValueError: If model_type invalid or worker_id negative
            RuntimeError: If model loading fails or torch not available
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/ultralytics not installed - cannot load models")

        # Validation
        if model_type not in ["segment", "detect", "classify"]:
            raise ValueError(f"Invalid model_type: {model_type}")

        if worker_id < 0:
            raise ValueError("worker_id must be non-negative")

        # Cache key
        cache_key = f"{model_type}_worker_{worker_id}"

        # Thread-safe singleton check
        with cls._lock:
            if cache_key not in cls._instances:
                logger.info(
                    "Loading model",
                    model_type=model_type,
                    worker_id=worker_id,
                )

                # Get model path (download from GCS if needed)
                model_path = cls._get_model_path(model_type)

                # Load model - for ONNX, explicitly pass task since it can't be inferred
                is_onnx = str(model_path).endswith(".onnx")
                if is_onnx:
                    # ONNX models need explicit task parameter for correct class mapping
                    model = YOLO(str(model_path), task=model_type)
                    logger.info("Loaded ONNX model with explicit task", task=model_type)
                else:
                    model = YOLO(str(model_path))

                # Determine device for inference
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_id = worker_id % gpu_count
                    device = f"cuda:{gpu_id}"
                    logger.info("Assigning model to GPU", device=device)
                else:
                    device = "cpu"
                    logger.info("GPU not available, using CPU", worker_id=worker_id)

                # For PyTorch models (.pt), move to device and fuse
                # For ONNX models, device is passed at inference time
                if not is_onnx:
                    model = model.to(device)
                    model.fuse()

                # Store device info for ONNX models to use at inference
                model._mlworker_device = device
                model._mlworker_is_onnx = is_onnx

                # Cache
                cls._instances[cache_key] = model
                logger.info(
                    "Model loaded and cached",
                    model_type=model_type,
                    worker_id=worker_id,
                    device=device,
                )

            return cls._instances[cache_key]

    @classmethod
    def _get_model_path(cls, model_type: ModelType) -> Path:
        """Get local path to model, downloading from GCS if needed.

        Args:
            model_type: Model type to load

        Returns:
            Local path to model file
        """
        # Determine model filename
        model_files = {
            "detect": settings.detection_model,
            "segment": settings.segmentation_model,
            "classify": "classify.pt",  # Add to settings if needed
        }
        model_filename = model_files.get(model_type, f"{model_type}.pt")

        # Check if model_path is GCS URL
        if settings.model_path.startswith("gs://"):
            return cls._download_from_gcs(settings.model_path, model_filename)

        # Local path
        local_path = Path(settings.model_path) / model_filename
        if not local_path.exists():
            raise FileNotFoundError(f"Model not found: {local_path}")

        return local_path

    @classmethod
    def _download_from_gcs(cls, gcs_prefix: str, model_filename: str) -> Path:
        """Download model from GCS to local temp directory.

        Args:
            gcs_prefix: GCS path prefix (gs://bucket/path)
            model_filename: Model filename to download

        Returns:
            Local path to downloaded model
        """
        if not GCS_AVAILABLE:
            raise RuntimeError("google-cloud-storage not installed")

        # Create local model directory if needed
        if cls._local_model_dir is None:
            cls._local_model_dir = Path(tempfile.mkdtemp(prefix="mlworker_models_"))
            logger.info("Created local model directory", path=str(cls._local_model_dir))

        local_path = cls._local_model_dir / model_filename

        # Check if already downloaded
        if local_path.exists():
            logger.debug("Model already downloaded", path=str(local_path))
            return local_path

        # Parse GCS URL
        # gs://bucket/path -> bucket, path
        gcs_path = gcs_prefix.replace("gs://", "")
        parts = gcs_path.split("/", 1)
        bucket_name = parts[0]
        blob_prefix = parts[1] if len(parts) > 1 else ""

        blob_path = f"{blob_prefix}/{model_filename}".lstrip("/")

        logger.info(
            "Downloading model from GCS",
            bucket=bucket_name,
            blob=blob_path,
            local_path=str(local_path),
        )

        # Download
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(str(local_path))

        logger.info("Model downloaded", path=str(local_path), size=local_path.stat().st_size)

        return local_path

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models and free GPU memory."""
        with cls._lock:
            model_count = len(cls._instances)
            cls._instances.clear()

            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")

            logger.info("Model cache cleared", models_removed=model_count)

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """Get information about cached models.

        Returns:
            Dict with cache statistics
        """
        with cls._lock:
            info = {
                "cached_models": list(cls._instances.keys()),
                "model_count": len(cls._instances),
                "local_model_dir": str(cls._local_model_dir) if cls._local_model_dir else None,
            }

            if TORCH_AVAILABLE and torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            else:
                info["gpu_available"] = False

            return info
