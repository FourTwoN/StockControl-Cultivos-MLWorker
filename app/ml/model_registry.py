"""Model Registry - Centralized model management and loading.

Provides a higher-level interface over ModelCache for:
- Model versioning
- Model metadata
- Health checks
"""

from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.infra.logging import get_logger
from app.ml.model_cache import ModelCache, ModelType

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Information about a loaded model."""

    model_type: ModelType
    worker_id: int
    device: str
    is_loaded: bool


class ModelRegistry:
    """High-level model management interface."""

    @staticmethod
    def get_detector(worker_id: int = 0) -> Any:
        """Get detection model.

        Args:
            worker_id: GPU worker ID

        Returns:
            Loaded YOLO detection model
        """
        return ModelCache.get_model("detect", worker_id)

    @staticmethod
    def get_segmenter(worker_id: int = 0) -> Any:
        """Get segmentation model.

        Args:
            worker_id: GPU worker ID

        Returns:
            Loaded YOLO segmentation model
        """
        return ModelCache.get_model("segment", worker_id)

    @staticmethod
    def get_classifier(worker_id: int = 0) -> Any:
        """Get classification model.

        Args:
            worker_id: GPU worker ID

        Returns:
            Loaded YOLO classification model
        """
        return ModelCache.get_model("classify", worker_id)

    @staticmethod
    def preload_models(worker_id: int = 0) -> list[str]:
        """Preload all models for faster inference.

        Args:
            worker_id: GPU worker ID

        Returns:
            List of loaded model types
        """
        loaded: list[str] = []

        try:
            ModelCache.get_model("detect", worker_id)
            loaded.append("detect")
        except Exception as e:
            logger.warning("Failed to preload detection model", error=str(e))

        try:
            ModelCache.get_model("segment", worker_id)
            loaded.append("segment")
        except Exception as e:
            logger.warning("Failed to preload segmentation model", error=str(e))

        logger.info("Models preloaded", loaded=loaded)
        return loaded

    @staticmethod
    def health_check() -> dict[str, Any]:
        """Check model system health.

        Returns:
            Health status dict
        """
        cache_info = ModelCache.get_cache_info()

        return {
            "healthy": True,
            "models_loaded": cache_info["model_count"],
            "gpu_available": cache_info.get("gpu_available", False),
            "model_path": settings.model_path,
        }

    @staticmethod
    def clear_all() -> None:
        """Clear all cached models."""
        ModelCache.clear_cache()
