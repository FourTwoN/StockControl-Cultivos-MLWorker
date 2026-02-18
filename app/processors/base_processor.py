"""Base Processor - Abstract base class for all ML processors.

Ported from DemeterAI-back with adaptations for async Cloud Tasks processing.

All processors integrate with ModelCache singleton for model persistence.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from app.infra.logging import get_logger

logger = get_logger(__name__)

# Generic type for processor output
T = TypeVar("T")


class BaseProcessor(ABC, Generic[T]):
    """Abstract base class for ML processors.

    All ML processors (Detector, Segmentador, Classifier, Estimator)
    inherit from this class and implement the `process()` method.

    Key Features:
        - Model caching via ModelCache singleton (per-tenant)
        - Async processing support
        - Resource management (load/unload models)
        - Structured logging

    Attributes:
        tenant_id: Tenant identifier for model loading
        worker_id: GPU worker ID for model assignment (default 0)
        confidence_threshold: Model confidence threshold (if applicable)

    Thread Safety:
        Processors are thread-safe. Model loading is synchronized via ModelCache.
    """

    def __init__(
        self,
        tenant_id: str,
        worker_id: int = 0,
        confidence_threshold: float = 0.25,
    ) -> None:
        """Initialize base processor.

        Args:
            tenant_id: Tenant identifier for model loading
            worker_id: GPU worker ID (0, 1, 2, ...) for model assignment
            confidence_threshold: Detection/classification confidence threshold
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")

        self.tenant_id = tenant_id
        self.worker_id = worker_id
        self.confidence_threshold = confidence_threshold
        self._model: Any = None  # Cached model instance

        logger.info(
            "Processor initialized",
            processor=self.__class__.__name__,
            tenant_id=tenant_id,
            worker_id=self.worker_id,
            confidence_threshold=self.confidence_threshold,
        )

    @abstractmethod
    async def process(self, *args: Any, **kwargs: Any) -> T:
        """Process input and return result.

        This method must be implemented by all subclasses.

        Args:
            *args: Positional arguments (varies by processor)
            **kwargs: Keyword arguments (varies by processor)

        Returns:
            Processor-specific output (generic type T)

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement process()")

    def load_model(self) -> Any:
        """Load ML model into memory.

        This method should be overridden by subclasses that use ML models.
        Default implementation returns None (for processors without models).

        Returns:
            Loaded model instance (varies by processor)
        """
        logger.debug(
            "load_model() not implemented",
            processor=self.__class__.__name__,
        )
        return None

    def unload_model(self) -> None:
        """Unload ML model from memory.

        This method should be overridden by subclasses that use ML models.
        Default implementation does nothing.
        """
        logger.debug(
            "unload_model() not implemented",
            processor=self.__class__.__name__,
        )

    def _validate_image_path(self, image_path: str | Path) -> Path:
        """Validate image path exists.

        Args:
            image_path: Path to image file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If path is not a file
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return path

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"tenant_id={self.tenant_id}, "
            f"worker_id={self.worker_id}, "
            f"confidence_threshold={self.confidence_threshold}"
            f")"
        )
