"""Classifier Processor - Classification of detected objects.

Classifies detected plants/objects into categories.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.infra.logging import get_logger
from app.processors.base_processor import BaseProcessor

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationResult:
    """Single classification result.

    Attributes:
        class_name: Predicted class name
        confidence: Classification confidence (0.0-1.0)
        class_id: Class index
        all_scores: Dict of all class scores (optional)
    """

    class_name: str
    confidence: float
    class_id: int
    all_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "all_scores": self.all_scores,
        }


class ClassifierProcessor(BaseProcessor[ClassificationResult]):
    """Classification processor for object categorization.

    Can use either YOLO classification or custom classification models.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        worker_id: int = 0,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialize classifier processor.

        Args:
            model_path: Path to classification model
            worker_id: GPU worker ID
            confidence_threshold: Minimum confidence score
        """
        super().__init__(model_path, worker_id, confidence_threshold)
        self._model: Any = None

    async def process(
        self,
        image_path: str | Path,
    ) -> ClassificationResult:
        """Classify an image.

        Args:
            image_path: Path to image file (usually a crop of detected object)

        Returns:
            ClassificationResult with predicted class

        Raises:
            FileNotFoundError: If image_path doesn't exist
            RuntimeError: If classification fails
        """
        # Validate image path
        image_path = self._validate_image_path(image_path)

        logger.debug(
            "Running classification",
            image=image_path.name,
            confidence_threshold=self.confidence_threshold,
        )

        # TODO: Implement actual classification
        # For now, return a stub result
        logger.warning(
            "Classification not implemented - returning stub",
            image=image_path.name,
        )

        return ClassificationResult(
            class_name="unknown",
            confidence=0.0,
            class_id=-1,
            all_scores=None,
        )

    async def classify_batch(
        self,
        image_paths: list[Path],
    ) -> list[ClassificationResult]:
        """Classify multiple images in batch.

        Args:
            image_paths: List of image paths

        Returns:
            List of ClassificationResult objects
        """
        results = []
        for image_path in image_paths:
            result = await self.process(image_path)
            results.append(result)
        return results
