"""Estimator Processor - Estimation/counting from detections.

Provides estimation logic for:
- Plant counting per segment
- Coverage estimation
- Yield estimation
"""

from dataclasses import dataclass
from typing import Any

from app.infra.logging import get_logger
from app.processors.base_processor import BaseProcessor
from app.processors.detector_processor import DetectionResult
from app.processors.segmentation_processor import SegmentResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class EstimationResult:
    """Estimation result for a processing session.

    Attributes:
        total_count: Total number of detected items
        count_per_segment: Count per segment (segment_idx -> count)
        coverage_percent: Estimated coverage percentage
        confidence_avg: Average detection confidence
        metadata: Additional estimation metadata
    """

    total_count: int
    count_per_segment: dict[int, int]
    coverage_percent: float
    confidence_avg: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_count": self.total_count,
            "count_per_segment": self.count_per_segment,
            "coverage_percent": self.coverage_percent,
            "confidence_avg": self.confidence_avg,
            "metadata": self.metadata,
        }


class EstimatorProcessor(BaseProcessor[EstimationResult]):
    """Estimator processor for counting and coverage estimation.

    Takes detection and segmentation results and produces estimates.
    """

    def __init__(
        self,
        model_path: str | None = None,
        worker_id: int = 0,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialize estimator processor.

        Args:
            model_path: Not used for estimation
            worker_id: Not used for estimation
            confidence_threshold: Minimum confidence for counting
        """
        super().__init__(model_path, worker_id, confidence_threshold)

    async def process(
        self,
        detections: list[DetectionResult],
        segments: list[SegmentResult] | None = None,
        image_area_px: float | None = None,
    ) -> EstimationResult:
        """Generate estimation from detections and segments.

        Args:
            detections: List of detection results
            segments: Optional list of segmentation results
            image_area_px: Total image area in pixels (for coverage calculation)

        Returns:
            EstimationResult with counts and coverage
        """
        logger.debug(
            "Running estimation",
            detection_count=len(detections),
            segment_count=len(segments) if segments else 0,
        )

        # Filter by confidence threshold
        valid_detections = [
            d for d in detections if d.confidence >= self.confidence_threshold
        ]

        # Count total
        total_count = len(valid_detections)

        # Count per segment
        count_per_segment: dict[int, int] = {}
        for detection in valid_detections:
            segment_idx = detection.segment_idx
            if segment_idx >= 0:
                count_per_segment[segment_idx] = count_per_segment.get(segment_idx, 0) + 1

        # Calculate average confidence
        confidence_avg = 0.0
        if valid_detections:
            confidence_avg = sum(d.confidence for d in valid_detections) / len(valid_detections)

        # Calculate coverage
        coverage_percent = 0.0
        if image_area_px and image_area_px > 0:
            detection_area = sum(d.width_px * d.height_px for d in valid_detections)
            coverage_percent = min(100.0, (detection_area / image_area_px) * 100)

        result = EstimationResult(
            total_count=total_count,
            count_per_segment=count_per_segment,
            coverage_percent=coverage_percent,
            confidence_avg=confidence_avg,
            metadata={
                "total_detections": len(detections),
                "filtered_detections": len(valid_detections),
                "confidence_threshold": self.confidence_threshold,
            },
        )

        logger.info(
            "Estimation complete",
            total_count=total_count,
            coverage_percent=f"{coverage_percent:.1f}%",
            confidence_avg=f"{confidence_avg:.2f}",
        )

        return result

    async def estimate_from_session(
        self,
        session_detections: dict[str, list[DetectionResult]],
        session_segments: dict[str, list[SegmentResult]] | None = None,
    ) -> dict[str, EstimationResult]:
        """Generate estimations for all images in a session.

        Args:
            session_detections: Dict of image_id -> detections
            session_segments: Optional dict of image_id -> segments

        Returns:
            Dict of image_id -> EstimationResult
        """
        results: dict[str, EstimationResult] = {}

        for image_id, detections in session_detections.items():
            segments = session_segments.get(image_id) if session_segments else None
            results[image_id] = await self.process(detections, segments)

        return results
