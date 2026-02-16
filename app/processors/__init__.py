"""ML Processors - Detection, segmentation, classification, estimation."""

from app.processors.base_processor import BaseProcessor
from app.processors.detector_processor import DetectorProcessor, DetectionResult
from app.processors.segmentation_processor import SegmentationProcessor, SegmentResult
from app.processors.classifier_processor import ClassifierProcessor, Classification
from app.processors.estimator_processor import EstimatorProcessor, EstimationResult

__all__ = [
    "BaseProcessor",
    "DetectorProcessor",
    "DetectionResult",
    "SegmentationProcessor",
    "SegmentResult",
    "ClassifierProcessor",
    "Classification",
    "EstimatorProcessor",
    "EstimationResult",
]
