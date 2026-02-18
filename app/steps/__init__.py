"""Pipeline steps for ML processing."""

from app.core.step_registry import StepRegistry
from app.steps.ml.detection_step import DetectionStep
from app.steps.ml.sahi_detection_step import SAHIDetectionStep

# Import ML steps
from app.steps.ml.segmentation_step import SegmentationStep
from app.steps.post.aggregate_detections import AggregateDetectionsStep

# Import post-processor steps
from app.steps.post.segment_filter import SegmentFilterStep
from app.steps.post.size_calculator import SizeCalculatorStep
from app.steps.post.species_distributor import SpeciesDistributorStep
from app.steps.post.upload_image import UploadImageStep
from app.steps.post.visualize_detections import VisualizeDetectionsStep


def register_all_steps() -> None:
    """Register all available pipeline steps with the StepRegistry.

    Registers both ML steps (segmentation, detection, sahi_detection) and
    post-processor steps (segment_filter, size_calculator, species_distributor,
    visualize_detections).

    This function should be called during application startup to ensure all
    steps are available for pipeline construction.
    """
    # ML Steps
    StepRegistry.register("segmentation", SegmentationStep)
    StepRegistry.register("detection", DetectionStep)
    StepRegistry.register("sahi_detection", SAHIDetectionStep)

    # Post-Processor Steps
    StepRegistry.register("segment_filter", SegmentFilterStep)
    StepRegistry.register("size_calculator", SizeCalculatorStep)
    StepRegistry.register("species_distributor", SpeciesDistributorStep)
    StepRegistry.register("aggregate_detections", AggregateDetectionsStep)
    StepRegistry.register("visualize_detections", VisualizeDetectionsStep)
    StepRegistry.register("upload_image", UploadImageStep)


__all__ = [
    "register_all_steps",
    "SegmentationStep",
    "DetectionStep",
    "SAHIDetectionStep",
    "SegmentFilterStep",
    "SizeCalculatorStep",
    "SpeciesDistributorStep",
    "AggregateDetectionsStep",
    "VisualizeDetectionsStep",
    "UploadImageStep",
]
