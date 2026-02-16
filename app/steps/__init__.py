"""Pipeline steps for ML processing."""

from app.core.step_registry import StepRegistry

# Import ML steps
from app.steps.ml.segmentation_step import SegmentationStep
from app.steps.ml.detection_step import DetectionStep
from app.steps.ml.sahi_detection_step import SAHIDetectionStep

# Import post-processor steps
from app.steps.post.segment_filter import SegmentFilterStep
from app.steps.post.size_calculator import SizeCalculatorStep
from app.steps.post.species_distributor import SpeciesDistributorStep


def register_all_steps() -> None:
    """Register all available pipeline steps with the StepRegistry.

    Registers both ML steps (segmentation, detection, sahi_detection) and
    post-processor steps (segment_filter, size_calculator, species_distributor).

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


__all__ = [
    "register_all_steps",
    "SegmentationStep",
    "DetectionStep",
    "SAHIDetectionStep",
    "SegmentFilterStep",
    "SizeCalculatorStep",
    "SpeciesDistributorStep",
]
