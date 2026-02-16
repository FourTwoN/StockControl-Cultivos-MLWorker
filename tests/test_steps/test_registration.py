"""Tests for step registration functionality."""

import pytest

from app.core.step_registry import StepRegistry
from app.steps import register_all_steps
from app.steps.ml.segmentation_step import SegmentationStep
from app.steps.ml.detection_step import DetectionStep
from app.steps.ml.sahi_detection_step import SAHIDetectionStep
from app.steps.post.segment_filter import SegmentFilterStep
from app.steps.post.size_calculator import SizeCalculatorStep
from app.steps.post.species_distributor import SpeciesDistributorStep


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the step registry before and after each test.

    This ensures test isolation by preventing state leakage between tests.
    """
    StepRegistry._steps.clear()
    yield
    StepRegistry._steps.clear()


def test_ml_steps_registered():
    """Verify that all ML steps are registered correctly."""
    register_all_steps()

    available = StepRegistry.available_steps()

    assert "segmentation" in available
    assert "detection" in available
    assert "sahi_detection" in available


def test_post_steps_registered():
    """Verify that all post-processor steps are registered correctly."""
    register_all_steps()

    available = StepRegistry.available_steps()

    assert "segment_filter" in available
    assert "size_calculator" in available
    assert "species_distributor" in available


def test_can_build_full_pipeline():
    """Verify that a complete pipeline can be constructed from registered steps."""
    register_all_steps()

    pipeline = StepRegistry.build_pipeline([
        "segmentation",
        "segment_filter",
        "detection",
        "size_calculator",
        "species_distributor",
    ])

    assert len(pipeline) == 5
    assert isinstance(pipeline[0], SegmentationStep)
    assert isinstance(pipeline[1], SegmentFilterStep)
    assert isinstance(pipeline[2], DetectionStep)
    assert isinstance(pipeline[3], SizeCalculatorStep)
    assert isinstance(pipeline[4], SpeciesDistributorStep)


def test_all_registered_steps_count():
    """Verify that exactly 6 steps are registered."""
    register_all_steps()

    available = StepRegistry.available_steps()

    assert len(available) == 6


def test_get_specific_step_instances():
    """Verify that we can retrieve specific step instances."""
    register_all_steps()

    segmentation = StepRegistry.get("segmentation")
    detection = StepRegistry.get("detection")
    sahi_detection = StepRegistry.get("sahi_detection")
    segment_filter = StepRegistry.get("segment_filter")
    size_calculator = StepRegistry.get("size_calculator")
    species_distributor = StepRegistry.get("species_distributor")

    assert isinstance(segmentation, SegmentationStep)
    assert isinstance(detection, DetectionStep)
    assert isinstance(sahi_detection, SAHIDetectionStep)
    assert isinstance(segment_filter, SegmentFilterStep)
    assert isinstance(size_calculator, SizeCalculatorStep)
    assert isinstance(species_distributor, SpeciesDistributorStep)


def test_registry_isolated_before_registration():
    """Verify that registry starts empty (due to fixture)."""
    available = StepRegistry.available_steps()

    assert len(available) == 0


def test_can_build_alternative_pipeline():
    """Verify that alternative pipeline configurations work."""
    register_all_steps()

    # SAHI detection instead of regular detection
    pipeline = StepRegistry.build_pipeline([
        "segmentation",
        "sahi_detection",
        "size_calculator",
    ])

    assert len(pipeline) == 3
    assert isinstance(pipeline[0], SegmentationStep)
    assert isinstance(pipeline[1], SAHIDetectionStep)
    assert isinstance(pipeline[2], SizeCalculatorStep)


def test_pipeline_order_preserved():
    """Verify that pipeline maintains step order."""
    register_all_steps()

    step_names = [
        "species_distributor",
        "segmentation",
        "size_calculator",
    ]

    pipeline = StepRegistry.build_pipeline(step_names)

    # Check order is preserved
    assert pipeline[0].name == "species_distributor"
    assert pipeline[1].name == "segmentation"
    assert pipeline[2].name == "size_calculator"
