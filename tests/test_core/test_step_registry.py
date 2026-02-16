"""Tests for StepRegistry."""

from typing import Generator

import pytest

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.step_registry import StepRegistry


class DummyStepA(PipelineStep):
    """Test step A."""

    @property
    def name(self) -> str:
        """Return step name."""
        return "step_a"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute step."""
        return ctx


class DummyStepB(PipelineStep):
    """Test step B."""

    @property
    def name(self) -> str:
        """Return step name."""
        return "step_b"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute step."""
        return ctx


class DummyStepC(PipelineStep):
    """Test step C."""

    @property
    def name(self) -> str:
        """Return step name."""
        return "step_c"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute step."""
        return ctx


@pytest.fixture(autouse=True)
def clear_registry() -> Generator[None, None, None]:
    """Clear registry before each test."""
    StepRegistry._steps = {}
    yield
    StepRegistry._steps = {}


def test_register_step() -> None:
    """Test registering a step."""
    StepRegistry.register("test_step", DummyStepA)

    assert "test_step" in StepRegistry._steps
    assert StepRegistry._steps["test_step"] == DummyStepA


def test_get_step() -> None:
    """Test getting a registered step."""
    StepRegistry.register("test_step", DummyStepA)

    step = StepRegistry.get("test_step")

    assert isinstance(step, DummyStepA)
    assert step.name == "step_a"


def test_get_unknown_raises() -> None:
    """Test getting unknown step raises ValueError."""
    with pytest.raises(ValueError, match="Step 'unknown_step' not found in registry"):
        StepRegistry.get("unknown_step")


def test_build_pipeline() -> None:
    """Test building pipeline from step names."""
    StepRegistry.register("step_a", DummyStepA)
    StepRegistry.register("step_b", DummyStepB)
    StepRegistry.register("step_c", DummyStepC)

    pipeline = StepRegistry.build_pipeline(["step_a", "step_b", "step_c"])

    assert len(pipeline) == 3
    assert isinstance(pipeline[0], DummyStepA)
    assert isinstance(pipeline[1], DummyStepB)
    assert isinstance(pipeline[2], DummyStepC)


def test_build_pipeline_preserves_order() -> None:
    """Test pipeline order matches step_names order."""
    StepRegistry.register("step_a", DummyStepA)
    StepRegistry.register("step_b", DummyStepB)
    StepRegistry.register("step_c", DummyStepC)

    pipeline = StepRegistry.build_pipeline(["step_c", "step_a", "step_b"])

    assert len(pipeline) == 3
    assert isinstance(pipeline[0], DummyStepC)
    assert isinstance(pipeline[1], DummyStepA)
    assert isinstance(pipeline[2], DummyStepB)


def test_available_steps() -> None:
    """Test getting list of available step names."""
    StepRegistry.register("step_a", DummyStepA)
    StepRegistry.register("step_b", DummyStepB)

    available = StepRegistry.available_steps()

    assert len(available) == 2
    assert "step_a" in available
    assert "step_b" in available


def test_available_steps_empty() -> None:
    """Test available_steps returns empty list when no steps registered."""
    available = StepRegistry.available_steps()

    assert available == []


def test_get_returns_new_instance_each_time() -> None:
    """Test that get() returns new instances, not singletons."""
    StepRegistry.register("test_step", DummyStepA)

    step1 = StepRegistry.get("test_step")
    step2 = StepRegistry.get("test_step")

    assert step1 is not step2
    assert isinstance(step1, DummyStepA)
    assert isinstance(step2, DummyStepA)


def test_build_pipeline_with_unknown_step_raises() -> None:
    """Test building pipeline with unknown step raises ValueError."""
    StepRegistry.register("step_a", DummyStepA)

    with pytest.raises(ValueError, match="Step 'unknown_step' not found in registry"):
        StepRegistry.build_pipeline(["step_a", "unknown_step"])
