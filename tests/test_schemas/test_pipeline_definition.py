"""Tests for pipeline definition Pydantic schemas."""

import pytest
from pydantic import ValidationError

from app.schemas.pipeline_definition import (
    ChainDefinition,
    ChordDefinition,
    GroupDefinition,
    PipelineDefinition,
    StepDefinition,
)


class TestStepDefinition:
    """Tests for StepDefinition schema."""

    def test_minimal_step(self) -> None:
        """Create step with just name."""
        step = StepDefinition(name="detection")

        assert step.type == "step"
        assert step.name == "detection"
        assert step.kwargs == {}

    def test_step_with_kwargs(self) -> None:
        """Create step with configuration kwargs."""
        step = StepDefinition(
            name="detection",
            kwargs={"segment_type": "cajon", "confidence": 0.85}
        )

        assert step.kwargs == {"segment_type": "cajon", "confidence": 0.85}

    def test_step_from_dict(self) -> None:
        """Create step from dict (JSON deserialization)."""
        data = {"type": "step", "name": "segmentation", "kwargs": {"model": "v2"}}

        step = StepDefinition.model_validate(data)

        assert step.name == "segmentation"
        assert step.kwargs == {"model": "v2"}

    def test_step_missing_name_raises(self) -> None:
        """Raise ValidationError when name is missing."""
        with pytest.raises(ValidationError) as exc_info:
            StepDefinition.model_validate({"type": "step"})

        assert "name" in str(exc_info.value)


class TestGroupDefinition:
    """Tests for GroupDefinition schema."""

    def test_group_with_steps(self) -> None:
        """Create group containing multiple steps."""
        group = GroupDefinition(
            steps=[
                StepDefinition(name="step_a"),
                StepDefinition(name="step_b"),
            ]
        )

        assert group.type == "group"
        assert len(group.steps) == 2

    def test_group_from_dict(self) -> None:
        """Create group from dict."""
        data = {
            "type": "group",
            "steps": [
                {"type": "step", "name": "a"},
                {"type": "step", "name": "b"},
            ]
        }

        group = GroupDefinition.model_validate(data)

        assert len(group.steps) == 2
        assert group.steps[0].name == "a"

    def test_empty_group_allowed(self) -> None:
        """Empty group is technically valid (though useless)."""
        group = GroupDefinition(steps=[])

        assert len(group.steps) == 0


class TestChordDefinition:
    """Tests for ChordDefinition schema."""

    def test_chord_with_callback(self) -> None:
        """Create chord with group and callback."""
        chord = ChordDefinition(
            group=GroupDefinition(
                steps=[
                    StepDefinition(name="branch_a"),
                    StepDefinition(name="branch_b"),
                ]
            ),
            callback=StepDefinition(name="aggregate")
        )

        assert chord.type == "chord"
        assert len(chord.group.steps) == 2
        assert chord.callback is not None
        assert chord.callback.name == "aggregate"

    def test_chord_without_callback(self) -> None:
        """Create chord without callback."""
        chord = ChordDefinition(
            group=GroupDefinition(
                steps=[StepDefinition(name="step_a")]
            )
        )

        assert chord.callback is None

    def test_chord_from_dict(self) -> None:
        """Create chord from dict."""
        data = {
            "type": "chord",
            "group": {
                "type": "group",
                "steps": [{"type": "step", "name": "x"}]
            },
            "callback": {"type": "step", "name": "cb"}
        }

        chord = ChordDefinition.model_validate(data)

        assert chord.callback.name == "cb"


class TestChainDefinition:
    """Tests for ChainDefinition schema."""

    def test_chain_with_mixed_elements(self) -> None:
        """Create chain with steps and groups."""
        chain = ChainDefinition(
            steps=[
                StepDefinition(name="first"),
                GroupDefinition(
                    steps=[
                        StepDefinition(name="parallel_a"),
                        StepDefinition(name="parallel_b"),
                    ]
                ),
                StepDefinition(name="last"),
            ]
        )

        assert chain.type == "chain"
        assert len(chain.steps) == 3
        assert isinstance(chain.steps[1], GroupDefinition)


class TestPipelineDefinition:
    """Tests for PipelineDefinition (root schema)."""

    def test_simple_pipeline(self) -> None:
        """Create simple sequential pipeline."""
        pipeline = PipelineDefinition(
            steps=[
                StepDefinition(name="a"),
                StepDefinition(name="b"),
            ]
        )

        assert pipeline.type == "chain"
        assert len(pipeline.steps) == 2

    def test_complex_pipeline_from_dict(self) -> None:
        """Create complex pipeline from dict (full integration test)."""
        data = {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "segmentation"},
                {"type": "step", "name": "segment_filter"},
                {
                    "type": "chord",
                    "group": {
                        "type": "group",
                        "steps": [
                            {
                                "type": "step",
                                "name": "sahi_detection",
                                "kwargs": {"segment_type": "claro-cajon"}
                            },
                            {
                                "type": "step",
                                "name": "detection",
                                "kwargs": {"segment_type": "cajon"}
                            }
                        ]
                    },
                    "callback": {"type": "step", "name": "aggregate_detections"}
                },
                {"type": "step", "name": "size_calculator"}
            ]
        }

        pipeline = PipelineDefinition.model_validate(data)

        assert len(pipeline.steps) == 4
        assert pipeline.steps[0].name == "segmentation"
        assert isinstance(pipeline.steps[2], ChordDefinition)

        chord = pipeline.steps[2]
        assert len(chord.group.steps) == 2
        assert chord.callback.name == "aggregate_detections"

    def test_nested_chains(self) -> None:
        """Pipeline can contain nested chains."""
        data = {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "outer_1"},
                {
                    "type": "chain",
                    "steps": [
                        {"type": "step", "name": "inner_1"},
                        {"type": "step", "name": "inner_2"},
                    ]
                },
                {"type": "step", "name": "outer_2"},
            ]
        }

        pipeline = PipelineDefinition.model_validate(data)

        assert len(pipeline.steps) == 3
        assert isinstance(pipeline.steps[1], ChainDefinition)


class TestDiscriminatedUnion:
    """Tests for type discriminator behavior."""

    def test_invalid_type_raises(self) -> None:
        """Raise ValidationError for invalid type value."""
        data = {
            "type": "chain",
            "steps": [
                {"type": "invalid_type", "name": "x"}
            ]
        }

        with pytest.raises(ValidationError) as exc_info:
            PipelineDefinition.model_validate(data)

        # Pydantic should mention the discriminator issue
        assert "type" in str(exc_info.value).lower()

    def test_missing_type_uses_default(self) -> None:
        """Steps without explicit type default to 'step'."""
        # StepDefinition has type="step" as default
        step = StepDefinition(name="test")
        assert step.type == "step"


class TestSerialization:
    """Tests for JSON serialization."""

    def test_roundtrip(self) -> None:
        """Pipeline survives JSON roundtrip."""
        original = PipelineDefinition(
            steps=[
                StepDefinition(name="a", kwargs={"x": 1}),
                ChordDefinition(
                    group=GroupDefinition(
                        steps=[StepDefinition(name="b")]
                    ),
                    callback=StepDefinition(name="c")
                )
            ]
        )

        json_str = original.model_dump_json()
        restored = PipelineDefinition.model_validate_json(json_str)

        assert len(restored.steps) == 2
        assert restored.steps[0].name == "a"
        assert restored.steps[0].kwargs == {"x": 1}

    def test_model_dump(self) -> None:
        """Pipeline converts to dict correctly."""
        pipeline = PipelineDefinition(
            steps=[StepDefinition(name="test")]
        )

        data = pipeline.model_dump()

        assert data["type"] == "chain"
        assert data["steps"][0]["name"] == "test"
