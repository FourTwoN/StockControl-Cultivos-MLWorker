"""Tests for PipelineParser - JSON to DSL conversion and validation."""

import pytest
from unittest.mock import MagicMock

from app.core.pipeline_dsl import Chain, Chord, Group, StepSignature
from app.core.pipeline_parser import PipelineParser, PipelineParserError
from app.schemas.pipeline_definition import (
    ChainDefinition,
    ChordDefinition,
    GroupDefinition,
    PipelineDefinition,
    StepDefinition,
)


class MockStepRegistry:
    """Mock StepRegistry for testing."""

    _steps = {"step_a", "step_b", "step_c", "segmentation", "detection", "aggregate"}

    @classmethod
    def available_steps(cls) -> list[str]:
        return list(cls._steps)


@pytest.fixture
def parser() -> PipelineParser:
    """Create parser with mock registry."""
    return PipelineParser(MockStepRegistry)


class TestParseSimpleChain:
    """Tests for parsing simple chain pipelines."""

    def test_parse_single_step(self, parser: PipelineParser) -> None:
        """Parse chain with single step."""
        definition = PipelineDefinition(
            steps=[StepDefinition(name="step_a")]
        )

        result = parser.parse(definition)

        assert isinstance(result, Chain)
        assert len(result.steps) == 1
        assert isinstance(result.steps[0], StepSignature)
        assert result.steps[0].name == "step_a"

    def test_parse_multiple_steps(self, parser: PipelineParser) -> None:
        """Parse chain with multiple sequential steps."""
        definition = PipelineDefinition(
            steps=[
                StepDefinition(name="step_a"),
                StepDefinition(name="step_b"),
                StepDefinition(name="step_c"),
            ]
        )

        result = parser.parse(definition)

        assert isinstance(result, Chain)
        assert len(result.steps) == 3
        assert result.steps[0].name == "step_a"
        assert result.steps[1].name == "step_b"
        assert result.steps[2].name == "step_c"

    def test_parse_step_with_kwargs(self, parser: PipelineParser) -> None:
        """Parse step with configuration kwargs."""
        definition = PipelineDefinition(
            steps=[
                StepDefinition(
                    name="detection",
                    kwargs={"segment_type": "cajon", "confidence": 0.8}
                )
            ]
        )

        result = parser.parse(definition)

        step = result.steps[0]
        assert isinstance(step, StepSignature)
        assert step.kwargs == {"segment_type": "cajon", "confidence": 0.8}


class TestParseGroup:
    """Tests for parsing group (parallel) structures."""

    def test_parse_group_in_chain(self, parser: PipelineParser) -> None:
        """Parse group nested in chain."""
        definition = PipelineDefinition(
            steps=[
                StepDefinition(name="step_a"),
                GroupDefinition(
                    steps=[
                        StepDefinition(name="step_b"),
                        StepDefinition(name="step_c"),
                    ]
                ),
            ]
        )

        result = parser.parse(definition)

        assert len(result.steps) == 2
        assert isinstance(result.steps[0], StepSignature)
        assert isinstance(result.steps[1], Group)

        group = result.steps[1]
        assert len(group.steps) == 2
        assert group.steps[0].name == "step_b"
        assert group.steps[1].name == "step_c"


class TestParseChord:
    """Tests for parsing chord (parallel + callback) structures."""

    def test_parse_chord_with_callback(self, parser: PipelineParser) -> None:
        """Parse chord with group and callback."""
        definition = PipelineDefinition(
            steps=[
                ChordDefinition(
                    group=GroupDefinition(
                        steps=[
                            StepDefinition(name="step_a"),
                            StepDefinition(name="step_b"),
                        ]
                    ),
                    callback=StepDefinition(name="aggregate"),
                )
            ]
        )

        result = parser.parse(definition)

        assert len(result.steps) == 1
        chord = result.steps[0]
        assert isinstance(chord, Chord)
        assert isinstance(chord.group, Group)
        assert len(chord.group.steps) == 2
        assert chord.callback is not None
        assert chord.callback.name == "aggregate"

    def test_parse_chord_without_callback(self, parser: PipelineParser) -> None:
        """Parse chord without callback (just parallel group)."""
        definition = PipelineDefinition(
            steps=[
                ChordDefinition(
                    group=GroupDefinition(
                        steps=[
                            StepDefinition(name="step_a"),
                            StepDefinition(name="step_b"),
                        ]
                    ),
                    callback=None,
                )
            ]
        )

        result = parser.parse(definition)

        chord = result.steps[0]
        assert isinstance(chord, Chord)
        assert chord.callback is None


class TestParseComplexPipeline:
    """Tests for parsing complex nested pipelines."""

    def test_parse_agro_full_style_pipeline(self, parser: PipelineParser) -> None:
        """Parse a realistic agro pipeline with chord."""
        definition = PipelineDefinition(
            steps=[
                StepDefinition(name="segmentation"),
                ChordDefinition(
                    group=GroupDefinition(
                        steps=[
                            StepDefinition(
                                name="detection",
                                kwargs={"segment_type": "cajon"}
                            ),
                            StepDefinition(
                                name="detection",
                                kwargs={"segment_type": "segmento"}
                            ),
                        ]
                    ),
                    callback=StepDefinition(name="aggregate"),
                ),
                StepDefinition(name="step_a"),
            ]
        )

        result = parser.parse(definition)

        assert len(result.steps) == 3
        assert isinstance(result.steps[0], StepSignature)
        assert result.steps[0].name == "segmentation"
        assert isinstance(result.steps[1], Chord)
        assert isinstance(result.steps[2], StepSignature)


class TestValidation:
    """Tests for step validation (fail-fast)."""

    def test_missing_step_raises_error(self, parser: PipelineParser) -> None:
        """Raise PipelineParserError when step not in registry."""
        definition = PipelineDefinition(
            steps=[
                StepDefinition(name="nonexistent_step")
            ]
        )

        with pytest.raises(PipelineParserError) as exc_info:
            parser.parse(definition)

        assert "nonexistent_step" in str(exc_info.value)
        assert "not found in registry" in str(exc_info.value)

    def test_missing_step_in_group_raises_error(self, parser: PipelineParser) -> None:
        """Raise error for missing step inside group."""
        definition = PipelineDefinition(
            steps=[
                GroupDefinition(
                    steps=[
                        StepDefinition(name="step_a"),
                        StepDefinition(name="bad_step"),
                    ]
                )
            ]
        )

        with pytest.raises(PipelineParserError) as exc_info:
            parser.parse(definition)

        assert "bad_step" in str(exc_info.value)

    def test_missing_callback_step_raises_error(self, parser: PipelineParser) -> None:
        """Raise error for missing step in chord callback."""
        definition = PipelineDefinition(
            steps=[
                ChordDefinition(
                    group=GroupDefinition(
                        steps=[StepDefinition(name="step_a")]
                    ),
                    callback=StepDefinition(name="bad_callback"),
                )
            ]
        )

        with pytest.raises(PipelineParserError) as exc_info:
            parser.parse(definition)

        assert "bad_callback" in str(exc_info.value)

    def test_multiple_missing_steps_all_reported(self, parser: PipelineParser) -> None:
        """Report all missing steps, not just the first one."""
        definition = PipelineDefinition(
            steps=[
                StepDefinition(name="missing_1"),
                StepDefinition(name="missing_2"),
            ]
        )

        with pytest.raises(PipelineParserError) as exc_info:
            parser.parse(definition)

        error_msg = str(exc_info.value)
        assert "missing_1" in error_msg
        assert "missing_2" in error_msg


class TestFromJson:
    """Tests for parsing from raw JSON (through Pydantic)."""

    def test_parse_from_dict(self, parser: PipelineParser) -> None:
        """Parse pipeline from dict (as would come from JSON)."""
        json_data = {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "step_a"},
                {"type": "step", "name": "step_b", "kwargs": {"foo": "bar"}},
            ]
        }

        definition = PipelineDefinition.model_validate(json_data)
        result = parser.parse(definition)

        assert len(result.steps) == 2
        assert result.steps[0].name == "step_a"
        assert result.steps[1].kwargs == {"foo": "bar"}

    def test_parse_complex_from_dict(self, parser: PipelineParser) -> None:
        """Parse complex pipeline from dict."""
        json_data = {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "segmentation"},
                {
                    "type": "chord",
                    "group": {
                        "type": "group",
                        "steps": [
                            {"type": "step", "name": "detection", "kwargs": {"segment_type": "cajon"}},
                            {"type": "step", "name": "detection", "kwargs": {"segment_type": "segmento"}},
                        ]
                    },
                    "callback": {"type": "step", "name": "aggregate"}
                }
            ]
        }

        definition = PipelineDefinition.model_validate(json_data)
        result = parser.parse(definition)

        assert len(result.steps) == 2
        assert isinstance(result.steps[1], Chord)
