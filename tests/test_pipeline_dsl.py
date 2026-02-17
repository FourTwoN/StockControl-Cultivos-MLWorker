"""Tests for pipeline DSL primitives."""

import pytest

from app.core.pipeline_dsl import (
    Chain,
    Chord,
    Group,
    StepSignature,
    chain,
    chord,
    group,
    step,
)


class TestStepSignature:
    """Tests for StepSignature dataclass."""

    def test_step_creates_signature(self):
        """step() creates StepSignature with name."""
        sig = step("detection")
        assert isinstance(sig, StepSignature)
        assert sig.name == "detection"
        assert sig.kwargs == {}

    def test_step_with_kwargs(self):
        """step() accepts keyword arguments."""
        sig = step("detection", segment_type="cajon", confidence=0.5)
        assert sig.name == "detection"
        assert sig.kwargs == {"segment_type": "cajon", "confidence": 0.5}

    def test_step_signature_is_frozen(self):
        """StepSignature is immutable."""
        sig = step("detection")
        with pytest.raises(AttributeError):
            sig.name = "other"


class TestChain:
    """Tests for Chain composition."""

    def test_chain_creates_chain(self):
        """chain() creates Chain with steps."""
        c = chain(step("a"), step("b"), step("c"))
        assert isinstance(c, Chain)
        assert len(c.steps) == 3

    def test_chain_preserves_order(self):
        """Chain preserves step order."""
        c = chain(step("first"), step("second"), step("third"))
        assert c.steps[0].name == "first"
        assert c.steps[1].name == "second"
        assert c.steps[2].name == "third"

    def test_chain_is_frozen(self):
        """Chain is immutable."""
        c = chain(step("a"))
        with pytest.raises(AttributeError):
            c.steps = ()

    def test_empty_chain(self):
        """Empty chain is valid."""
        c = chain()
        assert isinstance(c, Chain)
        assert len(c.steps) == 0


class TestGroup:
    """Tests for Group composition."""

    def test_group_creates_group(self):
        """group() creates Group with steps."""
        g = group(step("a"), step("b"))
        assert isinstance(g, Group)
        assert len(g.steps) == 2

    def test_group_is_frozen(self):
        """Group is immutable."""
        g = group(step("a"))
        with pytest.raises(AttributeError):
            g.steps = ()


class TestChord:
    """Tests for Chord composition."""

    def test_chord_creates_chord(self):
        """chord() creates Chord with group and callback."""
        g = group(step("a"), step("b"))
        c = chord(g, callback=step("aggregate"))

        assert isinstance(c, Chord)
        assert c.group == g
        assert c.callback.name == "aggregate"

    def test_chord_without_callback(self):
        """chord() works without callback."""
        g = group(step("a"))
        c = chord(g)

        assert c.callback is None

    def test_chord_is_frozen(self):
        """Chord is immutable."""
        g = group(step("a"))
        c = chord(g)
        with pytest.raises(AttributeError):
            c.group = group(step("b"))


class TestNestedComposition:
    """Tests for nested DSL structures."""

    def test_chain_with_chord(self):
        """chain() can contain chord."""
        pipeline = chain(
            step("segmentation"),
            chord(
                group(step("detection"), step("sahi_detection")),
                callback=step("aggregate"),
            ),
            step("postprocess"),
        )

        assert isinstance(pipeline, Chain)
        assert len(pipeline.steps) == 3
        assert isinstance(pipeline.steps[1], Chord)

    def test_agro_full_pipeline_structure(self):
        """Full agro pipeline has correct structure."""
        pipeline = chain(
            step("segmentation"),
            step("segment_filter"),
            chord(
                group(
                    step("sahi_detection", segment_type="claro-cajon"),
                    step("detection", segment_type="cajon"),
                ),
                callback=step("aggregate_detections"),
            ),
            step("size_calculator"),
            step("species_distributor"),
        )

        assert len(pipeline.steps) == 5
        assert pipeline.steps[0].name == "segmentation"
        assert pipeline.steps[1].name == "segment_filter"

        chord_step = pipeline.steps[2]
        assert isinstance(chord_step, Chord)
        assert len(chord_step.group.steps) == 2
        assert chord_step.callback.name == "aggregate_detections"

        # Check kwargs are preserved
        sahi_step = chord_step.group.steps[0]
        assert sahi_step.kwargs["segment_type"] == "claro-cajon"

        detection_step = chord_step.group.steps[1]
        assert detection_step.kwargs["segment_type"] == "cajon"
