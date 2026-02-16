"""Tests for post-processor steps."""

from pathlib import Path

import pytest

from app.core.processing_context import ProcessingContext
from app.steps.post.segment_filter import SegmentFilterStep
from app.steps.post.size_calculator import SizeCalculatorStep
from app.steps.post.species_distributor import SpeciesDistributorStep


@pytest.fixture
def base_context() -> ProcessingContext:
    """Create a base processing context for testing."""
    return ProcessingContext(
        tenant_id="test-tenant",
        image_id="test-image",
        session_id="test-session",
        image_path=Path("/tmp/test.jpg"),
        config={},
    )


class TestSegmentFilterStep:
    """Tests for SegmentFilterStep."""

    @pytest.fixture
    def step(self) -> SegmentFilterStep:
        """Create step instance."""
        return SegmentFilterStep()

    def test_name_property(self, step: SegmentFilterStep):
        """Test that step has correct name."""
        assert step.name == "segment_filter"

    @pytest.mark.asyncio
    async def test_no_filter_type_returns_unchanged(
        self, step: SegmentFilterStep, base_context: ProcessingContext
    ):
        """Test that missing filter type returns original segments."""
        segments = [
            {"type": "segmento", "area": 100},
            {"type": "claro-cajon", "area": 200},
        ]
        ctx = base_context.with_segments(segments)

        result_ctx = await step.execute(ctx)

        assert result_ctx.raw_segments == segments

    @pytest.mark.asyncio
    async def test_largest_claro_filter_keeps_largest_segmento(
        self, step: SegmentFilterStep, base_context: ProcessingContext
    ):
        """Test that largest_claro filter keeps only largest segmento."""
        segments = [
            {"type": "segmento", "area": 100},
            {"type": "segmento", "area": 300},
            {"type": "segmento", "area": 200},
            {"type": "other", "area": 50},
        ]
        ctx = base_context.with_segments(segments)
        ctx = ProcessingContext(
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            session_id=ctx.session_id,
            image_path=ctx.image_path,
            config={"segment_filter_type": "largest_claro"},
            raw_segments=segments,
        )

        result_ctx = await step.execute(ctx)

        assert len(result_ctx.raw_segments) == 2
        assert {"type": "segmento", "area": 300} in result_ctx.raw_segments
        assert {"type": "other", "area": 50} in result_ctx.raw_segments

    @pytest.mark.asyncio
    async def test_largest_claro_filter_keeps_largest_claro_cajon(
        self, step: SegmentFilterStep, base_context: ProcessingContext
    ):
        """Test that largest_claro filter keeps only largest claro-cajon."""
        segments = [
            {"type": "claro-cajon", "area": 100},
            {"type": "claro-cajon", "area": 300},
            {"type": "claro-cajon", "area": 200},
            {"type": "other", "area": 50},
        ]
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"segment_filter_type": "largest_claro"},
            raw_segments=segments,
        )

        result_ctx = await step.execute(ctx)

        assert len(result_ctx.raw_segments) == 2
        assert {"type": "claro-cajon", "area": 300} in result_ctx.raw_segments
        assert {"type": "other", "area": 50} in result_ctx.raw_segments

    @pytest.mark.asyncio
    async def test_largest_claro_filter_preserves_other_types(
        self, step: SegmentFilterStep, base_context: ProcessingContext
    ):
        """Test that largest_claro filter preserves all non-target types."""
        segments = [
            {"type": "segmento", "area": 100},
            {"type": "segmento", "area": 300},
            {"type": "bandeja", "area": 50},
            {"type": "planta", "area": 75},
        ]
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"segment_filter_type": "largest_claro"},
            raw_segments=segments,
        )

        result_ctx = await step.execute(ctx)

        assert len(result_ctx.raw_segments) == 3
        assert {"type": "segmento", "area": 300} in result_ctx.raw_segments
        assert {"type": "bandeja", "area": 50} in result_ctx.raw_segments
        assert {"type": "planta", "area": 75} in result_ctx.raw_segments

    @pytest.mark.asyncio
    async def test_largest_claro_filter_empty_segments(
        self, step: SegmentFilterStep, base_context: ProcessingContext
    ):
        """Test that filter handles empty segments list."""
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"segment_filter_type": "largest_claro"},
            raw_segments=[],
        )

        result_ctx = await step.execute(ctx)

        assert result_ctx.raw_segments == []


class TestSizeCalculatorStep:
    """Tests for SizeCalculatorStep."""

    @pytest.fixture
    def step(self) -> SizeCalculatorStep:
        """Create step instance."""
        return SizeCalculatorStep()

    def test_name_property(self, step: SizeCalculatorStep):
        """Test that step has correct name."""
        assert step.name == "size_calculator"

    @pytest.mark.asyncio
    async def test_calculates_sizes_using_z_scores(
        self, step: SizeCalculatorStep, base_context: ProcessingContext
    ):
        """Test that sizes are calculated using z-scores."""
        detections = [
            {"bbox": [0, 0, 100, 50]},  # area = 5000
            {"bbox": [0, 0, 200, 100]},  # area = 20000
            {"bbox": [0, 0, 150, 75]},  # area = 11250
            {"bbox": [0, 0, 50, 25]},  # area = 1250
        ]
        ctx = base_context.with_detections(detections)
        ctx = ProcessingContext(
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            session_id=ctx.session_id,
            image_path=ctx.image_path,
            config={"num_bands": 4, "image_height": 1000},
            raw_detections=detections,
        )

        result_ctx = await step.execute(ctx)

        assert "sizes" in result_ctx.results
        sizes = result_ctx.results["sizes"]
        assert len(sizes) == 4
        # All size values should be 1, 2, 3, or 4
        for size_id in sizes.values():
            assert size_id in [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_size_distribution_with_extreme_values(
        self, step: SizeCalculatorStep, base_context: ProcessingContext
    ):
        """Test size calculation with extreme outliers."""
        detections = [
            {"bbox": [0, 0, 10, 10]},  # Very small
            {"bbox": [0, 0, 100, 100]},  # Medium
            {"bbox": [0, 0, 100, 100]},  # Medium
            {"bbox": [0, 0, 500, 500]},  # Very large
        ]
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"num_bands": 4, "image_height": 1000},
            raw_detections=detections,
        )

        result_ctx = await step.execute(ctx)

        sizes = result_ctx.results["sizes"]
        # Should have different sizes assigned
        unique_sizes = set(sizes.values())
        assert len(unique_sizes) >= 2

    @pytest.mark.asyncio
    async def test_empty_detections_returns_empty_sizes(
        self, step: SizeCalculatorStep, base_context: ProcessingContext
    ):
        """Test that empty detections returns empty sizes dict."""
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"num_bands": 4, "image_height": 1000},
            raw_detections=[],
        )

        result_ctx = await step.execute(ctx)

        assert result_ctx.results["sizes"] == {}

    @pytest.mark.asyncio
    async def test_single_detection_gets_medium_size(
        self, step: SizeCalculatorStep, base_context: ProcessingContext
    ):
        """Test that single detection gets assigned medium size."""
        detections = [{"bbox": [0, 0, 100, 100]}]
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"num_bands": 4, "image_height": 1000},
            raw_detections=detections,
        )

        result_ctx = await step.execute(ctx)

        sizes = result_ctx.results["sizes"]
        assert sizes[0] == 2  # SIZE_M


class TestSpeciesDistributorStep:
    """Tests for SpeciesDistributorStep."""

    @pytest.fixture
    def step(self) -> SpeciesDistributorStep:
        """Create step instance."""
        return SpeciesDistributorStep()

    def test_name_property(self, step: SpeciesDistributorStep):
        """Test that step has correct name."""
        assert step.name == "species_distributor"

    @pytest.mark.asyncio
    async def test_distributes_detections_equitably(
        self, step: SpeciesDistributorStep, base_context: ProcessingContext
    ):
        """Test that detections are distributed equitably across species."""
        detections = [
            {"bbox": [0, 0, 100, 100]},
            {"bbox": [0, 0, 100, 100]},
            {"bbox": [0, 0, 100, 100]},
        ]
        sizes = {0: 1, 1: 2, 2: 3}
        ctx = base_context.with_detections(detections).with_results({"sizes": sizes})
        ctx = ProcessingContext(
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            session_id=ctx.session_id,
            image_path=ctx.image_path,
            config={"species": ["species_a", "species_b"]},
            raw_detections=detections,
            results={"sizes": sizes},
        )

        result_ctx = await step.execute(ctx)

        classifications = result_ctx.results["classifications"]
        assert len(classifications) == 3
        # Check that species are distributed
        species_counts = {}
        for cls in classifications:
            species = cls["species"]
            species_counts[species] = species_counts.get(species, 0) + 1

        # Should be roughly equal distribution
        assert "species_a" in species_counts
        assert "species_b" in species_counts

    @pytest.mark.asyncio
    async def test_includes_size_in_classifications(
        self, step: SpeciesDistributorStep, base_context: ProcessingContext
    ):
        """Test that classifications include size information."""
        detections = [{"bbox": [0, 0, 100, 100]}]
        sizes = {0: 3}
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"species": ["species_a"]},
            raw_detections=detections,
            results={"sizes": sizes},
        )

        result_ctx = await step.execute(ctx)

        classifications = result_ctx.results["classifications"]
        assert len(classifications) == 1
        assert classifications[0]["size_id"] == 3
        assert "species" in classifications[0]

    @pytest.mark.asyncio
    async def test_empty_detections_returns_empty_classifications(
        self, step: SpeciesDistributorStep, base_context: ProcessingContext
    ):
        """Test that empty detections returns empty classifications."""
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"species": ["species_a"]},
            raw_detections=[],
            results={"sizes": {}},
        )

        result_ctx = await step.execute(ctx)

        assert result_ctx.results["classifications"] == []

    @pytest.mark.asyncio
    async def test_single_species_assigns_all_to_that_species(
        self, step: SpeciesDistributorStep, base_context: ProcessingContext
    ):
        """Test that single species assigns all detections to it."""
        detections = [
            {"bbox": [0, 0, 100, 100]},
            {"bbox": [0, 0, 100, 100]},
        ]
        sizes = {0: 1, 1: 2}
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"species": ["species_a"]},
            raw_detections=detections,
            results={"sizes": sizes},
        )

        result_ctx = await step.execute(ctx)

        classifications = result_ctx.results["classifications"]
        assert len(classifications) == 2
        assert all(cls["species"] == "species_a" for cls in classifications)

    @pytest.mark.asyncio
    async def test_preserves_detection_index(
        self, step: SpeciesDistributorStep, base_context: ProcessingContext
    ):
        """Test that detection index is preserved in classifications."""
        detections = [
            {"bbox": [0, 0, 100, 100]},
            {"bbox": [0, 0, 200, 200]},
        ]
        sizes = {0: 1, 1: 3}
        ctx = ProcessingContext(
            tenant_id=base_context.tenant_id,
            image_id=base_context.image_id,
            session_id=base_context.session_id,
            image_path=base_context.image_path,
            config={"species": ["species_a", "species_b"]},
            raw_detections=detections,
            results={"sizes": sizes},
        )

        result_ctx = await step.execute(ctx)

        classifications = result_ctx.results["classifications"]
        assert len(classifications) == 2
        # Check that we have entries for both indices
        indices = [cls["detection_idx"] for cls in classifications]
        assert 0 in indices
        assert 1 in indices
