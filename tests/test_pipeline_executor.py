"""Tests for pipeline executor."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.pipeline_dsl import chain, chord, group, step
from app.core.pipeline_executor import PipelineExecutor
from app.core.processing_context import ProcessingContext


@pytest.fixture
def base_context(tmp_path: Path) -> ProcessingContext:
    """Create a base processing context for tests."""
    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake image")

    return ProcessingContext(
        tenant_id="test-tenant",
        image_id="test-image",
        session_id="test-session",
        image_path=image_path,
        config={},
    )


@pytest.fixture
def mock_step_registry():
    """Mock StepRegistry to return controllable step instances."""
    with patch("app.core.pipeline_executor.StepRegistry") as mock:
        yield mock


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""

    @pytest.mark.asyncio
    async def test_execute_single_step(
        self, base_context: ProcessingContext, mock_step_registry
    ):
        """Executor runs single step."""
        mock_step = MagicMock()
        mock_step.execute = AsyncMock(
            return_value=base_context.with_results({"key": "value"})
        )
        mock_step_registry.get.return_value = mock_step

        executor = PipelineExecutor()
        result = await executor.execute(step("test_step"), base_context)

        mock_step_registry.get.assert_called_once_with("test_step")
        mock_step.execute.assert_called_once()
        assert result.results["key"] == "value"

    @pytest.mark.asyncio
    async def test_execute_chain(
        self, base_context: ProcessingContext, mock_step_registry
    ):
        """Executor runs chain sequentially."""
        call_order = []

        def create_mock_step(name: str):
            mock = MagicMock()

            async def execute(ctx):
                call_order.append(name)
                return ctx.with_results({name: True})

            mock.execute = execute
            return mock

        mock_step_registry.get.side_effect = lambda n: create_mock_step(n)

        pipeline = chain(step("step1"), step("step2"), step("step3"))
        executor = PipelineExecutor()
        result = await executor.execute(pipeline, base_context)

        assert call_order == ["step1", "step2", "step3"]
        assert result.results["step1"] is True
        assert result.results["step2"] is True
        assert result.results["step3"] is True

    @pytest.mark.asyncio
    async def test_execute_group_parallel(
        self, base_context: ProcessingContext, mock_step_registry
    ):
        """Executor runs group in parallel."""
        execution_times = {}

        def create_mock_step(name: str, delay: float):
            mock = MagicMock()

            async def execute(ctx):
                start = asyncio.get_event_loop().time()
                await asyncio.sleep(delay)
                end = asyncio.get_event_loop().time()
                execution_times[name] = {"start": start, "end": end}
                return ctx.with_detections([{"source": name}])

            mock.execute = execute
            return mock

        mock_step_registry.get.side_effect = lambda n: create_mock_step(n, 0.1)

        pipeline = group(step("branch1"), step("branch2"))
        executor = PipelineExecutor()

        start = asyncio.get_event_loop().time()
        result = await executor.execute(pipeline, base_context)
        total_time = asyncio.get_event_loop().time() - start

        # If parallel, total time should be ~0.1s, not ~0.2s
        assert total_time < 0.15, "Group should execute in parallel"

        # Both detections should be merged
        assert len(result.raw_detections) == 2

    @pytest.mark.asyncio
    async def test_execute_chord(
        self, base_context: ProcessingContext, mock_step_registry
    ):
        """Executor runs chord: group then callback."""
        call_order = []

        def create_mock_step(name: str):
            mock = MagicMock()

            async def execute(ctx):
                call_order.append(name)
                if name == "aggregate":
                    return ctx.with_results({"aggregated": True})
                return ctx.with_detections([{"from": name}])

            mock.execute = execute
            return mock

        mock_step_registry.get.side_effect = lambda n: create_mock_step(n)

        pipeline = chord(
            group(step("branch1"), step("branch2")),
            callback=step("aggregate"),
        )
        executor = PipelineExecutor()
        result = await executor.execute(pipeline, base_context)

        # Callback should run after group
        assert "aggregate" in call_order
        assert call_order.index("aggregate") > call_order.index("branch1")
        assert call_order.index("aggregate") > call_order.index("branch2")

        assert result.results["aggregated"] is True
        assert len(result.raw_detections) == 2

    @pytest.mark.asyncio
    async def test_step_config_injection(
        self, base_context: ProcessingContext, mock_step_registry
    ):
        """Executor injects step kwargs into context.step_config."""
        captured_config = {}

        mock_step = MagicMock()

        async def capture_execute(ctx):
            captured_config.update(ctx.step_config)
            return ctx

        mock_step.execute = capture_execute
        mock_step_registry.get.return_value = mock_step

        pipeline = step("detection", segment_type="cajon", confidence=0.8)
        executor = PipelineExecutor()
        await executor.execute(pipeline, base_context)

        assert captured_config["segment_type"] == "cajon"
        assert captured_config["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_merge_contexts_combines_detections(
        self, base_context: ProcessingContext
    ):
        """_merge_contexts combines detections from all branches."""
        executor = PipelineExecutor()

        ctx1 = base_context.with_detections([{"id": 1}, {"id": 2}])
        ctx2 = base_context.with_detections([{"id": 3}])
        ctx3 = base_context.with_detections([{"id": 4}, {"id": 5}])

        merged = executor._merge_contexts(base_context, [ctx1, ctx2, ctx3])

        assert len(merged.raw_detections) == 5
        ids = [d["id"] for d in merged.raw_detections]
        assert ids == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_merge_contexts_preserves_segments(
        self, base_context: ProcessingContext
    ):
        """_merge_contexts preserves segments from original context."""
        executor = PipelineExecutor()

        original = base_context.with_segments([{"segment_idx": 0}])
        branch1 = original.with_detections([{"id": 1}])
        branch2 = original.with_detections([{"id": 2}])

        merged = executor._merge_contexts(original, [branch1, branch2])

        # Segments should be from original
        assert len(merged.raw_segments) == 1
        assert merged.raw_segments[0]["segment_idx"] == 0

    @pytest.mark.asyncio
    async def test_full_agro_pipeline_structure(
        self, base_context: ProcessingContext, mock_step_registry
    ):
        """Full agro pipeline executes in correct order."""
        call_order = []

        def create_mock_step(name: str):
            mock = MagicMock()

            async def execute(ctx):
                call_order.append(name)
                if name == "segmentation":
                    return ctx.with_segments([{"segment_idx": 0}])
                if name in ("detection", "sahi_detection"):
                    return ctx.with_detections([{"from": name}])
                return ctx.with_results({name: True})

            mock.execute = execute
            return mock

        mock_step_registry.get.side_effect = lambda n: create_mock_step(n)

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

        executor = PipelineExecutor()
        result = await executor.execute(pipeline, base_context)

        # Verify order
        assert call_order[0] == "segmentation"
        assert call_order[1] == "segment_filter"
        # Group steps can be in any order
        assert set(call_order[2:4]) == {"sahi_detection", "detection"}
        assert call_order[4] == "aggregate_detections"
        assert call_order[5] == "size_calculator"
        assert call_order[6] == "species_distributor"

        # Verify final context
        assert len(result.raw_detections) == 2
        assert result.results.get("size_calculator") is True
        assert result.results.get("species_distributor") is True
