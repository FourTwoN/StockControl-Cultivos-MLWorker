"""Tests for pipeline step interfaces."""

import pytest
from pathlib import Path

from app.core.processing_context import ProcessingContext
from app.core.pipeline_step import PipelineStep


class TestProcessingContext:
    """Test ProcessingContext dataclass."""

    def test_create_context(self) -> None:
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="img-123",
            session_id="sess-456",
            image_path=Path("/tmp/test.jpg"),
            config={"num_bands": 4},
        )
        assert ctx.tenant_id == "test-tenant"
        assert ctx.raw_segments == []
        assert ctx.results == {}

    def test_context_immutable_update(self) -> None:
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="img-123",
            session_id="sess-456",
            image_path=Path("/tmp/test.jpg"),
            config={},
        )
        new_ctx = ctx.with_results({"sizes": {0: 2}})

        assert ctx.results == {}
        assert new_ctx.results == {"sizes": {0: 2}}

    def test_context_with_segments(self) -> None:
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="img-123",
            session_id="sess-456",
            image_path=Path("/tmp/test.jpg"),
            config={},
        )
        segments = [{"class": "cajon", "bbox": [0, 0, 100, 100]}]
        new_ctx = ctx.with_segments(segments)

        assert new_ctx.raw_segments == segments


class TestPipelineStep:
    """Test PipelineStep abstract class."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            PipelineStep()

    def test_concrete_step_requires_name(self) -> None:
        class BadStep(PipelineStep):
            async def execute(self, ctx):
                return ctx

        with pytest.raises(TypeError):
            BadStep()
