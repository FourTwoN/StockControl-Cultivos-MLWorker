"""Tests for pipeline orchestrator."""

import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.pipeline import Pipeline, PipelineResult, StepResult
from app.core.industry_config import (
    IndustryConfig,
    ModelConfig,
    PipelineConfig,
)
from app.core.processor_registry import ProcessorRegistry, Processor


class MockSuccessProcessor(Processor):
    """Mock processor that always succeeds."""

    def __init__(self, result_data: dict[str, Any] | None = None, **kwargs: Any):
        self.result_data = result_data or {"detected": True}
        self.call_count = 0
        self.received_previous = None

    async def process(
        self,
        image_path: Path,
        previous_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.call_count += 1
        self.received_previous = previous_results
        return self.result_data


class MockFailingProcessor(Processor):
    """Mock processor that always fails."""

    def __init__(self, **kwargs: Any):
        pass

    async def process(
        self,
        image_path: Path,
        previous_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise RuntimeError("Processing failed")


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_successful_step_result(self):
        result = StepResult(
            step_name="detection",
            success=True,
            data={"count": 5},
            duration_ms=100,
        )
        assert result.step_name == "detection"
        assert result.success is True
        assert result.data == {"count": 5}
        assert result.error is None

    def test_failed_step_result(self):
        result = StepResult(
            step_name="detection",
            success=False,
            data=None,
            duration_ms=50,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_successful_pipeline_result(self):
        steps = [
            StepResult(step_name="detection", success=True, data={"count": 5}, duration_ms=100),
            StepResult(step_name="estimation", success=True, data={"total": 10}, duration_ms=50),
        ]
        result = PipelineResult(
            pipeline_name="FULL",
            success=True,
            steps=steps,
            total_duration_ms=150,
        )
        assert result.success is True
        assert len(result.steps) == 2
        assert result.error is None

    def test_get_step_data(self):
        steps = [
            StepResult(step_name="detection", success=True, data={"count": 5}, duration_ms=100),
            StepResult(step_name="estimation", success=True, data={"total": 10}, duration_ms=50),
        ]
        result = PipelineResult(
            pipeline_name="FULL",
            success=True,
            steps=steps,
            total_duration_ms=150,
        )

        detection_data = result.get_step_data("detection")
        assert detection_data == {"count": 5}

        estimation_data = result.get_step_data("estimation")
        assert estimation_data == {"total": 10}

    def test_get_step_data_not_found(self):
        result = PipelineResult(pipeline_name="EMPTY", success=True, steps=[])
        data = result.get_step_data("nonexistent")
        assert data is None

    def test_to_dict(self):
        steps = [
            StepResult(step_name="detection", success=True, data={"count": 5}, duration_ms=100),
        ]
        result = PipelineResult(
            pipeline_name="DETECTION",
            success=True,
            steps=steps,
            total_duration_ms=100,
        )

        result_dict = result.to_dict()

        assert "detection" in result_dict
        assert result_dict["detection"] == {"count": 5}
        assert result_dict["pipeline_name"] == "DETECTION"


class TestPipeline:
    """Tests for Pipeline orchestrator."""

    @pytest.fixture
    def sample_config(self) -> IndustryConfig:
        """Create a sample industry config."""
        return IndustryConfig(
            industry="test",
            version="1.0.0",
            models={
                "detection": ModelConfig(
                    path="detect.pt",
                    confidence_threshold=0.8,
                    enabled=True,
                ),
                "estimation": ModelConfig(
                    path="estimate.pt",
                    confidence_threshold=0.7,
                    enabled=True,
                ),
            },
            pipelines={
                "DETECTION": PipelineConfig(name="DETECTION", steps=("detection",)),
                "FULL": PipelineConfig(name="FULL", steps=("detection", "estimation")),
            },
            settings={},
        )

    @pytest.fixture
    def mock_registry(self) -> ProcessorRegistry:
        """Create a mock registry with test processors."""
        registry = ProcessorRegistry()
        return registry

    @pytest.mark.asyncio
    async def test_execute_single_step_pipeline(
        self, sample_config: IndustryConfig, mock_registry: ProcessorRegistry
    ):
        # Register mock processor
        detection_processor = MockSuccessProcessor({"detections": [{"label": "plant"}]})
        mock_registry.register("detection", lambda **kwargs: detection_processor)

        pipeline = Pipeline(config=sample_config, registry=mock_registry)

        result = await pipeline.execute(
            pipeline_name="DETECTION",
            image_path=Path("/tmp/test.jpg"),
        )

        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].step_name == "detection"
        assert detection_processor.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_multi_step_pipeline(
        self, sample_config: IndustryConfig, mock_registry: ProcessorRegistry
    ):
        detection_processor = MockSuccessProcessor({"detections": [{"label": "plant"}]})
        estimation_processor = MockSuccessProcessor({"total_count": 5})

        mock_registry.register("detection", lambda **kwargs: detection_processor)
        mock_registry.register("estimation", lambda **kwargs: estimation_processor)

        pipeline = Pipeline(config=sample_config, registry=mock_registry)

        result = await pipeline.execute(
            pipeline_name="FULL",
            image_path=Path("/tmp/test.jpg"),
        )

        assert result.success is True
        assert len(result.steps) == 2
        assert detection_processor.call_count == 1
        assert estimation_processor.call_count == 1

        # Both steps should have completed
        detection_data = result.get_step_data("detection")
        estimation_data = result.get_step_data("estimation")
        assert detection_data == {"detections": [{"label": "plant"}]}
        assert estimation_data == {"total_count": 5}

    @pytest.mark.asyncio
    async def test_execute_pipeline_step_failure_stops_execution(
        self, sample_config: IndustryConfig, mock_registry: ProcessorRegistry
    ):
        failing_processor = MockFailingProcessor()
        estimation_processor = MockSuccessProcessor({"total_count": 5})

        mock_registry.register("detection", lambda **kwargs: failing_processor)
        mock_registry.register("estimation", lambda **kwargs: estimation_processor)

        pipeline = Pipeline(config=sample_config, registry=mock_registry)

        result = await pipeline.execute(
            pipeline_name="FULL",
            image_path=Path("/tmp/test.jpg"),
        )

        assert result.success is False
        assert result.error is not None
        # Estimation should not have been called
        assert estimation_processor.call_count == 0

    @pytest.mark.asyncio
    async def test_execute_unknown_pipeline_returns_error(
        self, sample_config: IndustryConfig, mock_registry: ProcessorRegistry
    ):
        """Test that unknown pipeline returns an error result."""
        pipeline = Pipeline(config=sample_config, registry=mock_registry)

        result = await pipeline.execute(
            pipeline_name="NONEXISTENT",
            image_path=Path("/tmp/test.jpg"),
        )

        # Pipeline returns error result instead of raising
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "nonexistent" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_accumulates_results(
        self, sample_config: IndustryConfig, mock_registry: ProcessorRegistry
    ):
        detection_processor = MockSuccessProcessor({"boxes": [1, 2, 3]})
        estimation_processor = MockSuccessProcessor({"count": 3})

        mock_registry.register("detection", lambda **kwargs: detection_processor)
        mock_registry.register("estimation", lambda **kwargs: estimation_processor)

        pipeline = Pipeline(config=sample_config, registry=mock_registry)

        result = await pipeline.execute(
            pipeline_name="FULL",
            image_path=Path("/tmp/test.jpg"),
        )

        # Check accumulated results in to_dict
        result_dict = result.to_dict()
        assert "detection" in result_dict
        assert "estimation" in result_dict
        assert result_dict["detection"] == {"boxes": [1, 2, 3]}
        assert result_dict["estimation"] == {"count": 3}
