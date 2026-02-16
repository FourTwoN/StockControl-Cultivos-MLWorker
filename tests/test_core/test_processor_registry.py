"""Tests for processor registry."""

import pytest
from typing import Any
from pathlib import Path
from unittest.mock import MagicMock

from app.core.processor_registry import (
    ProcessorRegistry,
    get_processor_registry,
    Processor,
)
from app.core.industry_config import ModelConfig


class MockProcessor(Processor):
    """Mock processor for testing."""

    def __init__(self, model_config: ModelConfig | None = None, **kwargs: Any):
        self.model_config = model_config
        self.process_called = False

    async def process(
        self,
        image_path: Path,
        previous_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.process_called = True
        return {"mock": "result", "previous": previous_results}


class AnotherMockProcessor(Processor):
    """Another mock processor for testing."""

    def __init__(self, **kwargs: Any):
        pass

    async def process(
        self,
        image_path: Path,
        previous_results: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"another": "result"}


class TestProcessorRegistry:
    """Tests for ProcessorRegistry."""

    @pytest.fixture
    def registry(self) -> ProcessorRegistry:
        """Create a fresh registry instance."""
        return ProcessorRegistry()

    def test_register_processor(self, registry: ProcessorRegistry):
        registry.register("mock", MockProcessor)
        available = registry.get_available()
        assert "mock" in available

    def test_register_duplicate_warns(self, registry: ProcessorRegistry):
        """Test that registering a duplicate logs a warning but succeeds."""
        registry.register("mock", MockProcessor)
        # Second registration should work (overwrite)
        registry.register("mock", AnotherMockProcessor)
        # Should still be able to get processor
        processor = registry.get("mock")
        assert isinstance(processor, AnotherMockProcessor)

    def test_get_processor_creates_instance(self, registry: ProcessorRegistry):
        registry.register("mock", MockProcessor)
        processor = registry.get("mock")

        assert isinstance(processor, MockProcessor)

    def test_get_processor_with_model_config(self, registry: ProcessorRegistry):
        registry.register("mock", MockProcessor)

        model_config = ModelConfig(
            path="test.pt",
            confidence_threshold=0.8,
        )

        processor = registry.get("mock", model_config=model_config)

        assert isinstance(processor, MockProcessor)
        assert processor.model_config == model_config

    def test_get_processor_caches_instance(self, registry: ProcessorRegistry):
        registry.register("mock", MockProcessor)

        processor1 = registry.get("mock")
        processor2 = registry.get("mock")

        assert processor1 is processor2

    def test_get_nonexistent_processor_raises(self, registry: ProcessorRegistry):
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_get_available(self, registry: ProcessorRegistry):
        registry.register("mock1", MockProcessor)
        registry.register("mock2", AnotherMockProcessor)

        available = registry.get_available()

        assert "mock1" in available
        assert "mock2" in available
        assert len(available) == 2

    def test_clear_instances(self, registry: ProcessorRegistry):
        registry.register("mock", MockProcessor)
        processor = registry.get("mock")

        registry.clear_instances()

        # Getting again should create new instance
        processor2 = registry.get("mock")
        assert processor is not processor2


class TestGetProcessorRegistry:
    """Tests for singleton processor registry."""

    def test_returns_same_instance(self):
        registry1 = get_processor_registry()
        registry2 = get_processor_registry()
        assert registry1 is registry2

    def test_has_default_processors(self):
        registry = get_processor_registry()
        available = registry.get_available()

        # Should have the default processors registered
        assert "detection" in available
        assert "segmentation" in available
        assert "classification" in available
        assert "estimation" in available
