"""Tests for industry configuration loading."""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from app.core.industry_config import (
    IndustryConfig,
    ModelConfig,
    PipelineConfig,
    IndustryConfigLoader,
    get_config_loader,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_model_config_creation(self):
        config = ModelConfig(
            path="detect.pt",
            confidence_threshold=0.8,
            classes=("plant", "weed"),
            enabled=True,
        )
        assert config.path == "detect.pt"
        assert config.confidence_threshold == 0.8
        assert config.classes == ("plant", "weed")
        assert config.enabled is True

    def test_model_config_defaults(self):
        config = ModelConfig(
            path="detect.pt",
            confidence_threshold=0.5,
        )
        assert config.classes == ()
        assert config.enabled is True

    def test_model_config_from_dict(self):
        data = {
            "path": "detect.pt",
            "confidence_threshold": 0.75,
            "classes": ["plant", "weed", "pest"],
            "enabled": True,
        }
        config = ModelConfig.from_dict(data)
        assert config.path == "detect.pt"
        assert config.confidence_threshold == 0.75
        assert config.classes == ("plant", "weed", "pest")


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_pipeline_config_creation(self):
        config = PipelineConfig(
            name="DETECTION",
            steps=("detection", "segmentation", "estimation"),
        )
        assert config.name == "DETECTION"
        assert len(config.steps) == 3
        assert config.steps[0] == "detection"

    def test_pipeline_config_empty_steps(self):
        config = PipelineConfig(name="EMPTY", steps=())
        assert config.steps == ()

    def test_pipeline_config_from_dict(self):
        data = {"steps": ["detection", "estimation"]}
        config = PipelineConfig.from_dict("QUICK_COUNT", data)
        assert config.name == "QUICK_COUNT"
        assert config.steps == ("detection", "estimation")


class TestIndustryConfig:
    """Tests for IndustryConfig dataclass."""

    @pytest.fixture
    def sample_config(self) -> IndustryConfig:
        """Create a sample industry config."""
        return IndustryConfig(
            industry="agro",
            version="1.0.0",
            models={
                "detection": ModelConfig(
                    path="detect.pt",
                    confidence_threshold=0.8,
                    classes=("plant", "weed"),
                    enabled=True,
                ),
                "segmentation": ModelConfig(
                    path="segment.pt",
                    confidence_threshold=0.5,
                    classes=("field", "row"),
                    enabled=True,
                ),
            },
            pipelines={
                "DETECTION": PipelineConfig(name="DETECTION", steps=("detection",)),
                "FULL_PIPELINE": PipelineConfig(
                    name="FULL_PIPELINE",
                    steps=("segmentation", "detection", "estimation"),
                ),
            },
            settings={"default_pipeline": "DETECTION"},
        )

    def test_get_model_config(self, sample_config: IndustryConfig):
        model = sample_config.get_model_config("detection")
        assert model is not None
        assert model.path == "detect.pt"
        assert model.confidence_threshold == 0.8

    def test_get_model_config_not_found(self, sample_config: IndustryConfig):
        model = sample_config.get_model_config("nonexistent")
        assert model is None

    def test_get_pipeline(self, sample_config: IndustryConfig):
        pipeline = sample_config.get_pipeline("FULL_PIPELINE")
        assert pipeline is not None
        assert len(pipeline.steps) == 3
        assert pipeline.steps == ("segmentation", "detection", "estimation")

    def test_get_pipeline_not_found(self, sample_config: IndustryConfig):
        pipeline = sample_config.get_pipeline("NONEXISTENT")
        assert pipeline is None

    def test_get_available_pipelines(self, sample_config: IndustryConfig):
        pipelines = sample_config.get_available_pipelines()
        assert "DETECTION" in pipelines
        assert "FULL_PIPELINE" in pipelines
        assert len(pipelines) == 2

    def test_settings_dict_access(self, sample_config: IndustryConfig):
        """Test direct access to settings dictionary."""
        assert sample_config.settings.get("default_pipeline") == "DETECTION"
        assert sample_config.settings.get("nonexistent", "fallback") == "fallback"

    def test_from_yaml(self):
        yaml_content = """
industry: agro
version: "1.0.0"

models:
  detection:
    path: detect.pt
    confidence_threshold: 0.80
    classes:
      - plant
      - weed
    enabled: true

pipelines:
  DETECTION:
    steps:
      - detection
  FULL_PIPELINE:
    steps:
      - segmentation
      - detection
      - estimation

settings:
  default_pipeline: DETECTION
  use_sahi: false
"""
        config = IndustryConfig.from_yaml(yaml_content, "agro")

        assert config.industry == "agro"
        assert config.version == "1.0.0"
        assert "detection" in config.models
        assert config.models["detection"].confidence_threshold == 0.80
        assert "DETECTION" in config.pipelines
        assert "FULL_PIPELINE" in config.pipelines
        assert config.settings.get("use_sahi") is False


class TestIndustryConfigLoader:
    """Tests for IndustryConfigLoader."""

    @pytest.fixture
    def loader(self) -> IndustryConfigLoader:
        """Create a fresh loader instance."""
        return IndustryConfigLoader()

    def test_cache_functionality(self, loader: IndustryConfigLoader):
        """Test that configs are cached."""
        config = IndustryConfig(
            industry="test",
            version="1.0.0",
            models={},
            pipelines={},
        )
        loader._cache["test"] = config

        # Should return cached version
        cached = loader._cache.get("test")
        assert cached is config

    def test_clear_cache(self, loader: IndustryConfigLoader):
        config = IndustryConfig(
            industry="test",
            version="1.0.0",
            models={},
            pipelines={},
        )
        loader._cache["test"] = config

        loader.clear_cache()

        assert "test" not in loader._cache

    @pytest.mark.asyncio
    async def test_load_uses_cache(self, loader: IndustryConfigLoader):
        """Test that loader uses cache for subsequent calls."""
        # Pre-populate cache
        cached_config = IndustryConfig(
            industry="cached",
            version="1.0.0",
            models={},
            pipelines={},
        )
        loader._cache["cached"] = cached_config

        # Should return cached version without loading
        config = await loader.load("cached")

        assert config is cached_config
        assert config.industry == "cached"


class TestGetConfigLoader:
    """Tests for singleton config loader."""

    def test_returns_same_instance(self):
        loader1 = get_config_loader()
        loader2 = get_config_loader()
        assert loader1 is loader2
