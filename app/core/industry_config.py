"""Industry Configuration - Load and manage industry-specific config.

Configuration is loaded from:
1. GCS (production): gs://demeter-models/{industry}/config.yaml
2. Local file (development): config/{industry}.yaml
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model type."""

    path: str
    confidence_threshold: float = 0.80
    classes: tuple[str, ...] = field(default_factory=tuple)
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        classes = data.get("classes", [])
        return cls(
            path=data.get("path", ""),
            confidence_threshold=data.get("confidence_threshold", 0.80),
            classes=tuple(classes) if classes else (),
            enabled=data.get("enabled", True),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a processing pipeline."""

    name: str
    steps: tuple[str, ...]

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        steps = data.get("steps", [])
        return cls(name=name, steps=tuple(steps))


@dataclass(frozen=True)
class IndustryConfig:
    """Complete configuration for an industry.

    Loaded from YAML file either locally or from GCS.
    """

    industry: str
    version: str
    models: dict[str, ModelConfig]
    pipelines: dict[str, PipelineConfig]
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_content: str, industry: str) -> "IndustryConfig":
        """Parse YAML content into IndustryConfig.

        Args:
            yaml_content: Raw YAML string
            industry: Industry identifier (for logging)

        Returns:
            Parsed IndustryConfig
        """
        data = yaml.safe_load(yaml_content)

        # Parse models
        models: dict[str, ModelConfig] = {}
        for model_name, model_data in data.get("models", {}).items():
            models[model_name] = ModelConfig.from_dict(model_data)

        # Parse pipelines
        pipelines: dict[str, PipelineConfig] = {}
        for pipeline_name, pipeline_data in data.get("pipelines", {}).items():
            pipelines[pipeline_name] = PipelineConfig.from_dict(pipeline_name, pipeline_data)

        return cls(
            industry=data.get("industry", industry),
            version=data.get("version", "0.0.0"),
            models=models,
            pipelines=pipelines,
            settings=data.get("settings", {}),
        )

    def get_model_config(self, model_type: str) -> ModelConfig | None:
        """Get configuration for a specific model type."""
        config = self.models.get(model_type)
        if config and config.enabled:
            return config
        return None

    def get_pipeline(self, pipeline_name: str) -> PipelineConfig | None:
        """Get a pipeline by name."""
        return self.pipelines.get(pipeline_name)

    def get_available_pipelines(self) -> list[str]:
        """Get list of available pipeline names."""
        return list(self.pipelines.keys())


class IndustryConfigLoader:
    """Loads industry configuration from various sources."""

    def __init__(self) -> None:
        self._cache: dict[str, IndustryConfig] = {}

    async def load(self, industry: str | None = None) -> IndustryConfig:
        """Load industry configuration.

        Args:
            industry: Industry identifier. Defaults to settings.industry.

        Returns:
            Loaded IndustryConfig

        Raises:
            FileNotFoundError: If config not found
            ValueError: If config is invalid
        """
        industry = industry or settings.industry

        # Check cache
        if industry in self._cache:
            logger.debug("Using cached industry config", industry=industry)
            return self._cache[industry]

        # Try loading from different sources
        config = await self._load_from_source(industry)

        # Cache and return
        self._cache[industry] = config
        logger.info(
            "Industry config loaded",
            industry=industry,
            version=config.version,
            pipelines=list(config.pipelines.keys()),
            models=list(config.models.keys()),
        )

        return config

    async def _load_from_source(self, industry: str) -> IndustryConfig:
        """Load config from appropriate source based on environment."""
        # In production, load from GCS
        if settings.model_path.startswith("gs://"):
            return await self._load_from_gcs(industry)

        # In development, load from local file
        return self._load_from_local(industry)

    async def _load_from_gcs(self, industry: str) -> IndustryConfig:
        """Load config from Google Cloud Storage."""
        from google.cloud import storage

        # Parse GCS path
        # model_path format: gs://bucket/path
        gcs_path = settings.model_path.replace("gs://", "")
        parts = gcs_path.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # Build config path
        config_blob = f"{prefix}/{industry}/config.yaml".lstrip("/")

        logger.info(
            "Loading config from GCS",
            bucket=bucket_name,
            blob=config_blob,
        )

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(config_blob)

        if not blob.exists():
            raise FileNotFoundError(
                f"Config not found: gs://{bucket_name}/{config_blob}"
            )

        yaml_content = blob.download_as_text()
        return IndustryConfig.from_yaml(yaml_content, industry)

    def _load_from_local(self, industry: str) -> IndustryConfig:
        """Load config from local file system."""
        # Check multiple locations
        search_paths = [
            Path(settings.model_path) / industry / "config.yaml",
            Path("config") / f"{industry}.yaml",
            Path(__file__).parent.parent.parent / "config" / f"{industry}.yaml",
        ]

        for path in search_paths:
            if path.exists():
                logger.info("Loading config from local file", path=str(path))
                yaml_content = path.read_text()
                return IndustryConfig.from_yaml(yaml_content, industry)

        # Create default config if not found
        logger.warning(
            "Config not found, using defaults",
            industry=industry,
            searched=str(search_paths),
        )
        return self._create_default_config(industry)

    def _create_default_config(self, industry: str) -> IndustryConfig:
        """Create a default configuration for development."""
        return IndustryConfig(
            industry=industry,
            version="0.0.0-default",
            models={
                "detection": ModelConfig(
                    path="detect.pt",
                    confidence_threshold=settings.confidence_threshold,
                    classes=("object",),
                    enabled=True,
                ),
                "segmentation": ModelConfig(
                    path="segment.pt",
                    confidence_threshold=0.50,
                    classes=("region",),
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
        )

    def clear_cache(self) -> None:
        """Clear the config cache."""
        self._cache.clear()
        logger.info("Industry config cache cleared")


# Singleton loader
_loader: IndustryConfigLoader | None = None


def get_config_loader() -> IndustryConfigLoader:
    """Get the singleton config loader."""
    global _loader
    if _loader is None:
        _loader = IndustryConfigLoader()
    return _loader


async def load_industry_config(industry: str | None = None) -> IndustryConfig:
    """Convenience function to load industry config."""
    loader = get_config_loader()
    return await loader.load(industry)
