"""Application configuration using Pydantic Settings.

Reads configuration from environment variables with sensible defaults.
All secrets should be provided via environment variables (from Secret Manager).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application
    # =========================================================================
    environment: Literal["prod", "staging", "dev"] = Field(
        default="dev",
        description="Environment name",
    )
    industry: Literal["agro", "vending"] = Field(
        default="agro",
        description="Industry identifier for queue names",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # =========================================================================
    # Cloud Storage
    # =========================================================================
    gcs_bucket: str = Field(
        default="demeter-images-dev",
        description="Cloud Storage bucket for images",
    )
    use_local_storage: bool = Field(
        default=False,
        description="Use local filesystem instead of GCS (for development)",
    )
    local_storage_root: str = Field(
        default="./local_storage",
        description="Root directory for local storage when use_local_storage=True",
    )

    # =========================================================================
    # ML Models (per-tenant)
    # =========================================================================
    # Models are loaded per-tenant from: gs://{gcs_bucket}/models/{tenant_id}/
    # No global model settings needed - each tenant has their own models
    confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Default confidence threshold for detection",
    )

    # =========================================================================
    # Backend API
    # =========================================================================
    backend_url: str = Field(
        default="http://localhost:8080",
        description="Demeter backend URL for callbacks",
    )
    backend_timeout: float = Field(
        default=30.0,
        description="Backend request timeout in seconds",
    )

    # =========================================================================
    # Cloud Tasks
    # =========================================================================
    cloudtasks_location: str = Field(
        default="us-central1",
        description="Cloud Tasks queue location",
    )
    cloudtasks_strict_validation: bool = Field(
        default=True,
        description="Require Cloud Tasks headers in non-dev environments",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def queue_photo_processing(self) -> str:
        """Full queue path for photo processing."""
        return f"{self.industry}-photo-processing"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def queue_image_compress(self) -> str:
        """Full queue path for image compression."""
        return f"{self.industry}-image-compress"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def queue_reports(self) -> str:
        """Full queue path for reports."""
        return f"{self.industry}-reports"

    # =========================================================================
    # Logging
    # =========================================================================
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_json: bool = Field(
        default=True,
        description="Output logs as JSON (for Cloud Logging)",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Singleton instance for convenience
settings = get_settings()
