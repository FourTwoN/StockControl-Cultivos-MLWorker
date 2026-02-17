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
    # Database (Cloud SQL via Unix socket)
    # =========================================================================
    db_user: str = Field(
        default="demeter_app",
        description="Database user",
    )
    db_password: str = Field(
        default="",
        description="Database password (from Secret Manager)",
    )
    db_name: str = Field(
        default="demeter",
        description="Database name",
    )
    db_connection_name: str = Field(
        default="",
        description="Cloud SQL connection name (project:region:instance)",
    )
    db_host: str = Field(
        default="localhost",
        description="Database host (for local development)",
    )
    db_port: int = Field(
        default=5432,
        description="Database port (for local development)",
    )
    db_pool_size: int = Field(
        default=5,
        description="Database connection pool size",
    )
    db_pool_max_overflow: int = Field(
        default=10,
        description="Max overflow connections beyond pool size",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url(self) -> str:
        """Build database URL based on environment.

        In Cloud Run, uses Unix socket for Cloud SQL.
        In local dev, uses TCP connection.
        """
        if self.db_connection_name:
            # Cloud Run: Unix socket connection
            socket_path = f"/cloudsql/{self.db_connection_name}"
            return (
                f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
                f"@/{self.db_name}?host={socket_path}"
            )
        # Local development: TCP connection
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
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
    # ML Models
    # =========================================================================
    model_path: str = Field(
        default="./models",
        description="Path to ML models (local or gs:// for GCS)",
    )
    detection_model: str = Field(
        default="detect.pt",
        description="Detection model filename",
    )
    segmentation_model: str = Field(
        default="segment.pt",
        description="Segmentation model filename",
    )
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
