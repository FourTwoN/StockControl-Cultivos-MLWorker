"""Task schemas for the unified processing endpoint."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ProcessingRequest(BaseModel):
    """Unified request payload for all ML processing tasks.

    Sent by Backend via Cloud Tasks to the single /tasks/process endpoint.
    """

    tenant_id: str = Field(
        min_length=1,
        max_length=100,
        description="Tenant ID for multi-tenant isolation",
    )
    session_id: UUID = Field(description="Processing session UUID")
    image_id: UUID = Field(description="Image UUID")
    image_url: str = Field(
        min_length=1,
        description="GCS URL to image (gs://bucket/path)",
    )
    pipeline: str = Field(
        default="DETECTION",
        description="Pipeline name from industry config (e.g., DETECTION, FULL_PIPELINE)",
    )
    options: dict[str, Any] | None = Field(
        default=None,
        description="Optional processing overrides (confidence_override, etc.)",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional callback URL for results notification",
    )

    model_config = {"extra": "forbid"}

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """Validate that image_url is a valid GCS URL."""
        if not v.startswith("gs://"):
            raise ValueError("image_url must be a GCS URL (gs://bucket/path)")
        return v

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        """Validate tenant_id format."""
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid tenant_id format")
        return v

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline(cls, v: str) -> str:
        """Validate pipeline name format."""
        if not v or not v.replace("_", "").isalnum():
            raise ValueError("Pipeline name must be alphanumeric with underscores")
        return v.upper()


class ProcessingResponse(BaseModel):
    """Unified response payload for ML processing tasks."""

    success: bool = Field(description="Whether processing succeeded")
    tenant_id: str = Field(description="Tenant ID")
    session_id: UUID = Field(description="Processing session UUID")
    image_id: UUID = Field(description="Image UUID")
    pipeline: str = Field(description="Pipeline that was executed")

    # Results from each step (keyed by step name)
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Results from each pipeline step",
    )

    # Metadata
    duration_ms: int = Field(description="Total processing duration in milliseconds")
    steps_completed: int = Field(
        default=0,
        description="Number of steps successfully completed",
    )
    error: str | None = Field(default=None, description="Error message if failed")
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Processing timestamp",
    )

    model_config = {"extra": "forbid"}


class CompressionRequest(BaseModel):
    """Request for image compression task."""

    tenant_id: str = Field(
        min_length=1,
        max_length=100,
        description="Tenant ID for multi-tenant isolation",
    )
    image_id: UUID = Field(description="Image UUID")
    source_url: str = Field(description="GCS URL to source image")
    target_sizes: list[int] = Field(
        default=[256, 512, 1024],
        description="Target thumbnail sizes (longest edge in pixels)",
    )
    quality: int = Field(
        default=85,
        ge=1,
        le=100,
        description="JPEG quality (1-100)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, v: str) -> str:
        """Validate that source_url is a valid GCS URL."""
        if not v.startswith("gs://"):
            raise ValueError("source_url must be a GCS URL (gs://bucket/path)")
        return v


class CompressionResponse(BaseModel):
    """Response for image compression task."""

    success: bool = Field(description="Whether compression succeeded")
    tenant_id: str = Field(description="Tenant ID")
    image_id: UUID = Field(description="Image UUID")
    thumbnails: dict[int, str] = Field(
        default_factory=dict,
        description="Map of size -> GCS URL for generated thumbnails",
    )
    duration_ms: int = Field(description="Processing duration in milliseconds")
    error: str | None = Field(default=None, description="Error message if failed")

    model_config = {"extra": "forbid"}
