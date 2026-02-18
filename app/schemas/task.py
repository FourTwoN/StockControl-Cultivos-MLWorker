"""Task schemas for the unified processing endpoint."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.pipeline_definition import PipelineDefinition


class ProcessingRequest(BaseModel):
    """Unified request payload for all ML processing tasks.

    Sent by Backend via Cloud Tasks to the single /tasks/process endpoint.
    Pipeline definition is included directly in the request (no DB lookup).
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
    pipeline_definition: PipelineDefinition = Field(
        description="Full pipeline DSL definition",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Tenant settings for pipeline execution",
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


class ProcessingResponse(BaseModel):
    """Unified response payload for ML processing tasks."""

    success: bool = Field(description="Whether processing succeeded")
    tenant_id: str = Field(description="Tenant ID")
    session_id: UUID = Field(description="Processing session UUID")
    image_id: UUID = Field(description="Image UUID")
    pipeline_type: str = Field(description="Pipeline type that was executed (chain/group/chord)")

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
