"""Common schemas for API requests and responses."""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class TaskResult(BaseModel, Generic[T]):
    """Generic task result wrapper.

    Used for Cloud Tasks responses to indicate success/failure.
    """

    success: bool = Field(description="Whether the task completed successfully")
    data: T | None = Field(default=None, description="Task result data")
    error: str | None = Field(default=None, description="Error message if failed")
    error_type: str | None = Field(default=None, description="Error type/class name")
    duration_ms: int | None = Field(default=None, description="Processing duration in milliseconds")

    model_config = {"extra": "forbid"}


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    error_type: str = Field(description="Error type/class name")
    detail: dict[str, Any] | None = Field(default=None, description="Additional error details")

    model_config = {"extra": "forbid"}


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Health status (healthy, unhealthy)")
    version: str = Field(description="Service version")
    environment: str = Field(description="Environment name")
    checks: dict[str, bool] = Field(default_factory=dict, description="Individual health checks")

    model_config = {"extra": "forbid"}


class TaskMetadata(BaseModel):
    """Metadata extracted from Cloud Tasks headers."""

    task_name: str | None = Field(default=None, description="Cloud Tasks task name")
    queue_name: str | None = Field(default=None, description="Cloud Tasks queue name")
    retry_count: int = Field(default=0, description="Current retry attempt")
    execution_count: int = Field(default=0, description="Total execution count")

    model_config = {"extra": "forbid"}
