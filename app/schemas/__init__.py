"""Pydantic schemas for request/response validation."""

from app.schemas.common import TaskResult, ErrorResponse, HealthResponse
from app.schemas.pipeline_definition import (
    ChainDefinition,
    ChordDefinition,
    GroupDefinition,
    PipelineDefinition,
    PipelineElementDefinition,
    StepDefinition,
)
from app.schemas.task import (
    ProcessingRequest,
    ProcessingResponse,
    CompressionRequest,
    CompressionResponse,
)

__all__ = [
    "TaskResult",
    "ErrorResponse",
    "HealthResponse",
    "ProcessingRequest",
    "ProcessingResponse",
    "CompressionRequest",
    "CompressionResponse",
    "ChainDefinition",
    "ChordDefinition",
    "GroupDefinition",
    "PipelineDefinition",
    "PipelineElementDefinition",
    "StepDefinition",
]
