"""Callback schemas for Backend communication.

These schemas match the Java DTOs in DemeterAI-back:
- ProcessingResultRequest
- DetectionResultItem
- ClassificationResultItem
- EstimationResultItem
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResultItem(BaseModel):
    """Single detection result from ML inference."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    confidence: float
    boundingBox: BoundingBox | None = None


class ClassificationResultItem(BaseModel):
    """Single classification result from ML inference."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    confidence: float
    detectionId: UUID | None = None


class EstimationResultItem(BaseModel):
    """Estimation result (count, area, etc.) from ML inference."""

    model_config = ConfigDict(populate_by_name=True)

    estimationType: str
    value: float
    unit: str | None = None
    confidence: float | None = None


class ProcessingMetadata(BaseModel):
    """Processing metadata from ML Worker."""

    model_config = ConfigDict(populate_by_name=True)

    pipeline: str | None = None
    processingTimeMs: int | None = None
    modelVersion: str | None = None
    workerVersion: str | None = None


class ProcessingResultRequest(BaseModel):
    """Payload sent to Backend callback endpoint.

    Matches Java DTO: com.fortytwo.demeter.fotos.dto.ProcessingResultRequest
    """

    model_config = ConfigDict(populate_by_name=True)

    sessionId: UUID
    imageId: UUID
    detections: list[DetectionResultItem] = []
    classifications: list[ClassificationResultItem] = []
    estimations: list[EstimationResultItem] = []
    metadata: ProcessingMetadata | None = None


class ErrorReport(BaseModel):
    """Error report sent to Backend on processing failure."""

    model_config = ConfigDict(populate_by_name=True)

    sessionId: UUID
    imageId: UUID
    errorMessage: str
    errorType: str = "ProcessingError"
