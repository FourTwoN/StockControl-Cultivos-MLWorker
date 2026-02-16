"""Direct upload endpoint for local development testing.

Allows direct image upload and processing using the Pipeline architecture.
This is NOT for production - only for local POC testing.
"""

import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import base64
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field

from app.core.industry_config import load_industry_config
from app.core.pipeline import Pipeline, PipelineResult
from app.infra.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class LocalPipelineResponse(BaseModel):
    """Response from local ML pipeline execution."""

    success: bool
    image_id: str
    pipeline: str
    duration_ms: int
    steps_completed: int
    results: dict[str, Any]
    error: str | None = None


class ProcessImageJsonRequest(BaseModel):
    """Request body for JSON-based image processing."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(
        ..., alias="contentType", description="MIME type (image/jpeg, image/png, image/webp)"
    )
    image_base64: str = Field(
        ..., alias="imageBase64", description="Base64-encoded image data"
    )
    pipeline: str = Field(
        default="SEGMENT_DETECT", description="Pipeline to execute"
    )

    class Config:
        populate_by_name = True


@router.post(
    "/process-image",
    response_model=LocalPipelineResponse,
    summary="Upload and process image (dev only)",
    description="""
    Direct image upload and ML processing for local development.

    Uses the Pipeline architecture to process images through configured pipelines.

    Available pipelines (from agro.yaml):
    - `DETECTION`: Detection only
    - `SEGMENTATION`: Segmentation only
    - `SEGMENT_DETECT`: Segmentation + Detection (recommended for testing)
    - `FULL_PIPELINE`: Segmentation + Detection + Estimation
    - `QUICK_COUNT`: Detection + Estimation

    Returns pipeline results with all step outputs.
    """,
)
async def upload_and_process(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP)"),
    pipeline: str = Query(
        default="SEGMENT_DETECT",
        description="Pipeline to execute (from industry config)",
    ),
) -> LocalPipelineResponse:
    """Upload image and run ML pipeline.

    Args:
        file: Image file to process
        pipeline: Pipeline name from industry config

    Returns:
        LocalPipelineResponse with pipeline results
    """
    start_time = time.time()
    image_id = str(uuid4())
    temp_path: Path | None = None

    # Validate content type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}",
        )

    logger.info(
        "Received image for pipeline processing",
        image_id=image_id,
        filename=file.filename,
        content_type=file.content_type,
        pipeline=pipeline,
    )

    try:
        # Save to temp file
        file_bytes = await file.read()
        suffix = Path(file.filename or "image.jpg").suffix or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)

        logger.info(
            "Image saved to temp file",
            image_id=image_id,
            temp_path=str(temp_path),
            size_bytes=len(file_bytes),
        )

        # Load industry config
        config = await load_industry_config()

        # Validate pipeline exists
        pipeline_config = config.get_pipeline(pipeline.upper())
        if pipeline_config is None:
            available = config.get_available_pipelines()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pipeline '{pipeline}' not found. Available: {available}",
            )

        # Create and execute pipeline
        pipeline_executor = Pipeline(config=config)

        logger.info(
            "Executing pipeline",
            image_id=image_id,
            pipeline=pipeline.upper(),
            steps=pipeline_config.steps,
        )

        result: PipelineResult = await pipeline_executor.execute(
            pipeline_name=pipeline.upper(),
            image_path=temp_path,
        )

        logger.info(
            "Pipeline completed",
            image_id=image_id,
            pipeline=result.pipeline_name,
            success=result.success,
            duration_ms=result.total_duration_ms,
            steps_completed=len([s for s in result.steps if s.success]),
        )

        # Build response
        duration_ms = int((time.time() - start_time) * 1000)

        return LocalPipelineResponse(
            success=result.success,
            image_id=image_id,
            pipeline=result.pipeline_name,
            duration_ms=duration_ms,
            steps_completed=len([s for s in result.steps if s.success]),
            results=result.to_dict(),
            error=result.error,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Pipeline processing failed",
            image_id=image_id,
            error=str(e),
            exc_info=True,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        return LocalPipelineResponse(
            success=False,
            image_id=image_id,
            pipeline=pipeline.upper(),
            duration_ms=duration_ms,
            steps_completed=0,
            results={},
            error=str(e),
        )

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug("Temp file cleaned up", path=str(temp_path))
            except Exception:
                pass


@router.post(
    "/process-image-json",
    response_model=LocalPipelineResponse,
    summary="Process image from JSON/Base64 (dev only)",
    description="""
    Process image sent as Base64-encoded JSON for easier integration.

    This endpoint avoids multipart form-data complexity by accepting
    the image as a Base64-encoded string in JSON.
    """,
)
async def process_image_json(
    request: ProcessImageJsonRequest,
) -> LocalPipelineResponse:
    """Process image from JSON with Base64-encoded data.

    Args:
        request: JSON body with Base64-encoded image

    Returns:
        LocalPipelineResponse with pipeline results
    """
    start_time = time.time()
    image_id = str(uuid4())
    temp_path: Path | None = None

    # Validate content type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    if request.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type: {request.content_type}. Allowed: {allowed_types}",
        )

    logger.info(
        "Received JSON image for pipeline processing",
        image_id=image_id,
        filename=request.filename,
        content_type=request.content_type,
        pipeline=request.pipeline,
        base64_length=len(request.image_base64),
    )

    try:
        # Decode base64
        try:
            file_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 encoding: {e}",
            )

        # Save to temp file
        suffix = Path(request.filename).suffix or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = Path(tmp.name)

        logger.info(
            "Image decoded and saved to temp file",
            image_id=image_id,
            temp_path=str(temp_path),
            size_bytes=len(file_bytes),
        )

        # Load industry config
        config = await load_industry_config()

        # Validate pipeline exists
        pipeline = request.pipeline.upper()
        pipeline_config = config.get_pipeline(pipeline)
        if pipeline_config is None:
            available = config.get_available_pipelines()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Pipeline '{pipeline}' not found. Available: {available}",
            )

        # Create and execute pipeline
        pipeline_executor = Pipeline(config=config)

        logger.info(
            "Executing pipeline",
            image_id=image_id,
            pipeline=pipeline,
            steps=pipeline_config.steps,
        )

        result: PipelineResult = await pipeline_executor.execute(
            pipeline_name=pipeline,
            image_path=temp_path,
        )

        logger.info(
            "Pipeline completed",
            image_id=image_id,
            pipeline=result.pipeline_name,
            success=result.success,
            duration_ms=result.total_duration_ms,
            steps_completed=len([s for s in result.steps if s.success]),
        )

        # Build response
        duration_ms = int((time.time() - start_time) * 1000)

        return LocalPipelineResponse(
            success=result.success,
            image_id=image_id,
            pipeline=result.pipeline_name,
            duration_ms=duration_ms,
            steps_completed=len([s for s in result.steps if s.success]),
            results=result.to_dict(),
            error=result.error,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Pipeline processing failed",
            image_id=image_id,
            error=str(e),
            exc_info=True,
        )
        duration_ms = int((time.time() - start_time) * 1000)

        return LocalPipelineResponse(
            success=False,
            image_id=image_id,
            pipeline=request.pipeline.upper(),
            duration_ms=duration_ms,
            steps_completed=0,
            results={},
            error=str(e),
        )

    finally:
        # Cleanup temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug("Temp file cleaned up", path=str(temp_path))
            except Exception:
                pass


@router.get(
    "/pipelines",
    summary="List available pipelines",
    description="Returns all pipelines defined in the industry configuration",
)
async def list_pipelines() -> dict[str, Any]:
    """List available pipelines from industry config."""
    config = await load_industry_config()

    pipelines_info = {}
    for name in config.get_available_pipelines():
        pipeline_config = config.get_pipeline(name)
        if pipeline_config:
            pipelines_info[name] = {
                "steps": list(pipeline_config.steps),
            }

    return {
        "industry": config.industry,
        "version": config.version,
        "pipelines": pipelines_info,
    }


@router.get(
    "/models-info",
    summary="Get loaded models information",
    description="Returns information about cached ML models",
)
async def get_models_info() -> dict[str, Any]:
    """Get information about loaded ML models."""
    from app.ml.model_cache import ModelCache

    return ModelCache.get_cache_info()
