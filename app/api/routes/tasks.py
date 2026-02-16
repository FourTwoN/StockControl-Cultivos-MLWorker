"""Unified task endpoint for all ML processing.

Receives tasks from Cloud Tasks and routes them through the
appropriate pipeline based on the request.
"""

import io
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from PIL import Image

from app.api.deps import CloudTasksRequest, DbSession, Storage
from app.infra.logging import get_logger
from app.infra.storage import TenantPathError
from app.schemas.task import (
    CompressionRequest,
    CompressionResponse,
    ProcessingRequest,
    ProcessingResponse,
)
from app.services.agro_processing_service import AgroProcessingService
from app.services.processing_service import ProcessingService

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/process",
    response_model=ProcessingResponse,
    summary="Process image through ML pipeline",
    description="""
    Unified endpoint for all ML processing tasks.

    The `pipeline` field determines which processors run:
    - `DETECTION`: Run detection only
    - `SEGMENTATION`: Run segmentation only
    - `FULL_PIPELINE`: Run segmentation → detection → estimation
    - Custom pipelines defined in industry config

    Available pipelines depend on the industry configuration.
    """,
)
async def process_task(
    request: ProcessingRequest,
    storage: Storage,
    _: CloudTasksRequest,
) -> ProcessingResponse:
    """Process an image through the configured ML pipeline.

    This endpoint is called by Cloud Tasks with the processing request.
    It downloads the image, executes the requested pipeline, and saves
    results to the database.

    Args:
        request: Processing request with tenant_id, image_url, pipeline, etc.
        storage: Cloud Storage client
        _: Cloud Tasks request validation

    Returns:
        ProcessingResponse with pipeline results

    Raises:
        HTTPException 400: If tenant validation fails or pipeline not found
        HTTPException 500: If processing fails (Cloud Tasks will retry)
    """
    logger.info(
        "Received processing task",
        tenant_id=request.tenant_id,
        session_id=str(request.session_id),
        image_id=str(request.image_id),
        pipeline=request.pipeline,
    )

    try:
        # Create processing service (results sent to backend via HTTP callback)
        service = ProcessingService(storage_client=storage)

        response = await service.process(request)

        if response.success:
            logger.info(
                "Processing completed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                pipeline=request.pipeline,
                duration_ms=response.duration_ms,
                steps_completed=response.steps_completed,
            )
        else:
            logger.warning(
                "Processing failed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                pipeline=request.pipeline,
                error=response.error,
            )

        return response

    except TenantPathError as e:
        logger.error(
            "Tenant path validation failed",
            tenant_id=request.tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant validation failed: {e}",
        )

    except ValueError as e:
        # Pipeline not found or invalid configuration
        logger.error(
            "Invalid request",
            tenant_id=request.tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error(
            "Processing error",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {e}",
        )


@router.post(
    "/agro/process",
    response_model=ProcessingResponse,
    summary="Process image through Agro ML pipeline",
    description="""
    Agro-specific endpoint for plant detection and classification.

    Pipeline steps:
    1. Segmentation - Detect containers (cajon, segmento)
    2. Segment filtering - Keep only largest "claro" segment
    3. Detection - Standard YOLO for cajon, SAHI for large segments
    4. Coordinate transformation - Segment-relative to full-image coords
    5. Classification - Species distribution + size calculation
    6. DB persistence - INSERT detections, classifications

    Requires `species_config` in request payload for classification.
    """,
)
async def process_agro_task(
    request: ProcessingRequest,
    storage: Storage,
    db: DbSession,
    _: CloudTasksRequest,
) -> ProcessingResponse:
    """Process an image through the Agro ML pipeline.

    This endpoint handles agricultural plant detection with:
    - SAHI tiled detection for large segments (>1M pixels)
    - Equitable species distribution classification
    - Direct DB persistence (no HTTP callback)

    Args:
        request: Processing request with species_config for classification
        storage: Cloud Storage client
        db: Async database session with RLS context
        _: Cloud Tasks request validation

    Returns:
        ProcessingResponse with detection/classification results
    """
    logger.info(
        "Received agro processing task",
        tenant_id=request.tenant_id,
        session_id=str(request.session_id),
        image_id=str(request.image_id),
        pipeline=request.pipeline,
        has_species_config=request.species_config is not None,
    )

    try:
        service = AgroProcessingService(
            storage_client=storage,
            db_session=db,
        )

        response = await service.process(request)

        if response.success:
            logger.info(
                "Agro processing completed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                duration_ms=response.duration_ms,
                detections=response.results.get("total_detected", 0),
            )
        else:
            logger.warning(
                "Agro processing failed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                error=response.error,
            )

        return response

    except TenantPathError as e:
        logger.error(
            "Tenant path validation failed",
            tenant_id=request.tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant validation failed: {e}",
        )

    except Exception as e:
        logger.error(
            "Agro processing error",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agro processing failed: {e}",
        )


@router.post(
    "/compress",
    response_model=CompressionResponse,
    summary="Compress image and generate thumbnails",
    description="Generate thumbnails at specified sizes from a source image.",
)
async def compress_task(
    request: CompressionRequest,
    storage: Storage,
    _: CloudTasksRequest,
) -> CompressionResponse:
    """Compress an image and generate thumbnails.

    Args:
        request: Compression request with source_url and target sizes
        storage: Cloud Storage client
        _: Cloud Tasks request validation

    Returns:
        CompressionResponse with thumbnail URLs
    """
    start_time = time.time()
    local_path: Path | None = None

    logger.info(
        "Received compression task",
        tenant_id=request.tenant_id,
        image_id=str(request.image_id),
        target_sizes=request.target_sizes,
    )

    try:
        # Download source image
        local_path = await storage.download_to_tempfile(
            blob_path=request.source_url,
            tenant_id=request.tenant_id,
        )

        # Open and process image
        with Image.open(local_path) as img:
            # Convert to RGB if necessary
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            thumbnails: dict[int, str] = {}

            for size in request.target_sizes:
                # Create thumbnail
                thumb = _create_thumbnail(img, size)

                # Convert to bytes
                buffer = io.BytesIO()
                thumb.save(buffer, format="JPEG", quality=request.quality, optimize=True)
                buffer.seek(0)

                # Generate blob path
                blob_path = (
                    f"{request.tenant_id}/thumbnails/"
                    f"{request.image_id}_{size}.jpg"
                )

                # Upload to GCS
                url = await storage.upload_bytes(
                    data=buffer.read(),
                    blob_path=blob_path,
                    tenant_id=request.tenant_id,
                    content_type="image/jpeg",
                )

                thumbnails[size] = url

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Compression completed",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            thumbnails=len(thumbnails),
            duration_ms=duration_ms,
        )

        return CompressionResponse(
            success=True,
            tenant_id=request.tenant_id,
            image_id=request.image_id,
            thumbnails=thumbnails,
            duration_ms=duration_ms,
        )

    except TenantPathError as e:
        logger.error(
            "Tenant path validation failed",
            tenant_id=request.tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tenant validation failed: {e}",
        )

    except Exception as e:
        logger.error(
            "Compression failed",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compression failed: {e}",
        )

    finally:
        if local_path and local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass


def _create_thumbnail(img: Image.Image, max_size: int) -> Image.Image:
    """Create a thumbnail maintaining aspect ratio."""
    width, height = img.size

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
