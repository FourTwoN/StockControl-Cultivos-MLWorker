"""Unified task endpoint for all ML processing.

Receives tasks from Cloud Tasks and routes them through the
dynamic pipeline configured per tenant in the database.
"""

import io
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from PIL import Image

from app.api.deps import CloudTasksRequest, DbSession, Storage
from app.core.pipeline_executor import PipelineExecutor
from app.core.pipeline_parser import PipelineParser
from app.core.processing_context import ProcessingContext
from app.core.step_registry import StepRegistry
from app.core.tenant_config import get_tenant_cache
from app.infra.logging import get_logger
from app.infra.storage import TenantPathError
from app.schemas.task import (
    CompressionRequest,
    CompressionResponse,
    ProcessingRequest,
    ProcessingResponse,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/process",
    response_model=ProcessingResponse,
    summary="Process image through tenant-configured pipeline",
)
async def process_task(
    request: ProcessingRequest,
    storage: Storage,
    db: DbSession,
    _: CloudTasksRequest,
) -> ProcessingResponse:
    """Process image using dynamic pipeline per tenant.

    The pipeline is defined in tenant_config.pipeline_definition (JSONB)
    and supports full DSL composition: chain, group, chord for parallel execution.
    """
    start_time = time.time()
    local_path: Path | None = None

    # Log request received
    logger.info(
        "Processing request received",
        tenant_id=request.tenant_id,
        session_id=str(request.session_id),
        image_id=str(request.image_id),
        image_url=request.image_url,
    )

    try:
        # Get tenant config from cache (already validated at load time)
        config = await get_tenant_cache().get(request.tenant_id)
        if not config:
            logger.error(
                "Tenant config not found",
                tenant_id=request.tenant_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No pipeline config for tenant: {request.tenant_id}",
            )

        logger.info(
            "Tenant config loaded",
            tenant_id=request.tenant_id,
            pipeline_steps=len(config.pipeline_definition.steps),
            settings_keys=list(config.settings.keys()) if config.settings else [],
        )

        # Download image
        download_start = time.time()
        local_path = await storage.download_to_tempfile(
            blob_path=request.image_url,
            tenant_id=request.tenant_id,
        )
        download_ms = int((time.time() - download_start) * 1000)

        logger.info(
            "Image downloaded",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            local_path=str(local_path),
            file_size_bytes=local_path.stat().st_size if local_path.exists() else 0,
            download_ms=download_ms,
        )

        # Create initial context
        ctx = ProcessingContext(
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            session_id=str(request.session_id),
            image_path=local_path,
            config=config.settings,
        )

        # Parse pipeline definition to DSL structures
        parser = PipelineParser(StepRegistry)
        pipeline = parser.parse(config.pipeline_definition)

        logger.info(
            "Pipeline parsed, starting execution",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            pipeline_type=type(pipeline).__name__,
        )

        # Execute pipeline (always uses executor, supports parallelism)
        executor = PipelineExecutor()
        ctx = await executor.execute(pipeline, ctx)

        duration_ms = int((time.time() - start_time) * 1000)

        # Build complete results including raw ML data
        full_results = {
            **ctx.results,
            "segments": ctx.raw_segments,
            "detections": ctx.raw_detections,
            "classifications_raw": ctx.raw_classifications,
        }

        # Log success summary
        logger.info(
            "Processing completed SUCCESSFULLY",
            tenant_id=request.tenant_id,
            session_id=str(request.session_id),
            image_id=str(request.image_id),
            segments_found=len(ctx.raw_segments),
            detections_found=len(ctx.raw_detections),
            total_duration_ms=duration_ms,
            download_ms=download_ms,
            processing_ms=duration_ms - download_ms,
        )

        return ProcessingResponse(
            success=True,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            pipeline=config.pipeline_definition.model_dump_json(),
            results=full_results,
            duration_ms=duration_ms,
            steps_completed=len(config.pipeline_definition.steps),
        )

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "Processing FAILED",
            tenant_id=request.tenant_id,
            session_id=str(request.session_id),
            image_id=str(request.image_id),
            error_type=type(e).__name__,
            error_message=str(e),
            duration_ms=duration_ms,
            exc_info=True,
        )
        return ProcessingResponse(
            success=False,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            pipeline="",
            results={},
            duration_ms=duration_ms,
            steps_completed=0,
            error=str(e),
        )
    finally:
        if local_path and local_path.exists():
            local_path.unlink(missing_ok=True)
            logger.debug(
                "Temp file cleaned up",
                path=str(local_path),
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
