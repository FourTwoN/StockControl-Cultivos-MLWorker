"""Unified task endpoint for all ML processing.

Receives tasks from Cloud Tasks with pipeline_definition inline.
No database lookup - fully stateless.
"""

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CloudTasksRequest, Storage
from app.core.pipeline_executor import PipelineExecutor
from app.core.pipeline_parser import PipelineParser, PipelineParserError
from app.core.processing_context import ProcessingContext
from app.core.step_registry import StepRegistry
from app.infra.logging import get_logger
from app.schemas.task import ProcessingRequest, ProcessingResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/process",
    response_model=ProcessingResponse,
    summary="Process image through pipeline defined in request",
)
async def process_task(
    request: ProcessingRequest,
    storage: Storage,
    _: CloudTasksRequest,
) -> ProcessingResponse:
    """Process image using pipeline_definition from request.

    Pipeline is defined directly in the request payload - no DB lookup.
    Supports full DSL composition: chain, group, chord for parallel execution.
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
        pipeline_type=request.pipeline_definition.type,
    )

    try:
        # Parse and validate pipeline definition (fail-fast)
        parser = PipelineParser(StepRegistry)
        try:
            pipeline = parser.parse(request.pipeline_definition)
        except PipelineParserError as e:
            logger.error(
                "Invalid pipeline definition",
                tenant_id=request.tenant_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid pipeline: {e}",
            )

        logger.info(
            "Pipeline validated",
            tenant_id=request.tenant_id,
            pipeline_type=type(pipeline).__name__,
            steps_count=len(request.pipeline_definition.steps),
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

        # Create initial context with settings from request
        ctx = ProcessingContext(
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            session_id=str(request.session_id),
            image_path=local_path,
            config=request.settings,
        )

        logger.info(
            "Pipeline execution starting",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            pipeline_type=type(pipeline).__name__,
        )

        # Execute pipeline
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
            pipeline_type=request.pipeline_definition.type,
            results=full_results,
            duration_ms=duration_ms,
            steps_completed=len(request.pipeline_definition.steps),
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
            pipeline_type=request.pipeline_definition.type,
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
