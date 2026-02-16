"""Processing Service - Orchestrates ML pipeline execution.

This service coordinates the complete processing flow:
1. Load industry configuration
2. Download image from GCS
3. Execute the requested pipeline
4. Send results to backend via callback
5. Return response
"""

import time
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config import settings
from app.core.industry_config import IndustryConfig, load_industry_config
from app.core.pipeline import Pipeline, PipelineResult
from app.infra.logging import get_logger
from app.infra.storage import StorageClient, TenantPathError
from app.schemas.task import ProcessingRequest, ProcessingResponse
from app.services.backend_client import BackendClient, get_backend_client

logger = get_logger(__name__)


class ProcessingService:
    """Service for orchestrating ML pipeline execution."""

    def __init__(
        self,
        storage_client: StorageClient,
        backend_client: BackendClient | None = None,
    ) -> None:
        """Initialize processing service.

        Args:
            storage_client: Cloud Storage client
            backend_client: HTTP client for backend callbacks (optional)
        """
        self.storage = storage_client
        self.backend = backend_client or get_backend_client()
        self._config: IndustryConfig | None = None
        self._pipeline: Pipeline | None = None

    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process an image through the ML pipeline.

        Args:
            request: Processing request from Cloud Tasks

        Returns:
            ProcessingResponse with results

        Raises:
            TenantPathError: If image doesn't belong to tenant
        """
        start_time = time.time()
        local_path: Path | None = None

        try:
            logger.info(
                "Starting processing",
                tenant_id=request.tenant_id,
                session_id=str(request.session_id),
                image_id=str(request.image_id),
                pipeline=request.pipeline,
            )

            # Load industry config
            config = await self._get_config()

            # Validate pipeline exists
            pipeline_config = config.get_pipeline(request.pipeline)
            if pipeline_config is None:
                available = config.get_available_pipelines()
                raise ValueError(
                    f"Pipeline '{request.pipeline}' not found. "
                    f"Available: {available}"
                )

            # Download image from GCS
            local_path = await self.storage.download_to_tempfile(
                blob_path=request.image_url,
                tenant_id=request.tenant_id,
            )

            # Get or create pipeline executor
            pipeline = self._get_pipeline(config)

            # Apply option overrides if provided
            context: dict[str, Any] = {}
            if request.options:
                context["options"] = request.options

            # Execute pipeline
            result = await pipeline.execute(
                pipeline_name=request.pipeline,
                image_path=local_path,
                **context,
            )

            # Save results to database
            if result.success:
                await self._save_results(request, result)

            # Build response
            duration_ms = int((time.time() - start_time) * 1000)

            return ProcessingResponse(
                success=result.success,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                image_id=request.image_id,
                pipeline=request.pipeline,
                results=result.to_dict(),
                duration_ms=duration_ms,
                steps_completed=len([s for s in result.steps if s.success]),
                error=result.error,
            )

        except TenantPathError as e:
            logger.error(
                "Tenant path validation failed",
                tenant_id=request.tenant_id,
                image_url=request.image_url,
                error=str(e),
            )
            raise

        except Exception as e:
            logger.error(
                "Processing failed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                error=str(e),
                exc_info=True,
            )
            duration_ms = int((time.time() - start_time) * 1000)
            return ProcessingResponse(
                success=False,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                image_id=request.image_id,
                pipeline=request.pipeline,
                results={},
                duration_ms=duration_ms,
                steps_completed=0,
                error=str(e),
            )

        finally:
            # Cleanup temp file
            if local_path and local_path.exists():
                try:
                    local_path.unlink()
                except Exception:
                    pass

    async def _get_config(self) -> IndustryConfig:
        """Get or load industry configuration."""
        if self._config is None:
            self._config = await load_industry_config()
        return self._config

    def _get_pipeline(self, config: IndustryConfig) -> Pipeline:
        """Get or create pipeline executor."""
        if self._pipeline is None:
            self._pipeline = Pipeline(config=config)
        return self._pipeline

    async def _save_results(
        self,
        request: ProcessingRequest,
        result: PipelineResult,
    ) -> None:
        """Send pipeline results to backend via callback.

        Args:
            request: Original processing request
            result: Pipeline execution result
        """
        logger.info(
            "Sending results to backend",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            pipeline=request.pipeline,
        )

        # Collect results from pipeline steps
        detections = result.get_step_data("detection")
        classifications = result.get_step_data("classification")
        estimations = result.get_step_data("estimation")

        # Format estimations as list if single dict
        estimations_list = None
        if estimations:
            if isinstance(estimations, dict):
                estimations_list = [estimations]
            else:
                estimations_list = estimations

        # Build metadata
        metadata = {
            "pipeline": request.pipeline,
            "processingTimeMs": result.duration_ms,
            "workerVersion": settings.environment,
        }

        # Send to backend
        success = await self.backend.send_results(
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            detections=detections,
            classifications=classifications,
            estimations=estimations_list,
            metadata=metadata,
        )

        if success:
            logger.info(
                "Results sent successfully",
                image_id=str(request.image_id),
            )
        else:
            logger.warning(
                "Failed to send results to backend",
                image_id=str(request.image_id),
            )

    async def _report_error(
        self,
        request: ProcessingRequest,
        error: str,
        error_type: str | None = None,
    ) -> None:
        """Report processing error to backend.

        Args:
            request: Original processing request
            error: Error message
            error_type: Error type/category
        """
        await self.backend.report_error(
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            error_message=error,
            error_type=error_type,
        )
