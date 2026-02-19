"""Send final processing results to Backend via HTTP callback."""

import time
from uuid import UUID

from app.config import get_settings
from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger
from app.schemas.callback import (
    BoundingBox,
    DetectionResultItem,
    EstimationResultItem,
    ProcessingMetadata,
    ProcessingResultRequest,
)
from app.services.backend_client import BackendClient

logger = get_logger(__name__)


class SendResultsBackendStep(PipelineStep):
    """Send final ML processing results to Backend.

    This step collects all results from the ProcessingContext and sends
    them to the Backend via HTTP callback. Should typically be the last
    step in the pipeline.

    The step transforms:
        - ctx.raw_detections → DetectionResultItem[]
        - ctx.results (estimations) → EstimationResultItem[]

    Example pipeline DSL:
        {"type": "step", "name": "send_results_backend"}
    """

    @property
    def name(self) -> str:
        return "send_results_backend"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Send results to Backend.

        Args:
            ctx: Processing context with ML results

        Returns:
            Context with callback status added to results
        """
        step_start = time.perf_counter()
        settings = get_settings()

        logger.info(
            "Sending results to backend",
            tenant_id=ctx.tenant_id,
            session_id=ctx.session_id,
            image_id=ctx.image_id,
            detections_count=len(ctx.raw_detections),
        )

        # Transform raw detections to callback schema
        detections = self._transform_detections(ctx.raw_detections)

        # Transform estimations from results
        estimations = self._transform_estimations(ctx.results)

        # Build request payload
        request = ProcessingResultRequest(
            sessionId=UUID(ctx.session_id),
            imageId=UUID(ctx.image_id),
            detections=detections,
            classifications=[],  # TODO: Add when classification step exists
            estimations=estimations,
            metadata=ProcessingMetadata(
                pipeline=ctx.config.get("pipeline_name"),
                processingTimeMs=int((time.perf_counter() - step_start) * 1000),
            ),
        )

        # Send to backend
        client = BackendClient(
            base_url=settings.backend_url,
            timeout=settings.backend_timeout,
        )

        response = await client.send_results(
            tenant_id=ctx.tenant_id,
            results=request,
        )

        step_duration_ms = int((time.perf_counter() - step_start) * 1000)

        logger.info(
            "Results sent to backend successfully",
            tenant_id=ctx.tenant_id,
            session_id=ctx.session_id,
            image_id=ctx.image_id,
            detections_sent=len(detections),
            estimations_sent=len(estimations),
            duration_ms=step_duration_ms,
        )

        return ctx.with_results({
            "backend_callback": {
                "success": True,
                "response": response,
                "detections_sent": len(detections),
                "estimations_sent": len(estimations),
            }
        })

    def _transform_detections(
        self, raw_detections: list[dict]
    ) -> list[DetectionResultItem]:
        """Transform raw detections to callback schema.

        Args:
            raw_detections: Raw detection dicts from ML processor

        Returns:
            List of DetectionResultItem for callback
        """
        results = []
        for det in raw_detections:
            bbox = None
            if "bbox" in det:
                b = det["bbox"]
                bbox = BoundingBox(
                    x1=b.get("x1", 0),
                    y1=b.get("y1", 0),
                    x2=b.get("x2", 0),
                    y2=b.get("y2", 0),
                )

            results.append(
                DetectionResultItem(
                    label=det.get("label", det.get("class_name", "unknown")),
                    confidence=det.get("confidence", 0.0),
                    boundingBox=bbox,
                )
            )
        return results

    def _transform_estimations(
        self, results: dict
    ) -> list[EstimationResultItem]:
        """Transform results dict to estimation items.

        Extracts estimation data from results dict. Looks for:
            - results["size_distribution"] → size estimations
            - results["species_distribution"] → species counts
            - results["total_count"] → total detection count

        Args:
            results: Accumulated results dict from context

        Returns:
            List of EstimationResultItem for callback
        """
        estimations = []

        # Total count
        if "total_count" in results:
            estimations.append(
                EstimationResultItem(
                    estimationType="total_count",
                    value=float(results["total_count"]),
                    unit="units",
                )
            )

        # Size distribution (S, M, L, XL counts)
        if "size_distribution" in results:
            for size, count in results["size_distribution"].items():
                estimations.append(
                    EstimationResultItem(
                        estimationType=f"size_{size}",
                        value=float(count),
                        unit="units",
                    )
                )

        # Species distribution
        if "species_distribution" in results:
            for species, count in results["species_distribution"].items():
                estimations.append(
                    EstimationResultItem(
                        estimationType=f"species_{species}",
                        value=float(count),
                        unit="units",
                    )
                )

        return estimations
