"""HTTP client for Backend callback communication."""

from uuid import UUID

import httpx

from app.infra.logging import get_logger
from app.schemas.callback import ErrorReport, ProcessingResultRequest

logger = get_logger(__name__)


class BackendClient:
    """Client for sending ML results to Backend.

    Communicates with DemeterAI-back via HTTP callbacks:
    - POST /api/v1/processing-callback/results - send ML results
    - POST /api/v1/processing-callback/error - report errors
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize backend client.

        Args:
            base_url: Backend base URL (e.g., https://api.demeter.com)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def send_results(
        self,
        tenant_id: str,
        results: ProcessingResultRequest,
    ) -> dict:
        """Send ML processing results to Backend.

        Args:
            tenant_id: Tenant identifier for RLS
            results: Processing results payload

        Returns:
            Backend response as dict

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        url = f"{self.base_url}/api/v1/processing-callback/results"

        logger.info(
            "Sending results to backend",
            tenant_id=tenant_id,
            session_id=str(results.sessionId),
            image_id=str(results.imageId),
            detections_count=len(results.detections),
            estimations_count=len(results.estimations),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=results.model_dump(mode="json"),
                headers={
                    "X-Tenant-ID": tenant_id,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            logger.info(
                "Results sent successfully",
                tenant_id=tenant_id,
                session_id=str(results.sessionId),
                status_code=response.status_code,
            )

            return response.json()

    async def report_error(
        self,
        tenant_id: str,
        session_id: UUID,
        image_id: UUID,
        error_message: str,
        error_type: str = "ProcessingError",
    ) -> dict:
        """Report processing error to Backend.

        Args:
            tenant_id: Tenant identifier for RLS
            session_id: Processing session ID
            image_id: Image that failed processing
            error_message: Error description
            error_type: Error classification

        Returns:
            Backend response as dict
        """
        url = f"{self.base_url}/api/v1/processing-callback/error"

        error_report = ErrorReport(
            sessionId=session_id,
            imageId=image_id,
            errorMessage=error_message,
            errorType=error_type,
        )

        logger.warning(
            "Reporting error to backend",
            tenant_id=tenant_id,
            session_id=str(session_id),
            image_id=str(image_id),
            error_type=error_type,
            error_message=error_message,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=error_report.model_dump(mode="json"),
                headers={
                    "X-Tenant-ID": tenant_id,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            return response.json()
