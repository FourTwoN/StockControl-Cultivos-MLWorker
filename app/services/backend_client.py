"""Backend Client - HTTP client for calling the Demeter backend.

Sends processing results back to the backend via the callback endpoint.
Uses OIDC authentication when running in GCP.
"""

import httpx
from typing import Any
from uuid import UUID

from app.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)


class BackendClient:
    """HTTP client for Demeter backend API."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        """Initialize backend client.

        Args:
            base_url: Backend base URL (defaults to settings)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.backend_url
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send_results(
        self,
        tenant_id: str,
        session_id: UUID,
        image_id: UUID,
        detections: list[dict[str, Any]] | None = None,
        classifications: list[dict[str, Any]] | None = None,
        estimations: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Send processing results to backend.

        Args:
            tenant_id: Tenant identifier
            session_id: Photo processing session ID
            image_id: Image ID
            detections: List of detection results
            classifications: List of classification results
            estimations: List of estimation results
            metadata: Processing metadata

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()

        payload = {
            "sessionId": str(session_id),
            "imageId": str(image_id),
            "detections": self._format_detections(detections) if detections else None,
            "classifications": self._format_classifications(classifications) if classifications else None,
            "estimations": self._format_estimations(estimations) if estimations else None,
            "metadata": metadata,
        }

        try:
            response = await client.post(
                "/api/v1/processing-callback/results",
                json=payload,
                headers={"X-Tenant-ID": tenant_id},
            )
            response.raise_for_status()

            logger.info(
                "Results sent to backend",
                session_id=str(session_id),
                image_id=str(image_id),
                status_code=response.status_code,
            )
            return True

        except httpx.HTTPStatusError as e:
            logger.error(
                "Backend returned error",
                session_id=str(session_id),
                image_id=str(image_id),
                status_code=e.response.status_code,
                response_text=e.response.text[:500],
            )
            return False

        except Exception as e:
            logger.error(
                "Failed to send results to backend",
                session_id=str(session_id),
                image_id=str(image_id),
                error=str(e),
            )
            return False

    async def report_error(
        self,
        tenant_id: str,
        session_id: UUID,
        image_id: UUID,
        error_message: str,
        error_type: str | None = None,
    ) -> bool:
        """Report processing error to backend.

        Args:
            tenant_id: Tenant identifier
            session_id: Photo processing session ID
            image_id: Image ID
            error_message: Error description
            error_type: Error type/category

        Returns:
            True if successful, False otherwise
        """
        client = await self._get_client()

        payload = {
            "sessionId": str(session_id),
            "imageId": str(image_id),
            "errorMessage": error_message,
            "errorType": error_type,
        }

        try:
            response = await client.post(
                "/api/v1/processing-callback/error",
                json=payload,
                headers={"X-Tenant-ID": tenant_id},
            )
            response.raise_for_status()

            logger.info(
                "Error reported to backend",
                session_id=str(session_id),
                image_id=str(image_id),
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to report error to backend",
                session_id=str(session_id),
                image_id=str(image_id),
                error=str(e),
            )
            return False

    def _format_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format detections for backend API."""
        formatted = []
        for det in detections:
            formatted.append({
                "label": det.get("class_name") or det.get("label", "unknown"),
                "confidence": det.get("confidence", 0.0),
                "boundingBox": {
                    "x1": det.get("x1", det.get("center_x_px", 0) - det.get("width_px", 0) / 2),
                    "y1": det.get("y1", det.get("center_y_px", 0) - det.get("height_px", 0) / 2),
                    "x2": det.get("x2", det.get("center_x_px", 0) + det.get("width_px", 0) / 2),
                    "y2": det.get("y2", det.get("center_y_px", 0) + det.get("height_px", 0) / 2),
                } if any(k in det for k in ["x1", "center_x_px"]) else None,
            })
        return formatted

    def _format_classifications(self, classifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format classifications for backend API."""
        formatted = []
        for cls in classifications:
            formatted.append({
                "label": cls.get("class_name") or cls.get("label", "unknown"),
                "confidence": cls.get("confidence", 0.0),
                "detectionId": cls.get("detection_id"),
            })
        return formatted

    def _format_estimations(self, estimations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format estimations for backend API."""
        formatted = []
        for est in estimations:
            formatted.append({
                "estimationType": est.get("type") or est.get("estimation_type", "count"),
                "value": est.get("value") or est.get("total_count", 0),
                "unit": est.get("unit"),
                "confidence": est.get("confidence") or est.get("confidence_avg"),
            })
        return formatted


# Singleton instance
_backend_client: BackendClient | None = None


def get_backend_client() -> BackendClient:
    """Get backend client singleton."""
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendClient()
    return _backend_client
