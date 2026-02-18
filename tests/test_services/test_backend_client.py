"""Tests for BackendClient."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4

from app.schemas.callback import ProcessingResultRequest, DetectionResultItem
from app.services.backend_client import BackendClient


@pytest.fixture
def backend_client():
    return BackendClient(base_url="http://localhost:8080", timeout=5.0)


@pytest.fixture
def sample_results():
    return ProcessingResultRequest(
        sessionId=uuid4(),
        imageId=uuid4(),
        detections=[DetectionResultItem(label="plant", confidence=0.95, boundingBox=None)],
        classifications=[],
        estimations=[],
    )


@pytest.mark.asyncio
async def test_send_results_calls_correct_endpoint(backend_client, sample_results):
    """send_results should POST to /api/v1/processing-callback/results."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        await backend_client.send_results("tenant-123", sample_results)

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/v1/processing-callback/results" in call_args[0][0]
        assert call_args[1]["headers"]["X-Tenant-ID"] == "tenant-123"


@pytest.mark.asyncio
async def test_report_error_calls_error_endpoint(backend_client):
    """report_error should POST to /api/v1/processing-callback/error."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        await backend_client.report_error(
            tenant_id="tenant-123",
            session_id=uuid4(),
            image_id=uuid4(),
            error_message="Test error",
        )

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/api/v1/processing-callback/error" in call_args[0][0]
