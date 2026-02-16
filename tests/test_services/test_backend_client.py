"""Tests for backend client."""

import pytest
from uuid import UUID
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.services.backend_client import BackendClient, get_backend_client


class TestBackendClient:
    """Tests for BackendClient."""

    @pytest.fixture
    def client(self) -> BackendClient:
        """Create a backend client instance."""
        return BackendClient(base_url="http://test-backend:8080", timeout=10.0)

    @pytest.fixture
    def sample_session_id(self) -> UUID:
        return UUID("550e8400-e29b-41d4-a716-446655440000")

    @pytest.fixture
    def sample_image_id(self) -> UUID:
        return UUID("660e8400-e29b-41d4-a716-446655440001")

    @pytest.mark.asyncio
    async def test_send_results_success(
        self,
        client: BackendClient,
        sample_session_id: UUID,
        sample_image_id: UUID,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.send_results(
                tenant_id="test-tenant",
                session_id=sample_session_id,
                image_id=sample_image_id,
                detections=[
                    {"label": "plant", "confidence": 0.95, "x1": 10, "y1": 20, "x2": 100, "y2": 200}
                ],
            )

            assert result is True
            mock_http_client.post.assert_called_once()

            # Verify the payload structure
            call_args = mock_http_client.post.call_args
            assert call_args[0][0] == "/api/v1/processing-callback/results"
            payload = call_args[1]["json"]
            assert payload["sessionId"] == str(sample_session_id)
            assert payload["imageId"] == str(sample_image_id)
            assert payload["detections"] is not None

    @pytest.mark.asyncio
    async def test_send_results_http_error(
        self,
        client: BackendClient,
        sample_session_id: UUID,
        sample_image_id: UUID,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
        )

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.send_results(
                tenant_id="test-tenant",
                session_id=sample_session_id,
                image_id=sample_image_id,
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_send_results_connection_error(
        self,
        client: BackendClient,
        sample_session_id: UUID,
        sample_image_id: UUID,
    ):
        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_http_client

            result = await client.send_results(
                tenant_id="test-tenant",
                session_id=sample_session_id,
                image_id=sample_image_id,
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_report_error_success(
        self,
        client: BackendClient,
        sample_session_id: UUID,
        sample_image_id: UUID,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.report_error(
                tenant_id="test-tenant",
                session_id=sample_session_id,
                image_id=sample_image_id,
                error_message="Processing failed",
                error_type="RuntimeError",
            )

            assert result is True
            mock_http_client.post.assert_called_once()

            call_args = mock_http_client.post.call_args
            assert call_args[0][0] == "/api/v1/processing-callback/error"
            payload = call_args[1]["json"]
            assert payload["errorMessage"] == "Processing failed"
            assert payload["errorType"] == "RuntimeError"

    def test_format_detections(self, client: BackendClient):
        detections = [
            {
                "class_name": "plant",
                "confidence": 0.95,
                "center_x_px": 50,
                "center_y_px": 100,
                "width_px": 40,
                "height_px": 80,
            }
        ]

        formatted = client._format_detections(detections)

        assert len(formatted) == 1
        assert formatted[0]["label"] == "plant"
        assert formatted[0]["confidence"] == 0.95
        assert formatted[0]["boundingBox"] is not None
        # center_x=50, width=40 -> x1=30, x2=70
        assert formatted[0]["boundingBox"]["x1"] == 30
        assert formatted[0]["boundingBox"]["x2"] == 70

    def test_format_detections_with_x1_y1_format(self, client: BackendClient):
        detections = [
            {
                "label": "weed",
                "confidence": 0.85,
                "x1": 10,
                "y1": 20,
                "x2": 100,
                "y2": 200,
            }
        ]

        formatted = client._format_detections(detections)

        assert formatted[0]["boundingBox"]["x1"] == 10
        assert formatted[0]["boundingBox"]["y1"] == 20
        assert formatted[0]["boundingBox"]["x2"] == 100
        assert formatted[0]["boundingBox"]["y2"] == 200

    def test_format_classifications(self, client: BackendClient):
        classifications = [
            {"class_name": "coca_cola", "confidence": 0.92},
            {"label": "pepsi", "confidence": 0.88},
        ]

        formatted = client._format_classifications(classifications)

        assert len(formatted) == 2
        assert formatted[0]["label"] == "coca_cola"
        assert formatted[1]["label"] == "pepsi"

    def test_format_estimations(self, client: BackendClient):
        estimations = [
            {"type": "count", "value": 15, "unit": "plants", "confidence": 0.9},
            {"estimation_type": "coverage", "total_count": 0.75},
        ]

        formatted = client._format_estimations(estimations)

        assert len(formatted) == 2
        assert formatted[0]["estimationType"] == "count"
        assert formatted[0]["value"] == 15
        assert formatted[0]["unit"] == "plants"
        assert formatted[1]["estimationType"] == "coverage"

    @pytest.mark.asyncio
    async def test_close_client(self, client: BackendClient):
        # Create the internal client
        await client._get_client()
        assert client._client is not None

        # Close it
        await client.close()
        assert client._client is None


class TestGetBackendClient:
    """Tests for singleton backend client."""

    def test_returns_same_instance(self):
        # Note: This test may be affected by other tests since it uses a global singleton
        # Reset the singleton first
        import app.services.backend_client as bc
        bc._backend_client = None

        client1 = get_backend_client()
        client2 = get_backend_client()

        assert client1 is client2

        # Cleanup
        bc._backend_client = None
