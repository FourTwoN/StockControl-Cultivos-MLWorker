"""Tests for /tasks/process endpoint with inline pipeline_definition."""

import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from pathlib import Path

from app.schemas.pipeline_definition import PipelineDefinition, StepDefinition


class TestCloudTasksValidation:
    """Tests for Cloud Tasks header validation."""

    @pytest.fixture
    def valid_request_payload(self) -> dict:
        return {
            "tenant_id": "test-tenant-001",
            "session_id": str(uuid4()),
            "image_id": str(uuid4()),
            "image_url": "gs://test-bucket/test-tenant-001/images/test.jpg",
            "pipeline_definition": {
                "type": "chain",
                "steps": [{"type": "step", "name": "segmentation"}],
            },
            "settings": {"segment_filter_classes": ["segmento"]},
        }

    @pytest.fixture
    def cloud_tasks_headers(self) -> dict:
        return {
            "X-CloudTasks-TaskName": "test-task-123",
            "X-CloudTasks-QueueName": "test-queue",
        }

    @pytest.mark.asyncio
    async def test_rejects_requests_without_headers_in_prod(
        self, client: AsyncClient, valid_request_payload: dict
    ):
        """Test that requests without Cloud Tasks headers are rejected in production."""
        with patch("app.config.settings.environment", "prod"), \
             patch("app.config.settings.cloudtasks_strict_validation", True):
            response = await client.post(
                "/tasks/process",
                json=valid_request_payload,
            )

        assert response.status_code == 403
        assert "Cloud Tasks" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_rejects_requests_without_headers_in_staging(
        self, client: AsyncClient, valid_request_payload: dict
    ):
        """Test that requests without Cloud Tasks headers are rejected in staging."""
        with patch("app.config.settings.environment", "staging"), \
             patch("app.config.settings.cloudtasks_strict_validation", True):
            response = await client.post(
                "/tasks/process",
                json=valid_request_payload,
            )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_allows_requests_with_headers_in_prod(
        self, client: AsyncClient, valid_request_payload: dict, cloud_tasks_headers: dict
    ):
        """Test that requests with Cloud Tasks headers pass validation in production."""
        with patch("app.config.settings.environment", "prod"), \
             patch("app.config.settings.cloudtasks_strict_validation", True):
            response = await client.post(
                "/tasks/process",
                json=valid_request_payload,
                headers=cloud_tasks_headers,
            )

        # Should pass header validation (not 403)
        assert response.status_code != 403

    @pytest.mark.asyncio
    async def test_allows_requests_without_headers_in_dev(
        self, client: AsyncClient, valid_request_payload: dict
    ):
        """Test that requests without Cloud Tasks headers are allowed in dev."""
        with patch("app.config.settings.environment", "dev"), \
             patch("app.config.settings.cloudtasks_strict_validation", True):
            response = await client.post(
                "/tasks/process",
                json=valid_request_payload,
            )

        # Should pass validation (not 403) - may fail on other reasons
        assert response.status_code != 403

    @pytest.mark.asyncio
    async def test_allows_requests_when_strict_validation_disabled(
        self, client: AsyncClient, valid_request_payload: dict
    ):
        """Test that requests pass when strict validation is disabled."""
        with patch("app.config.settings.environment", "prod"), \
             patch("app.config.settings.cloudtasks_strict_validation", False):
            response = await client.post(
                "/tasks/process",
                json=valid_request_payload,
            )

        # Should pass validation even without headers
        assert response.status_code != 403


class TestProcessEndpoint:
    """Tests for /tasks/process endpoint with inline pipeline_definition."""

    @pytest.fixture
    def valid_request_payload(self) -> dict:
        return {
            "tenant_id": "tenant-001",
            "session_id": str(uuid4()),
            "image_id": str(uuid4()),
            "image_url": "gs://bucket/tenant-001/images/test.jpg",
            "pipeline_definition": {
                "type": "chain",
                "steps": [
                    {"type": "step", "name": "segmentation"},
                ],
            },
            "settings": {"segment_filter_classes": ["segmento"]},
        }

    @pytest.fixture
    def mock_storage(self):
        """Mock storage client."""
        storage = AsyncMock()
        mock_path = MagicMock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_size = 1024
        mock_path.unlink = MagicMock()
        storage.download_to_tempfile = AsyncMock(return_value=mock_path)
        return storage

    @pytest.mark.asyncio
    async def test_process_accepts_pipeline_definition(
        self, client: AsyncClient, valid_request_payload: dict, mock_storage
    ):
        """Endpoint should accept request with pipeline_definition."""
        mock_ctx = MagicMock()
        mock_ctx.results = {}
        mock_ctx.raw_segments = []
        mock_ctx.raw_detections = []
        mock_ctx.raw_classifications = []

        with patch("app.api.deps.get_storage_client", return_value=mock_storage):
            with patch("app.api.routes.tasks.PipelineParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse.return_value = MagicMock()
                mock_parser_class.return_value = mock_parser

                with patch("app.api.routes.tasks.PipelineExecutor") as mock_executor:
                    mock_executor.return_value.execute = AsyncMock(return_value=mock_ctx)

                    response = await client.post(
                        "/tasks/process",
                        json=valid_request_payload,
                        headers={"X-CloudTasks-TaskName": "test-task"},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["success"] is True
                    assert data["tenant_id"] == "tenant-001"

    @pytest.mark.asyncio
    async def test_process_rejects_invalid_pipeline(
        self, client: AsyncClient, mock_storage
    ):
        """Endpoint should reject invalid pipeline_definition."""
        payload = {
            "tenant_id": "tenant-001",
            "session_id": str(uuid4()),
            "image_id": str(uuid4()),
            "image_url": "gs://bucket/image.jpg",
            "pipeline_definition": {
                "type": "chain",
                "steps": [
                    {"type": "step", "name": "nonexistent_step"},  # Invalid
                ],
            },
        }

        from app.core.pipeline_parser import PipelineParserError

        with patch("app.api.deps.get_storage_client", return_value=mock_storage):
            with patch("app.api.routes.tasks.PipelineParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse.side_effect = PipelineParserError("Unknown step: nonexistent_step")
                mock_parser_class.return_value = mock_parser

                response = await client.post(
                    "/tasks/process",
                    json=payload,
                    headers={"X-CloudTasks-TaskName": "test-task"},
                )

                assert response.status_code == 400
                assert "Invalid pipeline" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_process_validates_request_schema(self, client: AsyncClient):
        """Test that invalid requests are rejected."""
        invalid_payload = {
            "tenant_id": "test-tenant",
            # Missing required fields
        }

        response = await client.post("/tasks/process", json=invalid_payload)
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_process_validates_uuid_format(self, client: AsyncClient):
        """Test that invalid UUIDs are rejected."""
        invalid_payload = {
            "tenant_id": "test-tenant",
            "session_id": "not-a-uuid",
            "image_id": "also-not-a-uuid",
            "image_url": "gs://bucket/path",
            "pipeline_definition": {
                "type": "chain",
                "steps": [{"type": "step", "name": "detection"}],
            },
        }

        response = await client.post("/tasks/process", json=invalid_payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_process_returns_error_on_failure(
        self, client: AsyncClient, valid_request_payload: dict, mock_storage
    ):
        """Endpoint should return success=False on processing failure."""
        with patch("app.api.deps.get_storage_client", return_value=mock_storage):
            with patch("app.api.routes.tasks.PipelineParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse.return_value = MagicMock()
                mock_parser_class.return_value = mock_parser

                with patch("app.api.routes.tasks.PipelineExecutor") as mock_executor:
                    mock_executor.return_value.execute = AsyncMock(
                        side_effect=RuntimeError("Pipeline execution failed")
                    )

                    response = await client.post(
                        "/tasks/process",
                        json=valid_request_payload,
                        headers={"X-CloudTasks-TaskName": "test-task"},
                    )

                    assert response.status_code == 200  # Returns 200 with success=False
                    data = response.json()
                    assert data["success"] is False
                    assert "Pipeline execution failed" in data["error"]
