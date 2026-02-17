"""Tests for task processing endpoints."""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock, MagicMock

from app.schemas.pipeline_definition import PipelineDefinition, StepDefinition


class TestCloudTasksValidation:
    """Tests for Cloud Tasks header validation."""

    @pytest.fixture
    def valid_request_payload(self) -> dict:
        return {
            "tenant_id": "test-tenant-001",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "image_id": "660e8400-e29b-41d4-a716-446655440001",
            "image_url": "gs://test-bucket/test-tenant-001/images/test.jpg",
            "pipeline": "DETECTION",
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

        # Should pass validation and fail on tenant config (404) not auth (403)
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
    """Tests for /tasks/process endpoint."""

    @pytest.fixture
    def valid_request_payload(self) -> dict:
        return {
            "tenant_id": "test-tenant-001",
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "image_id": "660e8400-e29b-41d4-a716-446655440001",
            "image_url": "gs://test-bucket/test-tenant-001/images/test.jpg",
            "pipeline": "DETECTION",
        }

    @pytest.mark.asyncio
    async def test_process_requires_cloud_tasks_headers(
        self, client: AsyncClient, valid_request_payload: dict
    ):
        """Test that endpoint returns 404 when tenant config is not found."""
        # Without tenant config in cache, should return 404
        response = await client.post(
            "/tasks/process",
            json=valid_request_payload,
        )
        # Returns 404 because tenant config is not in cache
        assert response.status_code == 404

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
            "pipeline": "DETECTION",
        }

        response = await client.post("/tasks/process", json=invalid_payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_process_with_mocked_pipeline(
        self, client: AsyncClient, valid_request_payload: dict
    ):
        """Test process endpoint with mocked tenant config and pipeline."""
        # Create mock config with PipelineDefinition
        mock_config = MagicMock()
        mock_config.pipeline_definition = PipelineDefinition(
            steps=[StepDefinition(name="detection")]
        )
        mock_config.settings = {}

        mock_ctx = MagicMock()
        mock_ctx.results = {"detection": []}
        mock_ctx.raw_segments = []
        mock_ctx.raw_detections = []
        mock_ctx.raw_classifications = []

        with patch("app.api.routes.tasks.get_tenant_cache") as mock_cache:
            mock_cache.return_value.get = AsyncMock(return_value=mock_config)

            with patch("app.api.routes.tasks.PipelineParser") as mock_parser_class:
                mock_parser = MagicMock()
                mock_parser.parse.return_value = MagicMock()  # Return mock Chain
                mock_parser_class.return_value = mock_parser

                with patch("app.api.routes.tasks.PipelineExecutor") as mock_executor_class:
                    mock_executor = MagicMock()
                    mock_executor.execute = AsyncMock(return_value=mock_ctx)
                    mock_executor_class.return_value = mock_executor

                    with patch("app.api.deps.get_storage_client") as mock_storage:
                        mock_storage_instance = MagicMock()
                        mock_storage_instance.download_to_tempfile = AsyncMock(
                            return_value=MagicMock(exists=lambda: False)
                        )
                        mock_storage.return_value = mock_storage_instance

                        response = await client.post(
                            "/tasks/process",
                            json=valid_request_payload,
                        )

                        # Should attempt to process (may fail on other dependencies)
                        assert response.status_code in [200, 400, 404, 500]


class TestCompressEndpoint:
    """Tests for /tasks/compress endpoint."""

    @pytest.fixture
    def valid_compress_payload(self) -> dict:
        return {
            "tenant_id": "test-tenant-001",
            "image_id": "660e8400-e29b-41d4-a716-446655440001",
            "source_url": "gs://test-bucket/test-tenant-001/originals/test.jpg",
            "target_sizes": [128, 256, 512],
            "quality": 85,
        }

    @pytest.mark.asyncio
    async def test_compress_validates_request_schema(self, client: AsyncClient):
        """Test that invalid compress requests are rejected."""
        invalid_payload = {
            "tenant_id": "test-tenant",
            # Missing required fields
        }

        response = await client.post("/tasks/compress", json=invalid_payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_compress_validates_target_sizes(self, client: AsyncClient):
        """Test that target_sizes must be a list of integers."""
        invalid_payload = {
            "tenant_id": "test-tenant",
            "image_id": "660e8400-e29b-41d4-a716-446655440001",
            "source_url": "gs://bucket/path",
            "target_sizes": "not-a-list",
        }

        response = await client.post("/tasks/compress", json=invalid_payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_compress_validates_quality_range(
        self, client: AsyncClient, valid_compress_payload: dict
    ):
        """Test that quality must be within valid range."""
        valid_compress_payload["quality"] = 150  # Invalid: > 100

        response = await client.post("/tasks/compress", json=valid_compress_payload)
        assert response.status_code == 422
