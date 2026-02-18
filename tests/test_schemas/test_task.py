"""Tests for ProcessingRequest schema with pipeline_definition."""

import pytest
from uuid import uuid4

from app.schemas.task import ProcessingRequest
from app.schemas.pipeline_definition import PipelineDefinition


class TestProcessingRequestWithPipeline:
    """Test ProcessingRequest accepts pipeline_definition."""

    def test_request_with_pipeline_definition(self):
        """Request should accept full pipeline_definition."""
        pipeline_def = {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "segmentation"},
                {"type": "step", "name": "detection"},
            ],
        }

        request = ProcessingRequest(
            tenant_id="tenant-001",
            session_id=uuid4(),
            image_id=uuid4(),
            image_url="gs://bucket/image.jpg",
            pipeline_definition=pipeline_def,
            settings={"key": "value"},
        )

        assert request.pipeline_definition.type == "chain"
        assert len(request.pipeline_definition.steps) == 2
        assert request.settings == {"key": "value"}

    def test_request_with_empty_settings(self):
        """Settings should default to empty dict."""
        request = ProcessingRequest(
            tenant_id="tenant-001",
            session_id=uuid4(),
            image_id=uuid4(),
            image_url="gs://bucket/image.jpg",
            pipeline_definition={
                "type": "chain",
                "steps": [{"type": "step", "name": "detection"}],
            },
        )

        assert request.settings == {}

    def test_request_rejects_missing_pipeline_definition(self):
        """Request should require pipeline_definition."""
        with pytest.raises(Exception):  # ValidationError
            ProcessingRequest(
                tenant_id="tenant-001",
                session_id=uuid4(),
                image_id=uuid4(),
                image_url="gs://bucket/image.jpg",
                # missing pipeline_definition
            )
