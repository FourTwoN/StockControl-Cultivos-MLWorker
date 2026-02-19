"""Tests for SendResultsBackendStep."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.processing_context import ProcessingContext
from app.steps.post.send_results_backend import SendResultsBackendStep


@pytest.fixture
def sample_context() -> ProcessingContext:
    """Create a sample processing context with ML results."""
    return ProcessingContext(
        tenant_id="test-tenant",
        image_id=str(uuid4()),
        session_id=str(uuid4()),
        image_path=Path("/tmp/test.jpg"),
        config={"pipeline_name": "test_pipeline"},
        raw_detections=[
            {
                "label": "plant_a",
                "confidence": 0.95,
                "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            },
            {
                "class_name": "plant_b",
                "confidence": 0.87,
                "bbox": {"x1": 50, "y1": 60, "x2": 150, "y2": 180},
            },
        ],
        results={
            "total_count": 42,
            "size_distribution": {"S": 10, "M": 20, "L": 10, "XL": 2},
            "species_distribution": {"species_a": 25, "species_b": 17},
        },
    )


@pytest.fixture
def step() -> SendResultsBackendStep:
    """Create a SendResultsBackendStep instance."""
    return SendResultsBackendStep()


def test_step_name(step: SendResultsBackendStep):
    """Verify step name is correct."""
    assert step.name == "send_results_backend"


def test_transform_detections(step: SendResultsBackendStep):
    """Verify raw detections are transformed correctly."""
    raw_detections = [
        {
            "label": "plant_a",
            "confidence": 0.95,
            "bbox": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
        },
        {
            "class_name": "plant_b",
            "confidence": 0.87,
        },
    ]

    result = step._transform_detections(raw_detections)

    assert len(result) == 2

    # First detection with bbox
    assert result[0].label == "plant_a"
    assert result[0].confidence == 0.95
    assert result[0].boundingBox is not None
    assert result[0].boundingBox.x1 == 10
    assert result[0].boundingBox.y2 == 200

    # Second detection uses class_name fallback, no bbox
    assert result[1].label == "plant_b"
    assert result[1].confidence == 0.87
    assert result[1].boundingBox is None


def test_transform_estimations(step: SendResultsBackendStep):
    """Verify results are transformed to estimations correctly."""
    results = {
        "total_count": 42,
        "size_distribution": {"S": 10, "M": 20},
        "species_distribution": {"species_a": 25},
        "other_key": "ignored",
    }

    estimations = step._transform_estimations(results)

    # total_count + 2 sizes + 1 species = 4 estimations
    assert len(estimations) == 4

    # Find total_count
    total = next(e for e in estimations if e.estimationType == "total_count")
    assert total.value == 42.0
    assert total.unit == "units"

    # Find size_S
    size_s = next(e for e in estimations if e.estimationType == "size_S")
    assert size_s.value == 10.0

    # Find species_a
    species_a = next(e for e in estimations if e.estimationType == "species_species_a")
    assert species_a.value == 25.0


def test_transform_estimations_empty_results(step: SendResultsBackendStep):
    """Verify empty results produce no estimations."""
    estimations = step._transform_estimations({})
    assert len(estimations) == 0


@pytest.mark.asyncio
async def test_execute_sends_results(
    step: SendResultsBackendStep,
    sample_context: ProcessingContext,
):
    """Verify execute sends results to backend."""
    mock_response = {"status": "ok", "id": "12345"}

    with patch(
        "app.steps.post.send_results_backend.BackendClient"
    ) as MockClient:
        mock_client = MockClient.return_value
        mock_client.send_results = AsyncMock(return_value=mock_response)

        result_ctx = await step.execute(sample_context)

        # Verify client was called
        mock_client.send_results.assert_called_once()
        call_args = mock_client.send_results.call_args

        assert call_args.kwargs["tenant_id"] == "test-tenant"

        # Verify results were added to context
        assert "backend_callback" in result_ctx.results
        callback_result = result_ctx.results["backend_callback"]
        assert callback_result["success"] is True
        assert callback_result["response"] == mock_response
        assert callback_result["detections_sent"] == 2
        assert callback_result["estimations_sent"] == 7  # 1 total + 4 sizes + 2 species


@pytest.mark.asyncio
async def test_execute_with_empty_detections(step: SendResultsBackendStep):
    """Verify execute handles empty detections gracefully."""
    ctx = ProcessingContext(
        tenant_id="test-tenant",
        image_id=str(uuid4()),
        session_id=str(uuid4()),
        image_path=Path("/tmp/test.jpg"),
        config={},
        raw_detections=[],
        results={},
    )

    with patch(
        "app.steps.post.send_results_backend.BackendClient"
    ) as MockClient:
        mock_client = MockClient.return_value
        mock_client.send_results = AsyncMock(return_value={"status": "ok"})

        result_ctx = await step.execute(ctx)

        callback_result = result_ctx.results["backend_callback"]
        assert callback_result["detections_sent"] == 0
        assert callback_result["estimations_sent"] == 0
