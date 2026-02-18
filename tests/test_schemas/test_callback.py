"""Tests for callback schemas."""

from uuid import uuid4

from app.schemas.callback import (
    BoundingBox,
    DetectionResultItem,
    ClassificationResultItem,
    EstimationResultItem,
    ProcessingMetadata,
    ProcessingResultRequest,
)


def test_bounding_box_serialization():
    """BoundingBox should serialize to camelCase JSON."""
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
    data = bbox.model_dump(mode="json")
    assert data == {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0}


def test_detection_result_item_serialization():
    """DetectionResultItem should serialize with camelCase."""
    item = DetectionResultItem(
        label="cactus",
        confidence=0.95,
        boundingBox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
    )
    data = item.model_dump(mode="json")
    assert data["label"] == "cactus"
    assert data["confidence"] == 0.95
    assert "boundingBox" in data


def test_processing_result_request_serialization():
    """ProcessingResultRequest should serialize for Java backend."""
    session_id = uuid4()
    image_id = uuid4()

    request = ProcessingResultRequest(
        sessionId=session_id,
        imageId=image_id,
        detections=[
            DetectionResultItem(label="plant", confidence=0.9, boundingBox=None)
        ],
        classifications=[],
        estimations=[],
    )

    data = request.model_dump(mode="json")
    assert data["sessionId"] == str(session_id)
    assert data["imageId"] == str(image_id)
    assert len(data["detections"]) == 1
