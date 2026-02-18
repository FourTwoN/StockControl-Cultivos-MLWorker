"""Tests for ML step wrappers.

Tests that ML steps correctly wrap processors and convert results to dicts.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.processing_context import ProcessingContext
from app.processors.detector_processor import DetectionResult
from app.processors.segmentation_processor import SegmentResult


@pytest.fixture
def mock_context() -> ProcessingContext:
    """Create a mock processing context."""
    return ProcessingContext(
        tenant_id="test-tenant",
        image_id="test-image-id",
        session_id="test-session-id",
        image_path=Path("/tmp/test.jpg"),
        config={},
    )


@pytest.fixture
def mock_segment_results() -> list[SegmentResult]:
    """Create mock segment results."""
    return [
        SegmentResult(
            segment_idx=0,
            class_name="field",
            confidence=0.95,
            bbox=(0.1, 0.2, 0.5, 0.6),
            area_px=1000.0,
            polygon=None,
            mask_rle=None,
        ),
        SegmentResult(
            segment_idx=1,
            class_name="field",
            confidence=0.88,
            bbox=(0.6, 0.3, 0.9, 0.7),
            area_px=800.0,
            polygon=None,
            mask_rle=None,
        ),
    ]


@pytest.fixture
def mock_detection_results() -> list[DetectionResult]:
    """Create mock detection results."""
    return [
        DetectionResult(
            center_x_px=100.0,
            center_y_px=150.0,
            width_px=50.0,
            height_px=60.0,
            confidence=0.92,
            class_name="plant",
            segment_idx=0,
            image_id="test-image",
        ),
        DetectionResult(
            center_x_px=200.0,
            center_y_px=250.0,
            width_px=55.0,
            height_px=65.0,
            confidence=0.87,
            class_name="plant",
            segment_idx=0,
            image_id="test-image",
        ),
    ]


class TestSegmentationStep:
    """Tests for SegmentationStep."""

    @pytest.mark.asyncio
    async def test_execute_calls_segmentation_processor(
        self, mock_context: ProcessingContext, mock_segment_results: list[SegmentResult]
    ) -> None:
        """Test that SegmentationStep calls the segmentation processor."""
        from app.steps.ml.segmentation_step import SegmentationStep

        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor.process.return_value = mock_segment_results

        with patch("app.steps.ml.segmentation_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = SegmentationStep()
            result_ctx = await step.execute(mock_context)

            # Verify processor was called with tenant_id
            mock_registry.return_value.get.assert_called_once_with(
                "segmentation", tenant_id="test-tenant"
            )
            mock_processor.process.assert_called_once_with(mock_context.image_path)

            # Verify results converted to dicts
            assert len(result_ctx.raw_segments) == 2
            assert result_ctx.raw_segments[0]["segment_idx"] == 0
            assert result_ctx.raw_segments[0]["class_name"] == "field"
            assert result_ctx.raw_segments[0]["confidence"] == 0.95
            assert result_ctx.raw_segments[1]["segment_idx"] == 1

    @pytest.mark.asyncio
    async def test_step_name(self) -> None:
        """Test that step has correct name."""
        from app.steps.ml.segmentation_step import SegmentationStep

        step = SegmentationStep()
        assert step.name == "segmentation"

    @pytest.mark.asyncio
    async def test_execute_with_empty_results(
        self, mock_context: ProcessingContext
    ) -> None:
        """Test execution with no segments found."""
        from app.steps.ml.segmentation_step import SegmentationStep

        mock_processor = AsyncMock()
        mock_processor.process.return_value = []

        with patch("app.steps.ml.segmentation_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = SegmentationStep()
            result_ctx = await step.execute(mock_context)

            assert result_ctx.raw_segments == []


class TestDetectionStep:
    """Tests for DetectionStep."""

    @pytest.mark.asyncio
    async def test_execute_calls_detection_processor(
        self, mock_context: ProcessingContext, mock_detection_results: list[DetectionResult]
    ) -> None:
        """Test that DetectionStep calls the detection processor."""
        from app.steps.ml.detection_step import DetectionStep

        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor.process.return_value = mock_detection_results

        with patch("app.steps.ml.detection_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = DetectionStep()
            result_ctx = await step.execute(mock_context)

            # Verify processor was called with tenant_id
            mock_registry.return_value.get.assert_called_once_with(
                "detection", tenant_id="test-tenant"
            )
            mock_processor.process.assert_called_once_with(mock_context.image_path)

            # Verify results converted to dicts
            assert len(result_ctx.raw_detections) == 2
            assert result_ctx.raw_detections[0]["center_x_px"] == 100.0
            assert result_ctx.raw_detections[0]["center_y_px"] == 150.0
            assert result_ctx.raw_detections[0]["width_px"] == 50.0
            assert result_ctx.raw_detections[0]["height_px"] == 60.0
            assert result_ctx.raw_detections[0]["confidence"] == 0.92
            assert result_ctx.raw_detections[0]["class_name"] == "plant"
            assert result_ctx.raw_detections[0]["segment_idx"] == 0
            assert result_ctx.raw_detections[0]["image_id"] == "test-image"

    @pytest.mark.asyncio
    async def test_step_name(self) -> None:
        """Test that step has correct name."""
        from app.steps.ml.detection_step import DetectionStep

        step = DetectionStep()
        assert step.name == "detection"

    @pytest.mark.asyncio
    async def test_execute_with_empty_results(
        self, mock_context: ProcessingContext
    ) -> None:
        """Test execution with no detections found."""
        from app.steps.ml.detection_step import DetectionStep

        mock_processor = AsyncMock()
        mock_processor.process.return_value = []

        with patch("app.steps.ml.detection_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = DetectionStep()
            result_ctx = await step.execute(mock_context)

            assert result_ctx.raw_detections == []


class TestSAHIDetectionStep:
    """Tests for SAHIDetectionStep."""

    @pytest.mark.asyncio
    async def test_execute_calls_sahi_detector(
        self, mock_context: ProcessingContext, mock_detection_results: list[DetectionResult]
    ) -> None:
        """Test that SAHIDetectionStep calls the SAHI detector."""
        from app.steps.ml.sahi_detection_step import SAHIDetectionStep

        # Add SAHI config to context
        context_with_config = ProcessingContext(
            tenant_id=mock_context.tenant_id,
            image_id=mock_context.image_id,
            session_id=mock_context.session_id,
            image_path=mock_context.image_path,
            config={
                "sahi_slice_height": 640,
                "sahi_slice_width": 640,
                "sahi_overlap_ratio": 0.3,
            },
        )

        # Mock the processor
        mock_processor = AsyncMock()
        mock_processor.process.return_value = mock_detection_results

        with patch("app.steps.ml.sahi_detection_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = SAHIDetectionStep()
            result_ctx = await step.execute(context_with_config)

            # Verify processor was called with tenant_id and config
            mock_registry.return_value.get.assert_called_once_with(
                "sahi_detection", tenant_id="test-tenant"
            )
            mock_processor.process.assert_called_once_with(
                context_with_config.image_path,
                slice_height=640,
                slice_width=640,
                overlap_ratio=0.3,
            )

            # Verify results converted to dicts
            assert len(result_ctx.raw_detections) == 2
            assert result_ctx.raw_detections[0]["center_x_px"] == 100.0

    @pytest.mark.asyncio
    async def test_step_name(self) -> None:
        """Test that step has correct name."""
        from app.steps.ml.sahi_detection_step import SAHIDetectionStep

        step = SAHIDetectionStep()
        assert step.name == "sahi_detection"

    @pytest.mark.asyncio
    async def test_execute_with_default_config(
        self, mock_context: ProcessingContext, mock_detection_results: list[DetectionResult]
    ) -> None:
        """Test execution with default SAHI config values."""
        from app.steps.ml.sahi_detection_step import SAHIDetectionStep

        mock_processor = AsyncMock()
        mock_processor.process.return_value = mock_detection_results

        with patch("app.steps.ml.sahi_detection_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = SAHIDetectionStep()
            result_ctx = await step.execute(mock_context)

            # Verify default values used
            mock_processor.process.assert_called_once_with(
                mock_context.image_path,
                slice_height=512,
                slice_width=512,
                overlap_ratio=0.25,
            )

            assert len(result_ctx.raw_detections) == 2

    @pytest.mark.asyncio
    async def test_execute_with_partial_config(
        self, mock_context: ProcessingContext, mock_detection_results: list[DetectionResult]
    ) -> None:
        """Test execution with partial SAHI config (defaults for missing values)."""
        from app.steps.ml.sahi_detection_step import SAHIDetectionStep

        # Add partial config to context
        context_with_config = ProcessingContext(
            tenant_id=mock_context.tenant_id,
            image_id=mock_context.image_id,
            session_id=mock_context.session_id,
            image_path=mock_context.image_path,
            config={
                "sahi_slice_height": 1024,
            },
        )

        mock_processor = AsyncMock()
        mock_processor.process.return_value = mock_detection_results

        with patch("app.steps.ml.sahi_detection_step.get_processor_registry") as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            step = SAHIDetectionStep()
            result_ctx = await step.execute(context_with_config)

            # Verify partial config with defaults
            mock_processor.process.assert_called_once_with(
                context_with_config.image_path,
                slice_height=1024,
                slice_width=512,
                overlap_ratio=0.25,
            )
