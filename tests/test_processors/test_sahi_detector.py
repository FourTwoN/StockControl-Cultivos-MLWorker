"""Tests for SAHI Detector Processor."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.processors.sahi_detector_processor import SAHIDetectorProcessor
from app.processors.detector_processor import DetectionResult


class TestSAHIDetectorProcessor:
    """Test suite for SAHIDetectorProcessor."""

    @pytest.fixture
    def processor(self) -> SAHIDetectorProcessor:
        """Create processor instance for testing."""
        return SAHIDetectorProcessor(
            model_path=None,
            worker_id=0,
            confidence_threshold=0.8,
        )

    @pytest.fixture
    def mock_sahi_result(self) -> MagicMock:
        """Create mock SAHI prediction result."""
        mock_bbox = MagicMock()
        mock_bbox.minx = 100
        mock_bbox.maxx = 200
        mock_bbox.miny = 150
        mock_bbox.maxy = 250

        mock_score = MagicMock()
        mock_score.value = 0.95

        mock_category = MagicMock()
        mock_category.name = "plant"

        mock_prediction = MagicMock()
        mock_prediction.bbox = mock_bbox
        mock_prediction.score = mock_score
        mock_prediction.category = mock_category

        mock_result = MagicMock()
        mock_result.object_prediction_list = [mock_prediction]

        return mock_result

    def test_init_default_values(self) -> None:
        """Test processor initialization with defaults."""
        processor = SAHIDetectorProcessor()
        assert processor.confidence_threshold == 0.80
        assert processor.worker_id == 0
        assert processor._model is None

    def test_init_custom_values(self) -> None:
        """Test processor initialization with custom values."""
        processor = SAHIDetectorProcessor(
            model_path="/custom/path.pt",
            worker_id=2,
            confidence_threshold=0.5,
        )
        assert processor.confidence_threshold == 0.5
        assert processor.worker_id == 2

    @pytest.mark.asyncio
    async def test_process_file_not_found(
        self,
        processor: SAHIDetectorProcessor,
    ) -> None:
        """Test that FileNotFoundError is raised for missing image."""
        with pytest.raises(FileNotFoundError):
            await processor.process("/nonexistent/image.jpg")

    @pytest.mark.asyncio
    async def test_process_success(
        self,
        processor: SAHIDetectorProcessor,
        temp_image_path: Path,
        mock_sahi_result: MagicMock,
    ) -> None:
        """Test successful SAHI detection."""
        with patch.object(processor, "_model", MagicMock()), \
             patch("app.processors.sahi_detector_processor.ModelCache") as mock_cache, \
             patch("app.processors.sahi_detector_processor.AutoDetectionModel") as mock_auto, \
             patch("app.processors.sahi_detector_processor.get_sliced_prediction") as mock_predict, \
             patch("app.processors.sahi_detector_processor.Image") as mock_pil:

            # Setup mocks
            mock_cache.get_model.return_value = MagicMock()
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_predict.return_value = mock_sahi_result

            mock_img = MagicMock()
            mock_img.size = (1024, 768)
            mock_img.__enter__ = MagicMock(return_value=mock_img)
            mock_img.__exit__ = MagicMock(return_value=False)
            mock_pil.open.return_value = mock_img

            # Execute
            results = await processor.process(temp_image_path)

            # Verify
            assert len(results) == 1
            assert isinstance(results[0], DetectionResult)
            assert results[0].confidence == 0.95
            assert results[0].class_name == "plant"
            assert results[0].center_x_px == 150.0  # (100 + 200) / 2
            assert results[0].center_y_px == 200.0  # (150 + 250) / 2
            assert results[0].width_px == 100.0  # 200 - 100
            assert results[0].height_px == 100.0  # 250 - 150

    def test_parse_sahi_results_empty(
        self,
        processor: SAHIDetectorProcessor,
    ) -> None:
        """Test parsing empty SAHI results."""
        mock_result = MagicMock()
        mock_result.object_prediction_list = []

        detections = processor._parse_sahi_results(mock_result)
        assert detections == []

    def test_parse_sahi_results_sorted_by_confidence(
        self,
        processor: SAHIDetectorProcessor,
    ) -> None:
        """Test that results are sorted by confidence descending."""
        # Create multiple predictions with different confidences
        predictions = []
        for conf in [0.5, 0.9, 0.7]:
            mock_bbox = MagicMock()
            mock_bbox.minx = mock_bbox.miny = 0
            mock_bbox.maxx = mock_bbox.maxy = 100

            mock_score = MagicMock()
            mock_score.value = conf

            mock_category = MagicMock()
            mock_category.name = "plant"

            pred = MagicMock()
            pred.bbox = mock_bbox
            pred.score = mock_score
            pred.category = mock_category
            predictions.append(pred)

        mock_result = MagicMock()
        mock_result.object_prediction_list = predictions

        detections = processor._parse_sahi_results(mock_result)

        assert len(detections) == 3
        assert detections[0].confidence == 0.9
        assert detections[1].confidence == 0.7
        assert detections[2].confidence == 0.5

    @pytest.mark.asyncio
    async def test_process_invalid_image_dimensions(
        self,
        processor: SAHIDetectorProcessor,
        temp_image_path: Path,
    ) -> None:
        """Test error handling for invalid image dimensions."""
        with patch("app.processors.sahi_detector_processor.Image") as mock_pil:
            mock_img = MagicMock()
            mock_img.size = (0, 0)  # Invalid dimensions
            mock_img.__enter__ = MagicMock(return_value=mock_img)
            mock_img.__exit__ = MagicMock(return_value=False)
            mock_pil.open.return_value = mock_img

            with pytest.raises(ValueError, match="Invalid image dimensions"):
                await processor.process(temp_image_path)

    @pytest.mark.asyncio
    async def test_process_uses_correct_slice_params(
        self,
        processor: SAHIDetectorProcessor,
        temp_image_path: Path,
        mock_sahi_result: MagicMock,
    ) -> None:
        """Test that custom slice parameters are passed correctly."""
        with patch.object(processor, "_model", MagicMock()), \
             patch("app.processors.sahi_detector_processor.ModelCache") as mock_cache, \
             patch("app.processors.sahi_detector_processor.AutoDetectionModel") as mock_auto, \
             patch("app.processors.sahi_detector_processor.get_sliced_prediction") as mock_predict, \
             patch("app.processors.sahi_detector_processor.Image") as mock_pil:

            mock_cache.get_model.return_value = MagicMock()
            mock_auto.from_pretrained.return_value = MagicMock()
            mock_predict.return_value = mock_sahi_result

            mock_img = MagicMock()
            mock_img.size = (2048, 2048)
            mock_img.__enter__ = MagicMock(return_value=mock_img)
            mock_img.__exit__ = MagicMock(return_value=False)
            mock_pil.open.return_value = mock_img

            await processor.process(
                temp_image_path,
                slice_height=640,
                slice_width=640,
                overlap_ratio=0.3,
            )

            # Verify slice parameters were passed
            call_kwargs = mock_predict.call_args.kwargs
            assert call_kwargs["slice_height"] == 640
            assert call_kwargs["slice_width"] == 640
            assert call_kwargs["overlap_height_ratio"] == 0.3
            assert call_kwargs["overlap_width_ratio"] == 0.3
