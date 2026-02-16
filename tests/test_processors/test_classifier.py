"""Tests for Classifier Processor."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.processors.classifier_processor import ClassifierProcessor, Classification
from app.processors.detector_processor import DetectionResult


class TestClassification:
    """Test suite for Classification dataclass."""

    def test_valid_classification(self) -> None:
        """Test creating a valid classification."""
        cls = Classification(
            detection_id=1,
            class_name="tomato",
            confidence=0.95,
            segment_idx=0,
            image_id="test-123",
            product_size_id=2,
        )
        assert cls.detection_id == 1
        assert cls.class_name == "tomato"
        assert cls.confidence == 0.95
        assert cls.product_size_id == 2

    def test_invalid_confidence_too_high(self) -> None:
        """Test that confidence > 1.0 raises error."""
        with pytest.raises(ValueError, match="confidence must be in"):
            Classification(
                class_name="tomato",
                confidence=1.5,
            )

    def test_invalid_confidence_negative(self) -> None:
        """Test that negative confidence raises error."""
        with pytest.raises(ValueError, match="confidence must be in"):
            Classification(
                class_name="tomato",
                confidence=-0.1,
            )

    def test_invalid_empty_class_name(self) -> None:
        """Test that empty class_name raises error."""
        with pytest.raises(ValueError, match="class_name must be a non-empty string"):
            Classification(
                class_name="",
                confidence=0.8,
            )

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        cls = Classification(
            detection_id=5,
            estimation_id=None,
            class_name="pepper",
            confidence=0.88,
            segment_idx=2,
            image_id="img-456",
            product_size_id=3,
        )
        result = cls.to_dict()

        assert result["detection_id"] == 5
        assert result["estimation_id"] is None
        assert result["class_name"] == "pepper"
        assert result["confidence"] == 0.88
        assert result["segment_idx"] == 2
        assert result["image_id"] == "img-456"
        assert result["product_size_id"] == 3

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        cls = Classification(class_name="test", confidence=0.5)
        assert cls.detection_id is None
        assert cls.estimation_id is None
        assert cls.segment_idx == -1
        assert cls.image_id == ""
        assert cls.product_size_id is None


class TestClassifierProcessor:
    """Test suite for ClassifierProcessor."""

    @pytest.fixture
    def processor(self) -> ClassifierProcessor:
        """Create processor instance for testing."""
        return ClassifierProcessor(
            model_path=None,
            worker_id=0,
            confidence_threshold=0.7,
        )

    @pytest.fixture
    def sample_detections(self) -> list[DetectionResult]:
        """Create sample detections for testing."""
        return [
            DetectionResult(
                center_x_px=100.0,
                center_y_px=100.0,
                width_px=50.0,
                height_px=60.0,
                confidence=0.9,
                class_name="plant",
            ),
            DetectionResult(
                center_x_px=200.0,
                center_y_px=200.0,
                width_px=45.0,
                height_px=55.0,
                confidence=0.85,
                class_name="plant",
            ),
            DetectionResult(
                center_x_px=300.0,
                center_y_px=300.0,
                width_px=40.0,
                height_px=50.0,
                confidence=0.8,
                class_name="plant",
            ),
        ]

    @pytest.fixture
    def sample_species_config(self) -> list[dict]:
        """Create sample species configuration."""
        return [
            {"product_name": "Tomato", "product_id": 1},
            {"product_name": "Pepper", "product_id": 2},
            {"product_name": "Lettuce", "product_id": 3},
        ]

    def test_init(self) -> None:
        """Test processor initialization."""
        processor = ClassifierProcessor(
            model_path="/path/to/model.pt",
            worker_id=1,
            confidence_threshold=0.6,
        )
        assert processor.worker_id == 1
        assert processor.confidence_threshold == 0.6

    @pytest.mark.asyncio
    async def test_process_empty_detections(
        self,
        processor: ClassifierProcessor,
        temp_image_path: Path,
    ) -> None:
        """Test processing with no detections returns empty list."""
        results = await processor.process(
            image_path=temp_image_path,
            detections=[],
            species_config=[{"product_name": "Test"}],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_process_no_species_config(
        self,
        processor: ClassifierProcessor,
        temp_image_path: Path,
        sample_detections: list[DetectionResult],
    ) -> None:
        """Test processing without species config returns empty or default."""
        results = await processor.process(
            image_path=temp_image_path,
            detections=sample_detections,
            species_config=None,
        )
        # With no species config, should return empty or handle gracefully
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_process_with_detections(
        self,
        processor: ClassifierProcessor,
        temp_image_path: Path,
        sample_detections: list[DetectionResult],
        sample_species_config: list[dict],
    ) -> None:
        """Test processing with detections and species config."""
        results = await processor.process(
            image_path=temp_image_path,
            detections=sample_detections,
            species_config=sample_species_config,
        )

        # Should have same number of classifications as detections
        assert len(results) == len(sample_detections)

        # Each result should be a Classification
        for result in results:
            assert isinstance(result, Classification)
            assert result.confidence >= 0.0
            assert result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_process_equitable_distribution(
        self,
        processor: ClassifierProcessor,
        temp_image_path: Path,
        sample_species_config: list[dict],
    ) -> None:
        """Test that classifications are distributed equitably across species."""
        # Create 9 detections (3 species, should get 3 each)
        detections = [
            DetectionResult(
                center_x_px=float(i * 100),
                center_y_px=float(i * 100),
                width_px=50.0,
                height_px=60.0,
                confidence=0.9,
                class_name="plant",
            )
            for i in range(9)
        ]

        results = await processor.process(
            image_path=temp_image_path,
            detections=detections,
            species_config=sample_species_config,
        )

        assert len(results) == 9

        # Count classifications per species
        species_counts: dict[str, int] = {}
        for cls in results:
            species_counts[cls.class_name] = species_counts.get(cls.class_name, 0) + 1

        # Should be roughly equitable (3 each for 3 species with 9 detections)
        for count in species_counts.values():
            assert count == 3

    @pytest.mark.asyncio
    async def test_process_file_not_found(
        self,
        processor: ClassifierProcessor,
    ) -> None:
        """Test that FileNotFoundError is raised for missing image."""
        with pytest.raises(FileNotFoundError):
            await processor.process(
                image_path="/nonexistent/image.jpg",
                detections=[],
            )

    @pytest.mark.asyncio
    async def test_classification_result_has_segment_info(
        self,
        processor: ClassifierProcessor,
        temp_image_path: Path,
        sample_species_config: list[dict],
    ) -> None:
        """Test that classifications include segment information when available."""
        detection = DetectionResult(
            center_x_px=100.0,
            center_y_px=100.0,
            width_px=50.0,
            height_px=60.0,
            confidence=0.9,
            class_name="plant",
            segment_idx=2,
            image_id="test-image-123",
        )

        results = await processor.process(
            image_path=temp_image_path,
            detections=[detection],
            species_config=sample_species_config,
        )

        if results:
            assert results[0].segment_idx == 2
            assert results[0].image_id == "test-image-123"
