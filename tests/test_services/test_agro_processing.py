"""Tests for Agro Processing Service."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID
import pytest

from app.services.agro_processing_service import AgroProcessingService
from app.processors.segmentation_processor import SegmentResult
from app.processors.detector_processor import DetectionResult
from app.processors.classifier_processor import Classification
from app.schemas.task import ProcessingRequest


class TestAgroProcessingService:
    """Test suite for AgroProcessingService."""

    @pytest.fixture
    def mock_storage(self) -> MagicMock:
        """Create mock storage client."""
        storage = MagicMock()
        storage.download_to_tempfile = AsyncMock()
        return storage

    @pytest.fixture
    def mock_db_session(self) -> MagicMock:
        """Create mock async database session."""
        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def sample_request(self) -> ProcessingRequest:
        """Create sample processing request."""
        return ProcessingRequest(
            tenant_id="test-tenant",
            session_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
            image_id=UUID("660e8400-e29b-41d4-a716-446655440001"),
            image_url="gs://bucket/test-tenant/images/test.jpg",
            pipeline="FULL_PIPELINE",
            species_config=[
                {"product_name": "Tomato", "product_id": 1},
                {"product_name": "Pepper", "product_id": 2},
            ],
        )

    @pytest.fixture
    def sample_segments(self) -> list[SegmentResult]:
        """Create sample segment results."""
        return [
            SegmentResult(
                segment_idx=0,
                class_name="cajon",
                confidence=0.95,
                bbox=(0.1, 0.1, 0.4, 0.4),
                area_px=500000,
            ),
            SegmentResult(
                segment_idx=1,
                class_name="segmento",
                confidence=0.90,
                bbox=(0.5, 0.5, 0.9, 0.9),
                area_px=1500000,  # Large segment - should use SAHI
            ),
        ]

    @pytest.fixture
    def sample_detections(self) -> list[DetectionResult]:
        """Create sample detection results."""
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
        ]

    def test_init(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test service initialization."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )
        assert service.storage == mock_storage
        assert service.db == mock_db_session
        assert service.industry == "agro"

    def test_industry_property(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test industry property returns 'agro'."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )
        assert service.industry == "agro"

    def test_filter_claro_segments_keeps_largest(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test that filter keeps only the largest claro segment."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        segments = [
            SegmentResult(
                segment_idx=0,
                class_name="segmento",
                confidence=0.9,
                bbox=(0.0, 0.0, 0.5, 0.5),
                area_px=500000,
            ),
            SegmentResult(
                segment_idx=1,
                class_name="segmento",
                confidence=0.85,
                bbox=(0.5, 0.5, 1.0, 1.0),
                area_px=800000,  # Largest
            ),
            SegmentResult(
                segment_idx=2,
                class_name="cajon",
                confidence=0.95,
                bbox=(0.2, 0.2, 0.3, 0.3),
                area_px=100000,
            ),
        ]

        filtered = service._filter_claro_segments(segments)

        # Should have cajon + largest segmento
        assert len(filtered) == 2
        segmento_segments = [s for s in filtered if s.class_name == "segmento"]
        assert len(segmento_segments) == 1
        assert segmento_segments[0].area_px == 800000

    def test_filter_claro_segments_single_claro(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test that filter doesn't remove single claro segment."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        segments = [
            SegmentResult(
                segment_idx=0,
                class_name="segmento",
                confidence=0.9,
                bbox=(0.0, 0.0, 0.5, 0.5),
                area_px=500000,
            ),
            SegmentResult(
                segment_idx=1,
                class_name="cajon",
                confidence=0.95,
                bbox=(0.5, 0.5, 0.8, 0.8),
                area_px=300000,
            ),
        ]

        filtered = service._filter_claro_segments(segments)
        assert len(filtered) == 2

    def test_transform_coordinates(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test coordinate transformation from segment to full image."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        detections = [
            DetectionResult(
                center_x_px=50.0,
                center_y_px=50.0,
                width_px=20.0,
                height_px=30.0,
                confidence=0.9,
                class_name="plant",
            )
        ]

        segment = SegmentResult(
            segment_idx=0,
            class_name="segmento",
            confidence=0.9,
            bbox=(0.1, 0.2, 0.5, 0.6),  # x1=10%, y1=20%
            area_px=500000,
        )

        # Image is 1000x1000
        transformed = service._transform_coordinates(
            detections=detections,
            segment=segment,
            img_width=1000,
            img_height=1000,
            segment_idx=0,
            image_id="test-123",
        )

        assert len(transformed) == 1
        # Original center (50, 50) + segment offset (100, 200)
        assert transformed[0].center_x_px == 150.0  # 50 + (0.1 * 1000)
        assert transformed[0].center_y_px == 250.0  # 50 + (0.2 * 1000)
        assert transformed[0].segment_idx == 0
        assert transformed[0].image_id == "test-123"

    def test_serialize_segment(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test segment serialization."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        segment = SegmentResult(
            segment_idx=0,
            class_name="cajon",
            confidence=0.95,
            bbox=(0.1, 0.2, 0.3, 0.4),
            area_px=100000,
        )

        serialized = service._serialize_segment(segment)

        assert serialized["class_name"] == "cajon"
        assert serialized["confidence"] == 0.95
        assert serialized["bbox"] == (0.1, 0.2, 0.3, 0.4)
        assert serialized["area_px"] == 100000

    def test_serialize_detection(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
    ) -> None:
        """Test detection serialization."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        detection = DetectionResult(
            center_x_px=100.0,
            center_y_px=200.0,
            width_px=50.0,
            height_px=60.0,
            confidence=0.85,
            class_name="plant",
            segment_idx=1,
            image_id="img-456",
        )

        serialized = service._serialize_detection(detection)

        assert serialized["center_x_px"] == 100.0
        assert serialized["center_y_px"] == 200.0
        assert serialized["width_px"] == 50.0
        assert serialized["height_px"] == 60.0
        assert serialized["confidence"] == 0.85
        assert serialized["class_name"] == "plant"
        assert serialized["segment_idx"] == 1
        assert serialized["image_id"] == "img-456"

    @pytest.mark.asyncio
    async def test_process_success(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
        sample_request: ProcessingRequest,
        sample_segments: list[SegmentResult],
        sample_detections: list[DetectionResult],
        temp_image_path: Path,
    ) -> None:
        """Test successful processing pipeline."""
        # Setup storage mock
        mock_storage.download_to_tempfile.return_value = temp_image_path

        # Setup processor mocks
        mock_segmentation = MagicMock()
        mock_segmentation.process = AsyncMock(return_value=sample_segments)

        mock_detection = MagicMock()
        mock_detection.process = AsyncMock(return_value=sample_detections)

        mock_sahi = MagicMock()
        mock_sahi.process = AsyncMock(return_value=sample_detections)

        mock_classifier = MagicMock()
        mock_classifier.process = AsyncMock(return_value=[
            Classification(
                detection_id=1,
                class_name="Tomato",
                confidence=0.9,
                segment_idx=0,
                image_id="test",
                product_size_id=2,
            ),
        ])

        mock_registry = MagicMock()
        mock_registry.get.side_effect = lambda name: {
            "segmentation": mock_segmentation,
            "detection": mock_detection,
            "sahi_detection": mock_sahi,
            "classification": mock_classifier,
        }.get(name)

        with patch(
            "app.services.agro_processing_service.get_processor_registry",
            return_value=mock_registry,
        ), patch(
            "app.services.agro_processing_service.Image"
        ) as mock_pil, patch.object(
            AgroProcessingService,
            "_crop_segment",
            new_callable=AsyncMock,
            return_value=temp_image_path,
        ):
            # Mock PIL Image
            mock_img = MagicMock()
            mock_img.size = (1000, 1000)
            mock_img.__enter__ = MagicMock(return_value=mock_img)
            mock_img.__exit__ = MagicMock(return_value=False)
            mock_pil.open.return_value = mock_img

            service = AgroProcessingService(
                storage_client=mock_storage,
                db_session=mock_db_session,
            )

            response = await service.process(sample_request)

            assert response.success is True
            assert response.tenant_id == "test-tenant"
            assert response.pipeline == "FULL_PIPELINE"
            assert "total_detected" in response.results

    @pytest.mark.asyncio
    async def test_process_no_segments_returns_empty(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
        sample_request: ProcessingRequest,
        temp_image_path: Path,
    ) -> None:
        """Test that empty segments returns empty results."""
        mock_storage.download_to_tempfile.return_value = temp_image_path

        mock_segmentation = MagicMock()
        mock_segmentation.process = AsyncMock(return_value=[])

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_segmentation

        with patch(
            "app.services.agro_processing_service.get_processor_registry",
            return_value=mock_registry,
        ):
            service = AgroProcessingService(
                storage_client=mock_storage,
                db_session=mock_db_session,
            )

            response = await service.process(sample_request)

            assert response.success is True
            assert response.results.get("total_detected", 0) == 0

    @pytest.mark.asyncio
    async def test_process_handles_download_error(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
        sample_request: ProcessingRequest,
    ) -> None:
        """Test error handling when image download fails."""
        mock_storage.download_to_tempfile.side_effect = RuntimeError("Download failed")

        with patch(
            "app.services.agro_processing_service.get_processor_registry"
        ):
            service = AgroProcessingService(
                storage_client=mock_storage,
                db_session=mock_db_session,
            )

            response = await service.process(sample_request)

            assert response.success is False
            assert "Download failed" in response.error

    @pytest.mark.asyncio
    async def test_uses_sahi_for_large_segments(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
        temp_image_path: Path,
    ) -> None:
        """Test that SAHI is used for segments larger than threshold."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        # Create large segment (> 1M pixels)
        large_segment = SegmentResult(
            segment_idx=0,
            class_name="segmento",
            confidence=0.9,
            bbox=(0.0, 0.0, 0.8, 0.8),
            area_px=2_000_000,  # > SAHI_THRESHOLD_PX
        )

        mock_sahi = MagicMock()
        mock_sahi.process = AsyncMock(return_value=[])

        mock_standard = MagicMock()
        mock_standard.process = AsyncMock(return_value=[])

        mock_registry = MagicMock()
        mock_registry.get.side_effect = lambda name: {
            "detection": mock_standard,
            "sahi_detection": mock_sahi,
        }.get(name)

        service.registry = mock_registry

        with patch.object(
            service,
            "_crop_segment",
            new_callable=AsyncMock,
            return_value=temp_image_path,
        ):
            await service._process_segmento_segments(
                image_path=temp_image_path,
                segmento_segments=[(0, large_segment)],
                img_width=2000,
                img_height=2000,
                image_id="test-123",
            )

            # SAHI should be called for large segment
            mock_sahi.process.assert_called_once()
            mock_standard.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_standard_detector_for_small_segments(
        self,
        mock_storage: MagicMock,
        mock_db_session: MagicMock,
        temp_image_path: Path,
    ) -> None:
        """Test that standard detector is used for small segments."""
        service = AgroProcessingService(
            storage_client=mock_storage,
            db_session=mock_db_session,
        )

        # Create small segment (< 1M pixels)
        small_segment = SegmentResult(
            segment_idx=0,
            class_name="segmento",
            confidence=0.9,
            bbox=(0.0, 0.0, 0.3, 0.3),
            area_px=500_000,  # < SAHI_THRESHOLD_PX
        )

        mock_sahi = MagicMock()
        mock_sahi.process = AsyncMock(return_value=[])

        mock_standard = MagicMock()
        mock_standard.process = AsyncMock(return_value=[])

        mock_registry = MagicMock()
        mock_registry.get.side_effect = lambda name: {
            "detection": mock_standard,
            "sahi_detection": mock_sahi,
        }.get(name)

        service.registry = mock_registry

        with patch.object(
            service,
            "_crop_segment",
            new_callable=AsyncMock,
            return_value=temp_image_path,
        ):
            await service._process_segmento_segments(
                image_path=temp_image_path,
                segmento_segments=[(0, small_segment)],
                img_width=1000,
                img_height=1000,
                image_id="test-123",
            )

            # Standard detector should be called for small segment
            mock_standard.process.assert_called_once()
            mock_sahi.process.assert_not_called()
