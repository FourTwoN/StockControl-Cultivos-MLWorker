"""Tests for upload_image step."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.steps.post.upload_image import UploadImageStep
from app.core.processing_context import ProcessingContext


class TestUploadImageStep:
    """Tests for UploadImageStep."""

    @pytest.fixture
    def step(self) -> UploadImageStep:
        return UploadImageStep()

    @pytest.fixture
    def mock_context(self, tmp_path: Path) -> ProcessingContext:
        """Create a mock context with a real temporary image."""
        # Create a minimal test image
        from PIL import Image

        test_image = tmp_path / "test_image.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(test_image, "JPEG")

        return ProcessingContext(
            tenant_id="test-tenant",
            image_id="test-image-123",
            session_id="test-session",
            image_path=test_image,
            config={},
            step_config={
                "source": "original",
                "dest_prefix": "originals",
                "thumbnail_sizes": [64, 128],
                "quality": 85,
            },
        )

    def test_step_name(self, step: UploadImageStep):
        """Test step has correct name."""
        assert step.name == "upload_image"

    @pytest.mark.asyncio
    async def test_upload_original_image(
        self, step: UploadImageStep, mock_context: ProcessingContext
    ):
        """Test uploading original image with thumbnails."""
        mock_storage = MagicMock()
        mock_storage.upload_bytes = AsyncMock(
            side_effect=lambda data, blob_path, **kwargs: f"gs://bucket/{blob_path}"
        )

        with patch("app.steps.post.upload_image.get_storage_client", return_value=mock_storage):
            result = await step.execute(mock_context)

        # Should have called upload 3 times (main + 2 thumbnails)
        assert mock_storage.upload_bytes.call_count == 3

        # Check result structure
        assert "original_urls" in result.results
        assert "main" in result.results["original_urls"]
        assert "thumbnails" in result.results["original_urls"]
        assert 64 in result.results["original_urls"]["thumbnails"]
        assert 128 in result.results["original_urls"]["thumbnails"]

    @pytest.mark.asyncio
    async def test_upload_processed_image(self, step: UploadImageStep, tmp_path: Path):
        """Test uploading processed image."""
        # Create processed image
        from PIL import Image

        processed_image = tmp_path / "processed.jpg"
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(processed_image, "JPEG")

        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="test-image-456",
            session_id="test-session",
            image_path=tmp_path / "original.jpg",  # Won't be used
            config={},
            results={"visualization_path": str(processed_image)},
            step_config={
                "source": "processed",
                "dest_prefix": "processed",
                "thumbnail_sizes": [128],
            },
        )

        mock_storage = MagicMock()
        mock_storage.upload_bytes = AsyncMock(
            side_effect=lambda data, blob_path, **kwargs: f"gs://bucket/{blob_path}"
        )

        with patch("app.steps.post.upload_image.get_storage_client", return_value=mock_storage):
            result = await step.execute(ctx)

        # Should have called upload 2 times (main + 1 thumbnail)
        assert mock_storage.upload_bytes.call_count == 2

        # Check result structure
        assert "processed_urls" in result.results
        assert "main" in result.results["processed_urls"]

    @pytest.mark.asyncio
    async def test_skip_if_missing(self, step: UploadImageStep, tmp_path: Path):
        """Test skip_if_missing flag when source doesn't exist."""
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="test-image",
            session_id="test-session",
            image_path=tmp_path / "nonexistent.jpg",
            config={},
            step_config={
                "source": "original",
                "skip_if_missing": True,
            },
        )

        # Should not raise, should return context unchanged
        result = await step.execute(ctx)
        assert result == ctx

    @pytest.mark.asyncio
    async def test_raises_when_source_missing(self, step: UploadImageStep, tmp_path: Path):
        """Test raises FileNotFoundError when source missing and skip_if_missing=False."""
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="test-image",
            session_id="test-session",
            image_path=tmp_path / "nonexistent.jpg",
            config={},
            step_config={
                "source": "original",
                "skip_if_missing": False,
            },
        )

        with pytest.raises(FileNotFoundError):
            await step.execute(ctx)

    @pytest.mark.asyncio
    async def test_blob_path_structure(
        self, step: UploadImageStep, mock_context: ProcessingContext
    ):
        """Test that blob paths are correctly structured."""
        uploaded_paths = []

        mock_storage = MagicMock()
        mock_storage.upload_bytes = AsyncMock(
            side_effect=lambda data, blob_path, **kwargs: (
                uploaded_paths.append(blob_path),
                f"gs://bucket/{blob_path}",
            )[1]
        )

        with patch("app.steps.post.upload_image.get_storage_client", return_value=mock_storage):
            await step.execute(mock_context)

        # Check path structure
        assert "test-tenant/originals/test-image-123.jpg" in uploaded_paths
        assert "test-tenant/originals_thumbnails/test-image-123_64.jpg" in uploaded_paths
        assert "test-tenant/originals_thumbnails/test-image-123_128.jpg" in uploaded_paths
