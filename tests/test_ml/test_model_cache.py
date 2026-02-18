"""Tests for per-tenant ModelCache."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestModelCachePerTenant:
    """Test ModelCache with tenant_id support."""

    def test_get_model_requires_tenant_id(self):
        """get_model should require tenant_id parameter."""
        from app.ml.model_cache import ModelCache

        with pytest.raises(TypeError):
            # Should fail without tenant_id
            ModelCache.get_model(model_type="detect", worker_id=0)

    def test_cache_key_includes_tenant_id(self):
        """Cache key should include tenant_id for isolation."""
        from app.ml.model_cache import ModelCache

        # Clear cache first
        ModelCache.clear_cache()

        # Mock the model loading
        with patch.object(ModelCache, "_get_tenant_model_path") as mock_path:
            with patch("app.ml.model_cache.YOLO") as mock_yolo:
                mock_path.return_value = Path("/tmp/model.onnx")
                mock_model = MagicMock()
                mock_yolo.return_value = mock_model

                with patch("app.ml.model_cache.torch") as mock_torch:
                    mock_torch.cuda.is_available.return_value = False

                    # Load model for tenant-001
                    ModelCache.get_model(
                        tenant_id="tenant-001",
                        model_type="detect",
                        worker_id=0,
                    )

        # Verify cache key includes tenant_id
        assert "tenant-001_detect_worker_0" in ModelCache._instances

    def test_different_tenants_get_different_models(self):
        """Different tenants should have separate cached models."""
        from app.ml.model_cache import ModelCache

        ModelCache.clear_cache()

        with patch.object(ModelCache, "_get_tenant_model_path") as mock_path:
            with patch("app.ml.model_cache.YOLO") as mock_yolo:
                mock_path.return_value = Path("/tmp/model.onnx")
                mock_yolo.return_value = MagicMock()

                with patch("app.ml.model_cache.torch") as mock_torch:
                    mock_torch.cuda.is_available.return_value = False

                    ModelCache.get_model("tenant-001", "detect", 0)
                    ModelCache.get_model("tenant-002", "detect", 0)

        assert "tenant-001_detect_worker_0" in ModelCache._instances
        assert "tenant-002_detect_worker_0" in ModelCache._instances
        assert len(ModelCache._instances) == 2
