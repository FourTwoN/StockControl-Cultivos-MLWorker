# Per-Tenant ML Models Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable each tenant to use their own ML models stored in GCS at `gs://{bucket}/models/{tenant_id}/`.

**Architecture:** ModelCache receives `tenant_id` and downloads models from tenant-specific GCS paths. Processors receive `tenant_id` from Steps via ProcessorRegistry. Cache key includes tenant_id for per-tenant model isolation.

**Tech Stack:** Python, YOLO/Ultralytics, GCS, existing ModelCache infrastructure

---

### Task 1: Update ModelCache to Accept tenant_id

**Files:**
- Modify: `app/ml/model_cache.py`
- Test: `tests/test_ml/test_model_cache.py` (create)

**Step 1: Write the failing test**

Create `tests/test_ml/__init__.py` and `tests/test_ml/test_model_cache.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_ml/test_model_cache.py -v`
Expected: FAIL (TypeError or different cache keys)

**Step 3: Implement ModelCache changes**

Modify `app/ml/model_cache.py`:

```python
"""ModelCache singleton for YOLO model management.

Per-tenant model loading from GCS.
"""

import tempfile
import threading
from pathlib import Path
from typing import Any, Literal

from app.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)

# Lazy imports for optional dependencies
try:
    import torch
    from ultralytics import YOLO

    TORCH_AVAILABLE = True
except ImportError:
    YOLO = None  # type: ignore
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    logger.warning("torch/ultralytics not available - ML features disabled")

try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    storage = None  # type: ignore
    GCS_AVAILABLE = False

ModelType = Literal["segment", "detect", "classify"]


class ModelCache:
    """Singleton cache for YOLO models (per tenant, per model type, per worker)."""

    _instances: dict[str, Any] = {}
    _lock: threading.Lock = threading.Lock()
    _local_model_dir: Path | None = None

    @classmethod
    def get_model(
        cls,
        tenant_id: str,
        model_type: ModelType,
        worker_id: int = 0,
    ) -> Any:
        """Get or create cached model instance for tenant.

        Args:
            tenant_id: Tenant identifier for model isolation
            model_type: "segment", "detect", or "classify"
            worker_id: GPU worker ID (0, 1, 2, etc.)

        Returns:
            Cached YOLO model instance

        Raises:
            ValueError: If model_type invalid or worker_id negative
            RuntimeError: If model loading fails or torch not available
            FileNotFoundError: If model not found for tenant
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/ultralytics not installed - cannot load models")

        # Validation
        if not tenant_id:
            raise ValueError("tenant_id is required")

        if model_type not in ["segment", "detect", "classify"]:
            raise ValueError(f"Invalid model_type: {model_type}")

        if worker_id < 0:
            raise ValueError("worker_id must be non-negative")

        # Cache key includes tenant_id
        cache_key = f"{tenant_id}_{model_type}_worker_{worker_id}"

        # Thread-safe singleton check
        with cls._lock:
            if cache_key not in cls._instances:
                logger.info(
                    "Loading model",
                    tenant_id=tenant_id,
                    model_type=model_type,
                    worker_id=worker_id,
                )

                # Get model path (download from GCS if needed)
                model_path = cls._get_tenant_model_path(tenant_id, model_type)

                # Load model
                is_onnx = str(model_path).endswith(".onnx")
                if is_onnx:
                    model = YOLO(str(model_path), task=model_type)
                    logger.info("Loaded ONNX model with explicit task", task=model_type)
                else:
                    model = YOLO(str(model_path))

                # Determine device for inference
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_id = worker_id % gpu_count
                    device = f"cuda:{gpu_id}"
                    logger.info("Assigning model to GPU", device=device)
                else:
                    device = "cpu"
                    logger.info("GPU not available, using CPU", worker_id=worker_id)

                # For PyTorch models (.pt), move to device and fuse
                if not is_onnx:
                    model = model.to(device)
                    model.fuse()

                # Store device info
                model._mlworker_device = device
                model._mlworker_is_onnx = is_onnx
                model._mlworker_tenant_id = tenant_id

                # Cache
                cls._instances[cache_key] = model
                logger.info(
                    "Model loaded and cached",
                    tenant_id=tenant_id,
                    model_type=model_type,
                    worker_id=worker_id,
                    device=device,
                    cache_key=cache_key,
                )

            return cls._instances[cache_key]

    @classmethod
    def _get_tenant_model_path(cls, tenant_id: str, model_type: ModelType) -> Path:
        """Get local path to tenant model, downloading from GCS if needed.

        Args:
            tenant_id: Tenant identifier
            model_type: Model type to load

        Returns:
            Local path to model file

        Raises:
            FileNotFoundError: If model not found for tenant
        """
        model_filename = f"{model_type}.onnx"

        # GCS path: gs://{bucket}/models/{tenant_id}/{model_type}.onnx
        gcs_url = f"gs://{settings.gcs_bucket}/models/{tenant_id}/{model_filename}"

        # Local filename includes tenant to avoid collisions
        local_filename = f"{tenant_id}_{model_filename}"

        return cls._download_from_gcs(gcs_url, local_filename)

    @classmethod
    def _download_from_gcs(cls, gcs_url: str, local_filename: str) -> Path:
        """Download model from GCS to local temp directory.

        Args:
            gcs_url: Full GCS URL (gs://bucket/path/file)
            local_filename: Local filename to save as

        Returns:
            Local path to downloaded model

        Raises:
            FileNotFoundError: If model not found in GCS
            RuntimeError: If GCS client not available
        """
        if not GCS_AVAILABLE:
            raise RuntimeError("google-cloud-storage not installed")

        # Create local model directory if needed
        if cls._local_model_dir is None:
            cls._local_model_dir = Path(tempfile.mkdtemp(prefix="mlworker_models_"))
            logger.info("Created local model directory", path=str(cls._local_model_dir))

        local_path = cls._local_model_dir / local_filename

        # Check if already downloaded
        if local_path.exists():
            logger.debug("Model already downloaded", path=str(local_path))
            return local_path

        # Parse GCS URL: gs://bucket/path/file
        gcs_path = gcs_url.replace("gs://", "")
        parts = gcs_path.split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        logger.info(
            "Downloading model from GCS",
            gcs_url=gcs_url,
            bucket=bucket_name,
            blob=blob_path,
            local_path=str(local_path),
        )

        # Download
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            if not blob.exists():
                raise FileNotFoundError(
                    f"Model not found for tenant. Expected at: {gcs_url}"
                )

            blob.download_to_filename(str(local_path))

            logger.info(
                "Model downloaded",
                path=str(local_path),
                size=local_path.stat().st_size,
            )

            return local_path

        except Exception as e:
            if "Not Found" in str(e) or isinstance(e, FileNotFoundError):
                raise FileNotFoundError(
                    f"Model not found for tenant. Expected at: {gcs_url}"
                ) from e
            raise

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached models and free GPU memory."""
        with cls._lock:
            model_count = len(cls._instances)
            cls._instances.clear()

            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cache cleared")

            logger.info("Model cache cleared", models_removed=model_count)

    @classmethod
    def get_cache_info(cls) -> dict[str, Any]:
        """Get information about cached models."""
        with cls._lock:
            info = {
                "cached_models": list(cls._instances.keys()),
                "model_count": len(cls._instances),
                "local_model_dir": str(cls._local_model_dir) if cls._local_model_dir else None,
            }

            if TORCH_AVAILABLE and torch.cuda.is_available():
                info["gpu_available"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                info["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            else:
                info["gpu_available"] = False

            return info
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_ml/test_model_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/ml/model_cache.py tests/test_ml/
git commit -m "feat: add per-tenant model loading to ModelCache"
```

---

### Task 2: Update BaseProcessor to Accept tenant_id

**Files:**
- Modify: `app/processors/base_processor.py`

**Step 1: Update the constructor**

```python
class BaseProcessor(ABC, Generic[T]):
    """Abstract base class for ML processors."""

    def __init__(
        self,
        tenant_id: str,                    # NEW: required
        worker_id: int = 0,
        confidence_threshold: float = 0.25,
    ) -> None:
        """Initialize base processor.

        Args:
            tenant_id: Tenant identifier for model loading
            worker_id: GPU worker ID (0, 1, 2, ...) for model assignment
            confidence_threshold: Detection/classification confidence threshold
        """
        if not tenant_id:
            raise ValueError("tenant_id is required")

        self.tenant_id = tenant_id
        self.worker_id = worker_id
        self.confidence_threshold = confidence_threshold
        self._model: Any = None

        logger.info(
            "Processor initialized",
            processor=self.__class__.__name__,
            tenant_id=tenant_id,
            worker_id=self.worker_id,
            confidence_threshold=self.confidence_threshold,
        )
```

**Step 2: Commit**

```bash
git add app/processors/base_processor.py
git commit -m "feat: add tenant_id to BaseProcessor constructor"
```

---

### Task 3: Update DetectorProcessor

**Files:**
- Modify: `app/processors/detector_processor.py`

**Step 1: Update to use tenant_id in ModelCache call**

Find the line where `ModelCache.get_model()` is called and update:

```python
# BEFORE
model = ModelCache.get_model("detect", worker_id=self.worker_id)

# AFTER
model = ModelCache.get_model(
    tenant_id=self.tenant_id,
    model_type="detect",
    worker_id=self.worker_id,
)
```

**Step 2: Commit**

```bash
git add app/processors/detector_processor.py
git commit -m "feat: pass tenant_id to ModelCache in DetectorProcessor"
```

---

### Task 4: Update SegmentationProcessor

**Files:**
- Modify: `app/processors/segmentation_processor.py`

**Step 1: Update to use tenant_id in ModelCache call**

```python
# BEFORE
model = ModelCache.get_model("segment", worker_id=self.worker_id)

# AFTER
model = ModelCache.get_model(
    tenant_id=self.tenant_id,
    model_type="segment",
    worker_id=self.worker_id,
)
```

**Step 2: Commit**

```bash
git add app/processors/segmentation_processor.py
git commit -m "feat: pass tenant_id to ModelCache in SegmentationProcessor"
```

---

### Task 5: Update SAHIDetectorProcessor

**Files:**
- Modify: `app/processors/sahi_detector_processor.py`

**Step 1: Update to use tenant_id in ModelCache call**

```python
# BEFORE
model = ModelCache.get_model("detect", worker_id=self.worker_id)

# AFTER
model = ModelCache.get_model(
    tenant_id=self.tenant_id,
    model_type="detect",
    worker_id=self.worker_id,
)
```

**Step 2: Commit**

```bash
git add app/processors/sahi_detector_processor.py
git commit -m "feat: pass tenant_id to ModelCache in SAHIDetectorProcessor"
```

---

### Task 6: Update DetectionStep

**Files:**
- Modify: `app/steps/ml/detection_step.py`

**Step 1: Pass tenant_id to processor registry**

```python
# BEFORE
processor = get_processor_registry().get("detection")

# AFTER
processor = get_processor_registry().get("detection", tenant_id=ctx.tenant_id)
```

Update ALL occurrences in the file (both in `_execute_full_image` and `_execute_segment_aware`).

**Step 2: Commit**

```bash
git add app/steps/ml/detection_step.py
git commit -m "feat: pass tenant_id to processor in DetectionStep"
```

---

### Task 7: Update SegmentationStep

**Files:**
- Modify: `app/steps/ml/segmentation_step.py`

**Step 1: Pass tenant_id to processor registry**

```python
# BEFORE
processor = get_processor_registry().get("segmentation")

# AFTER
processor = get_processor_registry().get("segmentation", tenant_id=ctx.tenant_id)
```

**Step 2: Commit**

```bash
git add app/steps/ml/segmentation_step.py
git commit -m "feat: pass tenant_id to processor in SegmentationStep"
```

---

### Task 8: Update SAHIDetectionStep

**Files:**
- Modify: `app/steps/ml/sahi_detection_step.py`

**Step 1: Pass tenant_id to processor registry**

```python
# BEFORE
processor = get_processor_registry().get("sahi_detection")

# AFTER
processor = get_processor_registry().get("sahi_detection", tenant_id=ctx.tenant_id)
```

**Step 2: Commit**

```bash
git add app/steps/ml/sahi_detection_step.py
git commit -m "feat: pass tenant_id to processor in SAHIDetectionStep"
```

---

### Task 9: Remove Global Model Settings from Config

**Files:**
- Modify: `app/config.py`

**Step 1: Remove these settings**

```python
# REMOVE these lines:
model_path: str = Field(...)
detection_model: str = Field(...)
segmentation_model: str = Field(...)
```

Keep `gcs_bucket` as it's used for model paths.

**Step 2: Commit**

```bash
git add app/config.py
git commit -m "chore: remove global model settings from config"
```

---

### Task 10: Update main.py - Remove Model Preloading

**Files:**
- Modify: `app/main.py`

**Step 1: Remove or update model preloading**

The current preloading code calls `ModelCache.get_model("detect")` without tenant_id. Options:
1. Remove preloading entirely (models load on first request per tenant)
2. Keep commented out with TODO

Recommend removing:

```python
# REMOVE these lines from lifespan():
if settings.environment != "dev":
    logger.info("Preloading ML models...")
    try:
        ModelCache.get_model("detect", worker_id=0)
        logger.info("Detection model preloaded")
    except Exception as e:
        logger.warning("Model preloading failed", error=str(e))
```

**Step 2: Commit**

```bash
git add app/main.py
git commit -m "refactor: remove global model preloading (now per-tenant)"
```

---

### Task 11: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Fix any broken tests**

Tests that mock ModelCache.get_model() need to be updated to include tenant_id parameter.

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "test: fix tests for per-tenant model loading"
```

---

### Task 12: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update model loading section**

Add section about per-tenant models:

```markdown
## Per-Tenant ML Models

Each tenant has their own ML models stored in GCS:

```
gs://{gcs_bucket}/models/
├── tenant-001/
│   ├── detect.onnx
│   └── segment.onnx
└── tenant-002/
    └── detect.onnx
```

Models are loaded on first request and cached per tenant.
If a model doesn't exist for a tenant, the request fails with HTTP 400.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add per-tenant model documentation to CLAUDE.md"
```

---

## Summary

| Task | Files | Change |
|------|-------|--------|
| 1 | `app/ml/model_cache.py` | Add tenant_id param, change GCS paths |
| 2 | `app/processors/base_processor.py` | Add tenant_id to constructor |
| 3 | `app/processors/detector_processor.py` | Pass tenant_id to ModelCache |
| 4 | `app/processors/segmentation_processor.py` | Pass tenant_id to ModelCache |
| 5 | `app/processors/sahi_detector_processor.py` | Pass tenant_id to ModelCache |
| 6 | `app/steps/ml/detection_step.py` | Pass ctx.tenant_id to registry |
| 7 | `app/steps/ml/segmentation_step.py` | Pass ctx.tenant_id to registry |
| 8 | `app/steps/ml/sahi_detection_step.py` | Pass ctx.tenant_id to registry |
| 9 | `app/config.py` | Remove global model settings |
| 10 | `app/main.py` | Remove model preloading |
| 11 | Tests | Fix mocks for tenant_id |
| 12 | `CLAUDE.md` | Update documentation |
