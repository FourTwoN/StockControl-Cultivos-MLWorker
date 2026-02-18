# Per-Tenant ML Models Design

**Date:** 2026-02-18
**Status:** Approved
**Author:** Claude + Franco

## Overview

Each tenant has its own ML models stored in GCS. Models are loaded per-tenant and cached in memory. No fallback to default models - if a tenant's model doesn't exist, the request fails with HTTP 400.

## GCS Structure

```
gs://{gcs_bucket}/models/
├── tenant-001/
│   ├── detect.onnx
│   └── segment.onnx
├── tenant-002/
│   ├── detect.onnx
│   └── segment.onnx
└── tenant-003/
    └── detect.onnx      # Only has detector
```

**Conventions:**
- Path: `gs://{bucket}/models/{tenant_id}/{model_type}.onnx`
- Fixed names: `detect.onnx`, `segment.onnx`, `classify.onnx`
- Model type determined by processor class (DetectorProcessor → "detect")

## ModelCache Changes

**Before:**
```python
ModelCache.get_model(model_type="detect", worker_id=0)
# Cache key: "detect_worker_0"
# Path: gs://{bucket}/models/detect.onnx (global)
```

**After:**
```python
ModelCache.get_model(tenant_id="tenant-001", model_type="detect", worker_id=0)
# Cache key: "tenant-001_detect_worker_0"
# Path: gs://{bucket}/models/tenant-001/detect.onnx
```

## Error Handling

- Model not found in GCS → `FileNotFoundError` → HTTP 400 Bad Request
- No silent fallbacks to default models
- Clear error message: "Model 'detect' not found for tenant 'tenant-001'"

## Data Flow

```
Request(tenant_id="tenant-001")
    ↓
ProcessingContext(tenant_id="tenant-001")
    ↓
DetectionStep.execute(ctx)
    ↓
DetectorProcessor(tenant_id=ctx.tenant_id)
    ↓
ModelCache.get_model(tenant_id="tenant-001", model_type="detect")
    ↓
Download from: gs://{bucket}/models/tenant-001/detect.onnx
    ↓
Cache with key: "tenant-001_detect_worker_0"
```

## Memory Management

- Cache key: `{tenant_id}_{model_type}_worker_{worker_id}`
- Models stay in memory (no LRU eviction)
- Suitable for <10 tenants with ~100MB models each
- `ModelCache.clear_cache()` removes all models

## Files to Modify

| File | Change |
|------|--------|
| `app/ml/model_cache.py` | Add `tenant_id` param, change GCS path logic |
| `app/processors/base_processor.py` | Add `tenant_id` to constructor |
| `app/processors/detector_processor.py` | Pass `tenant_id` to ModelCache |
| `app/processors/segmentation_processor.py` | Pass `tenant_id` to ModelCache |
| `app/processors/sahi_detector_processor.py` | Pass `tenant_id` to ModelCache |
| `app/steps/ml/detection_step.py` | Pass `ctx.tenant_id` to processor |
| `app/steps/ml/segmentation_step.py` | Pass `ctx.tenant_id` to processor |
| `app/steps/ml/sahi_detection_step.py` | Pass `ctx.tenant_id` to processor |
| `app/config.py` | Remove global model settings |

## Config Changes

**Remove from config.py:**
```python
# No longer needed (models are per-tenant in GCS)
model_path: str
detection_model: str
segmentation_model: str
```

**Keep:**
```python
gcs_bucket: str  # Used for model paths: gs://{gcs_bucket}/models/{tenant_id}/
```

## Benefits

1. **Tenant isolation**: Each tenant uses their own trained models
2. **Flexibility**: Different tenants can have different model types
3. **Stateless**: Model paths derived from tenant_id, no DB lookup
4. **Scalable**: Add new tenants by uploading models to GCS
