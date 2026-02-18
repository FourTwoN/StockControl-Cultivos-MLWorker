# Database Access Design - MLWorker

**Date:** 2026-02-18
**Status:** Approved
**Author:** Claude + Franco

## Overview

MLWorker is a **stateless ML processing service** that:
- **READS** configuration and catalog data from PostgreSQL
- **DOES NOT WRITE** to the database
- **SENDS** processing results to Backend via HTTP callback

The Backend (Java/Quarkus) handles all CRUD operations including StockMovement/StockBatch logic.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW                                         │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  1. Backend creates Cloud Task                                             │
│     └─▶ POST /tasks/process → MLWorker                                     │
│                                                                            │
│  2. MLWorker reads config from PostgreSQL (READ ONLY)                      │
│     └─▶ tenant_config, products, packaging_catalog, etc.                   │
│                                                                            │
│  3. MLWorker executes ML pipeline                                          │
│     └─▶ Segmentation → Detection → Classification → Post-processing       │
│                                                                            │
│  4. MLWorker sends results to Backend callback                             │
│     └─▶ POST /api/v1/processing-callback/results                           │
│         {sessionId, imageId, detections, classifications, estimations}     │
│                                                                            │
│  5. Backend persists results + creates StockMovements                      │
│     └─▶ Detection, Classification, Estimation, StockMovement → DB          │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Database Access Strategy

### Read-Only Models

MLWorker needs read access to catalog/config tables for ML pipeline execution:

| Table | Purpose |
|-------|---------|
| `tenant_config` | Pipeline definitions, settings |
| `products` | Product classification labels |
| `product_families` | Product hierarchy |
| `product_categories` | Product hierarchy |
| `product_sizes` | Size classification (S, M, L, XL) |
| `product_states` | State classification |
| `packaging_catalog` | Packaging detection |
| `packaging_types` | Packaging hierarchy |
| `packaging_materials` | Packaging hierarchy |
| `packaging_colors` | Packaging hierarchy |
| `density_parameters` | Estimation calibration |

### No Write Operations

MLWorker does NOT write to:
- `images` (s3_images)
- `photo_processing_sessions`
- `detections`
- `estimations`
- `classifications`
- `stock_movements`
- `stock_batches`
- `stock_batch_movements`

All writes are handled by Backend after receiving callback.

## Directory Structure

```
app/
├── models/                    # SQLAlchemy models (READ ONLY)
│   ├── __init__.py
│   ├── base.py               # DeclarativeBase + mixins
│   ├── product.py
│   ├── product_family.py
│   ├── product_category.py
│   ├── product_size.py
│   ├── product_state.py
│   ├── packaging_catalog.py
│   ├── packaging_type.py
│   ├── packaging_material.py
│   ├── packaging_color.py
│   └── density_parameter.py
│
├── schemas/
│   ├── __init__.py
│   ├── task.py               # Existing - input request
│   ├── pipeline_definition.py # Existing
│   └── callback.py           # NEW - backend callback payload
│
├── services/
│   ├── __init__.py
│   └── backend_client.py     # NEW - HTTP client for callback
│
└── ... (existing structure)
```

## Callback Schema

Matches Java DTO `ProcessingResultRequest`:

```python
class ProcessingResultRequest(BaseModel):
    sessionId: UUID
    imageId: UUID
    detections: list[DetectionResultItem] = []
    classifications: list[ClassificationResultItem] = []
    estimations: list[EstimationResultItem] = []
    metadata: ProcessingMetadata | None = None

class DetectionResultItem(BaseModel):
    label: str
    confidence: float
    boundingBox: BoundingBox | None = None

class ClassificationResultItem(BaseModel):
    label: str
    confidence: float
    detectionId: UUID | None = None

class EstimationResultItem(BaseModel):
    estimationType: str
    value: float
    unit: str | None = None
    confidence: float | None = None
```

## Backend Client

```python
class BackendClient:
    async def send_results(tenant_id: str, results: ProcessingResultRequest) -> dict
    async def report_error(tenant_id: str, session_id: UUID, image_id: UUID, error_message: str) -> dict
```

## Model Strategy

Since MLWorker shares the same database as DemeterAI-back:

1. **Table names**: Must match exactly (e.g., `products`, `packaging_catalog`)
2. **Column names**: Must match exactly
3. **Enums**: Reuse same values for compatibility
4. **Relationships**: Only declare relationships MLWorker needs (no back_populates to unused tables)
5. **Validators**: Minimal - trust database constraints
6. **FKs to unmodeled tables**: Declare as plain Integer without FK constraint

## Implementation Plan

### Phase 1: Models (Read-Only)
1. Create `app/models/base.py` with DeclarativeBase
2. Create product models (product, product_family, product_category)
3. Create size/state models (product_size, product_state)
4. Create packaging models (packaging_catalog, packaging_type, packaging_material, packaging_color)
5. Create density_parameter model
6. Update `app/models/__init__.py` exports

### Phase 2: Callback Infrastructure
1. Create `app/schemas/callback.py` with ProcessingResultRequest
2. Create `app/services/backend_client.py` with BackendClient
3. Add BACKEND_URL to config

### Phase 3: Pipeline Integration
1. Modify pipeline executor to collect results
2. Transform ProcessingContext → ProcessingResultRequest
3. Call BackendClient.send_results() after pipeline completes
4. Handle errors with BackendClient.report_error()

### Phase 4: Testing
1. Unit tests for models (read operations)
2. Unit tests for callback schema serialization
3. Integration tests for BackendClient (mock server)
4. E2E test with real backend callback

## Configuration

```bash
# .env additions
BACKEND_URL=https://api.demeter.com  # or Cloud Run URL
BACKEND_CALLBACK_TIMEOUT=30
```

## Notes

- MLWorker remains stateless regarding writes
- All business logic (StockMovement, StockBatch) stays in Backend
- Callback is synchronous - MLWorker waits for Backend response
- Error handling: report_error() callback on pipeline failure
