# Stateless Architecture Design - MLWorker

**Date:** 2026-02-18
**Status:** Approved
**Author:** Claude + Franco

## Overview

Remove all database access from MLWorker to make it completely stateless. Pipeline definitions come directly in the request payload instead of being fetched from a database cache.

## Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          BEFORE (with DB)                                  │
├────────────────────────────────────────────────────────────────────────────┤
│  Backend → Cloud Task(tenant_id, image_url)                                │
│                     ↓                                                      │
│  MLWorker → TenantConfigCache.get(tenant_id) → PostgreSQL                  │
│                     ↓                                                      │
│  Execute pipeline from DB config                                           │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│                          AFTER (stateless)                                 │
├────────────────────────────────────────────────────────────────────────────┤
│  Backend → Cloud Task(tenant_id, image_url, pipeline_definition, settings) │
│                     ↓                                                      │
│  MLWorker → Parse & validate pipeline_definition from request              │
│                     ↓                                                      │
│  Execute pipeline directly (no DB lookup)                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

## Request Schema Change

**Before:**
```python
class ProcessingRequest(BaseModel):
    tenant_id: str
    session_id: UUID
    image_id: UUID
    image_url: str
    pipeline: str  # Just a name like "DETECTION"
    callback_url: str | None = None
```

**After:**
```python
class ProcessingRequest(BaseModel):
    tenant_id: str
    session_id: UUID
    image_id: UUID
    image_url: str
    pipeline_definition: PipelineDefinition  # Full DSL
    settings: dict[str, Any] = Field(default_factory=dict)
    callback_url: str | None = None
```

## tenant_id Usage (unchanged)

The `tenant_id` field remains for:

1. **GCS Paths**: `gs://bucket/{tenant_id}/originals/`, `gs://bucket/{tenant_id}/processed/`
2. **Logging/Tracing**: Identify which tenant each request belongs to
3. **Backend Callback**: `X-Tenant-ID` header for RLS

**Model Loading** does NOT use tenant_id - models are global, loaded from `MODEL_PATH` env var.

## Files to Delete

| File | Reason |
|------|--------|
| `app/core/tenant_config.py` | DB cache no longer needed |
| `app/infra/database.py` | No DB access |
| `app/models/*.py` | SQLAlchemy models not needed |

## Files to Modify

| File | Change |
|------|--------|
| `app/schemas/task.py` | Add `pipeline_definition`, `settings`; remove `pipeline` |
| `app/api/routes/tasks.py` | Use request directly, remove cache lookup |
| `app/main.py` | Remove DB/cache initialization |
| `app/api/deps.py` | Remove DB session dependency |
| `pyproject.toml` | Remove SQLAlchemy, asyncpg dependencies |

## Dependencies to Remove

```toml
# Remove from pyproject.toml
sqlalchemy = "..."
asyncpg = "..."
greenlet = "..."
```

## Environment Variables to Remove

```bash
# No longer needed
DATABASE_URL=postgresql+asyncpg://...
```

## Validation Strategy

Pipeline validation happens at request time:
1. Pydantic validates JSON structure of `pipeline_definition`
2. `PipelineParser` validates all steps exist in `StepRegistry`
3. Invalid pipeline → HTTP 400 Bad Request (immediate feedback)

## Benefits

1. **Truly stateless**: No external dependencies except GCS and ML models
2. **Simpler deployment**: No DB connection management
3. **Faster cold starts**: No DB connection pool initialization
4. **Backend control**: Backend decides which pipeline to send
5. **Easier testing**: No need to mock DB, just mock request

## Notes

- Backend (Java/Quarkus) still owns the `tenant_config` table
- Backend includes `pipeline_definition` when creating Cloud Task
- MLWorker remains a pure ML processing service
