# Stateless Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all database access from MLWorker, making it fully stateless with pipeline definitions in requests.

**Architecture:** Backend sends `pipeline_definition` and `settings` directly in the Cloud Task request. MLWorker parses and executes immediately without any DB lookup. Files related to database (tenant_config.py, database.py, models/) are deleted.

**Tech Stack:** FastAPI, Pydantic v2, no SQLAlchemy/asyncpg

---

### Task 1: Update ProcessingRequest Schema

**Files:**
- Modify: `app/schemas/task.py:10-69`
- Test: `tests/test_schemas/test_task.py` (create)

**Step 1: Write the failing test**

Create `tests/test_schemas/test_task.py`:

```python
"""Tests for ProcessingRequest schema with pipeline_definition."""

import pytest
from uuid import uuid4

from app.schemas.task import ProcessingRequest
from app.schemas.pipeline_definition import PipelineDefinition


class TestProcessingRequestWithPipeline:
    """Test ProcessingRequest accepts pipeline_definition."""

    def test_request_with_pipeline_definition(self):
        """Request should accept full pipeline_definition."""
        pipeline_def = {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "segmentation"},
                {"type": "step", "name": "detection"},
            ],
        }

        request = ProcessingRequest(
            tenant_id="tenant-001",
            session_id=uuid4(),
            image_id=uuid4(),
            image_url="gs://bucket/image.jpg",
            pipeline_definition=pipeline_def,
            settings={"key": "value"},
        )

        assert request.pipeline_definition.type == "chain"
        assert len(request.pipeline_definition.steps) == 2
        assert request.settings == {"key": "value"}

    def test_request_with_empty_settings(self):
        """Settings should default to empty dict."""
        request = ProcessingRequest(
            tenant_id="tenant-001",
            session_id=uuid4(),
            image_id=uuid4(),
            image_url="gs://bucket/image.jpg",
            pipeline_definition={
                "type": "chain",
                "steps": [{"type": "step", "name": "detection"}],
            },
        )

        assert request.settings == {}

    def test_request_rejects_missing_pipeline_definition(self):
        """Request should require pipeline_definition."""
        with pytest.raises(Exception):  # ValidationError
            ProcessingRequest(
                tenant_id="tenant-001",
                session_id=uuid4(),
                image_id=uuid4(),
                image_url="gs://bucket/image.jpg",
                # missing pipeline_definition
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_task.py -v`
Expected: FAIL with "field required" or "unexpected keyword argument"

**Step 3: Implement the schema changes**

Modify `app/schemas/task.py`:

```python
"""Task schemas for the unified processing endpoint."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.schemas.pipeline_definition import PipelineDefinition


class ProcessingRequest(BaseModel):
    """Unified request payload for all ML processing tasks.

    Sent by Backend via Cloud Tasks to the single /tasks/process endpoint.
    Pipeline definition is included directly in the request (no DB lookup).
    """

    tenant_id: str = Field(
        min_length=1,
        max_length=100,
        description="Tenant ID for multi-tenant isolation",
    )
    session_id: UUID = Field(description="Processing session UUID")
    image_id: UUID = Field(description="Image UUID")
    image_url: str = Field(
        min_length=1,
        description="GCS URL to image (gs://bucket/path)",
    )
    pipeline_definition: PipelineDefinition = Field(
        description="Full pipeline DSL definition",
    )
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Tenant settings for pipeline execution",
    )
    callback_url: str | None = Field(
        default=None,
        description="Optional callback URL for results notification",
    )

    model_config = {"extra": "forbid"}

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v: str) -> str:
        """Validate that image_url is a valid GCS URL."""
        if not v.startswith("gs://"):
            raise ValueError("image_url must be a GCS URL (gs://bucket/path)")
        return v

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        """Validate tenant_id format."""
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid tenant_id format")
        return v


class ProcessingResponse(BaseModel):
    """Unified response payload for ML processing tasks."""

    success: bool = Field(description="Whether processing succeeded")
    tenant_id: str = Field(description="Tenant ID")
    session_id: UUID = Field(description="Processing session UUID")
    image_id: UUID = Field(description="Image UUID")
    pipeline_type: str = Field(description="Pipeline type that was executed (chain/group/chord)")

    # Results from each step (keyed by step name)
    results: dict[str, Any] = Field(
        default_factory=dict,
        description="Results from each pipeline step",
    )

    # Metadata
    duration_ms: int = Field(description="Total processing duration in milliseconds")
    steps_completed: int = Field(
        default=0,
        description="Number of steps successfully completed",
    )
    error: str | None = Field(default=None, description="Error message if failed")
    processed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Processing timestamp",
    )

    model_config = {"extra": "forbid"}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_task.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/schemas/task.py tests/test_schemas/test_task.py
git commit -m "feat: add pipeline_definition to ProcessingRequest schema"
```

---

### Task 2: Update /tasks/process Endpoint

**Files:**
- Modify: `app/api/routes/tasks.py`
- Test: `tests/test_api/test_tasks.py`

**Step 1: Write the failing test**

Create/update `tests/test_api/test_tasks.py`:

```python
"""Tests for /tasks/process endpoint with inline pipeline_definition."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from pathlib import Path

from fastapi.testclient import TestClient


@pytest.fixture
def mock_storage():
    """Mock storage client."""
    storage = AsyncMock()
    storage.download_to_tempfile = AsyncMock(return_value=Path("/tmp/test.jpg"))
    return storage


@pytest.fixture
def valid_request_payload():
    """Valid request with pipeline_definition."""
    return {
        "tenant_id": "tenant-001",
        "session_id": str(uuid4()),
        "image_id": str(uuid4()),
        "image_url": "gs://bucket/tenant-001/images/test.jpg",
        "pipeline_definition": {
            "type": "chain",
            "steps": [
                {"type": "step", "name": "segmentation"},
            ],
        },
        "settings": {"segment_filter_classes": ["segmento"]},
    }


class TestProcessEndpoint:
    """Test /tasks/process endpoint."""

    def test_process_accepts_pipeline_definition(self, valid_request_payload, mock_storage):
        """Endpoint should accept request with pipeline_definition."""
        from app.main import app

        with patch("app.api.routes.tasks.get_storage_client", return_value=mock_storage):
            with patch("app.api.routes.tasks.PipelineExecutor") as mock_executor:
                mock_executor.return_value.execute = AsyncMock(
                    return_value=MagicMock(
                        results={},
                        raw_segments=[],
                        raw_detections=[],
                        raw_classifications=[],
                    )
                )

                client = TestClient(app)
                response = client.post(
                    "/tasks/process",
                    json=valid_request_payload,
                    headers={"X-CloudTasks-TaskName": "test-task"},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["tenant_id"] == "tenant-001"

    def test_process_rejects_invalid_pipeline(self, mock_storage):
        """Endpoint should reject invalid pipeline_definition."""
        from app.main import app

        payload = {
            "tenant_id": "tenant-001",
            "session_id": str(uuid4()),
            "image_id": str(uuid4()),
            "image_url": "gs://bucket/image.jpg",
            "pipeline_definition": {
                "type": "chain",
                "steps": [
                    {"type": "step", "name": "nonexistent_step"},  # Invalid
                ],
            },
        }

        with patch("app.api.routes.tasks.get_storage_client", return_value=mock_storage):
            client = TestClient(app)
            response = client.post(
                "/tasks/process",
                json=payload,
                headers={"X-CloudTasks-TaskName": "test-task"},
            )

            assert response.status_code == 400
            assert "Invalid pipeline" in response.json()["detail"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api/test_tasks.py::TestProcessEndpoint -v`
Expected: FAIL (current endpoint uses get_tenant_cache)

**Step 3: Implement the endpoint changes**

Modify `app/api/routes/tasks.py`:

```python
"""Unified task endpoint for all ML processing.

Receives tasks from Cloud Tasks with pipeline_definition inline.
No database lookup - fully stateless.
"""

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CloudTasksRequest, Storage
from app.core.pipeline_executor import PipelineExecutor
from app.core.pipeline_parser import PipelineParser, PipelineParserError
from app.core.processing_context import ProcessingContext
from app.core.step_registry import StepRegistry
from app.infra.logging import get_logger
from app.schemas.task import ProcessingRequest, ProcessingResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/process",
    response_model=ProcessingResponse,
    summary="Process image through pipeline defined in request",
)
async def process_task(
    request: ProcessingRequest,
    storage: Storage,
    _: CloudTasksRequest,
) -> ProcessingResponse:
    """Process image using pipeline_definition from request.

    Pipeline is defined directly in the request payload - no DB lookup.
    Supports full DSL composition: chain, group, chord for parallel execution.
    """
    start_time = time.time()
    local_path: Path | None = None

    # Log request received
    logger.info(
        "Processing request received",
        tenant_id=request.tenant_id,
        session_id=str(request.session_id),
        image_id=str(request.image_id),
        image_url=request.image_url,
        pipeline_type=request.pipeline_definition.type,
    )

    try:
        # Parse and validate pipeline definition (fail-fast)
        parser = PipelineParser(StepRegistry)
        try:
            pipeline = parser.parse(request.pipeline_definition)
        except PipelineParserError as e:
            logger.error(
                "Invalid pipeline definition",
                tenant_id=request.tenant_id,
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid pipeline: {e}",
            )

        logger.info(
            "Pipeline validated",
            tenant_id=request.tenant_id,
            pipeline_type=type(pipeline).__name__,
            steps_count=len(request.pipeline_definition.steps),
        )

        # Download image
        download_start = time.time()
        local_path = await storage.download_to_tempfile(
            blob_path=request.image_url,
            tenant_id=request.tenant_id,
        )
        download_ms = int((time.time() - download_start) * 1000)

        logger.info(
            "Image downloaded",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            local_path=str(local_path),
            file_size_bytes=local_path.stat().st_size if local_path.exists() else 0,
            download_ms=download_ms,
        )

        # Create initial context with settings from request
        ctx = ProcessingContext(
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            session_id=str(request.session_id),
            image_path=local_path,
            config=request.settings,
        )

        logger.info(
            "Pipeline execution starting",
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            pipeline_type=type(pipeline).__name__,
        )

        # Execute pipeline
        executor = PipelineExecutor()
        ctx = await executor.execute(pipeline, ctx)

        duration_ms = int((time.time() - start_time) * 1000)

        # Build complete results including raw ML data
        full_results = {
            **ctx.results,
            "segments": ctx.raw_segments,
            "detections": ctx.raw_detections,
            "classifications_raw": ctx.raw_classifications,
        }

        # Log success summary
        logger.info(
            "Processing completed SUCCESSFULLY",
            tenant_id=request.tenant_id,
            session_id=str(request.session_id),
            image_id=str(request.image_id),
            segments_found=len(ctx.raw_segments),
            detections_found=len(ctx.raw_detections),
            total_duration_ms=duration_ms,
            download_ms=download_ms,
            processing_ms=duration_ms - download_ms,
        )

        return ProcessingResponse(
            success=True,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            pipeline_type=request.pipeline_definition.type,
            results=full_results,
            duration_ms=duration_ms,
            steps_completed=len(request.pipeline_definition.steps),
        )

    except HTTPException:
        raise
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "Processing FAILED",
            tenant_id=request.tenant_id,
            session_id=str(request.session_id),
            image_id=str(request.image_id),
            error_type=type(e).__name__,
            error_message=str(e),
            duration_ms=duration_ms,
            exc_info=True,
        )
        return ProcessingResponse(
            success=False,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            pipeline_type=request.pipeline_definition.type,
            results={},
            duration_ms=duration_ms,
            steps_completed=0,
            error=str(e),
        )
    finally:
        if local_path and local_path.exists():
            local_path.unlink(missing_ok=True)
            logger.debug(
                "Temp file cleaned up",
                path=str(local_path),
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api/test_tasks.py::TestProcessEndpoint -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/api/routes/tasks.py tests/test_api/test_tasks.py
git commit -m "feat: update /tasks/process to use inline pipeline_definition"
```

---

### Task 3: Clean Up main.py

**Files:**
- Modify: `app/main.py`

**Step 1: Review current imports and code**

Current imports to REMOVE:
```python
from app.infra.database import close_db_engine, verify_db_connection
from app.core.tenant_config import get_tenant_cache
from app.infra.database import get_db_session
```

Code blocks to REMOVE in lifespan:
- Lines 60-67: tenant_cache initialization
- Lines 69-72: verify_db_connection
- Lines 88-90: tenant_cache.stop()
- Line 92: close_db_engine()

**Step 2: Implement the cleanup**

Modify `app/main.py`:

```python
"""FastAPI application entry point.

ML Worker service for background processing via Cloud Tasks.
Fully stateless - no database access.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.infra.logging import get_logger, setup_logging
from app.ml.model_cache import ModelCache
from app.core.processor_registry import get_processor_registry
from app.steps import register_all_steps

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.tasks import router as tasks_router

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Startup:
    - Initialize processor registry
    - Register pipeline steps
    - Optionally preload ML models

    Shutdown:
    - Clear model caches
    """
    logger.info(
        "ML Worker starting",
        environment=settings.environment,
        industry=settings.industry,
    )

    # Initialize processor registry
    registry = get_processor_registry()
    logger.info("Processor registry initialized", processors=registry.get_available())

    # Register all pipeline steps
    register_all_steps()
    logger.info("Pipeline steps registered")

    # Preload models in production (skip in dev for faster startup)
    if settings.environment != "dev":
        logger.info("Preloading ML models...")
        try:
            ModelCache.get_model("detect", worker_id=0)
            logger.info("Detection model preloaded")
        except Exception as e:
            logger.warning("Model preloading failed", error=str(e))

    yield

    # Shutdown
    logger.info("ML Worker shutting down")
    ModelCache.clear_cache()
    get_processor_registry().clear_instances()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="StockControl ML Worker",
    description="Background ML processing service - Stateless",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.environment == "dev" else None,
    redoc_url=None,
)

# CORS middleware (mainly for local development)
if settings.environment == "dev":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests with context."""
    task_name = request.headers.get("X-CloudTasks-TaskName", "")
    queue_name = request.headers.get("X-CloudTasks-QueueName", "")
    retry_count = request.headers.get("X-CloudTasks-TaskRetryCount", "0")

    if task_name:
        logger.info(
            "Cloud Tasks request received",
            task_name=task_name,
            queue_name=queue_name,
            retry_count=retry_count,
            path=request.url.path,
        )

    response = await call_next(request)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_type": type(exc).__name__,
        },
    )


app.include_router(health_router, tags=["Health"])
app.include_router(tasks_router, prefix="/tasks", tags=["Tasks"])


@app.get("/")
async def root() -> dict:
    """Root endpoint - basic service info."""
    return {
        "service": "StockControl ML Worker",
        "version": "0.2.0",
        "environment": settings.environment,
        "industry": settings.industry,
        "stateless": True,
    }
```

**Step 3: Run existing tests to verify nothing broke**

Run: `pytest tests/test_api/test_health.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add app/main.py
git commit -m "refactor: remove database initialization from main.py"
```

---

### Task 4: Clean Up deps.py

**Files:**
- Modify: `app/api/deps.py`

**Step 1: Implement the cleanup**

Remove database-related code, keep only Storage and CloudTasksRequest:

```python
"""FastAPI dependencies for dependency injection.

Provides:
- Storage client
- Cloud Tasks request validation
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.infra.storage import StorageClient, get_storage_client
from app.infra.logging import get_logger

logger = get_logger(__name__)


async def get_storage() -> StorageClient:
    """Get storage client dependency."""
    return get_storage_client()


# Type alias for cleaner annotations
Storage = Annotated[StorageClient, Depends(get_storage)]


def validate_cloud_tasks_request(
    x_cloudtasks_taskname: Annotated[str | None, Header(alias="X-CloudTasks-TaskName")] = None,
    x_cloudtasks_queuename: Annotated[str | None, Header(alias="X-CloudTasks-QueueName")] = None,
) -> bool:
    """Validate that request comes from Cloud Tasks.

    In production (non-dev environments), requires Cloud Tasks headers.
    Cloud Run's IAM invoker role handles authentication, this is defense in depth.
    """
    from app.config import settings

    # In dev mode, allow requests without Cloud Tasks headers
    if settings.environment == "dev":
        if x_cloudtasks_taskname:
            logger.debug(
                "Cloud Tasks request (dev mode)",
                task_name=x_cloudtasks_taskname,
                queue_name=x_cloudtasks_queuename,
            )
        return True

    # In production/staging with strict validation enabled
    if settings.cloudtasks_strict_validation and not x_cloudtasks_taskname:
        logger.warning(
            "Rejected request: missing Cloud Tasks headers",
            has_task_name=bool(x_cloudtasks_taskname),
            has_queue_name=bool(x_cloudtasks_queuename),
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing required Cloud Tasks headers",
        )

    logger.info(
        "Cloud Tasks request validated",
        task_name=x_cloudtasks_taskname,
        queue_name=x_cloudtasks_queuename,
        environment=settings.environment,
    )
    return True


CloudTasksRequest = Annotated[bool, Depends(validate_cloud_tasks_request)]
```

**Step 2: Commit**

```bash
git add app/api/deps.py
git commit -m "refactor: remove database dependencies from deps.py"
```

---

### Task 5: Delete Database Files

**Files:**
- Delete: `app/infra/database.py`
- Delete: `app/core/tenant_config.py`
- Delete: `tests/test_core/test_tenant_config.py`

**Step 1: Delete the files**

```bash
rm app/infra/database.py
rm app/core/tenant_config.py
rm tests/test_core/test_tenant_config.py
```

**Step 2: Verify no imports remain**

Run: `grep -r "from app.infra.database" app/`
Expected: No output (no remaining imports)

Run: `grep -r "from app.core.tenant_config" app/`
Expected: No output

**Step 3: Run tests to verify nothing broke**

Run: `pytest tests/ -v --ignore=tests/test_models/`
Expected: PASS (ignoring model tests for now)

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: remove database.py and tenant_config.py"
```

---

### Task 6: Delete SQLAlchemy Models

**Files:**
- Delete: `app/models/` directory
- Delete: `tests/test_models/` directory

**Step 1: Delete the directories**

```bash
rm -rf app/models/
rm -rf tests/test_models/
```

**Step 2: Check for remaining imports**

Run: `grep -r "from app.models" app/`
Expected: No output

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: remove SQLAlchemy models directory"
```

---

### Task 7: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

**Step 1: Remove database dependencies**

Remove these lines from dependencies:
```toml
# Database - REMOVE
"sqlalchemy[asyncio]>=2.0.36",
"asyncpg>=0.30.0",
"greenlet>=3.1.0",
```

**Step 2: Update the file**

The dependencies section should become:

```toml
dependencies = [
    # Web Framework
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",

    # Google Cloud
    "google-cloud-storage>=2.18.0",
    "google-auth>=2.35.0",
    "google-cloud-tasks>=2.16.0",

    # ML / Computer Vision (torch CPU installed separately in Dockerfile)
    "ultralytics>=8.3.0",
    "pillow>=10.4.0",
    "numpy>=1.26.0",
    "opencv-python-headless>=4.10.0",

    # Observability
    "structlog>=24.4.0",
    "python-json-logger>=2.0.0",

    # Utilities
    "httpx>=0.27.0",
    "tenacity>=9.0.0",
    "pyyaml>=6.0.2",
    "python-multipart>=0.0.9",
]
```

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: remove SQLAlchemy dependencies from pyproject.toml"
```

---

### Task 8: Update config.py (Remove DATABASE_URL)

**Files:**
- Modify: `app/config.py`

**Step 1: Check current config**

Run: `grep -n "database\|DATABASE" app/config.py`

**Step 2: Remove DATABASE_URL if present**

Remove any database-related settings.

**Step 3: Commit**

```bash
git add app/config.py
git commit -m "chore: remove DATABASE_URL from config"
```

---

### Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run type check**

Run: `mypy app/`
Expected: No errors related to removed modules

**Step 3: Run linter**

Run: `ruff check app/`
Expected: No errors

**Step 4: Final commit**

```bash
git add -A
git commit -m "test: verify all tests pass after stateless refactor"
```

---

### Task 10: Update CLAUDE.md Documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update documentation**

Remove references to:
- `TenantConfigCache`
- Database connection
- `DATABASE_URL`
- `tenant_config` table

Add section about stateless architecture:
- Pipeline definitions come in request
- No database access
- Fully stateless service

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for stateless architecture"
```

---

## Summary

| Task | Files | Action |
|------|-------|--------|
| 1 | `app/schemas/task.py` | Add pipeline_definition, settings |
| 2 | `app/api/routes/tasks.py` | Use request directly |
| 3 | `app/main.py` | Remove DB initialization |
| 4 | `app/api/deps.py` | Remove DB dependencies |
| 5 | `app/infra/database.py`, `app/core/tenant_config.py` | DELETE |
| 6 | `app/models/`, `tests/test_models/` | DELETE |
| 7 | `pyproject.toml` | Remove SQLAlchemy deps |
| 8 | `app/config.py` | Remove DATABASE_URL |
| 9 | - | Run full test suite |
| 10 | `CLAUDE.md` | Update docs |
