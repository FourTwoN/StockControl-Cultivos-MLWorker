"""FastAPI application entry point.

ML Worker service for background processing via Cloud Tasks.
Uses configuration-driven pipeline orchestration.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.infra.database import close_db_engine, verify_db_connection
from app.infra.logging import get_logger, setup_logging
from app.ml.model_cache import ModelCache
from app.core.industry_config import load_industry_config, get_config_loader
from app.core.processor_registry import get_processor_registry
from app.core.tenant_config import get_tenant_cache
from app.infra.database import get_db_session
from app.steps import register_all_steps

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.tasks import router as tasks_router
from app.api.routes.upload import router as upload_router

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Startup:
    - Load industry configuration
    - Initialize processor registry
    - Verify database connection
    - Optionally preload ML models

    Shutdown:
    - Close database connections
    - Clear model and config caches
    """
    logger.info(
        "ML Worker starting",
        environment=settings.environment,
        industry=settings.industry,
    )

    # Load industry configuration
    try:
        config = await load_industry_config()
        logger.info(
            "Industry config loaded",
            industry=config.industry,
            version=config.version,
            pipelines=config.get_available_pipelines(),
        )
    except Exception as e:
        logger.warning("Failed to preload industry config", error=str(e))

    # Initialize processor registry
    registry = get_processor_registry()
    logger.info("Processor registry initialized", processors=registry.get_available())

    # Register all pipeline steps
    register_all_steps()
    logger.info("Pipeline steps registered")

    # Initialize tenant config cache
    tenant_cache = get_tenant_cache()
    try:
        async with get_db_session() as session:
            await tenant_cache.load_configs(session)
        logger.info("Tenant configs loaded into cache")
    except Exception as e:
        logger.warning("Failed to load tenant configs", error=str(e))

    # Verify database connection
    db_ok = await verify_db_connection()
    if not db_ok:
        logger.warning("Database connection failed - will retry on first request")

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

    # Stop tenant config refresh loop
    tenant_cache = get_tenant_cache()
    await tenant_cache.stop()

    await close_db_engine()
    ModelCache.clear_cache()
    get_config_loader().clear_cache()
    get_processor_registry().clear_instances()
    logger.info("Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="StockControl ML Worker",
    description="Background ML processing service for Demeter AI 2.0",
    version="0.1.0",
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


# =============================================================================
# Cloud Tasks OIDC Verification Middleware
# =============================================================================
# In production, Cloud Tasks sends OIDC tokens for authentication.
# The token is verified by Cloud Run automatically when using IAM invoker.
# Additional validation can be added here if needed.


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests with context."""
    # Extract Cloud Tasks headers for tracing
    task_name = request.headers.get("X-CloudTasks-TaskName", "")
    queue_name = request.headers.get("X-CloudTasks-QueueName", "")
    retry_count = request.headers.get("X-CloudTasks-TaskRetryCount", "0")

    # Bind context for all logs in this request
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


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions.

    Returns structured error response for Cloud Tasks to handle retries.
    """
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        exc_info=True,
    )

    # Return 500 for Cloud Tasks to retry
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_type": type(exc).__name__,
        },
    )


# =============================================================================
# Include Routers
# =============================================================================

app.include_router(health_router, tags=["Health"])
app.include_router(tasks_router, prefix="/tasks", tags=["Tasks"])
app.include_router(upload_router, prefix="/dev", tags=["Development"])


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get("/")
async def root() -> dict:
    """Root endpoint - basic service info."""
    return {
        "service": "StockControl ML Worker",
        "version": "0.1.0",
        "environment": settings.environment,
        "industry": settings.industry,
    }
