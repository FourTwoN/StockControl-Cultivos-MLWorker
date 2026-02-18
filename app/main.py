"""FastAPI application entry point.

ML Worker service for background processing via Cloud Tasks.
Fully stateless - no database access.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.tasks import router as tasks_router
from app.config import settings
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger, setup_logging
from app.ml.model_cache import ModelCache
from app.steps import register_all_steps

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Startup:
    - Initialize processor registry
    - Register pipeline steps

    Shutdown:
    - Clear model caches

    Note: Models are loaded per-tenant on first request, not preloaded globally.
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

    # Models are now loaded per-tenant on first request
    # No global preloading needed

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
