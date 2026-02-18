"""Health check endpoints.

Provides health status for Cloud Run probes and monitoring.
Stateless - no database checks.
"""

from fastapi import APIRouter

from app import __version__
from app.config import settings
from app.infra.logging import get_logger
from app.ml.model_registry import ModelRegistry
from app.schemas.common import HealthResponse

router = APIRouter()
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Basic health check.

    Returns 200 if service is running.
    Used by Cloud Run startup probe.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.environment,
        checks={},
    )


@router.get("/health/ready", response_model=HealthResponse)
async def health_ready() -> HealthResponse:
    """Readiness check.

    Verifies all dependencies are available:
    - Model availability

    Used by Cloud Run to determine if service can accept traffic.
    """
    checks: dict[str, bool] = {}

    # Check ML models
    try:
        model_health = ModelRegistry.health_check()
        checks["models"] = model_health.get("healthy", False)
    except Exception as e:
        logger.warning("Model health check failed", error=str(e))
        checks["models"] = False

    # Determine overall status
    all_healthy = all(checks.values()) if checks else True

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=__version__,
        environment=settings.environment,
        checks=checks,
    )


@router.get("/health/live", response_model=HealthResponse)
async def health_live() -> HealthResponse:
    """Liveness check.

    Basic check that service is responding.
    Used by Cloud Run liveness probe.
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        environment=settings.environment,
        checks={"alive": True},
    )
