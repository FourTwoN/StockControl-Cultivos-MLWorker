"""FastAPI dependencies for dependency injection.

Provides:
- Database session with RLS context
- Storage client
- Request validation
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_db_session
from app.infra.storage import StorageClient, get_storage_client
from app.infra.logging import get_logger

logger = get_logger(__name__)


async def get_tenant_id(
    x_tenant_id: Annotated[str | None, Header()] = None,
) -> str:
    """Extract tenant ID from request header.

    In production, Cloud Tasks includes tenant_id in the payload.
    This header is mainly for local development/testing.

    Args:
        x_tenant_id: Tenant ID header

    Returns:
        Tenant ID string

    Raises:
        HTTPException: If tenant ID is missing
    """
    if not x_tenant_id:
        # In production, tenant_id comes from request body
        # This dependency is optional - routes should validate from body
        logger.debug("No X-Tenant-Id header, will use body")
        return ""
    return x_tenant_id


async def get_db(
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session with RLS tenant context.

    Args:
        tenant_id: Tenant ID for RLS isolation

    Yields:
        AsyncSession with tenant context
    """
    async with get_db_session(tenant_id=tenant_id if tenant_id else None) as session:
        yield session


async def get_storage() -> StorageClient:
    """Get storage client dependency."""
    return get_storage_client()


# Type aliases for cleaner annotations
DbSession = Annotated[AsyncSession, Depends(get_db)]
Storage = Annotated[StorageClient, Depends(get_storage)]


def validate_cloud_tasks_request(
    x_cloudtasks_taskname: Annotated[str | None, Header(alias="X-CloudTasks-TaskName")] = None,
    x_cloudtasks_queuename: Annotated[str | None, Header(alias="X-CloudTasks-QueueName")] = None,
) -> bool:
    """Validate that request comes from Cloud Tasks.

    In production (non-dev environments), requires Cloud Tasks headers.
    Cloud Run's IAM invoker role handles authentication, this is defense in depth.

    Args:
        x_cloudtasks_taskname: Task name header (required in prod)
        x_cloudtasks_queuename: Queue name header

    Returns:
        True if valid Cloud Tasks request

    Raises:
        HTTPException: 403 if headers missing in production with strict validation
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
