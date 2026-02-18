"""FastAPI dependencies for dependency injection.

Provides:
- Storage client
- Cloud Tasks request validation
"""

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.infra.logging import get_logger
from app.infra.storage import StorageClient, get_storage_client

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
