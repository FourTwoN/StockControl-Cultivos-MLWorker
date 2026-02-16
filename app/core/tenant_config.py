"""Tenant configuration cache with periodic refresh from database.

This module provides in-memory caching of tenant pipeline configurations
loaded from the database, with automatic periodic refresh.
"""

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TenantPipelineConfig:
    """Immutable tenant pipeline configuration.

    Attributes:
        tenant_id: Unique tenant identifier
        pipeline_steps: List of pipeline step names to execute
        settings: Additional configuration settings for the tenant
    """

    tenant_id: str
    pipeline_steps: list[str]
    settings: dict[str, Any]


class TenantConfigCache:
    """In-memory cache for tenant pipeline configurations.

    Loads configurations from database and refreshes periodically.
    Thread-safe for concurrent access.
    """

    def __init__(self, refresh_interval_seconds: int = 300) -> None:
        """Initialize the tenant config cache.

        Args:
            refresh_interval_seconds: Interval between cache refreshes (default: 5 minutes)
        """
        self._cache: dict[str, TenantPipelineConfig] = {}
        self._refresh_interval = refresh_interval_seconds
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task[None] | None = None

        logger.info(
            "TenantConfigCache initialized",
            refresh_interval_seconds=refresh_interval_seconds,
        )

    async def get(self, tenant_id: str) -> TenantPipelineConfig | None:
        """Get tenant configuration from cache.

        Args:
            tenant_id: Tenant identifier

        Returns:
            TenantPipelineConfig if found, None otherwise
        """
        async with self._lock:
            config = self._cache.get(tenant_id)

        if config:
            logger.debug("Retrieved tenant config from cache", tenant_id=tenant_id)
        else:
            logger.debug("Tenant config not found in cache", tenant_id=tenant_id)

        return config

    async def load_configs(self, db_session: AsyncSession) -> None:
        """Load all tenant configurations from database.

        Args:
            db_session: Active database session
        """
        try:
            logger.info("Loading tenant configurations from database")

            query = text(
                "SELECT tenant_id, pipeline_steps, settings FROM tenant_config"
            )
            result = await db_session.execute(query)
            rows = result.fetchall()

            new_cache: dict[str, TenantPipelineConfig] = {}

            for row in rows:
                tenant_id, pipeline_steps, settings = row
                config = TenantPipelineConfig(
                    tenant_id=tenant_id,
                    pipeline_steps=pipeline_steps,
                    settings=settings or {},
                )
                new_cache[tenant_id] = config

            async with self._lock:
                self._cache = new_cache

            logger.info(
                "Loaded tenant configurations",
                count=len(new_cache),
                tenant_ids=list(new_cache.keys()),
            )

        except Exception as e:
            logger.error(
                "Failed to load tenant configurations from database",
                error=str(e),
                exc_info=True,
            )

    async def start_refresh_loop(
        self,
        db_session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Start periodic refresh loop in background.

        Args:
            db_session_factory: Function that returns an async context manager for database sessions
        """
        if self._refresh_task is not None:
            logger.warning("Refresh loop already running")
            return

        logger.info("Starting tenant config refresh loop")
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(db_session_factory)
        )

    async def stop(self) -> None:
        """Stop the periodic refresh loop."""
        if self._refresh_task is None:
            logger.debug("No refresh loop running")
            return

        logger.info("Stopping tenant config refresh loop")
        self._refresh_task.cancel()

        try:
            await self._refresh_task
        except asyncio.CancelledError:
            logger.info("Refresh loop cancelled successfully")

        self._refresh_task = None

    async def _refresh_loop(
        self,
        db_session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    ) -> None:
        """Internal refresh loop that runs periodically.

        Args:
            db_session_factory: Function that returns an async context manager for database sessions
        """
        while True:
            try:
                async with db_session_factory() as session:
                    await self.load_configs(session)

                logger.debug(
                    "Next refresh scheduled",
                    interval_seconds=self._refresh_interval,
                )

                await asyncio.sleep(self._refresh_interval)

            except asyncio.CancelledError:
                logger.info("Refresh loop cancelled")
                raise
            except Exception as e:
                logger.error(
                    "Error in refresh loop, will retry after interval",
                    error=str(e),
                    interval_seconds=self._refresh_interval,
                    exc_info=True,
                )
                await asyncio.sleep(self._refresh_interval)


# Global singleton instance
_tenant_cache: TenantConfigCache | None = None


def get_tenant_cache() -> TenantConfigCache:
    """Get or create the global tenant cache singleton.

    Returns:
        Global TenantConfigCache instance
    """
    global _tenant_cache

    if _tenant_cache is None:
        _tenant_cache = TenantConfigCache()
        logger.info("Created global TenantConfigCache singleton")

    return _tenant_cache
