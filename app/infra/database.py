"""Async database configuration with Row Level Security (RLS) support.

Provides:
- Async SQLAlchemy engine and session factory
- RLS tenant context setter for multi-tenant isolation
- Connection pooling optimized for Cloud Run
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)

# Type alias for dependency injection
DatabaseSession = AsyncSession

# Global engine (initialized on app startup)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Get or create the async database engine."""
    global _engine

    if _engine is None:
        logger.info(
            "Creating database engine",
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_pool_max_overflow,
        )

        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_pool_max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=1800,  # Recycle connections after 30 min
            echo=settings.debug,  # Log SQL in debug mode
        )

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory."""
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    return _session_factory


@asynccontextmanager
async def get_db_session(tenant_id: str | None = None) -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session with optional RLS tenant context.

    Args:
        tenant_id: Tenant ID for RLS isolation. If provided, sets the
                  `app.current_tenant` session variable for RLS policies.

    Yields:
        AsyncSession with tenant context set

    Example:
        async with get_db_session(tenant_id="tenant-123") as session:
            result = await session.execute(select(Detection))
            # RLS automatically filters to tenant-123

    Note:
        Database tables must have RLS policies configured:
        ```sql
        ALTER TABLE detections ENABLE ROW LEVEL SECURITY;
        CREATE POLICY tenant_isolation ON detections
            USING (tenant_id = current_setting('app.current_tenant'));
        ```
    """
    factory = get_session_factory()
    session = factory()

    try:
        if tenant_id:
            # Set RLS tenant context
            await session.execute(
                text("SET LOCAL app.current_tenant = :tenant_id"),
                {"tenant_id": tenant_id},
            )
            logger.debug("RLS tenant context set", tenant_id=tenant_id)

        yield session
        await session.commit()

    except Exception as e:
        await session.rollback()
        logger.error("Database session error", error=str(e), tenant_id=tenant_id)
        raise

    finally:
        await session.close()


async def close_db_engine() -> None:
    """Close the database engine and all connections.

    Call this during application shutdown.
    """
    global _engine, _session_factory

    if _engine is not None:
        logger.info("Closing database engine")
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def verify_db_connection() -> bool:
    """Verify database connectivity.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
            logger.info("Database connection verified")
            return True
    except Exception as e:
        logger.error("Database connection failed", error=str(e))
        return False
