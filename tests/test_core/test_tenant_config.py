"""Tests for TenantConfigCache."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant_config import (
    TenantConfigCache,
    TenantPipelineConfig,
    get_tenant_cache,
)


class TestTenantPipelineConfig:
    """Tests for TenantPipelineConfig dataclass."""

    def test_tenant_pipeline_config_creation(self):
        """Test creating a TenantPipelineConfig instance."""
        config = TenantPipelineConfig(
            tenant_id="tenant-001",
            pipeline_steps=["detection", "estimation"],
            settings={"confidence_threshold": 0.8},
        )

        assert config.tenant_id == "tenant-001"
        assert config.pipeline_steps == ["detection", "estimation"]
        assert config.settings == {"confidence_threshold": 0.8}

    def test_tenant_pipeline_config_immutability(self):
        """Test that TenantPipelineConfig is immutable."""
        config = TenantPipelineConfig(
            tenant_id="tenant-001",
            pipeline_steps=["detection"],
            settings={},
        )

        with pytest.raises(AttributeError):
            config.tenant_id = "tenant-002"  # type: ignore[misc]


class TestTenantConfigCache:
    """Tests for TenantConfigCache."""

    @pytest.fixture
    def mock_db_session(self) -> AsyncSession:
        """Create a mock database session."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.mark.asyncio
    async def test_cache_initialization(self):
        """Test cache initializes with empty data."""
        cache = TenantConfigCache(refresh_interval_seconds=300)

        assert cache._cache == {}
        assert cache._refresh_interval == 300
        assert cache._refresh_task is None

    @pytest.mark.asyncio
    async def test_get_existing_tenant(self, mock_db_session: AsyncSession):
        """Test retrieving an existing tenant config from cache."""
        cache = TenantConfigCache()

        # Load configs into cache
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("tenant-001", ["detection", "estimation"], {"threshold": 0.8})
        ]
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        await cache.load_configs(mock_db_session)

        # Get tenant config
        config = await cache.get("tenant-001")

        assert config is not None
        assert config.tenant_id == "tenant-001"
        assert config.pipeline_steps == ["detection", "estimation"]
        assert config.settings == {"threshold": 0.8}

    @pytest.mark.asyncio
    async def test_get_nonexistent_tenant(self):
        """Test retrieving a non-existent tenant returns None."""
        cache = TenantConfigCache()

        config = await cache.get("nonexistent-tenant")

        assert config is None

    @pytest.mark.asyncio
    async def test_load_configs_from_db(self, mock_db_session: AsyncSession):
        """Test loading configs from database."""
        cache = TenantConfigCache()

        # Mock database result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("tenant-001", ["detection"], {"conf": 0.7}),
            ("tenant-002", ["detection", "estimation"], {"conf": 0.9}),
        ]
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        await cache.load_configs(mock_db_session)

        # Verify both configs are cached
        config1 = await cache.get("tenant-001")
        config2 = await cache.get("tenant-002")

        assert config1 is not None
        assert config1.tenant_id == "tenant-001"
        assert config1.pipeline_steps == ["detection"]

        assert config2 is not None
        assert config2.tenant_id == "tenant-002"
        assert config2.pipeline_steps == ["detection", "estimation"]

    @pytest.mark.asyncio
    async def test_load_configs_with_empty_result(self, mock_db_session: AsyncSession):
        """Test loading configs when database returns no rows."""
        cache = TenantConfigCache()

        # Mock empty database result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        await cache.load_configs(mock_db_session)

        # Cache should be empty
        config = await cache.get("any-tenant")
        assert config is None

    @pytest.mark.asyncio
    async def test_load_configs_handles_db_error(self, mock_db_session: AsyncSession):
        """Test that load_configs handles database errors gracefully."""
        cache = TenantConfigCache()

        # Mock database error
        mock_db_session.execute = AsyncMock(side_effect=Exception("DB connection error"))

        # Should not raise, but log error
        await cache.load_configs(mock_db_session)

        # Cache should remain empty
        config = await cache.get("tenant-001")
        assert config is None

    @pytest.mark.asyncio
    async def test_start_refresh_loop(self, mock_db_session: AsyncSession):
        """Test starting the refresh loop."""
        cache = TenantConfigCache(refresh_interval_seconds=1)

        # Mock database result
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("tenant-001", ["detection"], {"conf": 0.8})
        ]
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        # Create async context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session_factory():
            yield mock_db_session

        # Start refresh loop
        await cache.start_refresh_loop(mock_session_factory)

        # Wait a bit for initial load
        await asyncio.sleep(0.2)

        # Verify config was loaded
        config = await cache.get("tenant-001")
        assert config is not None

        # Stop the loop
        await cache.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_refresh_task(self):
        """Test that stop cancels the refresh task."""
        cache = TenantConfigCache(refresh_interval_seconds=100)

        # Create async context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session_factory():
            mock_session = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.fetchall.return_value = []
            mock_session.execute = AsyncMock(return_value=mock_result)
            yield mock_session

        await cache.start_refresh_loop(mock_session_factory)

        # Wait a bit to ensure task is running
        await asyncio.sleep(0.1)

        # Task should be running
        assert cache._refresh_task is not None
        assert not cache._refresh_task.done()

        # Stop the task
        await cache.stop()

        # Task should be cancelled (note: after await stop(), the task is None)
        assert cache._refresh_task is None

    @pytest.mark.asyncio
    async def test_refresh_loop_periodic_reload(self, mock_db_session: AsyncSession):
        """Test that refresh loop reloads configs periodically."""
        cache = TenantConfigCache(refresh_interval_seconds=0.2)

        call_count = 0

        # Create async context manager
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def mock_session_factory():
            nonlocal call_count
            call_count += 1

            mock_session = AsyncMock(spec=AsyncSession)
            mock_result = MagicMock()
            mock_result.fetchall.return_value = [
                ("tenant-001", ["detection"], {"count": call_count})
            ]
            mock_session.execute = AsyncMock(return_value=mock_result)
            yield mock_session

        await cache.start_refresh_loop(mock_session_factory)

        # Wait for multiple refresh cycles
        await asyncio.sleep(0.6)

        # Verify config was reloaded multiple times
        assert call_count >= 2

        await cache.stop()

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self, mock_db_session: AsyncSession):
        """Test that concurrent get operations are thread-safe."""
        cache = TenantConfigCache()

        # Load initial config
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            ("tenant-001", ["detection"], {"conf": 0.8})
        ]
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        await cache.load_configs(mock_db_session)

        # Concurrent gets
        results = await asyncio.gather(
            cache.get("tenant-001"),
            cache.get("tenant-001"),
            cache.get("tenant-001"),
        )

        assert all(r is not None for r in results)
        assert all(r.tenant_id == "tenant-001" for r in results)  # type: ignore[union-attr]


class TestGetTenantCache:
    """Tests for get_tenant_cache singleton."""

    def test_get_tenant_cache_singleton(self):
        """Test that get_tenant_cache returns singleton instance."""
        cache1 = get_tenant_cache()
        cache2 = get_tenant_cache()

        assert cache1 is cache2

    def test_get_tenant_cache_returns_instance(self):
        """Test that get_tenant_cache returns TenantConfigCache instance."""
        cache = get_tenant_cache()

        assert isinstance(cache, TenantConfigCache)
