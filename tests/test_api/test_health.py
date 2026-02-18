"""Tests for health check endpoints."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoints:
    """Tests for health check routes."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, client: AsyncClient):
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_live_endpoint(self, client: AsyncClient):
        response = await client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_ready_endpoint(self, client: AsyncClient):
        response = await client.get("/health/ready")
        # May return 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]

        data = response.json()
        assert "status" in data
        assert "checks" in data
