# Post-Processors System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor MLWorker to separate generic ML processors from tenant-specific post-processors using a configurable pipeline system.

**Architecture:** Steps (ML + Post) implement a common `PipelineStep` interface. `TenantConfigCache` loads pipeline configs from DB at startup. `StepRegistry` builds pipelines dynamically per tenant. `ProcessingContext` flows through the chain accumulating results.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy async, Pydantic, pytest

---

## Task 1: Create Core Interfaces

**Files:**
- Create: `app/core/pipeline_step.py`
- Create: `app/core/processing_context.py`
- Test: `tests/test_core/test_pipeline_step.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_pipeline_step.py
"""Tests for pipeline step interfaces."""

import pytest
from dataclasses import replace
from pathlib import Path

from app.core.processing_context import ProcessingContext
from app.core.pipeline_step import PipelineStep


class TestProcessingContext:
    """Test ProcessingContext dataclass."""

    def test_create_context(self) -> None:
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="img-123",
            session_id="sess-456",
            image_path=Path("/tmp/test.jpg"),
            config={"num_bands": 4},
        )
        assert ctx.tenant_id == "test-tenant"
        assert ctx.raw_segments == []
        assert ctx.results == {}

    def test_context_immutable_update(self) -> None:
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="img-123",
            session_id="sess-456",
            image_path=Path("/tmp/test.jpg"),
            config={},
        )
        new_ctx = ctx.with_results({"sizes": {0: 2}})

        assert ctx.results == {}  # Original unchanged
        assert new_ctx.results == {"sizes": {0: 2}}

    def test_context_with_segments(self) -> None:
        ctx = ProcessingContext(
            tenant_id="test-tenant",
            image_id="img-123",
            session_id="sess-456",
            image_path=Path("/tmp/test.jpg"),
            config={},
        )
        segments = [{"class": "cajon", "bbox": [0, 0, 100, 100]}]
        new_ctx = ctx.with_segments(segments)

        assert new_ctx.raw_segments == segments


class TestPipelineStep:
    """Test PipelineStep abstract class."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            PipelineStep()

    def test_concrete_step_requires_name(self) -> None:
        class BadStep(PipelineStep):
            async def execute(self, ctx):
                return ctx

        with pytest.raises(TypeError):
            BadStep()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_pipeline_step.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write ProcessingContext**

```python
# app/core/processing_context.py
"""Processing context that flows through the pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessingContext:
    """Immutable context passed through pipeline steps.

    Each step receives context and returns a new one with updates.
    Original context is never mutated.
    """

    tenant_id: str
    image_id: str
    session_id: str
    image_path: Path
    config: dict[str, Any]

    # Raw results from ML processors
    raw_segments: list[dict[str, Any]] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    raw_classifications: list[dict[str, Any]] = field(default_factory=list)

    # Accumulated results from post-processors
    results: dict[str, Any] = field(default_factory=dict)

    def with_segments(self, segments: list[dict[str, Any]]) -> "ProcessingContext":
        """Return new context with segments."""
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=segments,
            raw_detections=self.raw_detections,
            raw_classifications=self.raw_classifications,
            results=self.results,
        )

    def with_detections(self, detections: list[dict[str, Any]]) -> "ProcessingContext":
        """Return new context with detections."""
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=detections,
            raw_classifications=self.raw_classifications,
            results=self.results,
        )

    def with_classifications(self, classifications: list[dict[str, Any]]) -> "ProcessingContext":
        """Return new context with classifications."""
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=self.raw_detections,
            raw_classifications=classifications,
            results=self.results,
        )

    def with_results(self, new_results: dict[str, Any]) -> "ProcessingContext":
        """Return new context with merged results."""
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=self.raw_detections,
            raw_classifications=self.raw_classifications,
            results={**self.results, **new_results},
        )
```

**Step 4: Write PipelineStep**

```python
# app/core/pipeline_step.py
"""Base class for all pipeline steps."""

from abc import ABC, abstractmethod

from app.core.processing_context import ProcessingContext


class PipelineStep(ABC):
    """Abstract base for ML processors and post-processors.

    All steps in the pipeline implement this interface,
    allowing them to be composed in any order.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this step."""
        pass

    @abstractmethod
    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute step and return updated context.

        Args:
            ctx: Current processing context

        Returns:
            New context with step results (original unchanged)
        """
        pass
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_core/test_pipeline_step.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add app/core/pipeline_step.py app/core/processing_context.py tests/test_core/test_pipeline_step.py
git commit -m "feat: add PipelineStep and ProcessingContext base classes"
```

---

## Task 2: Create StepRegistry

**Files:**
- Create: `app/core/step_registry.py`
- Test: `tests/test_core/test_step_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_step_registry.py
"""Tests for StepRegistry."""

import pytest
from pathlib import Path

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.step_registry import StepRegistry


class DummyStep(PipelineStep):
    """Test step for registry tests."""

    @property
    def name(self) -> str:
        return "dummy"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        return ctx.with_results({"dummy": True})


class AnotherStep(PipelineStep):
    """Another test step."""

    @property
    def name(self) -> str:
        return "another"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        return ctx.with_results({"another": True})


class TestStepRegistry:
    """Test StepRegistry functionality."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        StepRegistry._steps.clear()

    def test_register_step(self) -> None:
        StepRegistry.register("dummy", DummyStep)
        assert "dummy" in StepRegistry._steps

    def test_get_step(self) -> None:
        StepRegistry.register("dummy", DummyStep)
        step = StepRegistry.get("dummy")
        assert isinstance(step, DummyStep)
        assert step.name == "dummy"

    def test_get_unknown_step_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown step: unknown"):
            StepRegistry.get("unknown")

    def test_build_pipeline(self) -> None:
        StepRegistry.register("dummy", DummyStep)
        StepRegistry.register("another", AnotherStep)

        pipeline = StepRegistry.build_pipeline(["dummy", "another"])

        assert len(pipeline) == 2
        assert isinstance(pipeline[0], DummyStep)
        assert isinstance(pipeline[1], AnotherStep)

    def test_build_pipeline_preserves_order(self) -> None:
        StepRegistry.register("dummy", DummyStep)
        StepRegistry.register("another", AnotherStep)

        pipeline = StepRegistry.build_pipeline(["another", "dummy"])

        assert pipeline[0].name == "another"
        assert pipeline[1].name == "dummy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_step_registry.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write StepRegistry**

```python
# app/core/step_registry.py
"""Registry for pipeline steps."""

from typing import Type

from app.core.pipeline_step import PipelineStep
from app.infra.logging import get_logger

logger = get_logger(__name__)


class StepRegistry:
    """Registry of all available pipeline steps.

    Steps are registered by name and can be retrieved
    to build dynamic pipelines per tenant.
    """

    _steps: dict[str, Type[PipelineStep]] = {}

    @classmethod
    def register(cls, name: str, step_class: Type[PipelineStep]) -> None:
        """Register a step class by name.

        Args:
            name: Unique identifier for the step
            step_class: PipelineStep subclass
        """
        cls._steps[name] = step_class
        logger.debug("Registered step", name=name)

    @classmethod
    def get(cls, name: str) -> PipelineStep:
        """Get a step instance by name.

        Args:
            name: Step identifier

        Returns:
            New instance of the step

        Raises:
            ValueError: If step not found
        """
        if name not in cls._steps:
            raise ValueError(f"Unknown step: {name}")
        return cls._steps[name]()

    @classmethod
    def build_pipeline(cls, step_names: list[str]) -> list[PipelineStep]:
        """Build a pipeline from step names.

        Args:
            step_names: Ordered list of step identifiers

        Returns:
            List of step instances in order
        """
        return [cls.get(name) for name in step_names]

    @classmethod
    def available_steps(cls) -> list[str]:
        """Get list of registered step names."""
        return list(cls._steps.keys())
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_step_registry.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/step_registry.py tests/test_core/test_step_registry.py
git commit -m "feat: add StepRegistry for dynamic pipeline building"
```

---

## Task 3: Create TenantConfigCache

**Files:**
- Create: `app/core/tenant_config.py`
- Test: `tests/test_core/test_tenant_config.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_tenant_config.py
"""Tests for TenantConfigCache."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.tenant_config import TenantConfigCache, TenantPipelineConfig


class TestTenantPipelineConfig:
    """Test TenantPipelineConfig dataclass."""

    def test_create_config(self) -> None:
        config = TenantPipelineConfig(
            tenant_id="test-tenant",
            pipeline_steps=["segmentation", "detection"],
            settings={"num_bands": 4},
        )
        assert config.tenant_id == "test-tenant"
        assert config.pipeline_steps == ["segmentation", "detection"]
        assert config.settings["num_bands"] == 4

    def test_config_is_immutable(self) -> None:
        config = TenantPipelineConfig(
            tenant_id="test-tenant",
            pipeline_steps=["segmentation"],
            settings={},
        )
        with pytest.raises(AttributeError):
            config.tenant_id = "other"


class TestTenantConfigCache:
    """Test TenantConfigCache functionality."""

    @pytest.fixture
    def cache(self) -> TenantConfigCache:
        return TenantConfigCache(refresh_interval_seconds=300)

    @pytest.mark.asyncio
    async def test_get_returns_none_when_empty(self, cache: TenantConfigCache) -> None:
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_configs_from_db(self, cache: TenantConfigCache) -> None:
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            MagicMock(
                tenant_id="tenant-a",
                pipeline_steps=["segmentation", "detection"],
                settings={"num_bands": 4},
            ),
            MagicMock(
                tenant_id="tenant-b",
                pipeline_steps=["detection"],
                settings={},
            ),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        await cache.load_configs(mock_session)

        config_a = await cache.get("tenant-a")
        assert config_a is not None
        assert config_a.pipeline_steps == ["segmentation", "detection"]

        config_b = await cache.get("tenant-b")
        assert config_b is not None
        assert config_b.pipeline_steps == ["detection"]

    @pytest.mark.asyncio
    async def test_get_returns_cached_config(self, cache: TenantConfigCache) -> None:
        # Manually populate cache
        cache._cache["test-tenant"] = TenantPipelineConfig(
            tenant_id="test-tenant",
            pipeline_steps=["segmentation"],
            settings={"key": "value"},
        )

        result = await cache.get("test-tenant")

        assert result is not None
        assert result.tenant_id == "test-tenant"
        assert result.settings["key"] == "value"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_tenant_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write TenantConfigCache**

```python
# app/core/tenant_config.py
"""Tenant configuration cache with periodic refresh."""

import asyncio
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TenantPipelineConfig:
    """Immutable pipeline configuration for a tenant."""

    tenant_id: str
    pipeline_steps: list[str]
    settings: dict[str, Any]


class TenantConfigCache:
    """Cache of tenant configurations with periodic refresh.

    Loads all tenant configs from DB at startup and refreshes
    periodically in the background.
    """

    def __init__(self, refresh_interval_seconds: int = 300):
        """Initialize cache.

        Args:
            refresh_interval_seconds: How often to refresh (default 5 min)
        """
        self._cache: dict[str, TenantPipelineConfig] = {}
        self._refresh_interval = refresh_interval_seconds
        self._lock = asyncio.Lock()
        self._refresh_task: asyncio.Task | None = None

    async def get(self, tenant_id: str) -> TenantPipelineConfig | None:
        """Get config for tenant from cache.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Config if found, None otherwise
        """
        return self._cache.get(tenant_id)

    async def load_configs(self, db_session: AsyncSession) -> None:
        """Load all tenant configs from database.

        Args:
            db_session: Async database session
        """
        async with self._lock:
            try:
                result = await db_session.execute(
                    text("SELECT tenant_id, pipeline_steps, settings FROM tenant_config")
                )
                rows = result.fetchall()

                new_cache: dict[str, TenantPipelineConfig] = {}
                for row in rows:
                    new_cache[row.tenant_id] = TenantPipelineConfig(
                        tenant_id=row.tenant_id,
                        pipeline_steps=row.pipeline_steps,
                        settings=row.settings or {},
                    )

                self._cache = new_cache
                logger.info("Loaded tenant configs", count=len(self._cache))

            except Exception as e:
                logger.error("Failed to load tenant configs", error=str(e))
                raise

    async def start_refresh_loop(self, db_session_factory) -> None:
        """Start background refresh task.

        Args:
            db_session_factory: Callable that returns AsyncSession
        """
        self._refresh_task = asyncio.create_task(
            self._refresh_loop(db_session_factory)
        )
        logger.info(
            "Started tenant config refresh loop",
            interval_seconds=self._refresh_interval,
        )

    async def stop(self) -> None:
        """Stop the refresh loop."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped tenant config refresh loop")

    async def _refresh_loop(self, db_session_factory) -> None:
        """Background task that refreshes configs periodically."""
        while True:
            await asyncio.sleep(self._refresh_interval)
            try:
                async with db_session_factory() as session:
                    await self.load_configs(session)
            except Exception as e:
                logger.error("Refresh loop error", error=str(e))


# Global singleton
_tenant_cache: TenantConfigCache | None = None


def get_tenant_cache() -> TenantConfigCache:
    """Get the global tenant config cache."""
    global _tenant_cache
    if _tenant_cache is None:
        _tenant_cache = TenantConfigCache()
    return _tenant_cache
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_core/test_tenant_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/core/tenant_config.py tests/test_core/test_tenant_config.py
git commit -m "feat: add TenantConfigCache with periodic refresh"
```

---

## Task 4: Create Post-Processor Steps

**Files:**
- Create: `app/steps/__init__.py`
- Create: `app/steps/post/__init__.py`
- Create: `app/steps/post/segment_filter.py`
- Create: `app/steps/post/size_calculator.py`
- Create: `app/steps/post/species_distributor.py`
- Test: `tests/test_steps/__init__.py`
- Test: `tests/test_steps/test_post_processors.py`

**Step 1: Write the failing tests**

```python
# tests/test_steps/test_post_processors.py
"""Tests for post-processor steps."""

import pytest
from pathlib import Path

from app.core.processing_context import ProcessingContext
from app.steps.post.segment_filter import SegmentFilterStep
from app.steps.post.size_calculator import SizeCalculatorStep
from app.steps.post.species_distributor import SpeciesDistributorStep


@pytest.fixture
def base_context() -> ProcessingContext:
    return ProcessingContext(
        tenant_id="test-tenant",
        image_id="img-123",
        session_id="sess-456",
        image_path=Path("/tmp/test.jpg"),
        config={},
    )


class TestSegmentFilterStep:
    """Test SegmentFilterStep."""

    def test_name(self) -> None:
        step = SegmentFilterStep()
        assert step.name == "segment_filter"

    @pytest.mark.asyncio
    async def test_filter_keeps_largest_claro(self, base_context: ProcessingContext) -> None:
        segments = [
            {"class_name": "segmento", "area_px": 500000},
            {"class_name": "segmento", "area_px": 800000},  # Largest
            {"class_name": "cajon", "area_px": 100000},
        ]
        ctx = base_context.with_segments(segments)
        ctx = ProcessingContext(
            **{**ctx.__dict__, "config": {"segment_filter_type": "largest_claro"}}
        )

        result = await SegmentFilterStep().execute(ctx)

        claro_segments = [s for s in result.raw_segments if s["class_name"] == "segmento"]
        assert len(claro_segments) == 1
        assert claro_segments[0]["area_px"] == 800000

    @pytest.mark.asyncio
    async def test_filter_keeps_cajon(self, base_context: ProcessingContext) -> None:
        segments = [
            {"class_name": "segmento", "area_px": 500000},
            {"class_name": "cajon", "area_px": 100000},
        ]
        ctx = base_context.with_segments(segments)
        ctx = ProcessingContext(
            **{**ctx.__dict__, "config": {"segment_filter_type": "largest_claro"}}
        )

        result = await SegmentFilterStep().execute(ctx)

        cajon_segments = [s for s in result.raw_segments if s["class_name"] == "cajon"]
        assert len(cajon_segments) == 1


class TestSizeCalculatorStep:
    """Test SizeCalculatorStep."""

    def test_name(self) -> None:
        step = SizeCalculatorStep()
        assert step.name == "size_calculator"

    @pytest.mark.asyncio
    async def test_calculates_sizes(self, base_context: ProcessingContext) -> None:
        detections = [
            {"height_px": 50, "center_y_px": 100},
            {"height_px": 52, "center_y_px": 200},
            {"height_px": 48, "center_y_px": 300},
            {"height_px": 100, "center_y_px": 400},  # Outlier
        ]
        ctx = ProcessingContext(
            **{**base_context.__dict__,
               "raw_detections": detections,
               "config": {"num_bands": 4, "image_height": 1000}}
        )

        result = await SizeCalculatorStep().execute(ctx)

        assert "sizes" in result.results
        sizes = result.results["sizes"]
        assert len(sizes) == 4

    @pytest.mark.asyncio
    async def test_empty_detections_returns_empty(self, base_context: ProcessingContext) -> None:
        result = await SizeCalculatorStep().execute(base_context)
        assert result.results.get("sizes", {}) == {}


class TestSpeciesDistributorStep:
    """Test SpeciesDistributorStep."""

    def test_name(self) -> None:
        step = SpeciesDistributorStep()
        assert step.name == "species_distributor"

    @pytest.mark.asyncio
    async def test_distributes_equitably(self, base_context: ProcessingContext) -> None:
        detections = [{"id": i} for i in range(9)]
        ctx = ProcessingContext(
            **{**base_context.__dict__,
               "raw_detections": detections,
               "config": {"species": ["Tomato", "Pepper", "Lettuce"]},
               "results": {"sizes": {i: 2 for i in range(9)}}}
        )

        result = await SpeciesDistributorStep().execute(ctx)

        classifications = result.results["classifications"]
        assert len(classifications) == 9

        species_counts = {}
        for c in classifications:
            species_counts[c["class_name"]] = species_counts.get(c["class_name"], 0) + 1

        assert species_counts["Tomato"] == 3
        assert species_counts["Pepper"] == 3
        assert species_counts["Lettuce"] == 3

    @pytest.mark.asyncio
    async def test_no_species_returns_empty(self, base_context: ProcessingContext) -> None:
        ctx = ProcessingContext(
            **{**base_context.__dict__,
               "raw_detections": [{"id": 1}],
               "config": {}}
        )

        result = await SpeciesDistributorStep().execute(ctx)

        assert result.results.get("classifications", []) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_steps/test_post_processors.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create directory structure**

```bash
mkdir -p app/steps/post tests/test_steps
touch app/steps/__init__.py app/steps/post/__init__.py tests/test_steps/__init__.py
```

**Step 4: Write SegmentFilterStep**

```python
# app/steps/post/segment_filter.py
"""Segment filter post-processor."""

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SegmentFilterStep(PipelineStep):
    """Filters segments based on tenant configuration.

    Supports filter types:
    - largest_claro: Keep only the largest segmento/claro-cajon segment
    """

    @property
    def name(self) -> str:
        return "segment_filter"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        filter_type = ctx.config.get("segment_filter_type", "largest_claro")

        if filter_type == "largest_claro":
            filtered = self._filter_largest_claro(ctx.raw_segments)
        else:
            filtered = ctx.raw_segments

        logger.info(
            "Filtered segments",
            original=len(ctx.raw_segments),
            filtered=len(filtered),
            filter_type=filter_type,
        )

        return ctx.with_segments(filtered)

    def _filter_largest_claro(
        self,
        segments: list[dict],
    ) -> list[dict]:
        """Keep only largest claro segment, preserve others."""
        claro_types = ("segmento", "claro-cajon")
        claro_segments = [s for s in segments if s.get("class_name") in claro_types]

        if len(claro_segments) <= 1:
            return segments

        largest_claro = max(claro_segments, key=lambda s: s.get("area_px", 0))

        filtered = [s for s in segments if s.get("class_name") not in claro_types]
        filtered.append(largest_claro)

        return filtered
```

**Step 5: Write SizeCalculatorStep**

```python
# app/steps/post/size_calculator.py
"""Size calculator post-processor using z-scores."""

import numpy as np

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SizeCalculatorStep(PipelineStep):
    """Calculates plant sizes using z-score based algorithm.

    Algorithm:
    1. Normalize heights by band (compensate for perspective)
    2. Calculate global statistics (mean, std)
    3. Determine modal size based on median
    4. Assign sizes using z-scores:
       - z < -2.0: S (very small outlier)
       - z < -1.0: One size below modal
       - -1.0 <= z <= 1.0: Modal size (80-90%)
       - z > 1.0: One size above modal
       - z > 2.0: XL (very large outlier)
    """

    SIZE_S = 1
    SIZE_M = 2
    SIZE_L = 3
    SIZE_XL = 4

    @property
    def name(self) -> str:
        return "size_calculator"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        detections = ctx.raw_detections

        if not detections:
            logger.debug("No detections for size calculation")
            return ctx.with_results({"sizes": {}})

        num_bands = ctx.config.get("num_bands", 4)
        image_height = ctx.config.get("image_height", 4000)

        sizes = self._calculate_sizes(detections, num_bands, image_height)

        logger.info(
            "Calculated sizes",
            detections=len(detections),
            sizes_assigned=len(sizes),
        )

        return ctx.with_results({"sizes": sizes})

    def _calculate_sizes(
        self,
        detections: list[dict],
        num_bands: int,
        image_height: int,
    ) -> dict[int, int]:
        """Calculate size for each detection."""
        band_height = image_height / num_bands

        # Normalize heights by band
        normalized_heights = []
        for det in detections:
            height = det.get("height_px", 0)
            center_y = det.get("center_y_px", 0)
            band = min(int(center_y / band_height), num_bands - 1)

            # Simple normalization: scale by band position
            scale_factor = 1.0 + (band * 0.1)
            normalized_heights.append(height * scale_factor)

        if not normalized_heights:
            return {}

        # Calculate statistics
        heights_array = np.array(normalized_heights)
        mean_height = np.mean(heights_array)
        std_height = np.std(heights_array)

        if std_height < 1e-6:
            # All same size
            return {i: self.SIZE_M for i in range(len(detections))}

        # Determine modal size from median
        median_height = np.median(heights_array)
        if median_height < mean_height * 0.9:
            modal_size = self.SIZE_S
        elif median_height > mean_height * 1.1:
            modal_size = self.SIZE_L
        else:
            modal_size = self.SIZE_M

        # Assign sizes based on z-scores
        sizes = {}
        for i, height in enumerate(normalized_heights):
            z = (height - mean_height) / std_height

            if z < -2.0:
                sizes[i] = self.SIZE_S
            elif z < -1.0:
                sizes[i] = max(self.SIZE_S, modal_size - 1)
            elif z > 2.0:
                sizes[i] = self.SIZE_XL
            elif z > 1.0:
                sizes[i] = min(self.SIZE_XL, modal_size + 1)
            else:
                sizes[i] = modal_size

        return sizes
```

**Step 6: Write SpeciesDistributorStep**

```python
# app/steps/post/species_distributor.py
"""Species distributor post-processor."""

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SpeciesDistributorStep(PipelineStep):
    """Distributes detections equitably across configured species."""

    @property
    def name(self) -> str:
        return "species_distributor"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        species = ctx.config.get("species", [])
        detections = ctx.raw_detections
        sizes = ctx.results.get("sizes", {})

        if not species:
            logger.warning("No species configured, skipping distribution")
            return ctx.with_results({"classifications": []})

        if not detections:
            logger.debug("No detections to classify")
            return ctx.with_results({"classifications": []})

        classifications = self._distribute_equitably(
            detections=detections,
            species=species,
            sizes=sizes,
        )

        logger.info(
            "Distributed classifications",
            detections=len(detections),
            species=len(species),
            classifications=len(classifications),
        )

        return ctx.with_results({"classifications": classifications})

    def _distribute_equitably(
        self,
        detections: list[dict],
        species: list[str],
        sizes: dict[int, int],
    ) -> list[dict]:
        """Distribute detections equitably among species."""
        total = len(detections)
        num_species = len(species)
        base_count = total // num_species
        remainder = total % num_species

        classifications = []
        det_idx = 0

        for species_idx, species_name in enumerate(species):
            count = base_count + (1 if species_idx < remainder else 0)

            for _ in range(count):
                if det_idx >= total:
                    break

                det = detections[det_idx]
                classifications.append({
                    "detection_idx": det_idx,
                    "class_name": species_name,
                    "confidence": 1.0,
                    "product_size_id": sizes.get(det_idx),
                    "segment_idx": det.get("segment_idx", -1),
                    "image_id": det.get("image_id", ""),
                })
                det_idx += 1

        return classifications
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_steps/test_post_processors.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add app/steps/ tests/test_steps/
git commit -m "feat: add post-processor steps (filter, size, distributor)"
```

---

## Task 5: Create ML Steps (Wrappers)

**Files:**
- Create: `app/steps/ml/__init__.py`
- Create: `app/steps/ml/segmentation_step.py`
- Create: `app/steps/ml/detection_step.py`
- Create: `app/steps/ml/sahi_detection_step.py`
- Test: `tests/test_steps/test_ml_steps.py`

**Step 1: Write the failing test**

```python
# tests/test_steps/test_ml_steps.py
"""Tests for ML step wrappers."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from app.core.processing_context import ProcessingContext
from app.steps.ml.segmentation_step import SegmentationStep
from app.steps.ml.detection_step import DetectionStep
from app.steps.ml.sahi_detection_step import SAHIDetectionStep


@pytest.fixture
def base_context(temp_image_path: Path) -> ProcessingContext:
    return ProcessingContext(
        tenant_id="test-tenant",
        image_id="img-123",
        session_id="sess-456",
        image_path=temp_image_path,
        config={},
    )


class TestSegmentationStep:
    """Test SegmentationStep wrapper."""

    def test_name(self) -> None:
        assert SegmentationStep().name == "segmentation"

    @pytest.mark.asyncio
    async def test_executes_processor(self, base_context: ProcessingContext) -> None:
        mock_processor = MagicMock()
        mock_processor.process = AsyncMock(return_value=[
            MagicMock(to_dict=lambda: {"class_name": "cajon", "area_px": 100000})
        ])

        with patch(
            "app.steps.ml.segmentation_step.get_processor_registry"
        ) as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            result = await SegmentationStep().execute(base_context)

            assert len(result.raw_segments) == 1
            assert result.raw_segments[0]["class_name"] == "cajon"


class TestDetectionStep:
    """Test DetectionStep wrapper."""

    def test_name(self) -> None:
        assert DetectionStep().name == "detection"

    @pytest.mark.asyncio
    async def test_executes_processor(self, base_context: ProcessingContext) -> None:
        mock_processor = MagicMock()
        mock_processor.process = AsyncMock(return_value=[
            MagicMock(
                center_x_px=100, center_y_px=200,
                width_px=50, height_px=60,
                confidence=0.9, class_name="plant"
            )
        ])

        with patch(
            "app.steps.ml.detection_step.get_processor_registry"
        ) as mock_registry:
            mock_registry.return_value.get.return_value = mock_processor

            result = await DetectionStep().execute(base_context)

            assert len(result.raw_detections) == 1


class TestSAHIDetectionStep:
    """Test SAHIDetectionStep wrapper."""

    def test_name(self) -> None:
        assert SAHIDetectionStep().name == "sahi_detection"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_steps/test_ml_steps.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Create ML steps directory**

```bash
mkdir -p app/steps/ml
touch app/steps/ml/__init__.py
```

**Step 4: Write SegmentationStep**

```python
# app/steps/ml/segmentation_step.py
"""Segmentation ML step wrapper."""

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SegmentationStep(PipelineStep):
    """Wrapper around SegmentationProcessor."""

    @property
    def name(self) -> str:
        return "segmentation"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        registry = get_processor_registry()
        processor = registry.get("segmentation")

        segments = await processor.process(ctx.image_path)

        # Convert to dicts
        segment_dicts = [
            s.to_dict() if hasattr(s, "to_dict") else s.__dict__
            for s in segments
        ]

        logger.info(
            "Segmentation complete",
            segments=len(segment_dicts),
        )

        return ctx.with_segments(segment_dicts)
```

**Step 5: Write DetectionStep**

```python
# app/steps/ml/detection_step.py
"""Detection ML step wrapper."""

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


class DetectionStep(PipelineStep):
    """Wrapper around DetectorProcessor."""

    @property
    def name(self) -> str:
        return "detection"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        registry = get_processor_registry()
        processor = registry.get("detection")

        detections = await processor.process(ctx.image_path)

        # Convert to dicts
        detection_dicts = [
            {
                "center_x_px": d.center_x_px,
                "center_y_px": d.center_y_px,
                "width_px": d.width_px,
                "height_px": d.height_px,
                "confidence": d.confidence,
                "class_name": d.class_name,
                "segment_idx": getattr(d, "segment_idx", -1),
                "image_id": getattr(d, "image_id", ""),
            }
            for d in detections
        ]

        logger.info(
            "Detection complete",
            detections=len(detection_dicts),
        )

        return ctx.with_detections(detection_dicts)
```

**Step 6: Write SAHIDetectionStep**

```python
# app/steps/ml/sahi_detection_step.py
"""SAHI detection ML step wrapper."""

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SAHIDetectionStep(PipelineStep):
    """Wrapper around SAHIDetectorProcessor for large segments."""

    @property
    def name(self) -> str:
        return "sahi_detection"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        registry = get_processor_registry()
        processor = registry.get("sahi_detection")

        slice_height = ctx.config.get("sahi_slice_height", 512)
        slice_width = ctx.config.get("sahi_slice_width", 512)
        overlap_ratio = ctx.config.get("sahi_overlap_ratio", 0.25)

        detections = await processor.process(
            ctx.image_path,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_ratio=overlap_ratio,
        )

        # Convert to dicts
        detection_dicts = [
            {
                "center_x_px": d.center_x_px,
                "center_y_px": d.center_y_px,
                "width_px": d.width_px,
                "height_px": d.height_px,
                "confidence": d.confidence,
                "class_name": d.class_name,
            }
            for d in detections
        ]

        logger.info(
            "SAHI detection complete",
            detections=len(detection_dicts),
        )

        return ctx.with_detections(detection_dicts)
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_steps/test_ml_steps.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add app/steps/ml/
git commit -m "feat: add ML step wrappers (segmentation, detection, sahi)"
```

---

## Task 6: Register All Steps

**Files:**
- Modify: `app/steps/__init__.py`
- Test: `tests/test_steps/test_registration.py`

**Step 1: Write the failing test**

```python
# tests/test_steps/test_registration.py
"""Tests for step registration."""

import pytest

from app.core.step_registry import StepRegistry
from app.steps import register_all_steps


class TestStepRegistration:
    """Test that all steps are registered."""

    def setup_method(self) -> None:
        StepRegistry._steps.clear()
        register_all_steps()

    def test_ml_steps_registered(self) -> None:
        assert "segmentation" in StepRegistry.available_steps()
        assert "detection" in StepRegistry.available_steps()
        assert "sahi_detection" in StepRegistry.available_steps()

    def test_post_steps_registered(self) -> None:
        assert "segment_filter" in StepRegistry.available_steps()
        assert "size_calculator" in StepRegistry.available_steps()
        assert "species_distributor" in StepRegistry.available_steps()

    def test_can_build_full_pipeline(self) -> None:
        pipeline = StepRegistry.build_pipeline([
            "segmentation",
            "segment_filter",
            "detection",
            "size_calculator",
            "species_distributor",
        ])
        assert len(pipeline) == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_steps/test_registration.py -v`
Expected: FAIL

**Step 3: Write registration function**

```python
# app/steps/__init__.py
"""Pipeline steps registration."""

from app.core.step_registry import StepRegistry

from app.steps.ml.segmentation_step import SegmentationStep
from app.steps.ml.detection_step import DetectionStep
from app.steps.ml.sahi_detection_step import SAHIDetectionStep

from app.steps.post.segment_filter import SegmentFilterStep
from app.steps.post.size_calculator import SizeCalculatorStep
from app.steps.post.species_distributor import SpeciesDistributorStep


def register_all_steps() -> None:
    """Register all available pipeline steps."""
    # ML Steps
    StepRegistry.register("segmentation", SegmentationStep)
    StepRegistry.register("detection", DetectionStep)
    StepRegistry.register("sahi_detection", SAHIDetectionStep)

    # Post-Processor Steps
    StepRegistry.register("segment_filter", SegmentFilterStep)
    StepRegistry.register("size_calculator", SizeCalculatorStep)
    StepRegistry.register("species_distributor", SpeciesDistributorStep)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_steps/test_registration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add app/steps/__init__.py tests/test_steps/test_registration.py
git commit -m "feat: register all pipeline steps"
```

---

## Task 7: Database Migration

**Files:**
- Create: `migrations/versions/xxx_add_tenant_config.py` (or SQL file)

**Step 1: Create migration SQL**

```sql
-- migrations/add_tenant_config.sql

CREATE TABLE IF NOT EXISTS tenant_config (
    tenant_id VARCHAR(100) PRIMARY KEY,
    pipeline_steps JSONB NOT NULL,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_tenant_config_tenant_id ON tenant_config(tenant_id);

-- Example data for testing
INSERT INTO tenant_config (tenant_id, pipeline_steps, settings) VALUES
(
    'cultivos-demo',
    '["segmentation", "segment_filter", "detection", "size_calculator", "species_distributor"]',
    '{"num_bands": 4, "species": ["Tomato", "Pepper", "Lettuce"], "segment_filter_type": "largest_claro"}'
) ON CONFLICT (tenant_id) DO NOTHING;
```

**Step 2: Commit**

```bash
git add migrations/
git commit -m "feat: add tenant_config table migration"
```

---

## Task 8: Integrate with FastAPI Startup

**Files:**
- Modify: `app/main.py`

**Step 1: Add startup initialization**

```python
# Add to app/main.py

from app.core.tenant_config import get_tenant_cache
from app.steps import register_all_steps
from app.infra.database import get_db_session

@app.on_event("startup")
async def startup_event():
    # Register all pipeline steps
    register_all_steps()

    # Load tenant configs into cache
    cache = get_tenant_cache()
    async with get_db_session() as session:
        await cache.load_configs(session)

    # Start background refresh
    await cache.start_refresh_loop(get_db_session)

    logger.info("MLWorker started with tenant config cache")

@app.on_event("shutdown")
async def shutdown_event():
    cache = get_tenant_cache()
    await cache.stop()
```

**Step 2: Commit**

```bash
git add app/main.py
git commit -m "feat: initialize tenant config cache on startup"
```

---

## Task 9: Create New Unified Endpoint

**Files:**
- Modify: `app/api/routes/tasks.py`

**Step 1: Add new endpoint using pipeline system**

```python
# Add to app/api/routes/tasks.py

from app.core.tenant_config import get_tenant_cache
from app.core.step_registry import StepRegistry
from app.core.processing_context import ProcessingContext


@router.post(
    "/v2/process",
    response_model=ProcessingResponse,
    summary="Process image through tenant-configured pipeline",
)
async def process_task_v2(
    request: ProcessingRequest,
    storage: Storage,
    db: DbSession,
    _: CloudTasksRequest,
) -> ProcessingResponse:
    """Process image using dynamic pipeline per tenant."""
    start_time = time.time()
    local_path: Path | None = None

    try:
        # Get tenant config
        config = await get_tenant_cache().get(request.tenant_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No pipeline config for tenant: {request.tenant_id}",
            )

        # Download image
        local_path = await storage.download_to_tempfile(
            blob_path=request.image_url,
            tenant_id=request.tenant_id,
        )

        # Build pipeline
        steps = StepRegistry.build_pipeline(config.pipeline_steps)

        # Create context
        ctx = ProcessingContext(
            tenant_id=request.tenant_id,
            image_id=str(request.image_id),
            session_id=str(request.session_id),
            image_path=local_path,
            config=config.settings,
        )

        # Execute pipeline
        for step in steps:
            ctx = await step.execute(ctx)

        duration_ms = int((time.time() - start_time) * 1000)

        return ProcessingResponse(
            success=True,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            image_id=request.image_id,
            pipeline=",".join(config.pipeline_steps),
            results=ctx.results,
            duration_ms=duration_ms,
            steps_completed=len(config.pipeline_steps),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pipeline execution failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {e}",
        )
    finally:
        if local_path and local_path.exists():
            local_path.unlink(missing_ok=True)
```

**Step 2: Commit**

```bash
git add app/api/routes/tasks.py
git commit -m "feat: add /v2/process endpoint with dynamic pipeline"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Core interfaces | `pipeline_step.py`, `processing_context.py` |
| 2 | StepRegistry | `step_registry.py` |
| 3 | TenantConfigCache | `tenant_config.py` |
| 4 | Post-processor steps | `segment_filter.py`, `size_calculator.py`, `species_distributor.py` |
| 5 | ML step wrappers | `segmentation_step.py`, `detection_step.py`, `sahi_detection_step.py` |
| 6 | Step registration | `app/steps/__init__.py` |
| 7 | DB migration | `tenant_config` table |
| 8 | FastAPI startup | `main.py` |
| 9 | New endpoint | `/v2/process` |

**Estimated time:** 2-3 hours

**After completion:** Old endpoints remain for backwards compatibility. New `/v2/process` uses the configurable pipeline system.
