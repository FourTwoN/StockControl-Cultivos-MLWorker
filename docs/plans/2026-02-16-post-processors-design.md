# Post-Processors Architecture Design

**Date:** 2026-02-16
**Status:** Approved
**Author:** Claude + Franco

## Problem

El MLWorker tenía lógica de negocio específica por tenant mezclada con procesadores ML genéricos:
- Cálculo de tamaños con z-scores
- Distribución equitativa de especies
- Filtrado de segmentos "claro"

Esto hacía el código difícil de mantener y no escalable para múltiples tenants.

## Solution

Sistema de **Post-Processors** configurables por tenant que se intercalan con los procesadores ML en una cadena unificada.

## Design Decisions

| Decisión | Elección |
|----------|----------|
| Selección de post-processors | Por `tenant_id` |
| Ubicación de config | Base de datos (`tenant_config` table) |
| Carga de config | Cache al iniciar + refresh periódico (5 min) |
| Composición de processors | Mixto configurable (secuencial + paralelo) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MLWorker                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐     ┌──────────────────────────────────────┐ │
│  │  TenantConfigCache│────▶│  StepRegistry                        │ │
│  │  (refresh: 5min)  │     │  - ML Steps (segmentation, detect)   │ │
│  └──────────────────┘     │  - Post Steps (filter, sizer, dist)  │ │
│                            └──────────────────────────────────────┘ │
│                                           │                         │
│  ┌────────────────────────────────────────┼────────────────────────┐│
│  │              Dynamic Pipeline          ▼                        ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            ││
│  │  │ Step 1  │─▶│ Step 2  │─▶│ Step 3  │─▶│ Step N  │            ││
│  │  │ (ML/Post│  │ (ML/Post│  │ (ML/Post│  │ (ML/Post│            ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Core Interfaces

### PipelineStep (Base)

```python
class PipelineStep(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        pass
```

### ProcessingContext

```python
@dataclass
class ProcessingContext:
    tenant_id: str
    image_id: str
    session_id: str
    image_path: Path
    raw_segments: list[dict]
    raw_detections: list[dict]
    raw_classifications: list[dict]
    config: dict  # Tenant-specific settings
    results: dict[str, Any]  # Accumulated results
```

## Available Steps

### ML Steps (Generic)
| Step | Description |
|------|-------------|
| `segmentation` | Detecta segmentos (bbox, mask, class) |
| `detection` | Detecta objetos (bbox, confidence) |
| `sahi_detection` | Detección con tiling para imágenes grandes |
| `classification` | Clasificación raw (class, confidence) |

### Post-Processor Steps (Tenant-specific)
| Step | Description |
|------|-------------|
| `segment_filter` | Filtra segmentos (ej: solo el "claro" más grande) |
| `size_calculator` | Calcula tamaños S/M/L/XL con z-scores |
| `species_distributor` | Distribuye detecciones entre especies |

## Configuration

### Database Schema

```sql
CREATE TABLE tenant_config (
    tenant_id VARCHAR(100) PRIMARY KEY,
    pipeline_steps JSONB NOT NULL,
    settings JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Example
INSERT INTO tenant_config (tenant_id, pipeline_steps, settings) VALUES
(
    'cultivos-abc',
    '["segmentation", "segment_filter", "detection", "size_calculator", "species_distributor"]',
    '{"num_bands": 4, "species": ["Tomato", "Pepper"], "segment_filter_type": "largest_claro"}'
);
```

### TenantConfigCache

```python
class TenantConfigCache:
    def __init__(self, refresh_interval_seconds: int = 300):
        self._cache: dict[str, TenantPipelineConfig] = {}

    async def start(self, db_session) -> None:
        await self._load_all_configs(db_session)
        asyncio.create_task(self._refresh_loop(db_session))

    async def get(self, tenant_id: str) -> TenantPipelineConfig | None:
        return self._cache.get(tenant_id)
```

### StepRegistry

```python
class StepRegistry:
    _steps: dict[str, type[PipelineStep]] = {}

    @classmethod
    def register(cls, name: str, step_class: type[PipelineStep]) -> None:
        cls._steps[name] = step_class

    @classmethod
    def build_pipeline(cls, step_names: list[str]) -> list[PipelineStep]:
        return [cls.get(name) for name in step_names]
```

## Example Pipeline

**Tenant:** `cultivos-abc`

```
segmentation ─▶ segment_filter ─▶ detection ─▶ size_calculator ─▶ species_distributor
     │               │                │              │                    │
     ▼               ▼                ▼              ▼                    ▼
 [segments]    [filtered]       [detections]    [sizes]          [classifications]
```

## Endpoint Usage

```python
@router.post("/process")
async def process_task(request: ProcessingRequest, storage: Storage, db: DbSession):
    # 1. Get tenant config from cache
    config = await get_tenant_cache().get(request.tenant_id)
    if not config:
        raise HTTPException(404, "Tenant not configured")

    # 2. Build pipeline dynamically
    steps = StepRegistry.build_pipeline(config.pipeline_steps)

    # 3. Create initial context
    ctx = ProcessingContext(
        tenant_id=request.tenant_id,
        image_path=local_path,
        config=config.settings,
        results={},
    )

    # 4. Execute pipeline
    for step in steps:
        ctx = await step.execute(ctx)

    return ProcessingResponse(results=ctx.results)
```

## File Structure

```
app/
├── core/
│   ├── pipeline_step.py      # PipelineStep ABC
│   ├── processing_context.py # ProcessingContext dataclass
│   ├── step_registry.py      # StepRegistry
│   └── tenant_config.py      # TenantConfigCache
├── steps/
│   ├── __init__.py
│   ├── ml/
│   │   ├── segmentation_step.py
│   │   ├── detection_step.py
│   │   ├── sahi_detection_step.py
│   │   └── classification_step.py
│   └── post/
│       ├── segment_filter_step.py
│       ├── size_calculator_step.py
│       └── species_distributor_step.py
```

## Benefits

1. **Separation of concerns** - ML puro vs lógica de negocio
2. **Extensibility** - Agregar nuevos post-processors sin tocar código existente
3. **Per-tenant customization** - Cada tenant tiene su pipeline
4. **Testability** - Cada step se testea independientemente
5. **No spaghetti** - Cadena clara y predecible

## Next Steps

1. Crear tabla `tenant_config` en DB
2. Implementar `PipelineStep`, `ProcessingContext`
3. Implementar `StepRegistry`, `TenantConfigCache`
4. Migrar lógica existente a steps
5. Refactorizar endpoints para usar nuevo sistema
6. Tests para cada step
