# StockControl MLWorker

ML processing microservice for the cultivation industry. Runs segmentation, detection, and classification pipelines on plant images.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MLWorker                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐     ┌──────────────────────────────────────┐ │
│  │  TenantConfigCache│────▶│  StepRegistry                        │ │
│  │  (DB + 5min cache)│     │  - ML Steps (generic)                │ │
│  └──────────────────┘     │  - Post Steps (tenant-specific)      │ │
│                            └──────────────────────────────────────┘ │
│                                           │                         │
│  ┌────────────────────────────────────────┼────────────────────────┐│
│  │              Dynamic Pipeline          ▼                        ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            ││
│  │  │ Step 1  │─▶│ Step 2  │─▶│ Step 3  │─▶│ Step N  │            ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Pipeline Steps

All steps implement `PipelineStep` interface and can be composed in any order:

**ML Steps (generic, reusable):**
- `segmentation` - Detect segments (cajon, segmento)
- `detection` - Standard YOLO detection
- `sahi_detection` - Tiled detection for large images

**Post-Processor Steps (tenant-configurable):**
- `segment_filter` - Filter segments (e.g., keep largest claro)
- `size_calculator` - Calculate sizes with z-scores
- `species_distributor` - Distribute detections across species

### Tenant Configuration

Pipeline config stored in `tenant_config` table:
```sql
SELECT tenant_id, pipeline_steps, settings FROM tenant_config;
-- ('cultivos-abc', '["segmentation", "segment_filter", "detection"]', '{"species": [...]}')
```

Loaded into `TenantConfigCache` at startup, refreshed every 5 minutes.

## Directory Structure

```
app/
├── api/routes/          # FastAPI endpoints
│   ├── tasks.py         # /process, /v2/process, /compress
│   └── health.py        # Health checks
├── core/                # Core abstractions
│   ├── pipeline_step.py      # PipelineStep ABC
│   ├── processing_context.py # Immutable context
│   ├── step_registry.py      # Step registration
│   ├── tenant_config.py      # Config cache
│   └── processor_registry.py # ML processor registry
├── steps/               # Pipeline steps
│   ├── ml/              # ML step wrappers
│   │   ├── segmentation_step.py
│   │   ├── detection_step.py
│   │   └── sahi_detection_step.py
│   └── post/            # Post-processors
│       ├── segment_filter.py
│       ├── size_calculator.py
│       └── species_distributor.py
├── processors/          # Raw ML processors
│   ├── base_processor.py
│   ├── segmentation_processor.py
│   ├── detector_processor.py
│   ├── sahi_detector_processor.py
│   └── classifier_processor.py
├── services/            # Business logic
├── infra/               # Infrastructure (DB, storage, logging)
└── ml/                  # Model loading and caching
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /tasks/process` | Legacy endpoint (uses industry config) |
| `POST /tasks/v2/process` | **Preferred** - Dynamic pipeline per tenant |
| `POST /tasks/compress` | Image compression and thumbnails |
| `GET /health` | Health check |

## Adding a New Post-Processor

1. Create step in `app/steps/post/my_step.py`:
```python
from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext

class MyStep(PipelineStep):
    @property
    def name(self) -> str:
        return "my_step"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        # Process and return new context
        return ctx.with_results({"my_key": result})
```

2. Register in `app/steps/__init__.py`:
```python
from app.steps.post.my_step import MyStep
StepRegistry.register("my_step", MyStep)
```

3. Add to tenant config in DB:
```sql
UPDATE tenant_config
SET pipeline_steps = '["segmentation", "my_step", "detection"]'
WHERE tenant_id = 'tenant-xyz';
```

## Development

### Run locally
```bash
uvicorn app.main:app --reload --port 8080
```

### Run tests
```bash
pytest tests/ -v
pytest tests/test_steps/ -v  # Step tests only
```

### Environment Variables
```bash
DATABASE_URL=postgresql+asyncpg://...
GCS_BUCKET=ml-worker-bucket
ENVIRONMENT=development
```

## Conventions

- **Immutability**: `ProcessingContext` is frozen - always return new instance
- **Processors vs Steps**: Processors do raw ML inference, Steps wrap them for pipeline
- **No tenant logic in processors**: Keep ML code generic, put tenant logic in post-processors
- **TDD**: Write tests first, especially for new steps

## Database

### tenant_config table
```sql
CREATE TABLE tenant_config (
    tenant_id VARCHAR(100) PRIMARY KEY,
    pipeline_steps JSONB NOT NULL,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Common Tasks

### Add new ML model
1. Add processor in `app/processors/`
2. Register in `app/core/processor_registry.py`
3. Create step wrapper in `app/steps/ml/`
4. Register step in `app/steps/__init__.py`

### Debug pipeline execution
```python
# In endpoint or service
for step in steps:
    logger.info(f"Executing step: {step.name}")
    ctx = await step.execute(ctx)
    logger.info(f"Context results: {ctx.results.keys()}")
```

### Refresh tenant config manually
```python
cache = get_tenant_cache()
async with get_db_session() as session:
    await cache.load_configs(session)
```
