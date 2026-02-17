# StockControl MLWorker

ML processing microservice for the cultivation industry. Runs segmentation, detection, and classification pipelines on plant images.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MLWorker                                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐     ┌──────────────────────────────────────┐ │
│  │  TenantConfigCache│────▶│  PipelineParser                      │ │
│  │  (DB + 5min cache)│     │  JSON → DSL → Executable             │ │
│  └──────────────────┘     └──────────────────────────────────────┘ │
│                                           │                         │
│  ┌────────────────────────────────────────┼────────────────────────┐│
│  │              PipelineExecutor          ▼                        ││
│  │  ┌─────────┐  ┌───────────────────────┐  ┌─────────┐           ││
│  │  │ Chain   │─▶│ Chord(Group,Callback) │─▶│ Step N  │           ││
│  │  └─────────┘  │ ┌─────┐ ┌─────┐       │  └─────────┘           ││
│  │               │ │ Br1 │ │ Br2 │ → Agg │                        ││
│  │               │ └─────┘ └─────┘       │                        ││
│  │               └───────────────────────┘                        ││
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

**Post-Processor Steps (tenant-configurable via settings):**
- `segment_filter` - Filter segments by `class_name` and generate crops for parallel detection
- `aggregate_detections` - Merge detections from parallel branches (chord callback)
- `size_calculator` - Calculate sizes with z-scores (thresholds: S < -2σ, M ≤ 1σ, L ≤ 2σ, XL > 2σ)
- `species_distributor` - Distribute detections across species
- `visualize_detections` - Draw detection boxes on image
- `upload_image` - Upload image to GCS with thumbnail generation (reusable for original/processed)

### Pipeline DSL (Celery Canvas-inspired)

The system uses a serializable DSL stored as JSONB in the database. Supports:

**Primitives:**
- `chain` - Sequential execution
- `group` - Parallel execution with `asyncio.gather`
- `chord` - Parallel + aggregator callback
- `step` - Step reference with config (kwargs)

**Example tenant_config in DB:**
```json
{
  "pipeline_definition": {
    "type": "chain",
    "steps": [
      {"type": "step", "name": "upload_image", "kwargs": {
        "source": "original", "dest_prefix": "originals", "thumbnail_sizes": [256, 512, 1024]
      }},
      {"type": "step", "name": "segmentation"},
      {"type": "step", "name": "segment_filter"},
      {
        "type": "chord",
        "group": {
          "type": "group",
          "steps": [
            {"type": "step", "name": "sahi_detection", "kwargs": {"segment_type": "segmento"}},
            {"type": "step", "name": "detection", "kwargs": {"segment_type": "cajon"}}
          ]
        },
        "callback": {"type": "step", "name": "aggregate_detections"}
      },
      {"type": "step", "name": "size_calculator"},
      {"type": "step", "name": "visualize_detections"},
      {"type": "step", "name": "upload_image", "kwargs": {
        "source": "processed", "dest_prefix": "processed", "thumbnail_sizes": [256, 512]
      }}
    ]
  },
  "settings": {
    "segment_filter_type": "largest_by_class",
    "segment_filter_classes": ["segmento", "cajon"],
    "species": ["species_a", "species_b"]
  }
}
```

**`upload_image` step kwargs:**
| Param | Description | Default |
|-------|-------------|---------|
| `source` | `"original"` or `"processed"` | `"original"` |
| `dest_prefix` | GCS folder (e.g., `"originals"`, `"processed"`) | `"images"` |
| `thumbnail_sizes` | List of thumbnail sizes in pixels | `[256, 512]` |
| `quality` | JPEG quality 1-100 | `85` |
| `skip_if_missing` | Don't fail if source missing | `false` |

**GCS structure after upload:**
```
gs://{bucket}/{tenant_id}/
├── originals/{image_id}.jpg
├── originals_thumbnails/{image_id}_256.jpg
├── originals_thumbnails/{image_id}_512.jpg
├── processed/{image_id}.jpg
└── processed_thumbnails/{image_id}_256.jpg
```

**Execution flow:**
1. `TenantConfigCache` loads `pipeline_definition` (JSONB) from DB
2. `PipelineParser` validates JSON structure (Pydantic) and step names (StepRegistry)
3. `PipelineParser.parse()` converts JSON → DSL dataclasses (Chain, Group, Chord)
4. `PipelineExecutor.execute()` runs the pipeline with proper parallelism

### Tenant Configuration

Pipeline config stored in `tenant_config` table as JSON DSL:
```sql
SELECT tenant_id, pipeline_definition, settings FROM tenant_config;
```

Loaded into `TenantConfigCache` at startup, refreshed every 5 minutes.
**Validation is fail-fast:** Invalid pipeline definitions are rejected at load time.

## Directory Structure

```
app/
├── api/routes/          # FastAPI endpoints
│   ├── tasks.py         # /process, /compress
│   └── health.py        # Health checks
├── core/                # Core abstractions
│   ├── pipeline_step.py      # PipelineStep ABC
│   ├── pipeline_dsl.py       # DSL primitives (chain, group, chord)
│   ├── pipeline_parser.py    # JSON → DSL parser with validation
│   ├── pipeline_executor.py  # DSL executor with asyncio.gather
│   ├── processing_context.py # Immutable context
│   ├── step_registry.py      # Step registration
│   ├── tenant_config.py      # Config cache
│   └── processor_registry.py # ML processor registry
├── schemas/             # Pydantic schemas
│   └── pipeline_definition.py  # PipelineDefinition, StepDefinition, etc.
├── steps/               # Pipeline steps
│   ├── ml/              # ML step wrappers
│   │   ├── segmentation_step.py
│   │   ├── detection_step.py
│   │   └── sahi_detection_step.py
│   └── post/            # Post-processors
│       ├── segment_filter.py
│       ├── aggregate_detections.py
│       ├── size_calculator.py
│       └── species_distributor.py
├── processors/          # Raw ML processors
│   ├── base_processor.py
│   ├── segmentation_processor.py
│   ├── detector_processor.py
│   └── sahi_detector_processor.py
├── infra/               # Infrastructure (DB, storage, logging)
└── ml/                  # Model loading and caching
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /tasks/process` | Dynamic pipeline per tenant (uses TenantConfigCache) |
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

3. Add to tenant pipeline_definition in DB:
```sql
UPDATE tenant_config
SET pipeline_definition = '{
  "type": "chain",
  "steps": [
    {"type": "step", "name": "segmentation"},
    {"type": "step", "name": "my_step"},
    {"type": "step", "name": "detection"}
  ]
}'::jsonb
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
pytest tests/test_core/test_pipeline_parser.py -v  # Parser tests
pytest tests/test_schemas/test_pipeline_definition.py -v  # Schema tests
```

### Environment Variables
```bash
DATABASE_URL=postgresql+asyncpg://...
GCS_BUCKET=ml-worker-bucket
ENVIRONMENT=dev
USE_LOCAL_STORAGE=true        # Use local filesystem instead of GCS
LOCAL_STORAGE_ROOT=./local_storage  # Where to store/read images locally
```

### Local Storage (Development)
For development without GCS, set `USE_LOCAL_STORAGE=true`:
```bash
# Place test images in local_storage/
local_storage/
└── tenant-001/
    └── images/
        └── test.jpg

# The worker will read from local filesystem instead of GCS
```

## Conventions

- **Immutability**: `ProcessingContext` is frozen - always return new instance
- **Processors vs Steps**: Processors do raw ML inference, Steps wrap them for pipeline
- **No tenant logic in processors**: Keep ML code generic, put tenant logic in post-processors
- **No hardcoded pipelines**: All pipelines defined in DB as JSON DSL
- **TDD**: Write tests first, especially for new steps

## Database

### tenant_config table
```sql
CREATE TABLE tenant_config (
    tenant_id VARCHAR(100) PRIMARY KEY,
    pipeline_definition JSONB NOT NULL,  -- DSL as JSON
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
# PipelineExecutor logs each step
# Check logs for: "Executing step", "Executing chain", "Executing chord"
```

### Refresh tenant config manually
```python
cache = get_tenant_cache()
async with get_db_session() as session:
    await cache.load_configs(session)
```

### Validate pipeline definition
```python
from app.core.pipeline_parser import PipelineParser
from app.core.step_registry import StepRegistry
from app.schemas.pipeline_definition import PipelineDefinition

# From JSON dict
definition = PipelineDefinition.model_validate(json_dict)

# Parse and validate steps exist
parser = PipelineParser(StepRegistry)
pipeline = parser.parse(definition)  # Raises PipelineParserError if invalid
```
