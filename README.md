# StockControl-Cultivos-MLWorker

ML Worker for StockControl-Cultivos project.

# Demeter AI 2.0 — ML Worker

**Background ML processing service for photo analysis with Cloud Tasks orchestration.**

Python 3.11 · FastAPI · YOLO/Ultralytics · Cloud Tasks · Cloud Run

---

## What is ML Worker?

ML Worker is a background processing service that executes ML inference (detection, segmentation, classification, estimation) on uploaded photos. It receives HTTP requests from **Google Cloud Tasks** and processes them asynchronously, decoupling heavy ML computation from the synchronous Backend API.

### How it fits in the Demeter ecosystem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEMETER AI 2.0                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐   │
│  │   Frontend  │────▶│   Backend   │────▶│       Cloud Tasks           │   │
│  │   (React)   │     │  (Quarkus)  │     │   {industry}-ml-tasks       │   │
│  └─────────────┘     └─────────────┘     └──────────────┬──────────────┘   │
│                                                          │                  │
│                                                          ▼                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────────┐   │
│  │   Cloud     │◀────│  ML Worker  │◀────│     OIDC Token Auth         │   │
│  │   Storage   │     │  (FastAPI)  │     │   (Cloud Run IAM)           │   │
│  │  (images)   │     │             │     └─────────────────────────────┘   │
│  └─────────────┘     └──────┬──────┘                                       │
│                             │                                               │
│                             ▼                                               │
│                      ┌─────────────┐                                       │
│                      │  Cloud SQL  │                                       │
│                      │ (results)   │                                       │
│                      └─────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Cloud Tasks instead of Celery/Redis?

| Aspect | Celery + Redis | Cloud Tasks |
|--------|---------------|-------------|
| **Infrastructure** | Manage Redis cluster, Celery workers | Fully managed, serverless |
| **Scaling** | Manual worker scaling | Automatic scaling (0 to N) |
| **Cost** | Pay for always-on Redis | Pay per task execution |
| **Reliability** | Manual retry configuration | Built-in retry with exponential backoff |
| **Monitoring** | Custom setup | Cloud Monitoring integration |

---

## Architecture

### Pipeline Orchestrator Pattern

ML Worker uses a **configuration-driven pipeline orchestrator** that routes tasks through different ML processors based on the requested `processing_type`. The same code serves all tenants — only models and configuration differ per industry.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Cloud Tasks                                        │
│                    {industry}-ml-tasks (single queue)                        │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
    ┌───────────────────────────────────────────────────────────────────────┐
    │                      ML Worker Service                                 │
    │                                                                        │
    │  ┌──────────────────────────────────────────────────────────────────┐ │
    │  │                    Pipeline Orchestrator                          │ │
    │  │  • Receives task request                                          │ │
    │  │  • Loads industry config from GCS                                 │ │
    │  │  • Determines pipeline steps based on processing_type             │ │
    │  │  • Coordinates processor execution                                │ │
    │  │  • Aggregates results and saves to DB                             │ │
    │  └──────────────────────────────────────────────────────────────────┘ │
    │                              │                                         │
    │                              ▼                                         │
    │  ┌──────────────────────────────────────────────────────────────────┐ │
    │  │                   Processor Registry                              │ │
    │  │  (loaded dynamically based on pipeline config)                    │ │
    │  │                                                                    │ │
    │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐     │ │
    │  │  │ detection  │ │segmentation│ │classification│ │ estimation │     │ │
    │  │  │ processor  │ │ processor  │ │  processor  │ │ processor  │     │ │
    │  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘     │ │
    │  │                                                                    │ │
    │  └──────────────────────────────────────────────────────────────────┘ │
    └───────────────────────────────────────────────────────────────────────┘
                                   │
    ┌──────────────────────────────┴──────────────────────────────────────┐
    │                    Model Registry (GCS)                              │
    │  gs://demeter-models/                                                │
    │    ├── agro/                                                         │
    │    │   ├── detect.pt        # YOLO trained for plant detection       │
    │    │   ├── segment.pt       # Segmentation for field/row extraction  │
    │    │   └── config.yaml      # Thresholds, classes, pipelines         │
    │    │                                                                  │
    │    └── vending/                                                       │
    │        ├── detect.pt        # YOLO trained for product detection     │
    │        ├── segment.pt       # Shelf/section segmentation             │
    │        ├── classify.pt      # Product classification model           │
    │        └── config.yaml      # Different thresholds and pipelines     │
    └──────────────────────────────────────────────────────────────────────┘
```

### Industry Configuration

Each industry has its own `config.yaml` that defines:
- **Models**: Which models to load and their parameters
- **Pipelines**: Named sequences of processors for different use cases
- **Classes**: What the models detect/classify

```yaml
# gs://demeter-models/agro/config.yaml
industry: agro
version: "1.2.0"

models:
  detection:
    path: detect.pt
    confidence_threshold: 0.80
    classes: ["plant", "weed", "pest"]

  segmentation:
    path: segment.pt
    confidence_threshold: 0.50
    classes: ["field", "row", "cajon"]

  classification:
    enabled: false  # Not used in agro

pipelines:
  DETECTION:
    steps: [detection]

  FULL_PIPELINE:
    steps: [segmentation, detection, estimation]

  FIELD_ANALYSIS:
    steps: [segmentation, estimation]
```

```yaml
# gs://demeter-models/vending/config.yaml
industry: vending
version: "1.0.0"

models:
  detection:
    path: detect.pt
    confidence_threshold: 0.70
    classes: ["product", "empty_shelf", "price_tag"]

  segmentation:
    path: segment.pt
    confidence_threshold: 0.60
    classes: ["shelf", "section", "cooler"]

  classification:
    enabled: true
    path: classify.pt
    classes: ["coca_cola", "pepsi", "sprite", "water", ...]

pipelines:
  DETECTION:
    steps: [detection]

  FULL_PIPELINE:
    steps: [segmentation, detection, classification, estimation]

  SHELF_AUDIT:
    steps: [segmentation, detection, classification]
```

### Multi-Tenant Isolation

Every task includes a `tenant_id`. Isolation is enforced at three levels:

| Layer | Mechanism | What it does |
|-------|-----------|-------------|
| **Request** | Task payload validation | Extracts `tenant_id` from Cloud Tasks request body |
| **Storage** | Path prefix validation | Ensures image URLs match `gs://bucket/{tenant_id}/...` |
| **Database** | PostgreSQL RLS | `SET LOCAL app.current_tenant = '{tenant_id}'` on every connection |

The **same code** and **same models** serve all tenants within an industry. No tenant-specific code paths exist.

---

## Project Structure

```
StockControl-MLWorker/
├── pyproject.toml                      # Dependencies and build config
├── Dockerfile                          # Multi-stage build for Cloud Run
├── README.md
│
├── deploy/
│   └── terraform/                      # All infrastructure as code
│       ├── main.tf                     # Provider, APIs
│       ├── versions.tf                 # Terraform/provider versions
│       ├── variables.tf                # Input variables
│       ├── outputs.tf                  # Service URLs, queue names
│       ├── cloudtasks.tf               # Task queues
│       ├── cloudrun.tf                 # Cloud Run service
│       ├── iam.tf                      # Service accounts
│       ├── secrets.tf                  # Secret Manager
│       └── terraform.tfvars.example
│
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI app with lifespan
│   ├── config.py                       # Pydantic Settings
│   │
│   ├── core/                           # Core abstractions
│   │   ├── __init__.py
│   │   ├── pipeline_step.py            # PipelineStep ABC
│   │   ├── processing_context.py       # Immutable context
│   │   ├── step_registry.py            # Step registration
│   │   ├── tenant_config.py            # Tenant config cache (DB)
│   │   └── processor_registry.py       # ML processor registry
│   │
│   ├── steps/                          # Pipeline steps
│   │   ├── ml/                         # ML step wrappers
│   │   │   ├── segmentation_step.py
│   │   │   ├── detection_step.py
│   │   │   └── sahi_detection_step.py
│   │   └── post/                       # Post-processors
│   │       ├── segment_filter.py
│   │       ├── size_calculator.py
│   │       └── species_distributor.py
│   │
│   ├── processors/                     # ML processors (generic)
│   │   ├── __init__.py
│   │   ├── base_processor.py           # Abstract base class
│   │   ├── detector_processor.py       # YOLO detection
│   │   ├── segmentation_processor.py   # YOLO segmentation
│   │   ├── classifier_processor.py     # Classification
│   │   └── estimator_processor.py      # Counting/estimation
│   │
│   ├── ml/                             # Model management
│   │   ├── __init__.py
│   │   ├── model_cache.py              # Singleton model cache
│   │   └── model_registry.py           # High-level model access
│   │
│   ├── api/                            # FastAPI routes
│   │   ├── __init__.py
│   │   ├── deps.py                     # Dependencies (DB, storage)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py               # Health checks
│   │       └── tasks.py                # POST /tasks/process (unified)
│   │
│   ├── schemas/                        # Pydantic models
│   │   ├── __init__.py
│   │   ├── common.py                   # TaskResult, Error
│   │   └── task.py                     # ProcessingRequest/Response
│   │
│   └── infra/                          # Infrastructure adapters
│       ├── __init__.py
│       ├── database.py                 # Async SQLAlchemy + RLS
│       ├── storage.py                  # GCS with tenant validation
│       └── logging.py                  # structlog for Cloud Logging
│
└── tests/
    ├── conftest.py
    ├── test_api/
    └── test_processors/
```

### Design Decisions

#### 1. Unified Task Endpoint

Instead of separate endpoints per task type (`/tasks/process-photo`, `/tasks/compress`), we use a **single endpoint** with the pipeline defined in the request:

```python
# POST /tasks/process
{
  "tenant_id": "gobar",
  "session_id": "...",
  "image_id": "...",
  "image_url": "gs://bucket/gobar/originals/photo.jpg",
  "pipeline": "FULL_PIPELINE"  # or "DETECTION", "SHELF_AUDIT", etc.
}
```

**Why?** Pipeline definitions live in the industry config YAML, not hardcoded in route handlers. Adding a new pipeline is a config change, not a code change.

#### 2. Configuration-Driven Model Loading

Models are loaded based on the industry config, not environment variables:

```python
# Industry config determines which models to load
config = IndustryConfig.load(industry="agro")

# ModelCache uses config to find model paths
model = ModelCache.get_model(
    model_type="detect",
    config=config.models["detection"]
)
```

**Why?** Different industries may need different models, confidence thresholds, and classes. The config YAML captures all of this.

#### 3. Processors Are Generic

Processors don't know about industries. They receive:
- An image path
- Configuration parameters (confidence threshold, etc.)

```python
class DetectorProcessor:
    async def process(
        self,
        image_path: Path,
        confidence: float = 0.80,
        classes: list[str] | None = None,
    ) -> list[DetectionResult]:
        # Pure ML inference, no business logic
```

**Why?** Same processor code works for agro (detecting plants) and vending (detecting products). Only the model weights and config differ.

#### 4. Pipeline Orchestrator Pattern

The endpoint orchestrates the pipeline directly using `TenantConfigCache` and `StepRegistry`:

```python
async def process_task(request: ProcessingRequest, ...) -> ProcessingResponse:
    # 1. Get tenant config from cache
    config = await get_tenant_cache().get(request.tenant_id)

    # 2. Download image
    local_path = await storage.download_to_tempfile(request.image_url, request.tenant_id)

    # 3. Build pipeline dynamically from tenant config
    steps = StepRegistry.build_pipeline(config.pipeline_steps)

    # 4. Create immutable context
    ctx = ProcessingContext(
        tenant_id=request.tenant_id,
        image_path=local_path,
        config=config.settings,
    )

    # 5. Execute pipeline steps
    for step in steps:
        ctx = await step.execute(ctx)

    return ProcessingResponse(success=True, results=ctx.results)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for local development)
- GCP project (for deployment)

### Local Development

```bash
# 1. Clone and install
cd StockControl-MLWorker
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with local values

# 3. Run locally
uvicorn app.main:app --reload --port 8080

# 4. Test health endpoint
curl http://localhost:8080/health
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment name (dev, staging, prod) | `dev` |
| `INDUSTRY` | Industry identifier (agro, vending) | `agro` |
| `GCS_BUCKET` | Cloud Storage bucket for images | - |
| `MODEL_PATH` | Path to models (local or gs://) | `./models` |
| `DB_CONNECTION_NAME` | Cloud SQL connection name | - |
| `DB_USER` | Database user | `demeter_app` |
| `DB_PASSWORD` | Database password | - |
| `DB_NAME` | Database name | `demeter` |

---

## Deployment (Google Cloud Platform)

### Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GCP Cloud Platform                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Cloud Tasks                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │ agro-ml-tasks   │  │ agro-compress   │  │ agro-reports    │     │   │
│  │  │ (10 req/s, 5cc) │  │ (20 req/s, 10cc)│  │ (5 req/s, 3cc)  │     │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │   │
│  └───────────┼────────────────────┼────────────────────┼───────────────┘   │
│              │                    │                    │                    │
│              └────────────────────┼────────────────────┘                    │
│                                   │                                         │
│                                   ▼ OIDC Token                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Cloud Run                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              demeter-mlworker-prod                           │   │   │
│  │  │                                                              │   │   │
│  │  │  CPU: 4 vCPU    Memory: 8GB    Timeout: 30min               │   │   │
│  │  │  Min: 0         Max: 10        GPU: Optional (T4)           │   │   │
│  │  │                                                              │   │   │
│  │  │  ┌────────────────────────────────────────────────────────┐ │   │   │
│  │  │  │ ENV:                                                    │ │   │   │
│  │  │  │   INDUSTRY=agro                                         │ │   │   │
│  │  │  │   GCS_BUCKET=demeter-images-prod                        │ │   │   │
│  │  │  │   MODEL_PATH=gs://demeter-models/agro                   │ │   │   │
│  │  │  │   DB_USER, DB_PASSWORD, DB_NAME (from Secret Manager)   │ │   │   │
│  │  │  └────────────────────────────────────────────────────────┘ │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                   │                                         │
│           ┌───────────────────────┼───────────────────────┐                │
│           ▼                       ▼                       ▼                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │
│  │  Cloud Storage  │  │    Cloud SQL    │  │ Secret Manager  │           │
│  │                 │  │                 │  │                 │           │
│  │  Images bucket  │  │  PostgreSQL 17  │  │  DB credentials │           │
│  │  Models bucket  │  │  (shared with   │  │  API keys       │           │
│  │                 │  │   Backend)      │  │                 │           │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cloud Run Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| **CPU** | 4 vCPU | ML inference is CPU-intensive |
| **Memory** | 8 GB | YOLO models require ~2-4GB each |
| **Timeout** | 30 min | Large images may take time |
| **Min instances** | 0 | Scale to zero for cost savings |
| **Max instances** | 10 | Limit concurrent processing |
| **Concurrency** | 1 | One request per instance (ML is blocking) |

### Cloud Tasks Queue Configuration

| Queue | Rate | Concurrent | Retry | Use Case |
|-------|------|------------|-------|----------|
| `{industry}-ml-tasks` | 10/s | 5 | 5 attempts, 10s→300s backoff | ML inference |
| `{industry}-compress` | 20/s | 10 | 3 attempts, 5s→60s backoff | Thumbnails |
| `{industry}-reports` | 5/s | 3 | 3 attempts, 30s→600s backoff | Report generation |

### Service Accounts

| Account | Purpose | Roles |
|---------|---------|-------|
| `cloudtasks-invoker` | Backend creates tasks | `roles/cloudtasks.enqueuer` |
| `mlworker-runner` | ML Worker execution | `roles/run.invoker`, `roles/cloudsql.client`, `roles/storage.objectAdmin`, `roles/secretmanager.secretAccessor` |

### Deploy with Terraform

```bash
cd deploy/terraform

# 1. Configure variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars

# 2. Initialize and apply
terraform init
terraform plan
terraform apply
```

### Deploy Application

```bash
# Build and push image
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/demeter/mlworker:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/demeter/mlworker:latest

# Deploy to Cloud Run
gcloud run deploy demeter-mlworker-prod \
  --image us-central1-docker.pkg.dev/PROJECT_ID/demeter/mlworker:latest \
  --region us-central1
```

---

## API Reference

### POST /tasks/process

Process an image through the ML pipeline.

**Request:**
```json
{
  "tenant_id": "gobar",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_id": "660e8400-e29b-41d4-a716-446655440001",
  "image_url": "gs://demeter-images/gobar/originals/photo.jpg",
  "pipeline": "FULL_PIPELINE",
  "options": {
    "confidence_override": 0.85
  }
}
```

**Response:**
```json
{
  "success": true,
  "tenant_id": "gobar",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "image_id": "660e8400-e29b-41d4-a716-446655440001",
  "pipeline": "FULL_PIPELINE",
  "results": {
    "segmentation": {
      "segments": [...]
    },
    "detection": {
      "detections": [...]
    },
    "estimation": {
      "total_count": 42,
      "coverage_percent": 78.5
    }
  },
  "duration_ms": 2340,
  "processed_at": "2024-01-15T10:30:00Z"
}
```

### GET /health

Basic health check (startup probe).

### GET /health/ready

Readiness check (verifies DB and model availability).

### GET /health/live

Liveness check (basic response).

---

## Adding a New Industry

1. **Create models**: Train YOLO models for the new industry and upload to GCS:
   ```
   gs://demeter-models/new-industry/
     ├── detect.pt
     ├── segment.pt
     └── config.yaml
   ```

2. **Create config.yaml**:
   ```yaml
   industry: new-industry
   version: "1.0.0"

   models:
     detection:
       path: detect.pt
       confidence_threshold: 0.75
       classes: ["item_a", "item_b", "item_c"]

   pipelines:
     DETECTION:
       steps: [detection]
     FULL_PIPELINE:
       steps: [detection, estimation]
   ```

3. **Deploy a new ML Worker instance**:
   ```bash
   terraform apply -var="industry=new-industry"
   ```

4. **Configure Backend** to create tasks in the new queue.

No code changes required in ML Worker.

---

## Adding a New Pipeline

1. **Define the pipeline** in the industry's `config.yaml`:
   ```yaml
   pipelines:
     # ... existing pipelines ...

     NEW_PIPELINE:
       steps: [segmentation, detection, classification]
   ```

2. **Upload the updated config** to GCS.

3. **Use the new pipeline** in task requests:
   ```json
   {
     "pipeline": "NEW_PIPELINE",
     ...
   }
   ```

No code changes or redeployment required.

---

## Cloud Storage Integration

ML Worker downloads images from and uploads results to **Cloud Storage** using a provider-agnostic interface.

### Storage Structure

```
gs://stockcontrol-images/
└── {industry}/                    # e.g., cultivadores/
    └── {tenant_id}/               # e.g., cactus-mendoza/
        ├── sessions/
        │   └── {session_id}/
        │       ├── original/      # Source images (downloaded)
        │       ├── processed/     # ML output images (uploaded)
        │       ├── thumbnails/    # Generated thumbnails
        │       └── web/           # Web-optimized versions
        └── products/
```

### Image URLs

The ML Worker receives `gs://` URLs in task payloads:

```json
{
  "tenant_id": "cactus-mendoza",
  "session_id": "...",
  "image_id": "...",
  "image_url": "gs://stockcontrol-images/cultivadores/cactus-mendoza/sessions/.../original/photo.jpg",
  "pipeline": "SEGMENT_DETECT"
}
```

### Tenant Path Validation

Images are validated to ensure they belong to the requesting tenant:

```python
# TenantPathError raised if image_url doesn't match tenant_id
if not image_url.contains(f"/{tenant_id}/"):
    raise TenantPathError("Path does not belong to tenant")
```

---

## Artifact Registry

Docker images are organized by **industry**:

```
us-central1-docker.pkg.dev/PROJECT/
└── cultivadores/               # Industry-specific repo
    ├── mlworker:v1.0.0
    ├── mlworker:latest
    └── ...
```

### CPU-Only Optimization

The Docker image is optimized for **CPU-only inference** to reduce size:

```dockerfile
# Install CPU-only PyTorch (no CUDA dependencies)
RUN uv pip install --no-cache torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

**Image sizes:**
- With CUDA: ~8GB
- CPU-only: ~1.7GB

### Models Embedded in Image

ML models are embedded in the Docker image (not downloaded at runtime):

```dockerfile
COPY models/ /app/models/
```

This ensures:
- Faster cold starts (no model download)
- Consistent model versions per image tag
- Industry-specific models per image

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Framework | FastAPI 0.115 |
| ML Runtime | Ultralytics 8.3 (YOLO) |
| ML Framework | PyTorch 2.5 (CPU-only) |
| Database | Async SQLAlchemy + asyncpg |
| Storage | google-cloud-storage |
| Task Queue | Google Cloud Tasks |
| Compute | Google Cloud Run |
| Logging | structlog (JSON for Cloud Logging) |
| Validation | Pydantic 2.9 |
| Testing | pytest + pytest-asyncio |
| Registry | Google Artifact Registry |

---

## Troubleshooting

### Model loading fails

```bash
# Check model exists in GCS
gsutil ls gs://demeter-models/agro/

# Check ML Worker logs
gcloud run services logs read demeter-mlworker-prod --region=us-central1
```

### Tenant path validation fails

**Symptom:** `TenantPathError: Path 'X' does not belong to tenant 'Y'`

**Cause:** The `image_url` doesn't start with `{tenant_id}/`

**Solution:** Ensure images are stored with tenant prefix: `gs://bucket/{tenant_id}/originals/...`

### Cloud Tasks not reaching ML Worker

```bash
# Check queue status
gcloud tasks queues describe agro-ml-tasks --location=us-central1

# Check IAM bindings
gcloud run services get-iam-policy demeter-mlworker-prod --region=us-central1
```

### Out of memory during inference

**Symptom:** Container restarts, OOM errors in logs

**Solutions:**
1. Increase Cloud Run memory: `--memory 16Gi`
2. Reduce model size (use YOLO-nano instead of YOLO-large)
3. Process images in smaller batches

---

## Security Checklist

- [ ] Cloud SQL has no public IP
- [ ] All secrets in Secret Manager (not environment variables)
- [ ] ML Worker only accessible via Cloud Tasks (IAM invoker)
- [ ] Storage bucket has tenant-prefixed paths
- [ ] RLS policies active on all database tables
- [ ] Model files stored in separate bucket from user images
- [ ] Terraform state bucket has versioning enabled

---

## Cost Estimation

| Resource | Configuration | Monthly Cost (USD) |
|----------|---------------|--------------------|
| Cloud Run | 0-10 instances, 4 vCPU, 8GB | $0-200 (usage-based) |
| Cloud Tasks | 3 queues, ~100K tasks/month | ~$1 |
| Cloud Storage | Models ~5GB, Images variable | ~$5-50 |
| Secret Manager | 3 secrets | ~$0.10 |
| **Total** | | **~$10-250/month** |

*Scale-to-zero Cloud Run keeps costs minimal during low usage periods. Costs increase with task volume.*
