# Pipeline DSL Implementation Plan

## Objetivo

Implementar un DSL estilo Celery Canvas para orquestación de pipelines con soporte para:
- **chain**: Ejecución secuencial
- **group**: Ejecución paralela con `asyncio.gather`
- **chord**: Grupo paralelo + callback agregador

## Flujo Final del Pipeline Agro

```
Segmentation → segment_filter → chord(
                                    group(
                                        sahi_detection[segmento],
                                        detection[cajon]
                                    ),
                                    aggregate_detections
                                ) → size_calculator → species_distributor
```

---

## Fase 1: Pipeline DSL Primitives

### 1.1 Crear `app/core/pipeline_dsl.py`

```python
@dataclass
class StepSignature:
    name: str
    kwargs: dict[str, Any]

@dataclass
class Chain:
    steps: tuple[Any, ...]

@dataclass
class Group:
    steps: tuple[Any, ...]

@dataclass
class Chord:
    group: Group
    callback: StepSignature | None

# Helper functions
def step(name: str, **kwargs) -> StepSignature
def chain(*steps) -> Chain
def group(*steps) -> Group
def chord(grp: Group, callback: StepSignature | None = None) -> Chord
```

### 1.2 Crear `app/core/pipeline_executor.py`

```python
class PipelineExecutor:
    async def execute(pipeline: Chain, ctx: ProcessingContext) -> ProcessingContext
    async def _execute_chain(chain: Chain, ctx) -> ProcessingContext
    async def _execute_step(sig: StepSignature, ctx) -> ProcessingContext
    async def _execute_group(grp: Group, ctx) -> ProcessingContext  # asyncio.gather
    async def _execute_chord(chrd: Chord, ctx) -> ProcessingContext
    def _merge_contexts(original, results: list) -> ProcessingContext
```

---

## Fase 2: Modificar ProcessingContext

### 2.1 Agregar soporte para step config

```python
@dataclass(frozen=True)
class ProcessingContext:
    # ... existing fields ...
    step_config: dict[str, Any] = field(default_factory=dict)  # NEW

    def with_step_config(self, config: dict) -> ProcessingContext:
        """Merge step-specific config into context."""
```

### 2.2 Agregar campos para crops de segmentos

```python
    # Cropped images for parallel detection
    segment_crops: dict[int, Path] = field(default_factory=dict)  # segment_idx -> crop_path
```

---

## Fase 3: Modificar Steps para Segment-Aware Detection

### 3.1 Modificar `segment_filter.py`

- Separar segmentos por tipo: `segmento` vs `cajon`
- Generar crops de cada segmento
- Guardar crops en context

```python
async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
    # 1. Filter largest claro (existing)
    # 2. Crop each segment from original image
    # 3. Store crops in ctx.segment_crops
    # 4. Store segment metadata with crop paths
```

### 3.2 Modificar `detection_step.py`

- Leer `segment_type` de `ctx.step_config`
- Filtrar segmentos por tipo
- Procesar solo crops de ese tipo
- Transformar coordenadas: crop-relative → full-image

```python
async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
    segment_type = ctx.step_config.get("segment_type")
    if segment_type:
        segments = [s for s in ctx.raw_segments if s["class_name"] == segment_type]
    # Process each segment's crop
    # Transform coordinates back to full image
```

### 3.3 Modificar `sahi_detection_step.py`

- Mismo patrón que detection_step
- Solo procesa segmentos tipo `segmento`

---

## Fase 4: Crear Aggregate Step

### 4.1 Crear `app/steps/post/aggregate_detections.py`

```python
class AggregateDetectionsStep(PipelineStep):
    """Aggregates detections from parallel branches.

    This step is called after chord(group(...)) to merge
    detection results from multiple parallel detection branches.
    """

    @property
    def name(self) -> str:
        return "aggregate_detections"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        # Context already has merged detections from _merge_contexts
        # This step can add additional aggregation logic:
        # - Deduplicate overlapping detections (NMS)
        # - Sort by confidence
        # - Add statistics
        return ctx.with_results({
            "total_detections": len(ctx.raw_detections),
            "detection_sources": ["sahi", "standard"],
        })
```

### 4.2 Registrar step

```python
StepRegistry.register("aggregate_detections", AggregateDetectionsStep)
```

---

## Fase 5: Definir Pipelines

### 5.1 Crear `app/pipelines/__init__.py`

```python
from app.core.pipeline_dsl import chain, group, chord, step

AGRO_FULL_PIPELINE = chain(
    step("segmentation"),
    step("segment_filter"),
    chord(
        group(
            step("sahi_detection", segment_type="segmento"),
            step("detection", segment_type="cajon"),
        ),
        callback=step("aggregate_detections"),
    ),
    step("size_calculator"),
    step("species_distributor"),
)

PIPELINES = {
    "agro_full": AGRO_FULL_PIPELINE,
    "detection_only": chain(step("detection")),
    "segmentation_only": chain(step("segmentation")),
}
```

---

## Fase 6: Integrar con Endpoint

### 6.1 Modificar `app/api/routes/tasks.py`

```python
from app.core.pipeline_executor import PipelineExecutor
from app.pipelines import PIPELINES

@router.post("/process")
async def process_task(request: ProcessingRequest, ...):
    # Get pipeline from tenant config or default
    pipeline_name = config.pipeline_name or "agro_full"
    pipeline = PIPELINES.get(pipeline_name)

    if not pipeline:
        # Fallback: build from step list (backward compat)
        steps = StepRegistry.build_pipeline(config.pipeline_steps)
        # Execute linearly as before
    else:
        # Execute with DSL executor
        executor = PipelineExecutor()
        ctx = await executor.execute(pipeline, ctx)
```

---

## Fase 7: Update tenant_config Schema

### 7.1 Modificar schema

```sql
-- Add pipeline_name column
ALTER TABLE tenant_config
ADD COLUMN pipeline_name VARCHAR(50) DEFAULT NULL;

-- pipeline_name takes precedence over pipeline_steps if set
```

### 7.2 Opciones de config

```json
{
  "tenant_id": "cultivos-abc",
  "pipeline_name": "agro_full",  // Use predefined pipeline
  "pipeline_steps": null,        // Ignored if pipeline_name set
  "settings": {}
}
```

---

## Fase 8: Tests

### 8.1 Unit tests para DSL

- `test_pipeline_dsl.py`: Test chain, group, chord construction
- `test_pipeline_executor.py`: Test execution with mocked steps

### 8.2 Integration tests

- `test_agro_pipeline.py`: Full pipeline with real segments

---

## Archivos a Crear/Modificar

### Crear:
1. `app/core/pipeline_dsl.py` - DSL primitives
2. `app/core/pipeline_executor.py` - Executor
3. `app/steps/post/aggregate_detections.py` - Aggregate step
4. `app/pipelines/__init__.py` - Pipeline definitions
5. `tests/test_pipeline_dsl.py` - DSL tests
6. `tests/test_pipeline_executor.py` - Executor tests

### Modificar:
1. `app/core/processing_context.py` - Add step_config, segment_crops
2. `app/steps/post/segment_filter.py` - Generate crops
3. `app/steps/ml/detection_step.py` - Segment-aware detection
4. `app/steps/ml/sahi_detection_step.py` - Segment-aware SAHI
5. `app/steps/__init__.py` - Register aggregate step
6. `app/api/routes/tasks.py` - Use executor
7. `CLAUDE.md` - Update documentation

---

## Orden de Implementación

1. ✅ **pipeline_dsl.py** - Primitivas (sin dependencias)
2. ✅ **processing_context.py** - Agregar campos
3. ✅ **pipeline_executor.py** - Executor
4. ✅ **segment_filter.py** - Generar crops
5. ✅ **detection_step.py** - Segment-aware
6. ✅ **sahi_detection_step.py** - Segment-aware
7. ✅ **aggregate_detections.py** - Agregador
8. ✅ **pipelines/__init__.py** - Definiciones
9. ✅ **tasks.py** - Integración
10. ✅ **Tests** - Validar todo
11. ✅ **CLAUDE.md** - Documentar

---

## Estimación

- **Fase 1-2**: Core DSL + Context (~1 hora)
- **Fase 3-4**: Steps modifications (~1.5 horas)
- **Fase 5-7**: Integration (~1 hora)
- **Fase 8**: Tests (~1 hora)

**Total estimado**: ~4.5 horas de implementación
