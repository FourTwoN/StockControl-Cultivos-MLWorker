# Pipeline DSL - Sistema de Orquestación

## Resumen

Sistema de composición de pipelines inspirado en Celery Canvas que permite:
- Ejecución secuencial (chain)
- Ejecución paralela (group) con `asyncio.gather`
- Ejecución paralela con agregador (chord)

## Motivación

El pipeline original era lineal:
```
Step1 → Step2 → Step3 → Step4
```

El caso de uso Agro requiere bifurcación:
```
Segmentation → segment_filter → ┬─ sahi_detection (áreas grandes)  ─┬→ aggregate → post-process
                                └─ detection (áreas pequeñas)      ─┘
```

## Arquitectura

### Primitivas DSL (`app/core/pipeline_dsl.py`)

```python
@dataclass(frozen=True)
class StepSignature:
    """Referencia a un step con kwargs opcionales."""
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Chain:
    """Ejecución secuencial."""
    steps: tuple[Any, ...]

@dataclass(frozen=True)
class Group:
    """Ejecución paralela con asyncio.gather."""
    steps: tuple[Any, ...]

@dataclass(frozen=True)
class Chord:
    """Group + callback agregador."""
    group: Group
    callback: StepSignature | None = None
```

### Funciones helper

```python
step("detection", segment_type="cajon")  # StepSignature con kwargs
chain(step("a"), step("b"))              # Chain secuencial
group(step("a"), step("b"))              # Group paralelo
chord(group(...), callback=step("agg"))  # Chord con agregador
```

### Executor (`app/core/pipeline_executor.py`)

```python
class PipelineExecutor:
    async def execute(pipeline: PipelineElement, ctx: ProcessingContext) -> ProcessingContext
    async def _execute_chain(chain, ctx)   # for step in steps: ctx = await execute(step, ctx)
    async def _execute_group(group, ctx)   # asyncio.gather(*[execute(s, ctx) for s in steps])
    async def _execute_chord(chord, ctx)   # execute_group → execute_callback
    async def _execute_step(sig, ctx)      # StepRegistry.get(name).execute(ctx)
    def _merge_contexts(original, results) # Combina detections de branches paralelos
```

## Flujo de Datos

### ProcessingContext

Campos agregados para soportar DSL:

```python
@dataclass(frozen=True)
class ProcessingContext:
    # ... campos existentes ...

    # Step-specific config (inyectado por executor)
    step_config: dict[str, Any] = field(default_factory=dict)

    # Crops de segmentos para detección paralela
    segment_crops: dict[int, Path] = field(default_factory=dict)
```

### Inyección de kwargs

Cuando el executor ejecuta `step("detection", segment_type="cajon")`:

1. Crea contexto con `ctx.with_step_config({"segment_type": "cajon"})`
2. El step lee `ctx.step_config.get("segment_type")`
3. Filtra segmentos y procesa solo los de ese tipo

### Merge de contextos

Después de `group(step_a, step_b)`:

```python
def _merge_contexts(original, [result_a, result_b]):
    return ProcessingContext(
        raw_segments=original.raw_segments,        # Preserva original
        raw_detections=[*a.detections, *b.detections],  # Combina
        raw_classifications=[*a.classifications, *b.classifications],
        results={**original.results, **a.results, **b.results},
    )
```

## Pipelines Predefinidos (`app/pipelines/__init__.py`)

```python
AGRO_FULL_PIPELINE = chain(
    step("segmentation"),
    step("segment_filter"),
    chord(
        group(
            step("sahi_detection", segment_type="claro-cajon"),
            step("detection", segment_type="cajon"),
        ),
        callback=step("aggregate_detections"),
    ),
    step("size_calculator"),
    step("species_distributor"),
)

PIPELINES = {
    "agro_full": AGRO_FULL_PIPELINE,
    "agro_simple": AGRO_SIMPLE_PIPELINE,
    "detection_only": DETECTION_ONLY_PIPELINE,
    ...
}
```

## Integración con Endpoint

```python
# app/api/routes/tasks.py
pipeline_name = config.settings.get("pipeline_name")
pipeline = get_pipeline(pipeline_name) if pipeline_name else None

if pipeline:
    # DSL executor (soporta parallelismo)
    executor = PipelineExecutor()
    ctx = await executor.execute(pipeline, ctx)
else:
    # Fallback: lista lineal (backward compatible)
    steps = StepRegistry.build_pipeline(config.pipeline_steps)
    for step in steps:
        ctx = await step.execute(ctx)
```

## Uso

### Opción 1: Pipeline lineal (original, sin cambios)

```sql
INSERT INTO tenant_config (tenant_id, pipeline_steps, settings)
VALUES ('tenant-xyz', '["segmentation", "detection", "size_calculator"]', '{}');
```

### Opción 2: Pipeline DSL predefinido

```sql
INSERT INTO tenant_config (tenant_id, pipeline_steps, settings)
VALUES ('tenant-xyz', '[]', '{"pipeline_name": "agro_full"}');
```

### Opción 3: Nuevo pipeline DSL

```python
# En app/pipelines/__init__.py
MY_PIPELINE = chain(
    step("detection"),
    step("classification"),
)
PIPELINES["my_pipeline"] = MY_PIPELINE
```

---

## Limitaciones Actuales

### 1. Tipos de segmento hardcodeados

```python
# AGRO_FULL_PIPELINE hardcodea "claro-cajon" y "cajon"
step("sahi_detection", segment_type="claro-cajon"),
step("detection", segment_type="cajon"),
```

**Problema:** Otros tenants pueden tener tipos de segmento diferentes.

### 2. Pipelines definidos en código

Los pipelines están en `app/pipelines/__init__.py`, no en base de datos.

**Problema:** Agregar un pipeline requiere deploy.

### 3. Crop strategy fija

`segment_filter` genera crops usando bbox. No soporta:
- Crop con máscara de polígono
- Padding configurable
- Diferentes estrategias por tipo de segmento

---

## Propuesta de Mejora: Pipeline Configurable

### Fase 1: Tipos de segmento en config

```python
# En tenant settings
{
    "pipeline_name": "parallel_detection",
    "detection_branches": [
        {"segment_type": "claro-cajon", "detector": "sahi_detection"},
        {"segment_type": "cajon", "detector": "detection"}
    ]
}
```

Pipeline genérico que lee config:
```python
PARALLEL_DETECTION = chain(
    step("segmentation"),
    step("segment_filter"),
    step("dynamic_detection_branches"),  # Lee config y ejecuta branches
    step("aggregate_detections"),
)
```

### Fase 2: Pipeline DSL en base de datos

```sql
CREATE TABLE pipeline_definitions (
    name VARCHAR(100) PRIMARY KEY,
    definition JSONB NOT NULL,  -- DSL serializado
    created_at TIMESTAMP DEFAULT NOW()
);

-- Ejemplo
INSERT INTO pipeline_definitions (name, definition) VALUES (
    'custom_agro',
    '{
        "type": "chain",
        "steps": [
            {"type": "step", "name": "segmentation"},
            {"type": "chord", "group": {...}, "callback": {...}}
        ]
    }'
);
```

Parser que reconstruye DSL desde JSON:
```python
def parse_pipeline(definition: dict) -> PipelineElement:
    if definition["type"] == "chain":
        return chain(*[parse_pipeline(s) for s in definition["steps"]])
    elif definition["type"] == "step":
        return step(definition["name"], **definition.get("kwargs", {}))
    # ...
```

### Fase 3: Crop strategies configurables

```python
{
    "crop_config": {
        "claro-cajon": {"method": "bbox", "padding": 50},
        "cajon": {"method": "polygon_mask", "padding": 0}
    }
}
```

---

## Archivos Modificados/Creados

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `app/core/pipeline_dsl.py` | Nuevo | Primitivas DSL |
| `app/core/pipeline_executor.py` | Nuevo | Executor con asyncio.gather |
| `app/core/processing_context.py` | Mod | Campos step_config, segment_crops |
| `app/pipelines/__init__.py` | Nuevo | Pipelines predefinidos |
| `app/steps/post/aggregate_detections.py` | Nuevo | Agregador para chord |
| `app/steps/post/segment_filter.py` | Mod | Genera crops |
| `app/steps/ml/detection_step.py` | Mod | Segment-aware |
| `app/steps/ml/sahi_detection_step.py` | Mod | Segment-aware |
| `app/api/routes/tasks.py` | Mod | Integración executor |
| `tests/test_pipeline_dsl.py` | Nuevo | Tests DSL |
| `tests/test_pipeline_executor.py` | Nuevo | Tests executor |

---

## Decisiones de Diseño

### ¿Por qué no usar librerías existentes?

Se evaluaron:
- **Prefect/Airflow**: Overhead de infraestructura, orientados a DAGs distribuidos
- **Dask**: Demasiado complejo para pipelines in-process
- **Ray**: Requiere cluster, overkill para nuestro caso
- **Tawazi**: Proyecto pequeño, poco mantenido

**Decisión:** DSL custom de ~150 líneas con control total y sin dependencias.

### ¿Por qué frozen dataclasses?

- Inmutabilidad previene bugs
- Hasheable (puede usarse en sets/dicts)
- Consistente con ProcessingContext

### ¿Por qué asyncio.gather y no threads?

- Los pasos ML ya son async (usan threadpool interno para inference)
- Menos overhead que crear threads
- Más simple de razonar
