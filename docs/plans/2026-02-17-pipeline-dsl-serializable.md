# Pipeline DSL Serializable - Design Document

**Fecha:** 2026-02-17
**Estado:** Aprobado
**Autor:** Claude + Franco

## Problema

El sistema actual tiene dos mecanismos paralelos para configurar pipelines por tenant:

1. `pipeline_steps: list[str]` - Solo ejecución lineal, sin parallelismo
2. `pipeline_name: str` - Referencia a pipelines hardcodeados en Python

Esto causa:
- Cada variante de tenant requiere código nuevo
- No hay forma de expresar `group()` o `chord()` desde BD
- Dos paths de ejecución = mantenimiento duplicado

## Solución

Reemplazar ambos mecanismos con `pipeline_definition: JSONB` que serializa el DSL completo (chain, group, chord, step) y se parsea a estructuras ejecutables.

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                         tenant_config (BD)                          │
│  pipeline_definition: JSONB  ← DSL completo serializado             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    TenantConfigCache.load_configs()                 │
│  1. Pydantic valida estructura JSON (PipelineDefinition)            │
│  2. PipelineParser valida steps existen en StepRegistry             │
│  3. Cache guarda config validada                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Endpoint /process                           │
│  1. parser.parse(config.pipeline_definition) → Chain DSL            │
│  2. executor.execute(pipeline, ctx) → siempre con parallelismo      │
└─────────────────────────────────────────────────────────────────────┘
```

## Modelo de Datos

### Pydantic Schemas (`app/schemas/pipeline_definition.py`)

```python
class StepDefinition(BaseModel):
    type: Literal["step"] = "step"
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)

class GroupDefinition(BaseModel):
    type: Literal["group"] = "group"
    steps: list["PipelineElementDefinition"]

class ChordDefinition(BaseModel):
    type: Literal["chord"] = "chord"
    group: GroupDefinition
    callback: StepDefinition | None = None

class ChainDefinition(BaseModel):
    type: Literal["chain"] = "chain"
    steps: list["PipelineElementDefinition"]

PipelineElementDefinition = StepDefinition | GroupDefinition | ChordDefinition | ChainDefinition

class PipelineDefinition(BaseModel):
    type: Literal["chain"] = "chain"
    steps: list[PipelineElementDefinition]
```

### Pipeline Parser (`app/core/pipeline_parser.py`)

```python
class PipelineParser:
    def __init__(self, registry: type[StepRegistry]):
        self._registry = registry

    def parse(self, definition: PipelineDefinition) -> Chain:
        self._validate_all_steps_exist(definition)
        return self._parse_chain(definition)

    def _validate_all_steps_exist(self, definition: PipelineDefinition) -> None:
        step_names = self._collect_step_names(definition)
        available = set(self._registry.available_steps())
        missing = step_names - available
        if missing:
            raise ValueError(f"Steps not found in registry: {missing}")
```

## Cambios Requeridos

| Acción | Archivo | Descripción |
|--------|---------|-------------|
| Crear | `app/schemas/pipeline_definition.py` | Pydantic models para DSL JSON |
| Crear | `app/core/pipeline_parser.py` | Parser JSON → DSL + validación |
| Modificar | `app/core/tenant_config.py` | `pipeline_definition` en vez de `pipeline_steps` |
| Modificar | `app/api/routes/tasks.py` | Simplificar a un solo path |
| Eliminar | `app/pipelines/__init__.py` | Ya no hay pipelines hardcodeados |

## Ejemplo de Config Final

```json
{
  "tenant_id": "cultivos-abc",
  "pipeline_definition": {
    "type": "chain",
    "steps": [
      {"type": "step", "name": "segmentation"},
      {"type": "step", "name": "segment_filter"},
      {
        "type": "chord",
        "group": {
          "type": "group",
          "steps": [
            {"type": "step", "name": "sahi_detection", "kwargs": {"segment_type": "claro-cajon"}},
            {"type": "step", "name": "detection", "kwargs": {"segment_type": "cajon"}}
          ]
        },
        "callback": {"type": "step", "name": "aggregate_detections"}
      },
      {"type": "step", "name": "size_calculator"}
    ]
  },
  "settings": {"species": ["tomate", "lechuga"]}
}
```

## Migration SQL

```sql
ALTER TABLE tenant_config
  DROP COLUMN pipeline_steps,
  ADD COLUMN pipeline_definition JSONB NOT NULL;
```

## Decisiones de Diseño

1. **Opción A (Reemplazar)**: No hay tenants en prod, reemplazamos `pipeline_steps` directamente
2. **Opción C (Full validation)**: Validamos estructura Pydantic + steps existen en registry al cargar cache
3. **Un solo path**: Siempre usamos `PipelineExecutor`, eliminamos el fallback lineal
