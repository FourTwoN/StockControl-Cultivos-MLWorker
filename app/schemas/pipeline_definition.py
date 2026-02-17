"""Pydantic schemas for serializable pipeline DSL definitions.

These schemas allow pipeline definitions to be stored as JSON in the database
and validated with full type safety. The discriminated union pattern with
`type` field enables Pydantic to automatically validate the correct structure.

Example JSON:
    {
        "type": "chain",
        "steps": [
            {"type": "step", "name": "segmentation"},
            {
                "type": "chord",
                "group": {
                    "type": "group",
                    "steps": [
                        {"type": "step", "name": "detection", "kwargs": {"segment_type": "cajon"}}
                    ]
                },
                "callback": {"type": "step", "name": "aggregate_detections"}
            }
        ]
    }
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class StepDefinition(BaseModel):
    """A single pipeline step reference with optional configuration.

    Attributes:
        type: Discriminator field, always "step"
        name: Registered step name in StepRegistry
        kwargs: Step-specific configuration passed via ProcessingContext.step_config
    """

    type: Literal["step"] = "step"
    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class GroupDefinition(BaseModel):
    """Parallel execution of multiple steps.

    All steps run concurrently via asyncio.gather.
    Results are merged into a single context.

    Attributes:
        type: Discriminator field, always "group"
        steps: Steps to execute in parallel
    """

    type: Literal["group"] = "group"
    steps: list[PipelineElementDefinition]


class ChordDefinition(BaseModel):
    """Parallel execution with aggregator callback.

    The group runs in parallel, then the callback receives
    the merged results for final aggregation.

    Attributes:
        type: Discriminator field, always "chord"
        group: Steps to execute in parallel
        callback: Optional step to aggregate results
    """

    type: Literal["chord"] = "chord"
    group: GroupDefinition
    callback: StepDefinition | None = None


class ChainDefinition(BaseModel):
    """Sequential execution of steps.

    Steps are executed one after another, passing context through.

    Attributes:
        type: Discriminator field, always "chain"
        steps: Steps to execute in order
    """

    type: Literal["chain"] = "chain"
    steps: list[PipelineElementDefinition]


# Discriminated union of all pipeline element types
# Pydantic uses the `type` field to determine which model to use
PipelineElementDefinition = Annotated[
    StepDefinition | GroupDefinition | ChordDefinition | ChainDefinition,
    Field(discriminator="type"),
]

# Rebuild models to resolve forward references
GroupDefinition.model_rebuild()
ChordDefinition.model_rebuild()
ChainDefinition.model_rebuild()


class PipelineDefinition(BaseModel):
    """Root pipeline definition, always a chain.

    This is the top-level structure stored in tenant_config.pipeline_definition.
    A pipeline is always a chain at the root level, containing any combination
    of steps, groups, and chords.

    Attributes:
        type: Discriminator field, always "chain"
        steps: Top-level steps in the pipeline
    """

    type: Literal["chain"] = "chain"
    steps: list[PipelineElementDefinition]


__all__ = [
    "StepDefinition",
    "GroupDefinition",
    "ChordDefinition",
    "ChainDefinition",
    "PipelineElementDefinition",
    "PipelineDefinition",
]
