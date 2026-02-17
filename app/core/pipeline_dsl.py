"""Pipeline DSL primitives inspired by Celery Canvas.

Provides composable primitives for building complex pipelines:
- chain: Sequential execution
- group: Parallel execution with asyncio.gather
- chord: Group with aggregator callback

Example:
    AGRO_PIPELINE = chain(
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
    )
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepSignature:
    """Reference to a pipeline step with optional kwargs.

    Represents a single step to execute. The kwargs are passed
    to the step via ProcessingContext.step_config.
    """

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chain:
    """Sequential execution of steps.

    Steps are executed one after another, passing context through.
    """

    steps: tuple[Any, ...]


@dataclass(frozen=True)
class Group:
    """Parallel execution of steps.

    All steps run concurrently via asyncio.gather.
    Results are merged into a single context.
    """

    steps: tuple[Any, ...]


@dataclass(frozen=True)
class Chord:
    """Parallel execution with aggregator callback.

    The group runs in parallel, then the callback receives
    the merged results for final aggregation.
    """

    group: Group
    callback: StepSignature | None = None


# Type alias for any pipeline element
PipelineElement = StepSignature | Chain | Group | Chord


def step(name: str, **kwargs: Any) -> StepSignature:
    """Create a step signature.

    Args:
        name: Registered step name
        **kwargs: Step-specific configuration

    Returns:
        StepSignature for use in pipeline definitions
    """
    return StepSignature(name=name, kwargs=kwargs)


def chain(*steps: PipelineElement) -> Chain:
    """Create a sequential chain of steps.

    Args:
        *steps: Steps to execute in order

    Returns:
        Chain containing the steps
    """
    return Chain(steps=steps)


def group(*steps: PipelineElement) -> Group:
    """Create a parallel group of steps.

    Args:
        *steps: Steps to execute in parallel

    Returns:
        Group containing the steps
    """
    return Group(steps=steps)


def chord(grp: Group, callback: StepSignature | None = None) -> Chord:
    """Create a chord: parallel group with aggregator.

    Args:
        grp: Group to execute in parallel
        callback: Optional step to aggregate results

    Returns:
        Chord containing group and callback
    """
    return Chord(group=grp, callback=callback)
