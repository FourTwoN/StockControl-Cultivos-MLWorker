"""Parser for converting JSON pipeline definitions to DSL structures.

Converts PipelineDefinition (Pydantic models from JSON) into executable
DSL structures (Chain, Group, Chord, StepSignature) that PipelineExecutor
can run.

Includes validation that all referenced steps exist in the StepRegistry
at parse time (fail-fast approach).
"""

from typing import Protocol

from app.core.pipeline_dsl import (
    Chain,
    Chord,
    Group,
    PipelineElement,
    StepSignature,
    chain,
    chord,
    group,
    step,
)
from app.schemas.pipeline_definition import (
    ChainDefinition,
    ChordDefinition,
    GroupDefinition,
    PipelineDefinition,
    PipelineElementDefinition,
    StepDefinition,
)


class StepRegistryProtocol(Protocol):
    """Protocol for step registry compatibility."""

    @classmethod
    def available_steps(cls) -> list[str]: ...


class PipelineParserError(Exception):
    """Raised when pipeline parsing fails."""

    pass


class PipelineParser:
    """Converts JSON pipeline definitions to executable DSL structures.

    Performs full validation at parse time:
    1. Pydantic validates JSON structure (done before parser is called)
    2. Parser validates all step names exist in StepRegistry

    Usage:
        parser = PipelineParser(StepRegistry)
        pipeline = parser.parse(definition)
        # pipeline is now a Chain ready for PipelineExecutor
    """

    def __init__(self, registry: type[StepRegistryProtocol]) -> None:
        """Initialize parser with step registry.

        Args:
            registry: Class with available_steps() classmethod
        """
        self._registry = registry

    def parse(self, definition: PipelineDefinition) -> Chain:
        """Parse a pipeline definition into executable DSL.

        Validates all steps exist and converts to DSL structures.

        Args:
            definition: Pydantic-validated pipeline definition

        Returns:
            Chain ready for PipelineExecutor.execute()

        Raises:
            PipelineParserError: If any referenced step is not registered
        """
        self._validate_all_steps_exist(definition)
        return self._parse_chain(definition)

    def _validate_all_steps_exist(self, definition: PipelineDefinition) -> None:
        """Validate all step names are registered.

        Collects all step names from the definition tree and checks
        each one exists in the registry.

        Args:
            definition: Pipeline definition to validate

        Raises:
            PipelineParserError: If any step name is not found
        """
        step_names = self._collect_step_names(definition)
        available = set(self._registry.available_steps())
        missing = step_names - available

        if missing:
            raise PipelineParserError(
                f"Steps not found in registry: {sorted(missing)}. "
                f"Available steps: {sorted(available)}"
            )

    def _collect_step_names(self, definition: PipelineDefinition) -> set[str]:
        """Recursively collect all step names from definition.

        Traverses the entire pipeline tree to find all StepDefinition
        nodes and extracts their names.

        Args:
            definition: Pipeline definition to traverse

        Returns:
            Set of unique step names referenced in the definition
        """
        names: set[str] = set()
        self._collect_from_element(definition, names)
        return names

    def _collect_from_element(
        self,
        element: PipelineDefinition | PipelineElementDefinition,
        names: set[str],
    ) -> None:
        """Recursively collect step names from a single element.

        Args:
            element: Pipeline element to process
            names: Set to add found names to (mutated)
        """
        if isinstance(element, StepDefinition):
            names.add(element.name)

        elif isinstance(element, (ChainDefinition, GroupDefinition, PipelineDefinition)):
            for child in element.steps:
                self._collect_from_element(child, names)

        elif isinstance(element, ChordDefinition):
            self._collect_from_element(element.group, names)
            if element.callback:
                names.add(element.callback.name)

    def _parse_element(self, element: PipelineElementDefinition) -> PipelineElement:
        """Parse a single pipeline element by type.

        Dispatches to the appropriate type-specific parser.

        Args:
            element: Element to parse

        Returns:
            Corresponding DSL structure
        """
        match element:
            case StepDefinition():
                return self._parse_step(element)
            case ChainDefinition():
                return self._parse_chain(element)
            case GroupDefinition():
                return self._parse_group(element)
            case ChordDefinition():
                return self._parse_chord(element)

    def _parse_step(self, step_def: StepDefinition) -> StepSignature:
        """Parse a step definition to StepSignature.

        Args:
            step_def: Step definition from JSON

        Returns:
            StepSignature for the DSL
        """
        return step(step_def.name, **step_def.kwargs)

    def _parse_chain(
        self, chain_def: ChainDefinition | PipelineDefinition
    ) -> Chain:
        """Parse a chain definition to Chain.

        Args:
            chain_def: Chain or root pipeline definition

        Returns:
            Chain containing parsed child elements
        """
        parsed_steps = [self._parse_element(s) for s in chain_def.steps]
        return chain(*parsed_steps)

    def _parse_group(self, group_def: GroupDefinition) -> Group:
        """Parse a group definition to Group.

        Args:
            group_def: Group definition from JSON

        Returns:
            Group containing parsed child elements
        """
        parsed_steps = [self._parse_element(s) for s in group_def.steps]
        return group(*parsed_steps)

    def _parse_chord(self, chord_def: ChordDefinition) -> Chord:
        """Parse a chord definition to Chord.

        Args:
            chord_def: Chord definition from JSON

        Returns:
            Chord with parsed group and optional callback
        """
        parsed_group = self._parse_group(chord_def.group)
        parsed_callback = (
            self._parse_step(chord_def.callback) if chord_def.callback else None
        )
        return chord(parsed_group, callback=parsed_callback)


__all__ = ["PipelineParser", "PipelineParserError"]
