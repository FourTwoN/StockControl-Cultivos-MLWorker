"""Registry for dynamically building pipelines from step names."""

from app.core.pipeline_step import PipelineStep


class StepRegistry:
    """Registry for pipeline steps.

    Allows registering steps by name and building pipelines
    dynamically from a list of step names.
    """

    _steps: dict[str, type[PipelineStep]] = {}

    @classmethod
    def register(cls, name: str, step_class: type[PipelineStep]) -> None:
        """Register a pipeline step.

        Args:
            name: Unique identifier for the step
            step_class: PipelineStep class (not instance)

        Raises:
            ValueError: If name is empty or step_class is not a PipelineStep subclass
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"Step name must be a non-empty string, got {name}")

        if not isinstance(step_class, type) or not issubclass(step_class, PipelineStep):
            raise ValueError(
                f"step_class must be a PipelineStep subclass, got {step_class}"
            )

        cls._steps[name] = step_class

    @classmethod
    def get(cls, name: str) -> PipelineStep:
        """Get a new instance of a registered step.

        Args:
            name: Step identifier

        Returns:
            New instance of the step

        Raises:
            ValueError: If step not found in registry
        """
        if name not in cls._steps:
            raise ValueError(f"Step '{name}' not found in registry")

        step_class = cls._steps[name]
        return step_class()

    @classmethod
    def build_pipeline(cls, step_names: list[str]) -> list[PipelineStep]:
        """Build a pipeline from a list of step names.

        Args:
            step_names: List of step identifiers in execution order

        Returns:
            List of PipelineStep instances in the same order

        Raises:
            ValueError: If any step name is not found in registry
        """
        pipeline: list[PipelineStep] = []

        for name in step_names:
            step = cls.get(name)
            pipeline.append(step)

        return pipeline

    @classmethod
    def available_steps(cls) -> list[str]:
        """Get list of all registered step names.

        Returns:
            List of step identifiers
        """
        return list(cls._steps.keys())
