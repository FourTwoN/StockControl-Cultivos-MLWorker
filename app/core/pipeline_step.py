"""Base class for all pipeline steps."""

from abc import ABC, abstractmethod

from app.core.processing_context import ProcessingContext


class PipelineStep(ABC):
    """Abstract base for ML processors and post-processors.

    All steps in the pipeline implement this interface,
    allowing them to be composed in any order.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this step.

        Returns:
            String identifier for the step
        """
        pass

    @abstractmethod
    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Execute step and return updated context.

        Args:
            ctx: Current processing context

        Returns:
            New context with step results (original unchanged)
        """
        pass
