"""Processor Registry - Dynamic loading and management of ML processors.

Provides a central registry for all available processors, allowing
the pipeline orchestrator to load them by name.
"""

from typing import Any, Protocol, runtime_checkable

from app.infra.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class Processor(Protocol):
    """Protocol for ML processors."""

    async def process(self, *args: Any, **kwargs: Any) -> Any:
        """Process input and return result."""
        ...


class ProcessorRegistry:
    """Registry for ML processors.

    Processors are registered by name and can be retrieved dynamically
    based on pipeline configuration.
    """

    def __init__(self) -> None:
        self._processors: dict[str, type[Processor]] = {}
        self._instances: dict[str, Processor] = {}

    def register(self, name: str, processor_class: type[Processor]) -> None:
        """Register a processor class.

        Args:
            name: Processor name (e.g., "detection", "segmentation")
            processor_class: Processor class to register
        """
        self._processors[name] = processor_class
        logger.debug("Processor registered", name=name, cls=processor_class.__name__)

    def get(self, name: str, **init_kwargs: Any) -> Processor:
        """Get a processor instance by name.

        Instances are cached for reuse.

        Args:
            name: Processor name
            **init_kwargs: Arguments for processor initialization

        Returns:
            Processor instance

        Raises:
            KeyError: If processor not registered
        """
        # Create cache key from name and kwargs (handle nested dicts)
        def make_hashable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
            if isinstance(obj, list):
                return tuple(make_hashable(x) for x in obj)
            return obj

        cache_key = f"{name}:{hash(make_hashable(init_kwargs))}"

        if cache_key not in self._instances:
            if name not in self._processors:
                raise KeyError(f"Processor not registered: {name}")

            processor_class = self._processors[name]
            self._instances[cache_key] = processor_class(**init_kwargs)
            logger.debug(
                "Processor instance created",
                name=name,
                cache_key=cache_key,
            )

        return self._instances[cache_key]

    def get_available(self) -> list[str]:
        """Get list of registered processor names."""
        return list(self._processors.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a processor is registered."""
        return name in self._processors

    def clear_instances(self) -> None:
        """Clear cached processor instances."""
        self._instances.clear()
        logger.info("Processor instance cache cleared")


def create_default_registry() -> ProcessorRegistry:
    """Create a registry with default processors registered.

    Returns:
        ProcessorRegistry with all standard processors
    """
    from app.processors.detector_processor import DetectorProcessor
    from app.processors.sahi_detector_processor import SAHIDetectorProcessor
    from app.processors.segmentation_processor import SegmentationProcessor

    registry = ProcessorRegistry()

    # Register all standard processors
    registry.register("detection", DetectorProcessor)
    registry.register("segmentation", SegmentationProcessor)
    registry.register("sahi_detection", SAHIDetectorProcessor)

    logger.info(
        "Default processor registry created",
        processors=registry.get_available(),
    )

    return registry


# Singleton registry
_registry: ProcessorRegistry | None = None


def get_processor_registry() -> ProcessorRegistry:
    """Get the singleton processor registry."""
    global _registry
    if _registry is None:
        _registry = create_default_registry()
    return _registry
