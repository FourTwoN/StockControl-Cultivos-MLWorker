"""Pipeline - Orchestrates execution of ML processing steps.

The Pipeline class coordinates the execution of multiple processors
in sequence based on the pipeline configuration.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.industry_config import IndustryConfig, PipelineConfig
from app.core.processor_registry import ProcessorRegistry, get_processor_registry
from app.infra.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepResult:
    """Result from a single pipeline step."""

    step_name: str
    success: bool
    data: Any
    duration_ms: int
    error: str | None = None


@dataclass
class PipelineResult:
    """Result from executing a complete pipeline."""

    pipeline_name: str
    success: bool
    steps: list[StepResult] = field(default_factory=list)
    total_duration_ms: int = 0
    error: str | None = None

    def get_step_data(self, step_name: str) -> Any | None:
        """Get data from a specific step."""
        for step in self.steps:
            if step.step_name == step_name and step.success:
                return step.data
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "pipeline_name": self.pipeline_name,
            "success": self.success,
            "total_duration_ms": self.total_duration_ms,
        }

        if self.error:
            result["error"] = self.error

        # Include step results keyed by name
        for step in self.steps:
            if step.success and step.data is not None:
                result[step.step_name] = step.data

        return result


class Pipeline:
    """Orchestrates execution of ML processing pipeline.

    The Pipeline coordinates multiple processors, passing data between
    them as configured in the industry config.
    """

    def __init__(
        self,
        config: IndustryConfig,
        registry: ProcessorRegistry | None = None,
    ) -> None:
        """Initialize pipeline.

        Args:
            config: Industry configuration
            registry: Processor registry. Defaults to global registry.
        """
        self.config = config
        self.registry = registry or get_processor_registry()

    async def execute(
        self,
        pipeline_name: str,
        image_path: Path,
        **context: Any,
    ) -> PipelineResult:
        """Execute a named pipeline.

        Args:
            pipeline_name: Name of the pipeline to execute
            image_path: Path to the image to process
            **context: Additional context passed to processors

        Returns:
            PipelineResult with all step results
        """
        start_time = time.time()

        # Get pipeline config
        pipeline_config = self.config.get_pipeline(pipeline_name)
        if pipeline_config is None:
            return PipelineResult(
                pipeline_name=pipeline_name,
                success=False,
                error=f"Pipeline not found: {pipeline_name}",
            )

        logger.info(
            "Executing pipeline",
            pipeline=pipeline_name,
            steps=pipeline_config.steps,
            image=image_path.name,
        )

        steps: list[StepResult] = []
        accumulated_data: dict[str, Any] = {}

        # Execute each step
        for step_name in pipeline_config.steps:
            step_result = await self._execute_step(
                step_name=step_name,
                image_path=image_path,
                accumulated_data=accumulated_data,
                context=context,
            )
            steps.append(step_result)

            if step_result.success and step_result.data is not None:
                accumulated_data[step_name] = step_result.data
            elif not step_result.success:
                # Stop pipeline on failure
                logger.warning(
                    "Pipeline step failed, stopping",
                    pipeline=pipeline_name,
                    step=step_name,
                    error=step_result.error,
                )
                break

        # Calculate total duration
        total_duration_ms = int((time.time() - start_time) * 1000)

        # Determine overall success
        all_success = all(s.success for s in steps)

        result = PipelineResult(
            pipeline_name=pipeline_name,
            success=all_success,
            steps=steps,
            total_duration_ms=total_duration_ms,
            error=None if all_success else steps[-1].error if steps else "No steps executed",
        )

        logger.info(
            "Pipeline completed",
            pipeline=pipeline_name,
            success=all_success,
            duration_ms=total_duration_ms,
            steps_completed=len([s for s in steps if s.success]),
        )

        return result

    async def _execute_step(
        self,
        step_name: str,
        image_path: Path,
        accumulated_data: dict[str, Any],
        context: dict[str, Any],
    ) -> StepResult:
        """Execute a single pipeline step.

        Args:
            step_name: Name of the step (maps to processor and model config)
            image_path: Path to image
            accumulated_data: Results from previous steps
            context: Additional context

        Returns:
            StepResult with step outcome
        """
        start_time = time.time()

        try:
            # Get model config for this step
            model_config = self.config.get_model_config(step_name)

            # Check if this is the estimation step (special handling)
            if step_name == "estimation":
                return await self._execute_estimation_step(
                    accumulated_data=accumulated_data,
                    context=context,
                    start_time=start_time,
                )

            # Get processor
            if not self.registry.is_registered(step_name):
                raise ValueError(f"Processor not registered: {step_name}")

            # Build processor kwargs from model config
            init_kwargs: dict[str, Any] = {}
            if model_config:
                init_kwargs["confidence_threshold"] = model_config.confidence_threshold

            processor = self.registry.get(step_name, **init_kwargs)

            # Build process kwargs
            process_kwargs: dict[str, Any] = {}
            if model_config and model_config.classes:
                process_kwargs["classes"] = list(model_config.classes)

            # Execute processor
            logger.debug(
                "Executing step",
                step=step_name,
                confidence=model_config.confidence_threshold if model_config else None,
            )

            result_data = await processor.process(image_path, **process_kwargs)

            duration_ms = int((time.time() - start_time) * 1000)

            return StepResult(
                step_name=step_name,
                success=True,
                data=self._serialize_result(result_data),
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Step execution failed",
                step=step_name,
                error=str(e),
                exc_info=True,
            )
            return StepResult(
                step_name=step_name,
                success=False,
                data=None,
                duration_ms=duration_ms,
                error=str(e),
            )

    async def _execute_estimation_step(
        self,
        accumulated_data: dict[str, Any],
        context: dict[str, Any],
        start_time: float,
    ) -> StepResult:
        """Execute the estimation step with accumulated data.

        Estimation is special because it takes detection/segmentation
        results as input, not the raw image.
        """
        try:
            from app.processors.detector_processor import DetectionResult
            from app.processors.segmentation_processor import SegmentResult

            # Get processor
            processor = self.registry.get("estimation")

            # Get detections from previous step
            detections_data = accumulated_data.get("detection", [])
            detections = [
                DetectionResult(**d) if isinstance(d, dict) else d
                for d in detections_data
            ] if detections_data else []

            # Get segments if available
            segments_data = accumulated_data.get("segmentation", [])
            segments = [
                SegmentResult(**s) if isinstance(s, dict) else s
                for s in segments_data
            ] if segments_data else None

            # Execute estimation
            result_data = await processor.process(detections, segments)

            duration_ms = int((time.time() - start_time) * 1000)

            return StepResult(
                step_name="estimation",
                success=True,
                data=self._serialize_result(result_data),
                duration_ms=duration_ms,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error("Estimation step failed", error=str(e), exc_info=True)
            return StepResult(
                step_name="estimation",
                success=False,
                data=None,
                duration_ms=duration_ms,
                error=str(e),
            )

    def _serialize_result(self, result: Any) -> Any:
        """Serialize processor result for storage.

        Converts dataclass results to dictionaries.
        """
        if result is None:
            return None

        if isinstance(result, list):
            return [self._serialize_item(item) for item in result]

        return self._serialize_item(result)

    def _serialize_item(self, item: Any) -> Any:
        """Serialize a single item."""
        if hasattr(item, "to_dict"):
            return item.to_dict()
        if hasattr(item, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(item)
        return item
