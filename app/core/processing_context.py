"""Processing context that flows through the pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessingContext:
    """Immutable context passed through pipeline steps.

    Each step receives context and returns a new one with updates.
    Original context is never mutated.
    """

    tenant_id: str
    image_id: str
    session_id: str
    image_path: Path
    config: dict[str, Any]

    # Raw results from ML processors
    raw_segments: list[dict[str, Any]] = field(default_factory=list)
    raw_detections: list[dict[str, Any]] = field(default_factory=list)
    raw_classifications: list[dict[str, Any]] = field(default_factory=list)

    # Accumulated results from post-processors
    results: dict[str, Any] = field(default_factory=dict)

    # Step-specific configuration (injected by executor)
    step_config: dict[str, Any] = field(default_factory=dict)

    # Cropped images for parallel detection (segment_idx -> crop_path)
    segment_crops: dict[int, Path] = field(default_factory=dict)

    def with_segments(self, segments: list[dict[str, Any]]) -> "ProcessingContext":
        """Return new context with segments.

        Args:
            segments: Segment data to include

        Returns:
            New ProcessingContext with updated segments
        """
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=segments,
            raw_detections=self.raw_detections,
            raw_classifications=self.raw_classifications,
            results=self.results,
            step_config=self.step_config,
            segment_crops=self.segment_crops,
        )

    def with_detections(self, detections: list[dict[str, Any]]) -> "ProcessingContext":
        """Return new context with detections.

        Args:
            detections: Detection data to include

        Returns:
            New ProcessingContext with updated detections
        """
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=detections,
            raw_classifications=self.raw_classifications,
            results=self.results,
            step_config=self.step_config,
            segment_crops=self.segment_crops,
        )

    def with_classifications(
        self, classifications: list[dict[str, Any]]
    ) -> "ProcessingContext":
        """Return new context with classifications.

        Args:
            classifications: Classification data to include

        Returns:
            New ProcessingContext with updated classifications
        """
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=self.raw_detections,
            raw_classifications=classifications,
            results=self.results,
            step_config=self.step_config,
            segment_crops=self.segment_crops,
        )

    def with_results(self, new_results: dict[str, Any]) -> "ProcessingContext":
        """Return new context with merged results.

        Args:
            new_results: Results to merge with existing results

        Returns:
            New ProcessingContext with merged results
        """
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=self.raw_detections,
            raw_classifications=self.raw_classifications,
            results={**self.results, **new_results},
            step_config=self.step_config,
            segment_crops=self.segment_crops,
        )

    def with_step_config(self, config: dict[str, Any]) -> "ProcessingContext":
        """Return new context with step-specific configuration.

        Args:
            config: Step configuration to merge

        Returns:
            New ProcessingContext with merged step config
        """
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=self.raw_detections,
            raw_classifications=self.raw_classifications,
            results=self.results,
            step_config={**self.step_config, **config},
            segment_crops=self.segment_crops,
        )

    def with_segment_crops(self, crops: dict[int, Path]) -> "ProcessingContext":
        """Return new context with segment crop paths.

        Args:
            crops: Mapping of segment_idx to crop file path

        Returns:
            New ProcessingContext with segment crops
        """
        return ProcessingContext(
            tenant_id=self.tenant_id,
            image_id=self.image_id,
            session_id=self.session_id,
            image_path=self.image_path,
            config=self.config,
            raw_segments=self.raw_segments,
            raw_detections=self.raw_detections,
            raw_classifications=self.raw_classifications,
            results=self.results,
            step_config=self.step_config,
            segment_crops={**self.segment_crops, **crops},
        )
