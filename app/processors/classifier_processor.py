"""Classifier Processor - Plant type classification with size calculation.

This processor classifies plants by distributing them across configured species
and calculates plant sizes using a sophisticated z-score based algorithm.

Migrated from Demeter Backend.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.infra.logging import get_logger
from app.processors.base_processor import BaseProcessor

logger = get_logger(__name__)


@dataclass
class Classification:
    """Plant classification result.

    Attributes:
        detection_id: ID of detection being classified (optional)
        estimation_id: ID of estimation being classified (optional)
        class_name: Classification label (species name)
        confidence: Classification confidence score (0.0-1.0)
        segment_idx: Segment index this classification belongs to
        image_id: Image UUID this classification belongs to
        product_size_id: Product size ID (1=S, 2=M, 3=L, 4=XL) based on height analysis
    """

    detection_id: int | None = None
    estimation_id: int | None = None
    class_name: str = "unknown"
    confidence: float = 0.0
    segment_idx: int = -1
    image_id: str = ""
    product_size_id: int | None = None

    def __post_init__(self) -> None:
        """Validate classification fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        if not self.class_name or not isinstance(self.class_name, str):
            raise ValueError(f"class_name must be a non-empty string, got {self.class_name}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detection_id": self.detection_id,
            "estimation_id": self.estimation_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "segment_idx": self.segment_idx,
            "image_id": self.image_id,
            "product_size_id": self.product_size_id,
        }


class ClassifierProcessor(BaseProcessor[list[Classification]]):
    """Plant type classifier with equitable species distribution.

    This processor classifies plants by distributing them equitably across
    configured species and calculates sizes using z-score based algorithm.

    Key Features:
        - Distributes classifications equitably among species
        - Calculates plant sizes using band normalization + z-scores
        - ~80-90% of plants get modal size, rest are outliers

    Architecture:
        - Receives species config in payload (no DB queries)
        - Thread-safe (no shared state)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        worker_id: int = 0,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialize classifier processor.

        Args:
            model_path: Path to classification model (when available)
            worker_id: GPU worker ID
            confidence_threshold: Minimum confidence score
        """
        super().__init__(model_path, worker_id, confidence_threshold)
        logger.info("ClassifierProcessor initialized with equitable distribution strategy")

    async def process(
        self,
        image_path: str | Path,
        detections: list[Any] | None = None,
        estimations: list[Any] | None = None,
        species_config: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[Classification]:
        """Classify detected/estimated plants with equitable distribution.

        Args:
            image_path: Path to image
            detections: List of DetectionResult objects
            estimations: Optional list of estimation objects
            species_config: List of species dicts with 'product_name' key
                           (comes from Cloud Tasks payload)
            **kwargs: Additional arguments (ignored)

        Returns:
            List of Classification objects distributed equitably across species
        """
        detections = detections or []
        estimations = estimations or []

        logger.debug(
            "ClassifierProcessor.process() called",
            detections=len(detections),
            estimations=len(estimations),
            species_count=len(species_config) if species_config else 0,
        )

        if not species_config:
            logger.warning(
                "No species_config provided, returning empty classifications"
            )
            return []

        species = [cfg.get("product_name", "unknown") for cfg in species_config if cfg.get("product_name")]

        if not species:
            logger.warning("No product names found in species_config")
            return []

        logger.info(
            "Found species for classification",
            species=species,
            count=len(species),
        )

        logger.debug("Calculating plant sizes (predominant size with outliers)")
        size_assignments = self._calculate_plant_sizes(
            detections=detections,
            estimations=estimations,
            image_height=None,
        )

        all_items = list(detections)
        if estimations:
            all_items.extend(estimations)

        if not all_items:
            logger.debug("No items to classify, returning empty list")
            return []

        total_items = len(all_items)
        num_species = len(species)
        base_count = total_items // num_species
        remainder = total_items % num_species

        logger.debug(
            "Distributing items across species",
            total_items=total_items,
            num_species=num_species,
            base_count=base_count,
            remainder=remainder,
        )

        classifications = []
        item_idx = 0

        for species_idx, species_name in enumerate(species):
            count_for_species = base_count + (1 if species_idx < remainder else 0)

            for _ in range(count_for_species):
                if item_idx >= total_items:
                    break

                item = all_items[item_idx]

                segment_idx = getattr(item, "segment_idx", -1)
                image_id = getattr(item, "image_id", "")
                product_size_id = size_assignments.get(item_idx)

                classification = Classification(
                    detection_id=None,
                    estimation_id=None,
                    class_name=species_name,
                    confidence=1.0,
                    segment_idx=segment_idx,
                    image_id=image_id,
                    product_size_id=product_size_id,
                )

                classifications.append(classification)
                item_idx += 1

        logger.info(
            "Generated classifications",
            total=len(classifications),
            species_count=num_species,
        )

        return classifications

    def _calculate_plant_sizes(
        self,
        detections: list[Any],
        estimations: list[Any] | None = None,
        image_height: int | None = None,
    ) -> dict[int, int]:
        """Calculate plant sizes with a predominant modal size and outliers.

        This method assumes plants in the same processing batch have similar sizes,
        with most plants (80-90%) being the same size and only a few outliers.

        Algorithm:
        1. Normalize heights by band (compensate for perspective distortion)
        2. Calculate global statistics (mean, std) of normalized heights
        3. Determine modal size based on median of distribution
        4. Assign sizes using z-scores (standard deviations from mean):
           - z < -2.0: S (very small outlier)
           - z < -1.0: One size below modal
           - -1.0 <= z <= 1.0: Modal size (80-90% of plants)
           - z > 1.0: One size above modal
           - z > 2.0: XL (very large outlier)

        Args:
            detections: List of DetectionResult objects with height_px and center_y_px
            estimations: Optional list of estimation objects with band_number
            image_height: Optional image height to infer bands (defaults to 4000px)

        Returns:
            Dictionary mapping item index to product_size_id (1=S, 2=M, 3=L, 4=XL)
        """
        if not detections:
            logger.warning("No detections provided for size calculation")
            return {}

        if image_height is None:
            image_height = 4000
            logger.debug("No image_height provided, using default", height=image_height)

        num_bands = 4
        band_height = image_height // num_bands

        band_detections: dict[int, list[tuple[int, float]]] = defaultdict(list)
        band_heights: dict[int, list[float]] = defaultdict(list)

        for idx, det in enumerate(detections):
            height_px = getattr(det, "height_px", None)
            center_y_px = getattr(det, "center_y_px", None)

            if height_px is None or center_y_px is None:
                logger.warning(
                    "Detection missing height_px or center_y_px, skipping",
                    detection_idx=idx,
                )
                continue

            band_num = min(int(center_y_px // band_height) + 1, num_bands)

            band_detections[band_num].append((idx, float(height_px)))
            band_heights[band_num].append(float(height_px))

        logger.debug(
            "Grouped detections into bands",
            total=len(detections),
            bands={k: len(v) for k, v in band_detections.items()},
        )

        band_medians: dict[int, float] = {}
        for band_num, heights in band_heights.items():
            band_medians[band_num] = float(np.median(heights)) if heights else 50.0

        logger.debug("Band medians for normalization", medians=band_medians)

        all_heights = [h for heights in band_heights.values() for h in heights]
        if not all_heights:
            logger.warning("No valid heights found")
            return {}

        global_median = float(np.median(all_heights))
        logger.debug("Global median height", median=f"{global_median:.1f}")

        normalized_heights: list[tuple[int, float]] = []

        for band_num, det_list in band_detections.items():
            band_median = band_medians[band_num]
            normalization_factor = global_median / band_median if band_median > 0 else 1.0

            for det_idx, height in det_list:
                normalized_h = height * normalization_factor
                normalized_heights.append((det_idx, normalized_h))

        norm_height_values = np.array([h for _, h in normalized_heights])
        mean_height = float(np.mean(norm_height_values))
        std_height = float(np.std(norm_height_values))
        median_height = float(np.median(norm_height_values))

        logger.info(
            "Normalized height statistics",
            mean=f"{mean_height:.1f}",
            std=f"{std_height:.1f}",
            median=f"{median_height:.1f}",
        )

        if median_height < 45:
            modal_size = 1  # S
        elif median_height < 60:
            modal_size = 2  # M
        elif median_height < 80:
            modal_size = 3  # L
        else:
            modal_size = 4  # XL

        size_names = {1: "S", 2: "M", 3: "L", 4: "XL"}
        logger.info(
            "Determined modal size",
            modal=size_names[modal_size],
            median=f"{median_height:.1f}",
        )

        size_assignments: dict[int, int] = {}

        for det_idx, norm_h in normalized_heights:
            z_score = (norm_h - mean_height) / std_height if std_height > 0 else 0.0

            if z_score < -2.0:
                size_id = 1  # S
            elif z_score < -1.0:
                size_id = max(1, modal_size - 1)
            elif z_score <= 1.0:
                size_id = modal_size
            elif z_score <= 2.0:
                size_id = min(4, modal_size + 1)
            else:
                size_id = 4  # XL

            size_assignments[det_idx] = size_id

        if estimations:
            estimation_start_idx = len(detections)

            band_avg_normalized: dict[int, float] = {}
            for band_num in band_detections.keys():
                band_median = band_medians[band_num]
                normalization_factor = global_median / band_median if band_median > 0 else 1.0
                heights_in_band = band_heights[band_num]
                if heights_in_band:
                    avg_height_band = float(np.mean(heights_in_band))
                    band_avg_normalized[band_num] = avg_height_band * normalization_factor
                else:
                    band_avg_normalized[band_num] = mean_height

            for est_idx, est in enumerate(estimations):
                band_num = getattr(est, "band_number", None)

                if band_num is None or band_num not in band_avg_normalized:
                    logger.warning(
                        "Estimation has invalid band_number, assigning modal size",
                        estimation_idx=est_idx,
                        band_number=band_num,
                    )
                    size_assignments[estimation_start_idx + est_idx] = modal_size
                    continue

                avg_norm_height = band_avg_normalized[band_num]
                z_score = (avg_norm_height - mean_height) / std_height if std_height > 0 else 0.0

                if z_score < -2.0:
                    size_id = 1
                elif z_score < -1.0:
                    size_id = max(1, modal_size - 1)
                elif z_score <= 1.0:
                    size_id = modal_size
                elif z_score <= 2.0:
                    size_id = min(4, modal_size + 1)
                else:
                    size_id = 4

                size_assignments[estimation_start_idx + est_idx] = size_id

        size_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for size_id in size_assignments.values():
            size_counts[size_id] += 1

        total = len(size_assignments)
        if total > 0:
            logger.info(
                "Calculated sizes",
                total=total,
                modal=size_names[modal_size],
                S=f"{size_counts[1]} ({size_counts[1]/total*100:.1f}%)",
                M=f"{size_counts[2]} ({size_counts[2]/total*100:.1f}%)",
                L=f"{size_counts[3]} ({size_counts[3]/total*100:.1f}%)",
                XL=f"{size_counts[4]} ({size_counts[4]/total*100:.1f}%)",
            )

        return size_assignments
