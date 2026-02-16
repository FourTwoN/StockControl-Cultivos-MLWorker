"""Species distribution post-processor.

Distributes detections across species in an equitable manner.
"""

from typing import Any

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger

logger = get_logger(__name__)


class SpeciesDistributorStep(PipelineStep):
    """Distributes detections equitably across configured species.

    Uses round-robin distribution to ensure fair allocation of
    detections to each species in the configuration.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this step.

        Returns:
            String identifier "species_distributor"
        """
        return "species_distributor"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Distribute detections across species.

        Args:
            ctx: Current processing context with raw_detections and sizes

        Returns:
            New context with classifications added to results
        """
        detections = ctx.raw_detections
        sizes = ctx.results.get("sizes", {})
        species_list = ctx.config.get("species", [])

        if not detections:
            logger.debug("No detections to distribute")
            return ctx.with_results({"classifications": []})

        if not species_list:
            logger.warning("No species configured for distribution")
            return ctx.with_results({"classifications": []})

        classifications = self._distribute_species(
            detections, sizes, species_list
        )

        logger.info(
            "Distributed detections across species",
            detection_count=len(detections),
            species_count=len(species_list),
            distribution={
                species: sum(1 for c in classifications if c["species"] == species)
                for species in species_list
            },
        )

        return ctx.with_results({"classifications": classifications})

    def _distribute_species(
        self,
        detections: list[dict[str, Any]],
        sizes: dict[int, int],
        species_list: list[str],
    ) -> list[dict[str, Any]]:
        """Distribute detections across species using round-robin.

        Args:
            detections: List of detection dictionaries
            sizes: Dictionary mapping detection index to size ID
            species_list: List of species names

        Returns:
            List of classification dictionaries with species and size
        """
        classifications = []

        for idx in range(len(detections)):
            # Round-robin distribution
            species_idx = idx % len(species_list)
            species = species_list[species_idx]
            size_id = sizes.get(idx, 2)  # Default to SIZE_M if not found

            classifications.append(
                {
                    "detection_idx": idx,
                    "species": species,
                    "size_id": size_id,
                }
            )

        return classifications
