"""Agro Processing Service - ML pipeline for agricultural/cultivation industry.

Implements the complete ML workflow for plant detection and classification:
1. Segmentation - Detect containers (cajon, segmento)
2. Segment filtering - Keep only largest "claro" segment
3. Detection - Standard YOLO for cajon, SAHI for large segments
4. Coordinate transformation - Segment-relative to full-image coords
5. Classification - Species distribution + size calculation
6. DB persistence - INSERT detections, classifications

Migrated from Demeter Backend pipeline_coordinator.py
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.processor_registry import get_processor_registry
from app.infra.logging import get_logger
from app.infra.storage import StorageClient, TenantPathError
from app.processors.detector_processor import DetectionResult
from app.processors.segmentation_processor import SegmentResult
from app.schemas.task import ProcessingRequest, ProcessingResponse
from app.services.base_processing_service import BaseProcessingService

logger = get_logger(__name__)


class AgroProcessingService(BaseProcessingService):
    """Processing service for Agro/Cultivation industry.

    Implements the complete agro ML pipeline with:
    - Segment filtering (keep only largest "claro")
    - Parallel processing of cajon vs segmento types
    - SAHI for large segments (>1M pixels)
    - Coordinate transformation
    - Species classification with size calculation
    """

    # Configuration
    SAHI_THRESHOLD_PX = 1_000_000  # Use SAHI for segments > 1M pixels
    FILTER_CLARO_SEGMENTS = True

    def __init__(
        self,
        storage_client: StorageClient,
        db_session: AsyncSession,
    ) -> None:
        """Initialize agro processing service.

        Args:
            storage_client: Cloud Storage client
            db_session: Async SQLAlchemy session for DB operations
        """
        super().__init__(storage_client, db_session)
        self.registry = get_processor_registry()

    @property
    def industry(self) -> str:
        return "agro"

    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process image through complete agro ML pipeline.

        Args:
            request: Processing request from Cloud Tasks

        Returns:
            ProcessingResponse with detection/classification results
        """
        start_time = time.time()
        local_path: Path | None = None

        try:
            logger.info(
                "Starting agro processing",
                tenant_id=request.tenant_id,
                session_id=str(request.session_id),
                image_id=str(request.image_id),
                pipeline=request.pipeline,
            )

            # Mark session as PROCESSING
            if request.session_id:
                await self._update_session_status(
                    session_id=int(str(request.session_id).split("-")[0]) if "-" in str(request.session_id) else int(request.session_id),
                    status="processing",
                )

            # Download image
            local_path = await self._download_image(request)

            # Run ML pipeline
            results = await self._run_agro_pipeline(
                image_path=local_path,
                request=request,
            )

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Mark session as COMPLETED
            if request.session_id:
                await self._update_session_status(
                    session_id=int(str(request.session_id).split("-")[0]) if "-" in str(request.session_id) else int(request.session_id),
                    status="completed",
                )

            logger.info(
                "Agro processing completed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                duration_ms=duration_ms,
                detections=results.get("total_detected", 0),
            )

            return ProcessingResponse(
                success=True,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                image_id=request.image_id,
                pipeline=request.pipeline,
                results=results,
                duration_ms=duration_ms,
                steps_completed=results.get("steps_completed", 0),
                error=None,
            )

        except TenantPathError as e:
            logger.error(
                "Tenant path validation failed",
                tenant_id=request.tenant_id,
                error=str(e),
            )
            raise

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            logger.error(
                "Agro processing failed",
                tenant_id=request.tenant_id,
                image_id=str(request.image_id),
                error=str(e),
                exc_info=True,
            )

            # Mark session as FAILED
            if request.session_id:
                try:
                    await self._update_session_status(
                        session_id=int(str(request.session_id).split("-")[0]) if "-" in str(request.session_id) else int(request.session_id),
                        status="failed",
                        error_message=str(e),
                    )
                except Exception:
                    pass

            return ProcessingResponse(
                success=False,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                image_id=request.image_id,
                pipeline=request.pipeline,
                results={},
                duration_ms=duration_ms,
                steps_completed=0,
                error=str(e),
            )

        finally:
            await self._cleanup_temp_file(local_path)

    async def _run_agro_pipeline(
        self,
        image_path: Path,
        request: ProcessingRequest,
    ) -> dict[str, Any]:
        """Execute the complete agro ML pipeline.

        Args:
            image_path: Path to local image file
            request: Original processing request

        Returns:
            Dict with pipeline results
        """
        results: dict[str, Any] = {
            "steps_completed": 0,
            "segments": [],
            "detections": [],
            "classifications": [],
            "total_detected": 0,
            "total_classified": 0,
        }

        # =========================================================================
        # STEP 1: SEGMENTATION
        # =========================================================================
        logger.info("Step 1/4: Running segmentation", image=image_path.name)

        segmentation_processor = self.registry.get("segmentation")
        segments: list[SegmentResult] = await segmentation_processor.process(image_path)

        logger.info(
            "Segmentation complete",
            segments_found=len(segments),
        )

        if not segments:
            logger.warning("No segments detected, returning empty results")
            return results

        results["steps_completed"] = 1

        # =========================================================================
        # STEP 2: FILTER SEGMENTS (keep only largest "claro")
        # =========================================================================
        if self.FILTER_CLARO_SEGMENTS:
            segments = self._filter_claro_segments(segments)
            logger.info(
                "Segments filtered",
                remaining=len(segments),
            )

        results["segments"] = [self._serialize_segment(s) for s in segments]

        # =========================================================================
        # STEP 3: DETECTION (parallel cajon vs segmento processing)
        # =========================================================================
        logger.info("Step 2/4: Running detection", segments=len(segments))

        # Load image dimensions for coordinate transformation
        from PIL import Image
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # Separate segments by type
        cajon_segments = [(i, s) for i, s in enumerate(segments) if s.class_name == "cajon"]
        segmento_segments = [(i, s) for i, s in enumerate(segments) if s.class_name in ("segmento", "claro-cajon")]

        logger.info(
            "Segments by type",
            cajon=len(cajon_segments),
            segmento=len(segmento_segments),
        )

        # Process in parallel
        all_detections: list[DetectionResult] = []

        # Process cajon segments with standard detector
        if cajon_segments:
            cajon_detections = await self._process_cajon_segments(
                image_path, cajon_segments, img_width, img_height, str(request.image_id)
            )
            all_detections.extend(cajon_detections)

        # Process segmento segments (use SAHI for large ones)
        if segmento_segments:
            segmento_detections = await self._process_segmento_segments(
                image_path, segmento_segments, img_width, img_height, str(request.image_id)
            )
            all_detections.extend(segmento_detections)

        logger.info(
            "Detection complete",
            total_detections=len(all_detections),
        )

        results["steps_completed"] = 2
        results["detections"] = [self._serialize_detection(d) for d in all_detections]
        results["total_detected"] = len(all_detections)

        # =========================================================================
        # STEP 4: CLASSIFICATION
        # =========================================================================
        logger.info("Step 3/4: Running classification", detections=len(all_detections))

        classifier = self.registry.get("classification")
        classifications = await classifier.process(
            image_path=image_path,
            detections=all_detections,
            estimations=None,
            species_config=request.species_config,
        )

        logger.info(
            "Classification complete",
            classifications=len(classifications),
        )

        results["steps_completed"] = 3
        results["classifications"] = [c.to_dict() for c in classifications]
        results["total_classified"] = len(classifications)

        # =========================================================================
        # STEP 5: PERSIST TO DATABASE
        # =========================================================================
        logger.info("Step 4/4: Persisting results to DB")

        session_id = request.session_id
        if session_id:
            # Convert UUID to int if needed (simplified - real impl would use proper session ID)
            session_id_int = hash(str(session_id)) % (10 ** 9)

            # Save detections
            detection_dicts = [self._serialize_detection(d) for d in all_detections]
            await self._save_detections_to_db(session_id_int, detection_dicts)

            # Save classifications
            classification_dicts = [
                {
                    "product_id": c.detection_id,  # Will need proper mapping
                    "product_size_id": c.product_size_id,
                    "confidence": c.confidence,
                }
                for c in classifications
            ]
            await self._save_classifications_to_db(session_id_int, classification_dicts)

        results["steps_completed"] = 4

        return results

    def _filter_claro_segments(
        self,
        segments: list[SegmentResult],
    ) -> list[SegmentResult]:
        """Filter claro segments - keep only the largest one.

        If multiple "claro" or "segmento" segments are found, keep only
        the one with the largest pixel area.

        Args:
            segments: List of all detected segments

        Returns:
            Filtered list with at most one claro segment
        """
        claro_types = ("segmento", "claro-cajon")
        claro_segments = [s for s in segments if s.class_name in claro_types]

        if len(claro_segments) <= 1:
            return segments

        # Find largest claro segment by area
        largest_claro = max(claro_segments, key=lambda s: s.area_px)

        # Remove all claro segments except the largest
        filtered = [s for s in segments if s.class_name not in claro_types]
        filtered.append(largest_claro)

        logger.info(
            "Filtered claro segments",
            original_count=len(claro_segments),
            kept_area=largest_claro.area_px,
        )

        return filtered

    async def _process_cajon_segments(
        self,
        image_path: Path,
        cajon_segments: list[tuple[int, SegmentResult]],
        img_width: int,
        img_height: int,
        image_id: str,
    ) -> list[DetectionResult]:
        """Process cajon segments with standard YOLO detector.

        Args:
            image_path: Path to full image
            cajon_segments: List of (index, segment) tuples
            img_width: Full image width
            img_height: Full image height
            image_id: Image UUID for tracking

        Returns:
            List of detections with transformed coordinates
        """
        detector = self.registry.get("detection")
        all_detections: list[DetectionResult] = []

        for idx, segment in cajon_segments:
            try:
                # Crop segment
                crop_path = await self._crop_segment(image_path, segment, idx)

                # Run detection
                detections = await detector.process(crop_path)

                # Transform coordinates to full image
                transformed = self._transform_coordinates(
                    detections, segment, img_width, img_height, idx, image_id
                )
                all_detections.extend(transformed)

                # Cleanup crop
                await self._cleanup_temp_file(crop_path)

            except Exception as e:
                logger.warning(
                    "Cajon segment detection failed",
                    segment_idx=idx,
                    error=str(e),
                )

        return all_detections

    async def _process_segmento_segments(
        self,
        image_path: Path,
        segmento_segments: list[tuple[int, SegmentResult]],
        img_width: int,
        img_height: int,
        image_id: str,
    ) -> list[DetectionResult]:
        """Process segmento segments with SAHI for large ones.

        Args:
            image_path: Path to full image
            segmento_segments: List of (index, segment) tuples
            img_width: Full image width
            img_height: Full image height
            image_id: Image UUID for tracking

        Returns:
            List of detections with transformed coordinates
        """
        all_detections: list[DetectionResult] = []

        for idx, segment in segmento_segments:
            try:
                # Crop segment
                crop_path = await self._crop_segment(image_path, segment, idx)

                # Choose detector based on segment size
                if segment.area_px > self.SAHI_THRESHOLD_PX:
                    detector = self.registry.get("sahi_detection")
                    logger.debug(
                        "Using SAHI for large segment",
                        segment_idx=idx,
                        area_px=segment.area_px,
                    )
                else:
                    detector = self.registry.get("detection")

                # Run detection
                detections = await detector.process(crop_path)

                # Transform coordinates to full image
                transformed = self._transform_coordinates(
                    detections, segment, img_width, img_height, idx, image_id
                )
                all_detections.extend(transformed)

                # Cleanup crop
                await self._cleanup_temp_file(crop_path)

            except Exception as e:
                logger.warning(
                    "Segmento segment detection failed",
                    segment_idx=idx,
                    error=str(e),
                )

        return all_detections

    async def _crop_segment(
        self,
        image_path: Path,
        segment: SegmentResult,
        segment_idx: int,
    ) -> Path:
        """Crop segment from full image using mask.

        Args:
            image_path: Path to full image
            segment: Segment with bbox and polygon
            segment_idx: Index for temp file naming

        Returns:
            Path to cropped segment image
        """
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        img_height, img_width = img.shape[:2]

        # Get bbox coordinates
        x1, y1, x2, y2 = segment.bbox
        x1_px = int(x1 * img_width)
        y1_px = int(y1 * img_height)
        x2_px = int(x2 * img_width)
        y2_px = int(y2 * img_height)

        if segment.polygon:
            # Create mask from polygon for precise cropping
            polygon_px = [(int(x * img_width), int(y * img_height)) for x, y in segment.polygon]
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(polygon_px, dtype=np.int32)], 255)

            # Find tight bounding box around mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                raise RuntimeError(f"Empty mask for segment {segment_idx}")

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Crop using tight bbox
            crop = img[y_min:y_max, x_min:x_max]
            mask_crop = mask[y_min:y_max, x_min:x_max]

            # Apply mask (black out background)
            crop_masked = cv2.bitwise_and(crop, crop, mask=mask_crop)
        else:
            # Fallback to simple bbox crop when polygon not available
            crop_masked = img[y1_px:y2_px, x1_px:x2_px]

        # Save to temp file
        crop_path = image_path.parent / f"{image_path.stem}_segment{segment_idx}.jpg"
        cv2.imwrite(str(crop_path), crop_masked)

        return crop_path

    def _transform_coordinates(
        self,
        detections: list[DetectionResult],
        segment: SegmentResult,
        img_width: int,
        img_height: int,
        segment_idx: int,
        image_id: str,
    ) -> list[DetectionResult]:
        """Transform segment-relative coordinates to full-image coordinates.

        Args:
            detections: Detections in segment-relative coords
            segment: Source segment with bbox
            img_width: Full image width
            img_height: Full image height
            segment_idx: Segment index
            image_id: Image UUID

        Returns:
            Detections with transformed coordinates
        """
        x1, y1, x2, y2 = segment.bbox
        x1_px = int(x1 * img_width)
        y1_px = int(y1 * img_height)

        transformed = []
        for det in detections:
            new_det = DetectionResult(
                center_x_px=det.center_x_px + x1_px,
                center_y_px=det.center_y_px + y1_px,
                width_px=det.width_px,
                height_px=det.height_px,
                confidence=det.confidence,
                class_name=det.class_name,
                segment_idx=segment_idx,
                image_id=image_id,
            )
            transformed.append(new_det)

        return transformed

    def _serialize_segment(self, segment: SegmentResult) -> dict[str, Any]:
        """Serialize segment for response."""
        return {
            "class_name": segment.class_name,
            "confidence": segment.confidence,
            "bbox": segment.bbox,
            "area_px": segment.area_px,
        }

    def _serialize_detection(self, detection: DetectionResult) -> dict[str, Any]:
        """Serialize detection for response."""
        return {
            "center_x_px": detection.center_x_px,
            "center_y_px": detection.center_y_px,
            "width_px": detection.width_px,
            "height_px": detection.height_px,
            "confidence": detection.confidence,
            "class_name": detection.class_name,
            "segment_idx": getattr(detection, "segment_idx", -1),
            "image_id": getattr(detection, "image_id", ""),
        }
