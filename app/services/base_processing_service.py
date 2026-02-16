"""Base Processing Service - Abstract base for industry-specific processing.

Defines the common interface and shared functionality for all processing
services. Each industry (agro, vending, etc.) has its own ProcessingService
that inherits from this base.

Architecture:
    1 Processing Service = 1 Industry/Tenant
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.infra.logging import get_logger
from app.infra.storage import StorageClient, TenantPathError
from app.schemas.task import ProcessingRequest, ProcessingResponse

logger = get_logger(__name__)


class BaseProcessingService(ABC):
    """Abstract base class for industry-specific processing services.

    This class defines the common interface and shared functionality:
    - Image download from GCS
    - Result persistence to DB (INSERT directo)
    - Temporary file cleanup
    - Error handling and logging

    Subclasses implement the specific ML pipeline logic for each industry.
    """

    def __init__(
        self,
        storage_client: StorageClient,
        db_session: AsyncSession,
    ) -> None:
        """Initialize base processing service.

        Args:
            storage_client: Cloud Storage client for image download
            db_session: Async SQLAlchemy session for DB operations
        """
        self.storage = storage_client
        self.db = db_session

    @abstractmethod
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process an image through the ML pipeline.

        Each industry implements its specific processing logic.

        Args:
            request: Processing request from Cloud Tasks payload

        Returns:
            ProcessingResponse with results and metadata
        """
        pass

    @property
    @abstractmethod
    def industry(self) -> str:
        """Return the industry identifier for this service."""
        pass

    async def _download_image(
        self,
        request: ProcessingRequest,
    ) -> Path:
        """Download image from GCS to local temp file.

        Args:
            request: Processing request with image_url and tenant_id

        Returns:
            Path to local temporary file

        Raises:
            TenantPathError: If image doesn't belong to tenant
            RuntimeError: If download fails
        """
        logger.info(
            "Downloading image from GCS",
            tenant_id=request.tenant_id,
            image_url=request.image_url,
        )

        try:
            local_path = await self.storage.download_to_tempfile(
                blob_path=request.image_url,
                tenant_id=request.tenant_id,
            )

            logger.info(
                "Image downloaded successfully",
                tenant_id=request.tenant_id,
                local_path=str(local_path),
            )

            return local_path

        except TenantPathError:
            raise
        except Exception as e:
            logger.error(
                "Failed to download image",
                tenant_id=request.tenant_id,
                image_url=request.image_url,
                error=str(e),
            )
            raise RuntimeError(f"Image download failed: {e}") from e

    async def _cleanup_temp_file(self, local_path: Path | None) -> None:
        """Remove temporary local file after processing.

        Args:
            local_path: Path to temporary file (can be None)
        """
        if local_path is None:
            return

        try:
            if local_path.exists():
                local_path.unlink()
                logger.debug(
                    "Temporary file cleaned up",
                    path=str(local_path),
                )
        except Exception as e:
            logger.warning(
                "Failed to cleanup temporary file",
                path=str(local_path),
                error=str(e),
            )

    async def _update_session_status(
        self,
        session_id: int,
        status: str,
        error_message: str | None = None,
        **update_fields: Any,
    ) -> None:
        """Update photo_processing_session status in DB.

        Args:
            session_id: PhotoProcessingSession.id
            status: New status (processing, completed, failed)
            error_message: Optional error message for failed status
            **update_fields: Additional fields to update
        """
        from sqlalchemy import text

        try:
            if error_message:
                query = text("""
                    UPDATE photo_processing_session
                    SET status = :status, error_message = :error_message
                    WHERE id = :session_id
                """)
                await self.db.execute(
                    query,
                    {"status": status, "error_message": error_message, "session_id": session_id}
                )
            else:
                query = text("""
                    UPDATE photo_processing_session
                    SET status = :status
                    WHERE id = :session_id
                """)
                await self.db.execute(
                    query,
                    {"status": status, "session_id": session_id}
                )

            await self.db.commit()

            logger.info(
                "Session status updated",
                session_id=session_id,
                status=status,
            )

        except Exception as e:
            logger.error(
                "Failed to update session status",
                session_id=session_id,
                status=status,
                error=str(e),
            )
            await self.db.rollback()

    async def _save_detections_to_db(
        self,
        session_id: int,
        detections: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk insert detections to database.

        Args:
            session_id: PhotoProcessingSession.id
            detections: List of detection dicts

        Returns:
            List of created detection IDs
        """
        if not detections:
            return []

        from sqlalchemy import text

        detection_ids = []

        try:
            for det in detections:
                query = text("""
                    INSERT INTO detection (
                        session_id, center_x_px, center_y_px, width_px, height_px,
                        detection_confidence, is_empty_container, is_alive
                    ) VALUES (
                        :session_id, :center_x_px, :center_y_px, :width_px, :height_px,
                        :confidence, :is_empty, :is_alive
                    ) RETURNING id
                """)

                result = await self.db.execute(query, {
                    "session_id": session_id,
                    "center_x_px": det.get("center_x_px"),
                    "center_y_px": det.get("center_y_px"),
                    "width_px": det.get("width_px"),
                    "height_px": det.get("height_px"),
                    "confidence": det.get("confidence", 0.5),
                    "is_empty": det.get("is_empty", False),
                    "is_alive": det.get("is_alive", True),
                })

                row = result.fetchone()
                if row:
                    detection_ids.append(row[0])

            await self.db.commit()

            logger.info(
                "Detections saved to DB",
                session_id=session_id,
                count=len(detection_ids),
            )

            return detection_ids

        except Exception as e:
            logger.error(
                "Failed to save detections",
                session_id=session_id,
                error=str(e),
            )
            await self.db.rollback()
            raise

    async def _save_classifications_to_db(
        self,
        session_id: int,
        classifications: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk insert classifications to database.

        Args:
            session_id: PhotoProcessingSession.id
            classifications: List of classification dicts

        Returns:
            List of created classification IDs
        """
        if not classifications:
            return []

        from sqlalchemy import text

        classification_ids = []

        try:
            for cls in classifications:
                query = text("""
                    INSERT INTO classification (
                        product_id, product_size_id, product_state_id,
                        packaging_catalog_id, product_conf, product_size_conf
                    ) VALUES (
                        :product_id, :product_size_id, :product_state_id,
                        :packaging_catalog_id, :product_conf, :product_size_conf
                    ) RETURNING id
                """)

                result = await self.db.execute(query, {
                    "product_id": cls.get("product_id"),
                    "product_size_id": cls.get("product_size_id"),
                    "product_state_id": cls.get("product_state_id", 1),
                    "packaging_catalog_id": cls.get("packaging_catalog_id"),
                    "product_conf": cls.get("confidence", 95),
                    "product_size_conf": cls.get("size_confidence", 80),
                })

                row = result.fetchone()
                if row:
                    classification_ids.append(row[0])

            await self.db.commit()

            logger.info(
                "Classifications saved to DB",
                session_id=session_id,
                count=len(classification_ids),
            )

            return classification_ids

        except Exception as e:
            logger.error(
                "Failed to save classifications",
                session_id=session_id,
                error=str(e),
            )
            await self.db.rollback()
            raise
