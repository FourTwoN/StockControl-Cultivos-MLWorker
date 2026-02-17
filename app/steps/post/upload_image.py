"""Upload image step with thumbnail generation.

Generic step for uploading images to GCS with automatic thumbnail generation.
Configurable via pipeline kwargs for different use cases (original, processed, etc.).
"""

import io
import time
from pathlib import Path

from PIL import Image

from app.core.pipeline_step import PipelineStep
from app.core.processing_context import ProcessingContext
from app.infra.logging import get_logger
from app.infra.storage import get_storage_client

logger = get_logger(__name__)

# Default thumbnail sizes
DEFAULT_THUMBNAIL_SIZES = [256, 512]
DEFAULT_QUALITY = 85


class UploadImageStep(PipelineStep):
    """Upload image to GCS with thumbnail generation.

    Configurable via step_config (kwargs in pipeline DSL):
        - source: "original" or "processed" (default: "original")
        - dest_prefix: destination folder in GCS (default: "images")
        - thumbnail_sizes: list of thumbnail sizes (default: [256, 512])
        - quality: JPEG quality 1-100 (default: 85)
        - skip_if_missing: don't fail if source is missing (default: False)

    Example pipeline DSL:
        {"name": "upload_image", "kwargs": {
            "source": "original",
            "dest_prefix": "originals",
            "thumbnail_sizes": [256, 512, 1024]
        }}
    """

    @property
    def name(self) -> str:
        return "upload_image"

    async def execute(self, ctx: ProcessingContext) -> ProcessingContext:
        """Upload image and generate thumbnails.

        Args:
            ctx: Processing context with image paths and step_config

        Returns:
            Context with upload URLs added to results
        """
        step_start = time.perf_counter()

        # Get configuration from step_config
        source = ctx.step_config.get("source", "original")
        dest_prefix = ctx.step_config.get("dest_prefix", "images")
        thumbnail_sizes = ctx.step_config.get("thumbnail_sizes", DEFAULT_THUMBNAIL_SIZES)
        quality = ctx.step_config.get("quality", DEFAULT_QUALITY)
        skip_if_missing = ctx.step_config.get("skip_if_missing", False)

        logger.info(
            "Upload image step starting",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            source=source,
            dest_prefix=dest_prefix,
            thumbnail_sizes=thumbnail_sizes,
        )

        # Get source image path
        image_path = self._get_source_path(ctx, source)

        if image_path is None or not image_path.exists():
            if skip_if_missing:
                logger.warning(
                    "Source image not found, skipping upload",
                    source=source,
                    path=str(image_path) if image_path else "None",
                )
                return ctx
            raise FileNotFoundError(
                f"Source image not found: {source} -> {image_path}"
            )

        # Get storage client
        storage = get_storage_client()

        # Build destination paths
        base_blob_path = f"{ctx.tenant_id}/{dest_prefix}/{ctx.image_id}.jpg"
        thumbnails_prefix = f"{ctx.tenant_id}/{dest_prefix}_thumbnails"

        # Open image once for all operations
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, P modes)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            original_size = img.size
            logger.debug(
                "Source image loaded",
                size=original_size,
                mode=img.mode,
            )

            # Upload main image
            main_url = await self._upload_image(
                storage=storage,
                img=img,
                blob_path=base_blob_path,
                tenant_id=ctx.tenant_id,
                quality=quality,
            )

            # Generate and upload thumbnails
            thumbnail_urls: dict[int, str] = {}
            for size in thumbnail_sizes:
                thumb = self._create_thumbnail(img, size)
                thumb_blob_path = f"{thumbnails_prefix}/{ctx.image_id}_{size}.jpg"

                thumb_url = await self._upload_image(
                    storage=storage,
                    img=thumb,
                    blob_path=thumb_blob_path,
                    tenant_id=ctx.tenant_id,
                    quality=quality,
                )
                thumbnail_urls[size] = thumb_url

        step_duration_ms = int((time.perf_counter() - step_start) * 1000)

        logger.info(
            "Upload image step completed",
            tenant_id=ctx.tenant_id,
            image_id=ctx.image_id,
            source=source,
            dest_prefix=dest_prefix,
            main_url=main_url,
            thumbnails_count=len(thumbnail_urls),
            duration_ms=step_duration_ms,
        )

        # Build result key based on source type
        result_key = f"{source}_urls"
        upload_results = {
            result_key: {
                "main": main_url,
                "thumbnails": thumbnail_urls,
            }
        }

        return ctx.with_results(upload_results)

    def _get_source_path(self, ctx: ProcessingContext, source: str) -> Path | None:
        """Get the source image path based on source type.

        Args:
            ctx: Processing context
            source: Source type ("original" or "processed")

        Returns:
            Path to source image or None
        """
        if source == "original":
            return ctx.image_path
        elif source == "processed":
            viz_path = ctx.results.get("visualization_path")
            if viz_path:
                return Path(viz_path)
            return None
        else:
            logger.warning("Unknown source type", source=source)
            return None

    async def _upload_image(
        self,
        storage,
        img: Image.Image,
        blob_path: str,
        tenant_id: str,
        quality: int,
    ) -> str:
        """Upload PIL Image to GCS.

        Args:
            storage: Storage client
            img: PIL Image to upload
            blob_path: Destination path in GCS
            tenant_id: Tenant ID for validation
            quality: JPEG quality

        Returns:
            GCS URL of uploaded image
        """
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)

        url = await storage.upload_bytes(
            data=buffer.read(),
            blob_path=blob_path,
            tenant_id=tenant_id,
            content_type="image/jpeg",
        )

        return url

    def _create_thumbnail(self, img: Image.Image, max_size: int) -> Image.Image:
        """Create thumbnail maintaining aspect ratio.

        Args:
            img: Source PIL Image
            max_size: Maximum dimension (width or height)

        Returns:
            Resized PIL Image
        """
        width, height = img.size

        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
