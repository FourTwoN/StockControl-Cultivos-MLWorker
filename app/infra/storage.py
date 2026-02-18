"""Cloud Storage client with tenant-scoped path validation.

Provides:
- Upload/download images from GCS
- Tenant path validation for multi-tenant isolation
- Temporary file handling for ML processing
"""

import tempfile
from pathlib import Path
from typing import BinaryIO

from google.cloud import storage
from google.cloud.storage import Bucket

from app.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)


class TenantPathError(Exception):
    """Raised when a path doesn't match the expected tenant."""


class StorageClient:
    """Cloud Storage client with tenant isolation."""

    def __init__(self, bucket_name: str | None = None) -> None:
        """Initialize storage client.

        Args:
            bucket_name: GCS bucket name. Defaults to settings.gcs_bucket.
        """
        self._client: storage.Client | None = None
        self._bucket: Bucket | None = None
        self._bucket_name = bucket_name or settings.gcs_bucket

    @property
    def client(self) -> storage.Client:
        """Lazy-load the GCS client."""
        if self._client is None:
            self._client = storage.Client()
            logger.info("GCS client initialized")
        return self._client

    @property
    def bucket(self) -> Bucket:
        """Get the configured bucket."""
        if self._bucket is None:
            self._bucket = self.client.bucket(self._bucket_name)
            logger.info("GCS bucket configured", bucket=self._bucket_name)
        return self._bucket

    def validate_tenant_path(self, blob_path: str, tenant_id: str) -> None:
        """Validate that a blob path belongs to the specified tenant.

        Expected path format: {tenant_id}/originals/... or {tenant_id}/processed/...

        Args:
            blob_path: GCS blob path (without gs://bucket/)
            tenant_id: Expected tenant ID

        Raises:
            TenantPathError: If path doesn't match tenant
        """
        # Normalize path
        path = blob_path.lstrip("/")

        # Check tenant prefix
        if not path.startswith(f"{tenant_id}/"):
            logger.warning(
                "Tenant path validation failed",
                blob_path=blob_path,
                expected_tenant=tenant_id,
            )
            raise TenantPathError(
                f"Path '{blob_path}' does not belong to tenant '{tenant_id}'"
            )

    def parse_gs_url(self, gs_url: str) -> tuple[str, str]:
        """Parse a gs:// URL into bucket and blob path.

        Args:
            gs_url: Full GCS URL (gs://bucket/path/to/blob)

        Returns:
            Tuple of (bucket_name, blob_path)

        Raises:
            ValueError: If URL format is invalid
        """
        if not gs_url.startswith("gs://"):
            raise ValueError(f"Invalid GCS URL: {gs_url}")

        # Remove gs:// prefix
        path = gs_url[5:]

        # Split bucket and blob path
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid GCS URL (no blob path): {gs_url}")

        return parts[0], parts[1]

    async def download_to_tempfile(
        self,
        blob_path: str,
        tenant_id: str,
        suffix: str = ".jpg",
    ) -> Path:
        """Download a blob to a temporary file.

        Args:
            blob_path: GCS blob path or gs:// URL
            tenant_id: Tenant ID for validation
            suffix: File suffix for temp file

        Returns:
            Path to temporary file (caller must delete)

        Raises:
            TenantPathError: If path doesn't match tenant
            google.cloud.exceptions.NotFound: If blob doesn't exist
        """
        # Handle gs:// URLs
        if blob_path.startswith("gs://"):
            bucket_name, blob_path = self.parse_gs_url(blob_path)
            if bucket_name != self._bucket_name:
                logger.warning(
                    "Cross-bucket access attempted",
                    requested_bucket=bucket_name,
                    configured_bucket=self._bucket_name,
                )
                raise ValueError(f"Bucket mismatch: {bucket_name} != {self._bucket_name}")

        # Validate tenant owns this path
        self.validate_tenant_path(blob_path, tenant_id)

        # Download to temp file
        blob = self.bucket.blob(blob_path)

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)

        logger.info(
            "Downloading blob",
            blob_path=blob_path,
            tenant_id=tenant_id,
            temp_path=str(temp_path),
        )

        blob.download_to_filename(str(temp_path))

        return temp_path

    async def upload_file(
        self,
        local_path: Path | str,
        blob_path: str,
        tenant_id: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload a local file to GCS.

        Args:
            local_path: Path to local file
            blob_path: Destination blob path
            tenant_id: Tenant ID for validation
            content_type: MIME type

        Returns:
            gs:// URL to uploaded blob

        Raises:
            TenantPathError: If destination path doesn't match tenant
        """
        # Validate tenant owns destination path
        self.validate_tenant_path(blob_path, tenant_id)

        blob = self.bucket.blob(blob_path)

        logger.info(
            "Uploading file",
            local_path=str(local_path),
            blob_path=blob_path,
            tenant_id=tenant_id,
        )

        blob.upload_from_filename(str(local_path), content_type=content_type)

        return f"gs://{self._bucket_name}/{blob_path}"

    async def upload_bytes(
        self,
        data: bytes | BinaryIO,
        blob_path: str,
        tenant_id: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload bytes or file-like object to GCS.

        Args:
            data: Bytes or file-like object
            blob_path: Destination blob path
            tenant_id: Tenant ID for validation
            content_type: MIME type

        Returns:
            gs:// URL to uploaded blob
        """
        # Validate tenant owns destination path
        self.validate_tenant_path(blob_path, tenant_id)

        blob = self.bucket.blob(blob_path)

        if isinstance(data, bytes):
            blob.upload_from_string(data, content_type=content_type)
        else:
            blob.upload_from_file(data, content_type=content_type)

        logger.info(
            "Uploaded bytes",
            blob_path=blob_path,
            tenant_id=tenant_id,
            size=len(data) if isinstance(data, bytes) else "unknown",
        )

        return f"gs://{self._bucket_name}/{blob_path}"

    def get_blob_url(self, blob_path: str) -> str:
        """Get the gs:// URL for a blob path."""
        return f"gs://{self._bucket_name}/{blob_path}"

    async def exists(self, blob_path: str) -> bool:
        """Check if a blob exists."""
        blob = self.bucket.blob(blob_path)
        return blob.exists()

    async def delete(self, blob_path: str, tenant_id: str) -> None:
        """Delete a blob.

        Args:
            blob_path: GCS blob path
            tenant_id: Tenant ID for validation
        """
        self.validate_tenant_path(blob_path, tenant_id)
        blob = self.bucket.blob(blob_path)
        blob.delete()
        logger.info("Deleted blob", blob_path=blob_path, tenant_id=tenant_id)


class LocalStorageClient(StorageClient):
    """Local filesystem storage client for testing without GCS.

    Maps gs:// URLs to local filesystem paths.
    Use LOCAL_STORAGE_ROOT env var to set the base directory.
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        """Initialize local storage client.

        Args:
            root_dir: Root directory for local storage. Defaults to ./local_storage
        """
        super().__init__()
        self._root = Path(root_dir) if root_dir else Path("./local_storage")
        self._root.mkdir(parents=True, exist_ok=True)
        logger.info("Local storage client initialized", root=str(self._root))

    async def download_to_tempfile(
        self,
        blob_path: str,
        tenant_id: str,
        suffix: str = ".jpg",
    ) -> Path:
        """Download from local filesystem instead of GCS.

        The blob_path is mapped to: {root}/{blob_path}
        For gs:// URLs, extracts the path portion.
        """
        # Handle gs:// URLs - extract just the path
        if blob_path.startswith("gs://"):
            _, blob_path = self.parse_gs_url(blob_path)

        # Validate tenant path
        self.validate_tenant_path(blob_path, tenant_id)

        # Map to local path
        local_source = self._root / blob_path

        if not local_source.exists():
            # Try without tenant prefix (for simpler test setups)
            simple_path = self._root / Path(blob_path).name
            if simple_path.exists():
                local_source = simple_path
            else:
                raise FileNotFoundError(
                    f"Local file not found: {local_source} or {simple_path}"
                )

        # Copy to temp file (simulates download)
        import shutil
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = Path(temp_file.name)
        temp_file.close()

        shutil.copy(local_source, temp_path)

        logger.info(
            "Local file copied to temp",
            source=str(local_source),
            temp_path=str(temp_path),
            tenant_id=tenant_id,
        )

        return temp_path

    async def upload_bytes(
        self,
        data: bytes | BinaryIO,
        blob_path: str,
        tenant_id: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload to local filesystem instead of GCS."""
        self.validate_tenant_path(blob_path, tenant_id)

        local_dest = self._root / blob_path
        local_dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, bytes):
            local_dest.write_bytes(data)
        else:
            local_dest.write_bytes(data.read())

        logger.info(
            "Saved to local storage",
            path=str(local_dest),
            tenant_id=tenant_id,
        )

        return f"gs://local-bucket/{blob_path}"


# Singleton instance
_storage_client: StorageClient | None = None


def get_storage_client() -> StorageClient:
    """Get the singleton storage client.

    Returns LocalStorageClient in development mode,
    regular StorageClient in production.
    """
    global _storage_client
    if _storage_client is None:
        if settings.environment == "dev" and settings.use_local_storage:
            _storage_client = LocalStorageClient(
                root_dir=settings.local_storage_root
            )
        else:
            _storage_client = StorageClient()
    return _storage_client


def reset_storage_client() -> None:
    """Reset the singleton (useful for testing)."""
    global _storage_client
    _storage_client = None
