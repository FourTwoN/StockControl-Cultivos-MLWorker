"""Tests for PackagingCatalog model."""

# Import foundation models first to register with SQLAlchemy
from app.models.packaging_type import PackagingType  # noqa: F401
from app.models.packaging_material import PackagingMaterial  # noqa: F401
from app.models.packaging_color import PackagingColor  # noqa: F401
from app.models.packaging_catalog import PackagingCatalog


def test_packaging_catalog_tablename():
    """PackagingCatalog should map to packaging_catalog table."""
    assert PackagingCatalog.__tablename__ == "packaging_catalog"


def test_packaging_catalog_has_fk_columns():
    """PackagingCatalog should have FK columns to foundation tables."""
    columns = {c.name for c in PackagingCatalog.__table__.columns}
    assert "packaging_type_id" in columns
    assert "packaging_material_id" in columns
    assert "packaging_color_id" in columns


def test_packaging_catalog_has_dimension_columns():
    """PackagingCatalog should have volume, diameter, height columns."""
    columns = {c.name for c in PackagingCatalog.__table__.columns}
    assert "volume_liters" in columns
    assert "diameter_cm" in columns
    assert "height_cm" in columns
