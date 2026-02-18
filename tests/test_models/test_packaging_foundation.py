"""Tests for packaging foundation models."""

from app.models.packaging_type import PackagingType
from app.models.packaging_material import PackagingMaterial
from app.models.packaging_color import PackagingColor


def test_packaging_type_tablename():
    """PackagingType should map to packaging_types table."""
    assert PackagingType.__tablename__ == "packaging_types"


def test_packaging_material_tablename():
    """PackagingMaterial should map to packaging_materials table."""
    assert PackagingMaterial.__tablename__ == "packaging_materials"


def test_packaging_color_tablename():
    """PackagingColor should map to packaging_colors table."""
    assert PackagingColor.__tablename__ == "packaging_colors"


def test_packaging_color_has_hex_code():
    """PackagingColor should have hex_code column."""
    columns = {c.name for c in PackagingColor.__table__.columns}
    assert "hex_code" in columns
