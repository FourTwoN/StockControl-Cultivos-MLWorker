"""Tests for ProductSize model."""

from app.models.product_size import ProductSize


def test_product_size_tablename():
    """ProductSize should map to product_sizes table."""
    assert ProductSize.__tablename__ == "product_sizes"


def test_product_size_has_required_columns():
    """ProductSize should have id, code, name, sort_order columns."""
    columns = {c.name for c in ProductSize.__table__.columns}
    assert "id" in columns
    assert "code" in columns
    assert "name" in columns
    assert "sort_order" in columns


def test_product_size_has_height_columns():
    """ProductSize should have min/max height columns."""
    columns = {c.name for c in ProductSize.__table__.columns}
    assert "min_height_cm" in columns
    assert "max_height_cm" in columns
