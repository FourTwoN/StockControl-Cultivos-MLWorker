"""Tests for ProductCategory model."""

from app.models.product_category import ProductCategory


def test_product_category_tablename():
    """ProductCategory should map to product_categories table."""
    assert ProductCategory.__tablename__ == "product_categories"


def test_product_category_has_required_columns():
    """ProductCategory should have id, code, name columns."""
    columns = {c.name for c in ProductCategory.__table__.columns}
    assert "id" in columns
    assert "code" in columns
    assert "name" in columns


def test_product_category_has_optional_columns():
    """ProductCategory should have description, active columns."""
    columns = {c.name for c in ProductCategory.__table__.columns}
    assert "description" in columns
    assert "active" in columns
