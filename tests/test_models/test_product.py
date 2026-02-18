"""Tests for Product model."""

# Import all models in hierarchy order to register with SQLAlchemy
from app.models.product_category import ProductCategory  # noqa: F401
from app.models.product_family import ProductFamily  # noqa: F401
from app.models.product import Product


def test_product_tablename():
    """Product should map to products table."""
    assert Product.__tablename__ == "products"


def test_product_has_required_columns():
    """Product should have id, family_id, sku columns."""
    columns = {c.name for c in Product.__table__.columns}
    assert "id" in columns
    assert "family_id" in columns
    assert "sku" in columns
    assert "common_name" in columns


def test_product_has_family_relationship():
    """Product should have relationship to ProductFamily."""
    assert "family" in Product.__mapper__.relationships
