"""Tests for ProductFamily model."""

from app.models.product_family import ProductFamily


def test_product_family_tablename():
    """ProductFamily should map to product_families table."""
    assert ProductFamily.__tablename__ == "product_families"


def test_product_family_has_required_columns():
    """ProductFamily should have id, category_id, name columns."""
    columns = {c.name for c in ProductFamily.__table__.columns}
    assert "id" in columns
    assert "category_id" in columns
    assert "name" in columns


def test_product_family_has_category_relationship():
    """ProductFamily should have relationship to ProductCategory."""
    from app.models.product_category import ProductCategory
    # Relationship should be declared
    assert "category" in ProductFamily.__mapper__.relationships
