"""Tests for ProductState model."""

from app.models.product_state import ProductState


def test_product_state_tablename():
    """ProductState should map to product_states table."""
    assert ProductState.__tablename__ == "product_states"


def test_product_state_has_required_columns():
    """ProductState should have id, code, name, is_sellable columns."""
    columns = {c.name for c in ProductState.__table__.columns}
    assert "id" in columns
    assert "code" in columns
    assert "name" in columns
    assert "is_sellable" in columns
