"""Tests for base model infrastructure."""

from sqlalchemy.orm import DeclarativeBase

from app.models.base import Base, TimestampMixin


def test_base_is_declarative_base():
    """Base should be a SQLAlchemy DeclarativeBase."""
    assert hasattr(Base, "metadata")
    assert issubclass(Base, DeclarativeBase)


def test_timestamp_mixin_has_created_at():
    """TimestampMixin should provide created_at column."""
    assert hasattr(TimestampMixin, "created_at")


def test_timestamp_mixin_has_updated_at():
    """TimestampMixin should provide updated_at column."""
    assert hasattr(TimestampMixin, "updated_at")
