"""ProductCategory model - product taxonomy root."""

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ProductCategory(Base, TimestampMixin):
    """Product category - root of product taxonomy hierarchy.

    Maps to existing `product_categories` table in DemeterAI database.
    """

    __tablename__ = "product_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<ProductCategory(id={self.id}, code='{self.code}')>"
