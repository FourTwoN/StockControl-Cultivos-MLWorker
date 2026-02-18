"""ProductFamily model - product taxonomy level 2."""

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.product_category import ProductCategory


class ProductFamily(Base, TimestampMixin):
    """Product family - level 2 of product taxonomy.

    Maps to existing `product_families` table in DemeterAI database.
    """

    __tablename__ = "product_families"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("product_categories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    scientific_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    category: Mapped["ProductCategory"] = relationship(
        "ProductCategory",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ProductFamily(id={self.id}, name='{self.name}')>"
