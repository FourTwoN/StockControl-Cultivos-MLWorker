"""Product model - leaf of product taxonomy."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.product_family import ProductFamily


class Product(Base, TimestampMixin):
    """Product - leaf of product taxonomy (Category -> Family -> Product).

    Maps to existing `products` table in DemeterAI database.
    """

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    family_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("product_families.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sku: Mapped[str | None] = mapped_column(String(20), unique=True, nullable=True, index=True)
    common_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    scientific_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    custom_attributes: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True, default=dict)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    family: Mapped["ProductFamily"] = relationship(
        "ProductFamily",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, sku='{self.sku}')>"
