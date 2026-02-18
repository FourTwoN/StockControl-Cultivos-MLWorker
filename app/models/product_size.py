"""ProductSize model - size classification catalog."""

from decimal import Decimal

from sqlalchemy import Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ProductSize(Base, TimestampMixin):
    """Product size classification (S, M, L, XL, etc.).

    Maps to existing `product_sizes` table in DemeterAI database.
    Used by ML pipeline for size classification based on detected dimensions.
    """

    __tablename__ = "product_sizes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    min_height_cm: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    max_height_cm: Mapped[Decimal | None] = mapped_column(Numeric(6, 2), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=99, index=True)

    def __repr__(self) -> str:
        return f"<ProductSize(id={self.id}, code='{self.code}')>"
