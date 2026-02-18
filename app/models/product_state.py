"""ProductState model - product lifecycle state catalog."""

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ProductState(Base, TimestampMixin):
    """Product lifecycle state (seedling, juvenile, adult, etc.).

    Maps to existing `product_states` table in DemeterAI database.
    """

    __tablename__ = "product_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_sellable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=99, index=True)

    def __repr__(self) -> str:
        return f"<ProductState(id={self.id}, code='{self.code}')>"
