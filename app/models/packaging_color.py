"""PackagingColor model - container color catalog."""

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PackagingColor(Base):
    """Packaging color with hex code for UI display.

    Maps to existing `packaging_colors` table in DemeterAI database.
    """

    __tablename__ = "packaging_colors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    hex_code: Mapped[str] = mapped_column(String(7), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<PackagingColor(id={self.id}, code='{self.code}', hex='{self.hex_code}')>"
