"""PackagingType model - container type catalog."""

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PackagingType(Base):
    """Packaging type (pot, tray, box, etc.).

    Maps to existing `packaging_types` table in DemeterAI database.
    """

    __tablename__ = "packaging_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<PackagingType(id={self.id}, code='{self.code}')>"
