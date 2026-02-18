"""PackagingCatalog model - complete packaging specification."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.packaging_color import PackagingColor
    from app.models.packaging_material import PackagingMaterial
    from app.models.packaging_type import PackagingType


class PackagingCatalog(Base):
    """Complete packaging specification (type + material + color + dimensions).

    Maps to existing `packaging_catalog` table in DemeterAI database.
    """

    __tablename__ = "packaging_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    packaging_type_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("packaging_types.id"),
        nullable=False,
        index=True,
    )
    packaging_material_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("packaging_materials.id"),
        nullable=False,
        index=True,
    )
    packaging_color_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("packaging_colors.id"),
        nullable=False,
        index=True,
    )
    sku: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    volume_liters: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    diameter_cm: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    height_cm: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    packaging_type: Mapped["PackagingType"] = relationship("PackagingType", lazy="selectin")
    packaging_material: Mapped["PackagingMaterial"] = relationship("PackagingMaterial", lazy="selectin")
    packaging_color: Mapped["PackagingColor"] = relationship("PackagingColor", lazy="selectin")

    def __repr__(self) -> str:
        return f"<PackagingCatalog(id={self.id}, sku='{self.sku}')>"
