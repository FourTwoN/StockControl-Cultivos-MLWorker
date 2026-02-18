"""DensityParameter model - density-based estimation calibration."""

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Numeric, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.packaging_catalog import PackagingCatalog
    from app.models.product import Product


class DensityParameter(Base, TimestampMixin):
    """Density parameters for plant count estimation.

    Maps to existing `density_parameters` table in DemeterAI database.
    Used by estimation pipeline for count calculations.
    """

    __tablename__ = "density_parameters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    storage_bin_type_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    product_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("products.id"),
        nullable=False,
        index=True,
    )
    packaging_catalog_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("packaging_catalog.id"),
        nullable=False,
        index=True,
    )
    avg_area_per_plant_cm2: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    plants_per_m2: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    overlap_adjustment_factor: Mapped[Decimal] = mapped_column(
        Numeric(3, 2),
        nullable=False,
        default=Decimal("0.85"),
    )
    avg_diameter_cm: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    product: Mapped["Product"] = relationship("Product", lazy="selectin")
    packaging_catalog: Mapped["PackagingCatalog"] = relationship("PackagingCatalog", lazy="selectin")

    def __repr__(self) -> str:
        return f"<DensityParameter(id={self.id}, plants_per_m2={self.plants_per_m2})>"
