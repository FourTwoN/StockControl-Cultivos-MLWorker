"""Tests for DensityParameter model."""

# Import all related models to register with SQLAlchemy
from app.models.product_category import ProductCategory  # noqa: F401
from app.models.product_family import ProductFamily  # noqa: F401
from app.models.product import Product  # noqa: F401
from app.models.packaging_type import PackagingType  # noqa: F401
from app.models.packaging_material import PackagingMaterial  # noqa: F401
from app.models.packaging_color import PackagingColor  # noqa: F401
from app.models.packaging_catalog import PackagingCatalog  # noqa: F401
from app.models.density_parameter import DensityParameter


def test_density_parameter_tablename():
    """DensityParameter should map to density_parameters table."""
    assert DensityParameter.__tablename__ == "density_parameters"


def test_density_parameter_has_density_columns():
    """DensityParameter should have density calculation columns."""
    columns = {c.name for c in DensityParameter.__table__.columns}
    assert "avg_area_per_plant_cm2" in columns
    assert "plants_per_m2" in columns
    assert "overlap_adjustment_factor" in columns
