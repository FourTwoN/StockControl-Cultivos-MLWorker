"""SQLAlchemy models for MLWorker.

All models are READ-ONLY. MLWorker does not write to the database.
Results are sent to Backend via HTTP callback.
"""

from app.models.base import Base, TimestampMixin
from app.models.density_parameter import DensityParameter
from app.models.packaging_catalog import PackagingCatalog
from app.models.packaging_color import PackagingColor
from app.models.packaging_material import PackagingMaterial
from app.models.packaging_type import PackagingType
from app.models.product import Product
from app.models.product_category import ProductCategory
from app.models.product_family import ProductFamily
from app.models.product_size import ProductSize
from app.models.product_state import ProductState

__all__ = [
    "Base",
    "TimestampMixin",
    "DensityParameter",
    "PackagingCatalog",
    "PackagingColor",
    "PackagingMaterial",
    "PackagingType",
    "Product",
    "ProductCategory",
    "ProductFamily",
    "ProductSize",
    "ProductState",
]
