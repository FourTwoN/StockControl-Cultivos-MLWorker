# Database Models & Backend Callback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement read-only SQLAlchemy models for catalog data and HTTP callback client to send ML results to Backend.

**Architecture:** MLWorker reads catalog data (products, packaging, sizes) from PostgreSQL for ML classification. After pipeline execution, sends results via HTTP POST to Backend's `/api/v1/processing-callback/results` endpoint. No writes to database - Backend handles all persistence.

**Tech Stack:** SQLAlchemy 2.0 (async), Pydantic v2, httpx (async HTTP client), pytest-asyncio

---

## Phase 1: Base Model Infrastructure

### Task 1: Create Base Model

**Files:**
- Create: `app/models/base.py`
- Test: `tests/test_models/test_base.py`

**Step 1: Create test directory and init file**

```bash
mkdir -p tests/test_models
touch tests/test_models/__init__.py
```

**Step 2: Write the failing test**

Create `tests/test_models/test_base.py`:

```python
"""Tests for base model infrastructure."""

import pytest
from sqlalchemy import Column, Integer, String

from app.models.base import Base, TimestampMixin


def test_base_is_declarative_base():
    """Base should be a SQLAlchemy DeclarativeBase."""
    assert hasattr(Base, "metadata")
    assert hasattr(Base, "__tablename__")


def test_timestamp_mixin_has_created_at():
    """TimestampMixin should provide created_at column."""
    assert hasattr(TimestampMixin, "created_at")


def test_timestamp_mixin_has_updated_at():
    """TimestampMixin should provide updated_at column."""
    assert hasattr(TimestampMixin, "updated_at")
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_models/test_base.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'app.models.base'"

**Step 4: Write minimal implementation**

Create `app/models/base.py`:

```python
"""Base model infrastructure for SQLAlchemy models."""

from datetime import datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.

    Compatible with existing DemeterAI-back database schema.
    """
    pass


class TimestampMixin:
    """Mixin providing created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_models/test_base.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add app/models/base.py tests/test_models/
git commit -m "feat(models): add base model infrastructure with TimestampMixin"
```

---

### Task 2: Create Product Category Model

**Files:**
- Create: `app/models/product_category.py`
- Test: `tests/test_models/test_product_category.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_product_category.py`:

```python
"""Tests for ProductCategory model."""

import pytest

from app.models.product_category import ProductCategory


def test_product_category_tablename():
    """ProductCategory should map to product_categories table."""
    assert ProductCategory.__tablename__ == "product_categories"


def test_product_category_has_required_columns():
    """ProductCategory should have id, code, name columns."""
    columns = {c.name for c in ProductCategory.__table__.columns}
    assert "id" in columns
    assert "code" in columns
    assert "name" in columns


def test_product_category_has_optional_columns():
    """ProductCategory should have description, active columns."""
    columns = {c.name for c in ProductCategory.__table__.columns}
    assert "description" in columns
    assert "active" in columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_product_category.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `app/models/product_category.py`:

```python
"""ProductCategory model - product taxonomy root."""

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class ProductCategory(Base, TimestampMixin):
    """Product category - root of product taxonomy hierarchy.

    Maps to existing `product_categories` table in DemeterAI database.
    """

    __tablename__ = "product_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<ProductCategory(id={self.id}, code='{self.code}')>"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_product_category.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add app/models/product_category.py tests/test_models/test_product_category.py
git commit -m "feat(models): add ProductCategory model"
```

---

### Task 3: Create Product Family Model

**Files:**
- Create: `app/models/product_family.py`
- Test: `tests/test_models/test_product_family.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_product_family.py`:

```python
"""Tests for ProductFamily model."""

import pytest

from app.models.product_family import ProductFamily


def test_product_family_tablename():
    """ProductFamily should map to product_families table."""
    assert ProductFamily.__tablename__ == "product_families"


def test_product_family_has_required_columns():
    """ProductFamily should have id, category_id, name columns."""
    columns = {c.name for c in ProductFamily.__table__.columns}
    assert "id" in columns
    assert "category_id" in columns
    assert "name" in columns


def test_product_family_has_category_relationship():
    """ProductFamily should have relationship to ProductCategory."""
    from app.models.product_category import ProductCategory
    # Relationship should be declared
    assert "category" in ProductFamily.__mapper__.relationships
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_product_family.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/models/product_family.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_product_family.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add app/models/product_family.py tests/test_models/test_product_family.py
git commit -m "feat(models): add ProductFamily model with category relationship"
```

---

### Task 4: Create Product Model

**Files:**
- Create: `app/models/product.py`
- Test: `tests/test_models/test_product.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_product.py`:

```python
"""Tests for Product model."""

import pytest

from app.models.product import Product


def test_product_tablename():
    """Product should map to products table."""
    assert Product.__tablename__ == "products"


def test_product_has_required_columns():
    """Product should have id, family_id, sku columns."""
    columns = {c.name for c in Product.__table__.columns}
    assert "id" in columns
    assert "family_id" in columns
    assert "sku" in columns
    assert "common_name" in columns


def test_product_has_family_relationship():
    """Product should have relationship to ProductFamily."""
    assert "family" in Product.__mapper__.relationships
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_product.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/models/product.py`:

```python
"""Product model - leaf of product taxonomy."""

from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.product_family import ProductFamily


class Product(Base, TimestampMixin):
    """Product - leaf of product taxonomy (Category → Family → Product).

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_product.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add app/models/product.py tests/test_models/test_product.py
git commit -m "feat(models): add Product model with family relationship"
```

---

### Task 5: Create Product Size Model

**Files:**
- Create: `app/models/product_size.py`
- Test: `tests/test_models/test_product_size.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_product_size.py`:

```python
"""Tests for ProductSize model."""

import pytest

from app.models.product_size import ProductSize


def test_product_size_tablename():
    """ProductSize should map to product_sizes table."""
    assert ProductSize.__tablename__ == "product_sizes"


def test_product_size_has_required_columns():
    """ProductSize should have id, code, name, sort_order columns."""
    columns = {c.name for c in ProductSize.__table__.columns}
    assert "id" in columns
    assert "code" in columns
    assert "name" in columns
    assert "sort_order" in columns


def test_product_size_has_height_columns():
    """ProductSize should have min/max height columns."""
    columns = {c.name for c in ProductSize.__table__.columns}
    assert "min_height_cm" in columns
    assert "max_height_cm" in columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_product_size.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/models/product_size.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_product_size.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add app/models/product_size.py tests/test_models/test_product_size.py
git commit -m "feat(models): add ProductSize model for size classification"
```

---

### Task 6: Create Product State Model

**Files:**
- Create: `app/models/product_state.py`
- Test: `tests/test_models/test_product_state.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_product_state.py`:

```python
"""Tests for ProductState model."""

import pytest

from app.models.product_state import ProductState


def test_product_state_tablename():
    """ProductState should map to product_states table."""
    assert ProductState.__tablename__ == "product_states"


def test_product_state_has_required_columns():
    """ProductState should have id, code, name, is_sellable columns."""
    columns = {c.name for c in ProductState.__table__.columns}
    assert "id" in columns
    assert "code" in columns
    assert "name" in columns
    assert "is_sellable" in columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_product_state.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/models/product_state.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_product_state.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add app/models/product_state.py tests/test_models/test_product_state.py
git commit -m "feat(models): add ProductState model for lifecycle states"
```

---

### Task 7: Create Packaging Foundation Models

**Files:**
- Create: `app/models/packaging_type.py`
- Create: `app/models/packaging_material.py`
- Create: `app/models/packaging_color.py`
- Test: `tests/test_models/test_packaging_foundation.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_packaging_foundation.py`:

```python
"""Tests for packaging foundation models."""

import pytest

from app.models.packaging_type import PackagingType
from app.models.packaging_material import PackagingMaterial
from app.models.packaging_color import PackagingColor


def test_packaging_type_tablename():
    """PackagingType should map to packaging_types table."""
    assert PackagingType.__tablename__ == "packaging_types"


def test_packaging_material_tablename():
    """PackagingMaterial should map to packaging_materials table."""
    assert PackagingMaterial.__tablename__ == "packaging_materials"


def test_packaging_color_tablename():
    """PackagingColor should map to packaging_colors table."""
    assert PackagingColor.__tablename__ == "packaging_colors"


def test_packaging_color_has_hex_code():
    """PackagingColor should have hex_code column."""
    columns = {c.name for c in PackagingColor.__table__.columns}
    assert "hex_code" in columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_packaging_foundation.py -v`
Expected: FAIL

**Step 3: Write minimal implementations**

Create `app/models/packaging_type.py`:

```python
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
```

Create `app/models/packaging_material.py`:

```python
"""PackagingMaterial model - container material catalog."""

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class PackagingMaterial(Base):
    """Packaging material (plastic, terracotta, biodegradable, etc.).

    Maps to existing `packaging_materials` table in DemeterAI database.
    """

    __tablename__ = "packaging_materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<PackagingMaterial(id={self.id}, code='{self.code}')>"
```

Create `app/models/packaging_color.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_packaging_foundation.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add app/models/packaging_type.py app/models/packaging_material.py app/models/packaging_color.py tests/test_models/test_packaging_foundation.py
git commit -m "feat(models): add packaging foundation models (type, material, color)"
```

---

### Task 8: Create Packaging Catalog Model

**Files:**
- Create: `app/models/packaging_catalog.py`
- Test: `tests/test_models/test_packaging_catalog.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_packaging_catalog.py`:

```python
"""Tests for PackagingCatalog model."""

import pytest

from app.models.packaging_catalog import PackagingCatalog


def test_packaging_catalog_tablename():
    """PackagingCatalog should map to packaging_catalog table."""
    assert PackagingCatalog.__tablename__ == "packaging_catalog"


def test_packaging_catalog_has_fk_columns():
    """PackagingCatalog should have FK columns to foundation tables."""
    columns = {c.name for c in PackagingCatalog.__table__.columns}
    assert "packaging_type_id" in columns
    assert "packaging_material_id" in columns
    assert "packaging_color_id" in columns


def test_packaging_catalog_has_dimension_columns():
    """PackagingCatalog should have volume, diameter, height columns."""
    columns = {c.name for c in PackagingCatalog.__table__.columns}
    assert "volume_liters" in columns
    assert "diameter_cm" in columns
    assert "height_cm" in columns
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_packaging_catalog.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/models/packaging_catalog.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_packaging_catalog.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add app/models/packaging_catalog.py tests/test_models/test_packaging_catalog.py
git commit -m "feat(models): add PackagingCatalog model with relationships"
```

---

### Task 9: Create Density Parameter Model

**Files:**
- Create: `app/models/density_parameter.py`
- Test: `tests/test_models/test_density_parameter.py`

**Step 1: Write the failing test**

Create `tests/test_models/test_density_parameter.py`:

```python
"""Tests for DensityParameter model."""

import pytest

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_density_parameter.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/models/density_parameter.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models/test_density_parameter.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add app/models/density_parameter.py tests/test_models/test_density_parameter.py
git commit -m "feat(models): add DensityParameter model for estimation calibration"
```

---

### Task 10: Create Models Package Export

**Files:**
- Modify: `app/models/__init__.py`

**Step 1: Write the implementation**

Create/update `app/models/__init__.py`:

```python
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
```

**Step 2: Run all model tests**

Run: `pytest tests/test_models/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add app/models/__init__.py
git commit -m "feat(models): export all models from package"
```

---

## Phase 2: Callback Infrastructure

### Task 11: Create Callback Schema

**Files:**
- Create: `app/schemas/callback.py`
- Test: `tests/test_schemas/test_callback.py`

**Step 1: Write the failing test**

Create `tests/test_schemas/test_callback.py`:

```python
"""Tests for callback schemas."""

import pytest
from uuid import uuid4

from app.schemas.callback import (
    BoundingBox,
    DetectionResultItem,
    ClassificationResultItem,
    EstimationResultItem,
    ProcessingMetadata,
    ProcessingResultRequest,
)


def test_bounding_box_serialization():
    """BoundingBox should serialize to camelCase JSON."""
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
    data = bbox.model_dump(mode="json")
    assert data == {"x1": 10.0, "y1": 20.0, "x2": 100.0, "y2": 200.0}


def test_detection_result_item_serialization():
    """DetectionResultItem should serialize with camelCase."""
    item = DetectionResultItem(
        label="cactus",
        confidence=0.95,
        boundingBox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
    )
    data = item.model_dump(mode="json")
    assert data["label"] == "cactus"
    assert data["confidence"] == 0.95
    assert "boundingBox" in data


def test_processing_result_request_serialization():
    """ProcessingResultRequest should serialize for Java backend."""
    session_id = uuid4()
    image_id = uuid4()

    request = ProcessingResultRequest(
        sessionId=session_id,
        imageId=image_id,
        detections=[
            DetectionResultItem(label="plant", confidence=0.9, boundingBox=None)
        ],
        classifications=[],
        estimations=[],
    )

    data = request.model_dump(mode="json")
    assert data["sessionId"] == str(session_id)
    assert data["imageId"] == str(image_id)
    assert len(data["detections"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas/test_callback.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Create `app/schemas/callback.py`:

```python
"""Callback schemas for Backend communication.

These schemas match the Java DTOs in DemeterAI-back:
- ProcessingResultRequest
- DetectionResultItem
- ClassificationResultItem
- EstimationResultItem
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    model_config = ConfigDict(populate_by_name=True)

    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResultItem(BaseModel):
    """Single detection result from ML inference."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    confidence: float
    boundingBox: BoundingBox | None = None


class ClassificationResultItem(BaseModel):
    """Single classification result from ML inference."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    confidence: float
    detectionId: UUID | None = None


class EstimationResultItem(BaseModel):
    """Estimation result (count, area, etc.) from ML inference."""

    model_config = ConfigDict(populate_by_name=True)

    estimationType: str
    value: float
    unit: str | None = None
    confidence: float | None = None


class ProcessingMetadata(BaseModel):
    """Processing metadata from ML Worker."""

    model_config = ConfigDict(populate_by_name=True)

    pipeline: str | None = None
    processingTimeMs: int | None = None
    modelVersion: str | None = None
    workerVersion: str | None = None


class ProcessingResultRequest(BaseModel):
    """Payload sent to Backend callback endpoint.

    Matches Java DTO: com.fortytwo.demeter.fotos.dto.ProcessingResultRequest
    """

    model_config = ConfigDict(populate_by_name=True)

    sessionId: UUID
    imageId: UUID
    detections: list[DetectionResultItem] = []
    classifications: list[ClassificationResultItem] = []
    estimations: list[EstimationResultItem] = []
    metadata: ProcessingMetadata | None = None


class ErrorReport(BaseModel):
    """Error report sent to Backend on processing failure."""

    model_config = ConfigDict(populate_by_name=True)

    sessionId: UUID
    imageId: UUID
    errorMessage: str
    errorType: str = "ProcessingError"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas/test_callback.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add app/schemas/callback.py tests/test_schemas/test_callback.py
git commit -m "feat(schemas): add callback schemas matching Java DTOs"
```

---

### Task 12: Create Backend Client

**Files:**
- Create: `app/services/__init__.py`
- Create: `app/services/backend_client.py`
- Test: `tests/test_services/test_backend_client.py`

**Step 1: Create service directory**

```bash
mkdir -p app/services tests/test_services
touch app/services/__init__.py tests/test_services/__init__.py
```

**Step 2: Write the failing test**

Create `tests/test_services/test_backend_client.py`:

```python
"""Tests for BackendClient."""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.schemas.callback import ProcessingResultRequest, DetectionResultItem
from app.services.backend_client import BackendClient


@pytest.fixture
def backend_client():
    return BackendClient(base_url="http://localhost:8080", timeout=5.0)


@pytest.fixture
def sample_results():
    return ProcessingResultRequest(
        sessionId=uuid4(),
        imageId=uuid4(),
        detections=[DetectionResultItem(label="plant", confidence=0.95, boundingBox=None)],
        classifications=[],
        estimations=[],
    )


@pytest.mark.asyncio
async def test_send_results_calls_correct_endpoint(backend_client, sample_results):
    """send_results should POST to /api/v1/processing-callback/results."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = {"status": "ok"}
        mock_post.return_value.raise_for_status = lambda: None

        await backend_client.send_results("tenant-123", sample_results)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/v1/processing-callback/results" in call_args[0][0]
        assert call_args[1]["headers"]["X-Tenant-ID"] == "tenant-123"


@pytest.mark.asyncio
async def test_report_error_calls_error_endpoint(backend_client):
    """report_error should POST to /api/v1/processing-callback/error."""
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = {"status": "ok"}
        mock_post.return_value.raise_for_status = lambda: None

        await backend_client.report_error(
            tenant_id="tenant-123",
            session_id=uuid4(),
            image_id=uuid4(),
            error_message="Test error",
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/api/v1/processing-callback/error" in call_args[0][0]
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/test_services/test_backend_client.py -v`
Expected: FAIL

**Step 4: Write minimal implementation**

Create `app/services/backend_client.py`:

```python
"""HTTP client for Backend callback communication."""

from uuid import UUID

import httpx

from app.infra.logging import get_logger
from app.schemas.callback import ErrorReport, ProcessingResultRequest

logger = get_logger(__name__)


class BackendClient:
    """Client for sending ML results to Backend.

    Communicates with DemeterAI-back via HTTP callbacks:
    - POST /api/v1/processing-callback/results - send ML results
    - POST /api/v1/processing-callback/error - report errors
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        """Initialize backend client.

        Args:
            base_url: Backend base URL (e.g., https://api.demeter.com)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def send_results(
        self,
        tenant_id: str,
        results: ProcessingResultRequest,
    ) -> dict:
        """Send ML processing results to Backend.

        Args:
            tenant_id: Tenant identifier for RLS
            results: Processing results payload

        Returns:
            Backend response as dict

        Raises:
            httpx.HTTPStatusError: If request fails
        """
        url = f"{self.base_url}/api/v1/processing-callback/results"

        logger.info(
            "Sending results to backend",
            tenant_id=tenant_id,
            session_id=str(results.sessionId),
            image_id=str(results.imageId),
            detections_count=len(results.detections),
            estimations_count=len(results.estimations),
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=results.model_dump(mode="json"),
                headers={
                    "X-Tenant-ID": tenant_id,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            logger.info(
                "Results sent successfully",
                tenant_id=tenant_id,
                session_id=str(results.sessionId),
                status_code=response.status_code,
            )

            return response.json()

    async def report_error(
        self,
        tenant_id: str,
        session_id: UUID,
        image_id: UUID,
        error_message: str,
        error_type: str = "ProcessingError",
    ) -> dict:
        """Report processing error to Backend.

        Args:
            tenant_id: Tenant identifier for RLS
            session_id: Processing session ID
            image_id: Image that failed processing
            error_message: Error description
            error_type: Error classification

        Returns:
            Backend response as dict
        """
        url = f"{self.base_url}/api/v1/processing-callback/error"

        error_report = ErrorReport(
            sessionId=session_id,
            imageId=image_id,
            errorMessage=error_message,
            errorType=error_type,
        )

        logger.warning(
            "Reporting error to backend",
            tenant_id=tenant_id,
            session_id=str(session_id),
            image_id=str(image_id),
            error_type=error_type,
            error_message=error_message,
        )

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=error_report.model_dump(mode="json"),
                headers={
                    "X-Tenant-ID": tenant_id,
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            return response.json()
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_services/test_backend_client.py -v`
Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add app/services/ tests/test_services/
git commit -m "feat(services): add BackendClient for callback communication"
```

---

### Task 13: Add Backend URL to Config

**Files:**
- Modify: `app/config.py`
- Update: `.env.local.example`

**Step 1: Read current config**

Run: `cat app/config.py` to see current structure

**Step 2: Add backend config fields**

Add to `app/config.py` Settings class:

```python
# Backend callback configuration
backend_url: str = Field(
    default="http://localhost:8080",
    description="Backend API base URL for callbacks",
)
backend_callback_timeout: float = Field(
    default=30.0,
    description="Timeout for backend callback requests in seconds",
)
```

**Step 3: Update .env.local.example**

Add to `.env.local.example`:

```bash
# Backend callback
BACKEND_URL=http://localhost:8080
BACKEND_CALLBACK_TIMEOUT=30
```

**Step 4: Commit**

```bash
git add app/config.py .env.local.example
git commit -m "feat(config): add backend callback configuration"
```

---

### Task 14: Run Full Test Suite

**Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests PASS

**Step 2: Run type checking**

```bash
mypy app/models app/schemas app/services --ignore-missing-imports
```

Expected: No errors

**Step 3: Final commit with passing tests**

```bash
git add -A
git commit -m "test: verify all models and callback infrastructure pass"
```

---

## Checkpoint: Phase 1 & 2 Complete

At this point you should have:
- [ ] 10 read-only SQLAlchemy models
- [ ] Callback schemas matching Java DTOs
- [ ] BackendClient for HTTP communication
- [ ] All tests passing
- [ ] ~13 commits

**Next:** Phase 3 - Pipeline Integration (separate plan)
