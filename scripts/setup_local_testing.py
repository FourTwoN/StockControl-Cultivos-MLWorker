#!/usr/bin/env python
"""Setup local testing environment for ML Worker.

This script:
1. Creates a test tenant configuration in the database
2. Configures the storage client for local file access

Usage:
    # Setup test tenant with default pipeline
    python scripts/setup_local_testing.py --tenant test-tenant

    # Setup with custom pipeline steps
    python scripts/setup_local_testing.py \
        --tenant test-tenant \
        --steps segmentation,segment_filter,detection

    # List available steps
    python scripts/setup_local_testing.py --list-steps
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.step_registry import StepRegistry
from app.steps import register_all_steps
from app.infra.database import get_db_session
from app.infra.logging import setup_logging, get_logger
from sqlalchemy import text


setup_logging()
logger = get_logger(__name__)


# Default pipeline configurations for testing
DEFAULT_PIPELINES = {
    "detection_only": ["detection"],
    "segmentation_only": ["segmentation"],
    "full_agro": [
        "segmentation",
        "segment_filter",
        "detection",
        "size_calculator",
        "species_distributor",
    ],
    "sahi_detection": ["segmentation", "segment_filter", "sahi_detection"],
}


async def ensure_tenant_config_table() -> bool:
    """Ensure tenant_config table exists in the database."""
    try:
        async with get_db_session() as session:
            # Create table if not exists
            create_table_sql = text("""
                CREATE TABLE IF NOT EXISTS tenant_config (
                    tenant_id VARCHAR(100) PRIMARY KEY,
                    pipeline_steps JSONB NOT NULL,
                    settings JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            await session.execute(create_table_sql)
            await session.commit()
            logger.info("tenant_config table ensured")
            return True
    except Exception as e:
        logger.error("Failed to create tenant_config table", error=str(e))
        return False


async def create_tenant_config(
    tenant_id: str,
    pipeline_steps: list[str],
    settings: dict | None = None,
) -> bool:
    """Create or update tenant configuration in the database.

    Args:
        tenant_id: Tenant identifier
        pipeline_steps: List of pipeline step names
        settings: Optional settings dict

    Returns:
        True if successful
    """
    import json

    settings = settings or {}

    # Ensure table exists first
    if not await ensure_tenant_config_table():
        return False

    try:
        async with get_db_session() as session:
            # Upsert tenant config
            query = text("""
                INSERT INTO tenant_config (tenant_id, pipeline_steps, settings)
                VALUES (:tenant_id, :pipeline_steps, :settings)
                ON CONFLICT (tenant_id) DO UPDATE SET
                    pipeline_steps = EXCLUDED.pipeline_steps,
                    settings = EXCLUDED.settings,
                    updated_at = NOW()
            """)

            await session.execute(
                query,
                {
                    "tenant_id": tenant_id,
                    "pipeline_steps": json.dumps(pipeline_steps),
                    "settings": json.dumps(settings),
                },
            )
            await session.commit()

            logger.info(
                "Tenant config created/updated",
                tenant_id=tenant_id,
                pipeline_steps=pipeline_steps,
            )
            return True

    except Exception as e:
        logger.error("Failed to create tenant config", error=str(e))
        return False


async def list_available_steps() -> list[str]:
    """List all registered pipeline steps."""
    register_all_steps()
    return StepRegistry.available_steps()


async def verify_pipeline_steps(steps: list[str]) -> tuple[bool, list[str]]:
    """Verify that all pipeline steps are registered.

    Args:
        steps: List of step names to verify

    Returns:
        Tuple of (all_valid, list_of_invalid_steps)
    """
    register_all_steps()
    available = StepRegistry.available_steps()
    invalid = [s for s in steps if s not in available]
    return len(invalid) == 0, invalid


async def get_tenant_config(tenant_id: str) -> dict | None:
    """Get existing tenant configuration."""
    try:
        async with get_db_session() as session:
            query = text("""
                SELECT tenant_id, pipeline_steps, settings
                FROM tenant_config
                WHERE tenant_id = :tenant_id
            """)
            result = await session.execute(query, {"tenant_id": tenant_id})
            row = result.fetchone()

            if row:
                return {
                    "tenant_id": row[0],
                    "pipeline_steps": row[1],
                    "settings": row[2],
                }
            return None

    except Exception as e:
        logger.error("Failed to get tenant config", error=str(e))
        return None


async def list_all_tenants() -> list[dict]:
    """List all configured tenants."""
    try:
        async with get_db_session() as session:
            query = text("""
                SELECT tenant_id, pipeline_steps, settings, updated_at
                FROM tenant_config
                ORDER BY updated_at DESC
            """)
            result = await session.execute(query)
            rows = result.fetchall()

            return [
                {
                    "tenant_id": row[0],
                    "pipeline_steps": row[1],
                    "settings": row[2],
                    "updated_at": row[3],
                }
                for row in rows
            ]

    except Exception as e:
        logger.error("Failed to list tenants", error=str(e))
        return []


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Setup local testing environment for ML Worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--tenant",
        type=str,
        help="Tenant ID to create/update",
    )
    parser.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of pipeline steps",
    )
    parser.add_argument(
        "--preset",
        choices=list(DEFAULT_PIPELINES.keys()),
        help="Use a preset pipeline configuration",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="{}",
        help="JSON settings string (default: {})",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all available pipeline steps",
    )
    parser.add_argument(
        "--list-tenants",
        action="store_true",
        help="List all configured tenants",
    )
    parser.add_argument(
        "--show",
        type=str,
        metavar="TENANT_ID",
        help="Show configuration for a specific tenant",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # List available steps
    if args.list_steps:
        print("\nAvailable pipeline steps:")
        print("-" * 40)
        steps = await list_available_steps()
        for step in sorted(steps):
            print(f"  - {step}")
        print(f"\nTotal: {len(steps)} steps")

        print("\nPreset pipelines:")
        print("-" * 40)
        for name, steps in DEFAULT_PIPELINES.items():
            print(f"  {name}: {' -> '.join(steps)}")
        return 0

    # List all tenants
    if args.list_tenants:
        print("\nConfigured tenants:")
        print("-" * 60)
        tenants = await list_all_tenants()
        if not tenants:
            print("  No tenants configured")
        for t in tenants:
            print(f"  {t['tenant_id']}")
            print(f"    Steps: {t['pipeline_steps']}")
            print(f"    Updated: {t['updated_at']}")
        return 0

    # Show specific tenant
    if args.show:
        config = await get_tenant_config(args.show)
        if config:
            import json
            print(f"\nTenant: {config['tenant_id']}")
            print("-" * 40)
            print(f"Pipeline steps: {config['pipeline_steps']}")
            print(f"Settings: {json.dumps(config['settings'], indent=2)}")
        else:
            print(f"Tenant not found: {args.show}")
        return 0

    # Create/update tenant config
    if not args.tenant:
        print("Error: --tenant is required (or use --list-steps, --list-tenants)")
        return 1

    # Determine pipeline steps
    if args.preset:
        pipeline_steps = DEFAULT_PIPELINES[args.preset]
    elif args.steps:
        pipeline_steps = [s.strip() for s in args.steps.split(",")]
    else:
        pipeline_steps = DEFAULT_PIPELINES["detection_only"]

    # Verify steps are valid
    valid, invalid = await verify_pipeline_steps(pipeline_steps)
    if not valid:
        print(f"Error: Unknown pipeline steps: {invalid}")
        print("Use --list-steps to see available steps")
        return 1

    # Parse settings
    import json
    try:
        settings = json.loads(args.settings)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid settings JSON: {e}")
        return 1

    # Create tenant config
    print(f"\nCreating tenant configuration:")
    print(f"  Tenant: {args.tenant}")
    print(f"  Pipeline: {' -> '.join(pipeline_steps)}")
    print(f"  Settings: {settings}")
    print()

    success = await create_tenant_config(
        tenant_id=args.tenant,
        pipeline_steps=pipeline_steps,
        settings=settings,
    )

    if success:
        print("Tenant configuration created successfully!")
        print(f"\nYou can now test with:")
        print(f"  python scripts/simulate_cloud_task.py --tenant {args.tenant} --image ./test.jpg")
        return 0
    else:
        print("Failed to create tenant configuration")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
