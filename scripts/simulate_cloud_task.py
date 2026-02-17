#!/usr/bin/env python
"""Simulate Cloud Tasks requests for local testing.

This script sends HTTP requests to the ML Worker endpoint simulating
how Cloud Tasks would call it in production.

Usage:
    # Basic usage with a local image
    python scripts/simulate_cloud_task.py --image ./test_image.jpg

    # Specify tenant and pipeline
    python scripts/simulate_cloud_task.py \
        --image ./test_image.jpg \
        --tenant test-tenant \
        --pipeline FULL_PIPELINE

    # With species config (for agro classification)
    python scripts/simulate_cloud_task.py \
        --image ./test_image.jpg \
        --species '[{"product_name": "Tomato", "product_id": 1}]'
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

import httpx


# Cloud Tasks headers that would be sent in production
CLOUD_TASKS_HEADERS = {
    "X-CloudTasks-TaskName": "simulated-task-{task_id}",
    "X-CloudTasks-QueueName": "local-test-queue",
    "X-CloudTasks-TaskRetryCount": "0",
    "X-CloudTasks-TaskExecutionCount": "1",
    "X-CloudTasks-TaskETA": "0",
}


async def send_process_request(
    base_url: str,
    tenant_id: str,
    image_path: Path,
    pipeline: str,
    species_config: list[dict] | None = None,
    options: dict | None = None,
) -> dict:
    """Send a simulated Cloud Tasks request to /tasks/process.

    Args:
        base_url: ML Worker base URL (e.g., http://localhost:8080)
        tenant_id: Tenant ID for multi-tenant isolation
        image_path: Path to local image file
        pipeline: Pipeline name (e.g., DETECTION, FULL_PIPELINE)
        species_config: Optional species configuration for classification
        options: Optional processing overrides

    Returns:
        Response JSON from the ML Worker
    """
    task_id = str(uuid.uuid4())[:8]
    session_id = str(uuid.uuid4())
    image_id = str(uuid.uuid4())

    # Build headers simulating Cloud Tasks
    headers = {
        k: v.format(task_id=task_id)
        for k, v in CLOUD_TASKS_HEADERS.items()
    }
    headers["Content-Type"] = "application/json"

    # For local testing, we use a fake GCS URL
    # The storage client should be mocked or configured for local files
    fake_gcs_url = f"gs://test-bucket/{tenant_id}/images/{image_path.name}"

    payload = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "image_id": image_id,
        "image_url": fake_gcs_url,
        "pipeline": pipeline,
    }

    if species_config:
        payload["species_config"] = species_config
    if options:
        payload["options"] = options

    print(f"\n{'='*60}")
    print(f"Simulating Cloud Task: {task_id}")
    print(f"{'='*60}")
    print(f"Tenant:    {tenant_id}")
    print(f"Pipeline:  {pipeline}")
    print(f"Image:     {image_path}")
    print(f"Image ID:  {image_id}")
    print(f"Session:   {session_id}")
    print(f"{'='*60}\n")

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{base_url}/tasks/process",
            json=payload,
            headers=headers,
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success')}")
            print(f"Duration: {result.get('duration_ms')}ms")
            print(f"Steps completed: {result.get('steps_completed')}")
            if result.get('error'):
                print(f"Error: {result.get('error')}")
            print(f"\nResults keys: {list(result.get('results', {}).keys())}")
            return result
        else:
            print(f"Error: {response.text}")
            return {"error": response.text, "status_code": response.status_code}


async def check_health(base_url: str) -> bool:
    """Check if the ML Worker is running and healthy."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate Cloud Tasks requests for local ML Worker testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to local image file",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="ML Worker base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--tenant",
        default="test-tenant",
        help="Tenant ID (default: test-tenant)",
    )
    parser.add_argument(
        "--pipeline",
        default="DETECTION",
        help="Pipeline name (default: DETECTION)",
    )
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Species config JSON (e.g., '[{\"product_name\": \"Tomato\", \"product_id\": 1}]')",
    )
    parser.add_argument(
        "--skip-health",
        action="store_true",
        help="Skip health check before sending request",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save response JSON to file",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate image exists
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        return 1

    # Check health unless skipped
    if not args.skip_health:
        print(f"Checking ML Worker health at {args.url}...")
        if not await check_health(args.url):
            print("Error: ML Worker is not healthy or not running")
            print("Make sure the server is running: uvicorn app.main:app --port 8080")
            return 1
        print("ML Worker is healthy!\n")

    # Parse species config if provided
    species_config = None
    if args.species:
        try:
            species_config = json.loads(args.species)
        except json.JSONDecodeError as e:
            print(f"Error parsing species config JSON: {e}")
            return 1

    # Send process request
    result = await send_process_request(
        base_url=args.url,
        tenant_id=args.tenant,
        image_path=args.image,
        pipeline=args.pipeline,
        species_config=species_config,
    )

    # Save output if requested
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nResponse saved to: {args.output}")

    # Return exit code based on success
    if result.get("success"):
        print("\nTask completed successfully!")
        return 0
    else:
        print("\nTask failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
