"""
Script to execute Phase E re-ingestion for the_king_crusher using production pipeline.
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

# Set up paths
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from app import config
from app.routers.ingest import _run_ingest, _jobs
from app.services import storage

async def main():
    username = "the_king_crusher"
    key = username.lower()
    _jobs[key] = {"status": "queued", "username": key, "progress": 0.0}

    print(f"Starting re-ingest for '{username}'...")
    await _run_ingest(username)
    status_payload = _jobs[key]
    print("\n--- INGEST STATUS PAYLOAD ---")
    print(json.dumps(status_payload, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
