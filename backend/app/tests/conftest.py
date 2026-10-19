"""Pytest fixtures for backend test suite.

Ensures required environment configurations and base test model checkpoints exist.
"""

from pathlib import Path
import pytest
import torch

from app import config
from app.ml.model_v2 import ChessPolicyNetV2


@pytest.fixture(scope="session", autouse=True)
def ensure_base_checkpoint():
    """Ensure base_v1.pt exists for the test suite."""
    base_path = config.DATA_ROOT / "base" / "base_v1_fast" / "base_v1.pt"
    if not base_path.exists():
        base_path.parent.mkdir(parents=True, exist_ok=True)
        model = ChessPolicyNetV2(
            channels=96,
            num_blocks=6,
            policy_dim=32,
            meta_dim=10,
            num_planes=22,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": model.to_config(),
            },
            base_path,
        )
    return base_path
