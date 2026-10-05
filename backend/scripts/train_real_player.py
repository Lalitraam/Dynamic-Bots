"""
Standalone script to train a policy network for The_King_Crusher using real production defaults.
"""
import sys
import time
from pathlib import Path
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.training_runner import train_player_model

# Hook into ReduceLROnPlateau.step to log epoch progress in real time
_orig_step = ReduceLROnPlateau.step
epoch_counter = 0
epoch_start = time.time()

def _step_with_logging(self, metrics, epoch=None):
    global epoch_counter, epoch_start
    epoch_counter += 1
    duration = time.time() - epoch_start
    current_lr = self.optimizer.param_groups[0]["lr"]
    print(
        f"[EPOCH {epoch_counter}] finished in {duration:.1f}s | "
        f"val_loss: {metrics:.4f} | lr: {current_lr:.6f}",
        flush=True
    )
    epoch_start = time.time()
    return _orig_step(self, metrics, epoch)

ReduceLROnPlateau.step = _step_with_logging


def main():
    username = "the_king_crusher"
    cuda_available = torch.cuda.is_available()
    print("=" * 60)
    print(f"Starting training for '{username}'")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")
    print("Using production defaults from app.config (no overrides)")
    print("=" * 60, flush=True)

    t0 = time.time()
    result = train_player_model(username, overrides=None)
    total_time = time.time() - t0

    print("=" * 60)
    print(f"Training completed in {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"Epochs run: {result.get('epochs_run')}")
    print(f"Best val loss: {result.get('best_val_loss')}")
    print(f"Device recorded in training_info: {result.get('device')}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
