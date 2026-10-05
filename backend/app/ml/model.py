"""
CNN architecture for chess player-cloning (Milestone 3).

Class: ChessPolicyNet
---------------------
Input
  board : (B, 18, 8, 8) uint8 **or** float32  -- cast to float32 internally
  meta  : (B, META_DIM) float32

Trunk
  initial_conv  : Conv2d(18 -> channels, 3x3, pad 1) + BN + ReLU
  res_blocks    : num_res_blocks x ResidualBlock (two 3x3 convs, BN, ReLU, skip)
  dropout2d     : Dropout2d(dropout)
  flatten       : channels*8*8 units
  meta_mlp      : Linear(META_DIM -> 32) + ReLU + Linear(32 -> 32) + ReLU
  concat        : [flattened_trunk | meta_embedding]
  shared_head   : Linear(channels*64 + 32 -> hidden_dim) + ReLU + Dropout(dropout)

Heads (all from shared hidden_dim)
  from_sq_logits     : Linear(hidden_dim, 64)
  to_sq_logits       : Linear(hidden_dim, 64)
  promo_logits       : Linear(hidden_dim, 4)
  piece_type_logits  : Linear(hidden_dim, 6)
  is_capture_logit   : Linear(hidden_dim, 1)
  is_check_logit     : Linear(hidden_dim, 1)
  is_castle_logit    : Linear(hidden_dim, 1)

forward() returns a dict of raw logits (no softmax/sigmoid applied).

Weight initialisation
---------------------
Kaiming-uniform (fan_in, relu) for all Conv2d and Linear layers.
BatchNorm2d weights set to 1, biases to 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class _ResidualBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs with BN+ReLU and a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class ChessPolicyNet(nn.Module):
    """Per-player chess CNN.  See module docstring for full architecture description."""

    def __init__(
        self,
        channels:       int   = config.MODEL_CHANNELS,
        num_res_blocks: int   = config.MODEL_RES_BLOCKS,
        hidden_dim:     int   = config.MODEL_HIDDEN_DIM,
        dropout:        float = config.MODEL_DROPOUT,
        meta_dim:       int   = config.META_DIM,
    ) -> None:
        super().__init__()

        self.channels   = channels
        self.hidden_dim = hidden_dim

        # --- Trunk ---
        self.initial_conv = nn.Conv2d(
            in_channels=config.BOARD_PLANES,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.initial_bn   = nn.BatchNorm2d(channels)
        self.initial_relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(
            *[_ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.trunk_dropout = nn.Dropout2d(p=dropout)

        trunk_flat_dim = channels * 8 * 8  # 4096 for default channels=64

        # --- Meta MLP ---
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # --- Shared head ---
        combined_dim   = trunk_flat_dim + 32
        self.shared_fc   = nn.Linear(combined_dim, hidden_dim)
        self.shared_relu = nn.ReLU(inplace=True)
        self.shared_drop = nn.Dropout(p=dropout)

        # --- Output heads ---
        self.from_sq_logits    = nn.Linear(hidden_dim, 64)
        self.to_sq_logits      = nn.Linear(hidden_dim, 64)
        self.promo_logits      = nn.Linear(hidden_dim, config.PROMO_CLASSES)
        self.piece_type_logits = nn.Linear(hidden_dim, 6)
        self.is_capture_logit  = nn.Linear(hidden_dim, 1)
        self.is_check_logit    = nn.Linear(hidden_dim, 1)
        self.is_castle_logit   = nn.Linear(hidden_dim, 1)

        # Weight initialisation (Kaiming-uniform, fan_in, relu nonlinearity)
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Initialise weights: Kaiming-uniform for conv/linear; BN to identity."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        board: torch.Tensor,
        meta:  torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        board : (B, 18, 8, 8) -- uint8 or float32 (cast to float32 internally)
        meta  : (B, META_DIM) float32

        Returns
        -------
        dict with keys:
            from_sq_logits, to_sq_logits, promo_logits,
            piece_type_logits, is_capture_logit, is_check_logit, is_castle_logit
        All values are raw logits (no softmax/sigmoid applied).
        """
        # Accept uint8 (compact Dataset storage) or float32 -- cast internally
        x = board.float()  # (B, 18, 8, 8)

        # Trunk
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))  # (B, C, 8, 8)
        x = self.res_blocks(x)                                         # (B, C, 8, 8)
        x = self.trunk_dropout(x)                                      # (B, C, 8, 8)
        x = x.flatten(start_dim=1)                                     # (B, C*64)

        # Meta embedding
        m = self.meta_mlp(meta)                                        # (B, 32)

        # Combine
        combined = torch.cat([x, m], dim=1)                            # (B, C*64+32)
        h = self.shared_drop(self.shared_relu(self.shared_fc(combined)))  # (B, hidden_dim)

        return {
            "from_sq_logits":    self.from_sq_logits(h),     # (B, 64)
            "to_sq_logits":      self.to_sq_logits(h),       # (B, 64)
            "promo_logits":      self.promo_logits(h),        # (B, 4)
            "piece_type_logits": self.piece_type_logits(h),   # (B, 6)
            "is_capture_logit":  self.is_capture_logit(h),    # (B, 1)
            "is_check_logit":    self.is_check_logit(h),      # (B, 1)
            "is_castle_logit":   self.is_castle_logit(h),     # (B, 1)
        }
