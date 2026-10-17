"""
ChessPolicyNetV2: ResNet trunk with FiLM metadata conditioning and square-factored policy head.
Plan C2 / Decision D-01 architecture for Phase B / Phase C imitation learning.
Pure nn.Module with no dataset or framework dependencies.
"""
from __future__ import annotations

import math
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Pre/post-activation residual block for 8x8 spatial chess boards."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ChessPolicyNetV2(nn.Module):
    """
    ChessPolicyNetV2 architecture:
      - Stem: Conv3x3(P -> channels) + BN + ReLU
      - Trunk: num_blocks x ResBlock(channels); after each block, FiLM(meta)
      - Square representation: x -> (B, 64, channels) row-major (sq = rank*8 + file)
      - Policy: query = Linear(channels->policy_dim), key = Linear(channels->policy_dim)
                logits[b, f, t] = q[b,f] . k[b,t] / sqrt(policy_dim) + bias_pair[f*64+t]
                -> (B, 4096)
      - Promo: Linear(2*channels -> 4)(concat(sq[f], sq[t]))
    """

    def __init__(
        self,
        channels: int = 96,
        num_blocks: int = 6,
        policy_dim: int = 32,
        meta_dim: int = 10,
        num_planes: int = 22,
    ):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.policy_dim = policy_dim
        self.meta_dim = meta_dim
        self.num_planes = num_planes

        # Stem
        self.stem_conv = nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        # Meta conditioning MLP (10 -> 64 -> 64)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Trunk blocks and per-block FiLM layers
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])
        self.film_generators = nn.ModuleList([
            nn.Linear(64, 2 * channels) for _ in range(num_blocks)
        ])

        # Initialize FiLM layers to identity (gamma=0, beta=0)
        for film in self.film_generators:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

        # Policy projections
        self.q_proj = nn.Linear(channels, policy_dim)
        self.k_proj = nn.Linear(channels, policy_dim)
        self.bias_pair = nn.Parameter(torch.zeros(4096))

        # Promotion head (takes concat of from_sq and to_sq features -> 4 classes)
        self.promo_head = nn.Linear(2 * channels, 4)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ChessPolicyNetV2":
        """Instantiate model from a configuration dictionary."""
        return cls(
            channels=int(cfg.get("channels", 96)),
            num_blocks=int(cfg.get("num_blocks", 6)),
            policy_dim=int(cfg.get("policy_dim", 32)),
            meta_dim=int(cfg.get("meta_dim", 10)),
            num_planes=int(cfg.get("num_planes", 22)),
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize configuration parameters to dictionary."""
        return {
            "channels": self.channels,
            "num_blocks": self.num_blocks,
            "policy_dim": self.policy_dim,
            "meta_dim": self.meta_dim,
            "num_planes": self.num_planes,
        }

    def forward(
        self,
        board: torch.Tensor,
        meta: torch.Tensor,
        target_move_idx: Optional[torch.Tensor] = None,
        target_from: Optional[torch.Tensor] = None,
        target_to: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        board : torch.Tensor
            Board tensor (B, P, 8, 8), P in {18, 22}.
        meta : torch.Tensor
            Metadata tensor (B, meta_dim).
        target_move_idx : torch.Tensor, optional
            Target move indices (B,) in [0, 4095] for promotion feature gathering.
        target_from : torch.Tensor, optional
            Target from-square indices (B,) in [0, 63].
        target_to : torch.Tensor, optional
            Target to-square indices (B,) in [0, 63].

        Returns
        -------
        dict[str, torch.Tensor]
            {"policy_logits": (B, 4096), "promo_logits": (B, 4)}
        """
        B = board.shape[0]

        # Convert to float if necessary
        if not board.is_floating_point():
            board = board.float()
        if not meta.is_floating_point():
            meta = meta.float()

        # Handle plane mismatch (e.g. 18 planes passed to 22-plane model)
        if board.shape[1] < self.num_planes:
            pad = torch.zeros(
                B, self.num_planes - board.shape[1], 8, 8,
                device=board.device, dtype=board.dtype
            )
            board = torch.cat([board, pad], dim=1)
        elif board.shape[1] > self.num_planes:
            board = board[:, :self.num_planes]

        # Stem
        x = F.relu(self.stem_bn(self.stem_conv(board)))

        # Meta conditioning embedding
        meta_emb = self.meta_mlp(meta)  # (B, 64)

        # Trunk with FiLM
        for block, film in zip(self.blocks, self.film_generators):
            x = block(x)
            gb = film(meta_emb)  # (B, 2 * channels)
            gamma, beta = gb.chunk(2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            x = x * (1.0 + gamma) + beta

        # Square embeddings: (B, C, 8, 8) -> permute to (B, 8, 8, C) -> (B, 64, C)
        # matches square index = rank * 8 + file (0=a1, 63=h8)
        sq = x.permute(0, 2, 3, 1).reshape(B, 64, self.channels)

        # Policy logits via query-key dot product
        q = self.q_proj(sq)  # (B, 64, policy_dim)
        k = self.k_proj(sq)  # (B, 64, policy_dim)
        scale = 1.0 / math.sqrt(self.policy_dim)
        logits_matrix = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, 64, 64)
        policy_logits = logits_matrix.reshape(B, 4096) + self.bias_pair  # (B, 4096)

        # Promo logits from (sq[f], sq[t])
        if target_from is not None and target_to is not None:
            f = target_from
            t = target_to
        elif target_move_idx is not None:
            f = target_move_idx // 64
            t = target_move_idx % 64
        else:
            chosen = policy_logits.argmax(dim=-1)
            f = chosen // 64
            t = chosen % 64

        b_idx = torch.arange(B, device=sq.device)
        sq_f = sq[b_idx, f]  # (B, channels)
        sq_t = sq[b_idx, t]  # (B, channels)
        promo_in = torch.cat([sq_f, sq_t], dim=-1)  # (B, 2 * channels)
        promo_logits = self.promo_head(promo_in)  # (B, 4)

        return {
            "policy_logits": policy_logits,
            "promo_logits": promo_logits,
        }
