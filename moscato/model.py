"""MOSP network with an optional State-Action Interaction (SAI) module.

The model consumes a per-video feature sequence ``x`` of shape ``[T, D]`` and
predicts, for every frame, the active states of every object: a logit tensor of
shape ``[T, O, S]`` (O objects, S states).  Training is multi-label BCE against
per-frame object-state (pseudo-)labels.

  * MOSP  = input projection -> sinusoidal positions -> Transformer encoder ->
            linear head to O*S logits.
  * SAI   = per-frame action embeddings that the frame features cross-attend to,
            explicitly injecting action context before the state head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_positions(length: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class SAIModule(nn.Module):
    """Cross-attention from frame features to per-frame action embeddings."""

    def __init__(self, hidden: int, num_actions: int, nhead: int = 4):
        super().__init__()
        # +1 index reserved for the "no action" background label.
        self.action_emb = nn.Embedding(num_actions + 1, hidden)
        self.attn = nn.MultiheadAttention(hidden, nhead, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        # h: [B, T, H]   action_ids: [B, T]
        a = self.action_emb(action_ids)               # [B, T, H]
        attn_out, _ = self.attn(h, a, a)              # frames attend over action context
        h = self.norm(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h


class MOSP(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_objects: int,
        num_states: int,
        num_actions: int,
        hidden: int = 256,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        use_sai: bool = True,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.num_states = num_states
        self.hidden = hidden
        self.use_sai = use_sai

        self.input_proj = nn.Linear(feature_dim, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.sai = SAIModule(hidden, num_actions, nhead) if use_sai else None
        self.head = nn.Linear(hidden, num_objects * num_states)

    def forward(self, features: torch.Tensor, action_ids: torch.Tensor) -> torch.Tensor:
        # features: [B, T, D]   action_ids: [B, T]  ->  logits [B, T, O, S]
        B, T, _ = features.shape
        h = self.input_proj(features)
        h = h + _sinusoidal_positions(T, self.hidden, features.device).unsqueeze(0)
        h = self.encoder(h)
        if self.sai is not None:
            h = self.sai(h, action_ids)
        logits = self.head(h)                          # [B, T, O*S]
        return logits.view(B, T, self.num_objects, self.num_states)
