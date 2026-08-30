"""Factorization Machine and DeepFM (shared-embedding variant).

FM is the integrated, cleanly-initialised counterpart to the trust-anchor
``fm_torch.FM``; DeepFM adds a DNN tower over the same shared field embedding. The
organizer flags raw model capacity as a near-dead-end, so DeepFM is here mainly to
confirm that finding cheaply — the real levers are loss alignment and sequence modeling.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FM(nn.Module):
    """Factorization Machine over the official 5-field global-offset encoding."""

    def __init__(self, dim: int, k: int = 16, aux_watch: bool = False, cont_dim: int = 0):
        super().__init__()
        self.V = nn.Embedding(dim, k)
        nn.init.normal_(self.V.weight, 0.0, 0.01)
        self.W = nn.Embedding(dim, 1)
        nn.init.zeros_(self.W.weight)
        self.b = nn.Parameter(torch.zeros(1))
        self.aux_head = nn.Linear(k, 1) if aux_watch else None
        self.cont_lin = nn.Linear(cont_dim, 1, bias=False) if cont_dim > 0 else None

    def forward(self, x: torch.Tensor, cont: torch.Tensor | None = None) -> torch.Tensor:
        e = self.V(x)                       # (B, F, k)
        s = e.sum(dim=1)                    # (B, k)
        inter = 0.5 * (s.square().sum(dim=1) - e.square().sum(dim=(1, 2)))
        lin = self.W(x).sum(dim=1).squeeze(1)  # (B,)
        out = self.b + lin + inter
        if cont is not None and self.cont_lin is not None:
            out = out + self.cont_lin(cont).squeeze(1)
        return out

    def aux_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CWM watch-fraction head over the shared field embedding (mean-pooled)."""
        return self.aux_head(self.V(x).mean(dim=1)).squeeze(1)


class DeepFM(nn.Module):
    """DeepFM: FM pairwise term + DNN tower over the shared field embeddings."""

    def __init__(self, dim: int, n_fields: int, k: int = 16,
                 dnn_hidden=(64, 32), dropout: float = 0.0, aux_watch: bool = False,
                 cont_dim: int = 0):
        super().__init__()
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, 0.0, 0.01)
        self.lin = nn.Embedding(dim, 1)
        nn.init.zeros_(self.lin.weight)
        self.b = nn.Parameter(torch.zeros(1))
        self.aux_head = nn.Linear(k, 1) if aux_watch else None

        layers = []
        in_dim = n_fields * k + cont_dim
        for h in dnn_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cont: torch.Tensor | None = None) -> torch.Tensor:
        e = self.emb(x)                     # (B, F, k)
        s = e.sum(dim=1)
        inter = 0.5 * (s.square().sum(dim=1) - e.square().sum(dim=(1, 2)))
        lin = self.lin(x).sum(dim=1).squeeze(1)
        fm = self.b + lin + inter
        flat = e.reshape(x.size(0), -1)
        dnn_in = torch.cat([flat, cont], dim=1) if cont is not None else flat
        dnn = self.mlp(dnn_in).squeeze(1)
        return fm + dnn

    def aux_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CWM watch-fraction head over the shared field embedding (mean-pooled)."""
        return self.aux_head(self.emb(x).mean(dim=1)).squeeze(1)
