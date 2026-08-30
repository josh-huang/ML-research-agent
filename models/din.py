"""DIN: Deep Interest Network over the user's past behavior sequence.

The official 5-field encoding ignores the behavior sequence entirely; DIN is the
organizer's headroom direction #2. For each row, the candidate video embedding attends
over the user's time-ordered past video ids (``hist``), and the pooled interest vector
feeds a wide-and-deep scorer together with the other fields.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DIN(nn.Module):
    def __init__(self, dim: int, n_fields: int, k: int = 16,
                 hidden=(64, 32), dropout: float = 0.0, aux_watch: bool = False,
                 cont_dim: int = 0):
        super().__init__()
        self.k = k
        self.other_idx = [i for i in range(n_fields) if i != 1]
        self.emb = nn.Embedding(dim, k)
        nn.init.normal_(self.emb.weight, 0.0, 0.01)
        self.lin = nn.Embedding(dim, 1)
        nn.init.zeros_(self.lin.weight)
        self.b = nn.Parameter(torch.zeros(1))
        self.aux_head = nn.Linear(k, 1) if aux_watch else None

        # DIN attention: score q vs h via [q, h, q-h, q*h] -> MLP -> scalar.
        self.attn = nn.Sequential(nn.Linear(4 * k, 16), nn.ReLU(), nn.Linear(16, 1))

        # deep scorer input: candidate (k) + pooled history (k) + other fields (n_fields-1)*k
        in_dim = k + k + (n_fields - 1) * k + cont_dim
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, hist=None, hist_mask=None, cont=None) -> torch.Tensor:
        e = self.emb(x)                     # (B, F, k)
        q = e[:, 1]                         # candidate = field 1 (video_id)
        wide = self.lin(x).sum(dim=1).squeeze(1) + self.b

        if hist is not None and hist_mask is not None:
            h = self.emb(hist)              # (B, K, k)
            bsz, K, _ = h.shape
            qe = q.unsqueeze(1).expand(bsz, K, self.k)
            cat = torch.cat([qe, h, qe - h, qe * h], dim=-1)  # (B, K, 4k)
            scores = self.attn(cat).squeeze(-1)               # (B, K)
            scores = scores.masked_fill(hist_mask < 0.5, -1e9)
            w = F.softmax(scores, dim=1)                       # (B, K)
            pooled = (w.unsqueeze(-1) * h).sum(dim=1)          # (B, k)
            has_hist = (hist_mask.sum(dim=1, keepdim=True) > 0).float()
            pooled = pooled * has_hist
        else:
            pooled = torch.zeros_like(q)

        other = e[:, self.other_idx].reshape(x.size(0), -1)    # (B, (F-1)*k)
        deep_in = torch.cat([q, pooled, other], dim=-1)
        if cont is not None:
            deep_in = torch.cat([deep_in, cont], dim=-1)
        deep = self.mlp(deep_in).squeeze(1)
        return wide + deep

    def aux_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CWM watch-fraction head over the shared candidate embedding (mean-pooled)."""
        return self.aux_head(self.emb(x).mean(dim=1)).squeeze(1)
