"""PyTorch port of the official FM baseline — the trust anchor for Phase 3.

Consumes the exact same ``data.load()`` + ``data.encode()`` output as the numpy
baseline and trains a torch FM with identical hyperparameters. Reproducing valid
primary ≈ 0.6015 proves our training loop / eval wiring / early-stopping are correct,
so every later model (DeepFM / DIN / multi-task / CWM) has a verified foundation.

Fidelity notes vs ``baseline.FM``:
  * ``V`` init and the per-epoch shuffle are generated with numpy RNG seeded the same
    way, so the initial state and batch order match the numpy baseline.
  * ``V``/``W`` use Adam (b1=0.9, b2=0.999, eps=1e-8, lr=0.001); L2 (1e-6) is added
    only to ``V`` and ``W`` (the baseline does not regularize the bias).
  * The global bias ``b`` is rank-irrelevant (a constant shift leaves within-user
    GAUC / nDCG unchanged), so we let Adam optimize it rather than replicating the
    baseline's SGD-on-bias quirk — the ``V``/``W`` trajectory is unaffected.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_KIT = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit")
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)

from data import load, encode  # noqa: E402
from evaluate import evaluate  # noqa: E402

L2 = 1e-6


class FM(nn.Module):
    """Factorization Machine mirroring ``baseline.FM`` (k=16, l2=1e-6)."""

    def __init__(self, dim: int, k: int = 16, seed: int = 0):
        super().__init__()
        v_init = np.random.default_rng(seed).normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = nn.Embedding(dim, k)
        self.V.weight.data.copy_(torch.from_numpy(v_init))
        self.W = nn.Embedding(dim, 1)
        nn.init.zeros_(self.W.weight)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.V(x)                           # (B, F, k)
        s = e.sum(dim=1)                        # (B, k)
        inter = 0.5 * (s.square().sum(dim=1) - e.square().sum(dim=(1, 2)))
        lin = self.W(x).sum(dim=1).squeeze(1)   # (B,)
        return self.b + lin + inter


def train_epoch(model, optimizer, X, y, bs, rng):
    idx = rng.permutation(len(y))
    losses = []
    for i in range(0, len(idx), bs):
        batch = idx[i:i + bs]
        xb = torch.from_numpy(X[batch])
        yb = torch.from_numpy(y[batch])
        optimizer.zero_grad()
        logits = model(xb)
        bce = F.binary_cross_entropy_with_logits(logits, yb)
        reg = 0.5 * ((model.V.weight ** 2).sum() + (model.W.weight ** 2).sum())
        loss = bce + L2 * reg
        loss.backward()
        optimizer.step()
        losses.append(bce.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, X, bs=200_000):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + bs])).numpy() for i in range(0, len(X), bs)]
    model.train()
    return np.concatenate(out)


def run_fm_torch(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0):
    enc, dim = encode(splits)
    (Xtr, ytr, _), (Xva, yva, uva), (Xte, yte, ute) = enc['train'], enc['valid'], enc['test']
    model = FM(dim, k=k, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, optimizer, Xtr, ytr, bs, rng)
        va = evaluate(uva, yva, predict(model, Xva))
        print(f"  epoch {ep:2d} | loss {loss:.4f} | valid GAUC {va['GAUC']:.4f} "
              f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)
    model.eval()
    return {'valid': evaluate(uva, yva, predict(model, Xva)),
            'test': evaluate(ute, yte, predict(model, Xte))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()})
    res = run_fm_torch(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)
    print(f"\n=== fm_torch (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
