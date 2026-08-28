"""Unified training loop for Phase 3 models.

Drives FM / DeepFM / DIN with pointwise BCE, pairwise BPR, or listwise softmax losses.
Config is a plain dict (project convention); a thin ``main()`` exposes the common knobs
as CLI flags for manual runs, while the Agent calls ``run_experiment`` programmatically.

Invoke from the project root::

    python -m models.train --data_dir .../KuaiRand-Pure/data --model fm --loss bpr
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.harness import evaluate  # noqa: E402
from models.losses import bce_loss, bpr_loss, listwise_softmax_loss  # noqa: E402
from models import deepfm, din  # noqa: E402


def make_model(config, dim, n_fields):
    name = config["model"]
    k = config.get("k", 16)
    if name == "fm":
        return deepfm.FM(dim, k=k)
    if name == "deepfm":
        return deepfm.DeepFM(dim, n_fields, k=k,
                             dnn_hidden=config.get("dnn_hidden", (64, 32)),
                             dropout=config.get("dropout", 0.0))
    if name == "din":
        return din.DIN(dim, n_fields, k=k,
                       hidden=config.get("dnn_hidden", (64, 32)),
                       dropout=config.get("dropout", 0.0))
    raise ValueError(f"unknown model {name!r}")


def _row_batches(n, bs, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]


def _user_batches(user_idx, target_rows):
    """Group rows by user (stable), then pack consecutive users into row-bounded batches.

    Returns a list of ``(idx, segments)`` where ``segments`` marks user boundaries
    within the batch (shape U+1, first 0, last batch size).
    """
    order = np.argsort(user_idx, kind="stable")
    suid = user_idx[order]
    change = np.where(suid[1:] != suid[:-1])[0] + 1
    boundaries = np.concatenate([[0], change, [len(order)]]).astype(np.int64)
    n_users = len(boundaries) - 1
    batches = []
    u = 0
    while u < n_users:
        s = boundaries[u]
        e = u + 1
        while e < n_users and boundaries[e] - s < target_rows:
            e += 1
        idx = order[s:boundaries[e]]
        seg = np.concatenate([boundaries[u:e] - s, [boundaries[e] - s]]).astype(np.int64)
        batches.append((idx, seg))
        u = e
    return batches


def _forward(model, xb, split, idx, device):
    if isinstance(model, din.DIN):
        hist = torch.from_numpy(split["hist"][idx]).to(device)
        mask = torch.from_numpy(split["hist_mask"][idx]).to(device)
        return model(xb, hist, mask)
    return model(xb)


@torch.no_grad()
def _predict(model, split, X, device, bs=200_000):
    model.eval()
    outs = []
    for i in range(0, len(X), bs):
        sl = slice(i, i + bs)
        xb = torch.from_numpy(X[sl]).to(device)
        outs.append(_forward(model, xb, split, sl, device).cpu().numpy())
    model.train()
    return np.concatenate(outs)


def run_experiment(data, config, verbose=True):
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dim, n_fields = data["dim"], data["n_fields"]
    model = make_model(config, dim, n_fields).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    loss_name = config.get("loss", "bce")
    bs = config.get("bs", 8192)
    epochs = config.get("epochs", 40)
    patience = config.get("patience", 4)

    tr, va, te = data["train"], data["valid"], data["test"]
    Xtr, ytr = tr["X"], tr["y"].astype(np.float32)

    if loss_name == "bce":
        batches = _row_batches(len(ytr), bs, seed)
    else:
        batches = _user_batches(Xtr[:, 0], bs)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, used = 0.0, 0
        for batch in batches:
            optimizer.zero_grad()
            if loss_name == "bce":
                idx, seg = batch, None
            else:
                idx, seg = batch
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            logits = _forward(model, xb, tr, idx, device)
            if loss_name == "bce":
                loss = bce_loss(logits, yb)
            elif loss_name == "bpr":
                loss = bpr_loss(logits, yb, torch.from_numpy(seg).to(device))
            else:
                loss = listwise_softmax_loss(logits, yb, torch.from_numpy(seg).to(device))
            if loss is not None:
                loss.backward()
                optimizer.step()
                total += loss.item()
                used += 1
        vam = evaluate(va["users"], va["y"], _predict(model, va, va["X"], device))
        if verbose:
            print(f"  epoch {ep:2d} | loss {total / max(used, 1):.4f} | valid GAUC {vam['GAUC']:.4f} "
                  f"nDCG@5 {vam['nDCG@5']:.4f} primary {vam['primary']:.4f} | {time.time() - t0:.1f}s")
        if vam["primary"] > best + 1e-5:
            best, bad = vam["primary"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    model.eval()
    te_scores = _predict(model, te, te["X"], device)
    return {
        "valid": evaluate(va["users"], va["y"], _predict(model, va, va["X"], device)),
        "test": evaluate(te["users"], te["y"], te_scores),
        "test_scores": te_scores,   # raw per-row test scores (submission.csv)
        "best_valid": best,
        "epochs_run": ep,
        "config": config,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="kuairand-starter-kit/kuairand-starter-kit/KuaiRand-Pure/data")
    ap.add_argument("--model", default="fm", choices=["fm", "deepfm", "din"])
    ap.add_argument("--loss", default="bce", choices=["bce", "bpr", "listwise"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--dnn_hidden", type=str, default="64,32")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    from models.data_loader import load_extended
    data = load_extended(a.data_dir)
    config = {
        "model": a.model, "loss": a.loss, "k": a.k, "lr": a.lr,
        "epochs": a.epochs, "bs": a.bs, "seed": a.seed,
        "dnn_hidden": tuple(int(x) for x in a.dnn_hidden.split(",")),
        "dropout": a.dropout,
    }
    if a.device:
        config["device"] = a.device
    print(f"config: {config}")
    res = run_experiment(data, config)
    print(f"\n=== {a.model} ({a.loss}, seed={a.seed}) ===")
    for sp in ("valid", "test"):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")


if __name__ == "__main__":
    main()
