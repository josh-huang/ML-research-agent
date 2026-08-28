"""Ranking losses for within-user pairwise / listwise training.

The official baseline uses pointwise BCE. The organizer README lists loss alignment
(pointwise -> pairwise BPR / listwise) as headroom direction #1. These losses require
samples to be grouped by user, so they take a ``segments`` tensor of user boundaries.

Convention: ``segments`` is a LongTensor of shape (U+1,) indexing into a flattened
(N,) array, where user u's impressions occupy [segments[u], segments[u+1]). Users with
no ranking signal (all-positive or all-negative) contribute zero to the loss.

Both ranking losses are fully vectorised (scatter_add / scatter_reduce, no Python loop)
because a per-user loop over ~21k users/epoch with CUDA kernels is ~100x too slow.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pointwise BCE (matches the official baseline's loss)."""
    return F.binary_cross_entropy_with_logits(logits, labels)


def _seg_meta(segments):
    """Return (seg_id, lengths) where seg_id maps each row to its user index."""
    lengths = segments[1:] - segments[:-1]
    seg_id = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=segments.device), lengths)
    return seg_id, lengths


def listwise_softmax_loss(logits, labels, segments):
    """ListNet-style softmax CE within each user's impression list (vectorised).

    softmax over the user's full list; loss = -mean over positive positions of log p_c.
    Users with no positives (or all positives) are skipped.
    """
    n = logits.shape[0]
    device = logits.device
    U = segments.shape[0] - 1
    seg_id, lengths = _seg_meta(segments)

    # stable per-segment logsumexp
    max_per_seg = torch.full((U,), -float("inf"), device=device)
    max_per_seg.scatter_reduce_(0, seg_id, logits, reduce="amax")
    exp_sum = torch.zeros(U, device=device)
    exp_sum.scatter_add_(0, seg_id, torch.exp(logits - max_per_seg[seg_id]))
    lse = max_per_seg + torch.log(exp_sum)          # (U,)

    pos = labels > 0
    npos = torch.zeros(U, device=device)
    npos.scatter_add_(0, seg_id, pos.float())
    valid = (npos > 0) & (npos < lengths.float())
    if valid.sum() == 0:
        return None

    log_softmax = logits - lse[seg_id]              # (N,)
    pos_logsum = torch.zeros(U, device=device)
    pos_logsum.scatter_add_(0, seg_id, log_softmax * pos.float())
    per_seg = -pos_logsum / npos.clamp(min=1)       # -mean over positives of log p
    return (per_seg * valid.float()).sum() / valid.float().sum()


def bpr_loss(logits, labels, segments, num_neg=1):
    """Bayesian Personalized Ranking: -log(sigmoid(z_pos - z_neg)), vectorised.

    Samples ``num_neg`` negatives per positive within each user, mapping each positive
    to a uniformly-random negative in the same user via a precomputed negative-index
    table (no Python loop).
    """
    n = logits.shape[0]
    device = logits.device
    U = segments.shape[0] - 1
    seg_id, lengths = _seg_meta(segments)

    pos = labels > 0
    npos = torch.zeros(U, device=device)
    npos.scatter_add_(0, seg_id, pos.float())
    nneg = torch.zeros(U, device=device)
    nneg.scatter_add_(0, seg_id, (~pos).float())
    valid = (npos > 0) & (nneg > 0)
    if valid.sum() == 0:
        return None

    # negative rows sorted by row index -> non-decreasing seg_id
    neg_rows = torch.nonzero(~pos).squeeze(1)       # (Nn,)
    neg_seg = seg_id[neg_rows]                      # (Nn,)
    first_neg = torch.zeros(U, dtype=torch.long, device=device)
    if len(neg_rows) > 0:
        idx = torch.arange(len(neg_rows), device=device)
        first_neg.scatter_reduce_(0, neg_seg, idx, reduce="amin", include_self=False)

    pos_rows = torch.nonzero(pos).squeeze(1)        # (P,)
    pos_seg = seg_id[pos_rows]                      # (P,)
    nneg_v = nneg[pos_seg].long()                   # (P,)
    keep = nneg_v > 0                               # drop positives in all-negative-less users
    if keep.sum() == 0:
        return None
    pos_rows, pos_seg, nneg_v = pos_rows[keep], pos_seg[keep], nneg_v[keep]

    P = len(pos_rows)
    rand_ord = torch.randint(0, 1 << 30, (P, num_neg), device=device)
    neg_ord = (rand_ord % nneg_v.unsqueeze(1))      # (P, num_neg)
    neg_row = neg_rows[first_neg[pos_seg].unsqueeze(1) + neg_ord]  # (P, num_neg)

    z_pos = logits[pos_rows].unsqueeze(1)           # (P, 1)
    z_neg = logits[neg_row]                         # (P, num_neg)
    return -F.logsigmoid(z_pos - z_neg).mean()


def censored_watch_time_loss(aux_logits, play_time, duration):
    """CWM-style censored regression on normalized watch fraction (auxiliary task).

    ``play_time`` / ``duration`` are raw ms. ``watch_frac = clip(play_time/duration, 0, 1)``
    is right-censored at 1: a completed play (``play_time >= duration``) means the observed
    watch time was truncated by the video length, so the true fraction is >= 1 and we use a
    one-sided loss (penalize under-prediction only) for those rows. Non-completed rows are
    exact and get squared error. This is the censored-regression idea from CWM (KDD'24), NOT
    the soft-label regression that already failed — here watch time is an *aux* task sharing
    the main model's embedding, not a replacement for the binary ``long_view`` target.
    """
    watch_frac = torch.clamp(play_time / duration.clamp(min=1.0), 0.0, 1.0)
    pred = torch.sigmoid(aux_logits)                # watch fraction in [0, 1]
    complete = (play_time >= duration).float()      # 1 = censored (completed play)
    err = watch_frac - pred
    uncensored = (1.0 - complete) * err.square()
    censored = complete * torch.relu(err).square()  # under-predict -> penalty, over -> 0
    return (uncensored + censored).mean()
