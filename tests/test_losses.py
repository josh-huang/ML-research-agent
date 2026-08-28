"""Numerical tests for the vectorised ranking losses.

These pin down two regressions found during development:
  * listwise softmax must return a *positive* loss (a sign error returned its negative,
    which minimized the positive-class probability instead of maximizing it);
  * BPR must not divide-by-zero on users with no negatives.
"""
import pytest
import torch

from models.losses import (
    bce_loss, bpr_loss, censored_watch_time_loss, listwise_softmax_loss)


def test_listwise_positive_and_rank_consistent():
    labels = torch.tensor([1.0, 0.0, 0.0])
    seg = torch.tensor([0, 3])
    good = listwise_softmax_loss(torch.tensor([2.0, 0.0, -1.0]), labels, seg)
    bad = listwise_softmax_loss(torch.tensor([-2.0, 1.0, 1.0]), labels, seg)
    assert good.item() > 0.0          # sign: loss must be positive
    assert good.item() < bad.item()   # better ranking -> lower loss


def test_listwise_returns_none_for_all_negative_user():
    assert listwise_softmax_loss(
        torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0]),
        torch.tensor([0, 2])) is None


def test_bpr_rank_consistent():
    labels = torch.tensor([1.0, 0.0])
    seg = torch.tensor([0, 2])
    good = bpr_loss(torch.tensor([2.0, 0.0]), labels, seg)
    bad = bpr_loss(torch.tensor([0.0, 2.0]), labels, seg)
    assert good.item() < bad.item()


def test_bpr_all_positive_user_no_crash():
    # user 0 has a negative; user 1 is all-positive (no negative -> must be skipped).
    logits = torch.tensor([2.0, 0.0, 3.0, 4.0])
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0])
    seg = torch.tensor([0, 2, 4])
    loss = bpr_loss(logits, labels, seg)
    assert loss is not None and torch.isfinite(loss)


def test_bce_matches_reference():
    logits = torch.tensor([1.0, -1.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 0.5])
    expected = -(labels * torch.log(torch.sigmoid(logits))
                 + (1 - labels) * torch.log(1 - torch.sigmoid(logits))).mean().item()
    assert bce_loss(logits, labels).item() == pytest.approx(expected, abs=1e-6)


def test_censored_watch_time_one_sided_for_completed():
    # Completed play (play_time == duration): over-prediction is free, under-prediction costs.
    pt = torch.tensor([1000.0])
    du = torch.tensor([1000.0])
    over = censored_watch_time_loss(torch.tensor([10.0]), pt, du)   # sigmoid ~1.0
    under = censored_watch_time_loss(torch.tensor([-10.0]), pt, du)  # sigmoid ~0.0
    assert over.item() < 1e-3
    assert under.item() > 0.9


def test_censored_watch_time_symmetric_for_incomplete():
    # Incomplete play (watch_frac 0.5): squared error is symmetric around the truth.
    pt = torch.tensor([500.0])
    du = torch.tensor([1000.0])
    exact = censored_watch_time_loss(torch.tensor([0.0]), pt, du)   # sigmoid 0.5 -> 0
    high = censored_watch_time_loss(torch.tensor([2.0]), pt, du)    # over-predict
    low = censored_watch_time_loss(torch.tensor([-2.0]), pt, du)    # under-predict
    assert exact.item() < 1e-4
    assert abs(high.item() - low.item()) < 1e-4  # symmetric (both penalize)
