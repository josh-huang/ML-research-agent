"""Sanity tests for the official evaluation harness (via models.harness)."""
import pytest

from models.harness import evaluate


def test_perfect_ranking_and_zero_positive_user():
    # user 'a': positive ranked above negative -> GAUC 1.0, nDCG 1.0
    # user 'b': all-negative -> nDCG 0.0, excluded from GAUC
    r = evaluate(['a', 'a', 'b', 'b'], [1, 0, 0, 0], [0.9, 0.1, 0.5, 0.5])
    assert r['GAUC'] == pytest.approx(1.0)
    assert r['nDCG@5'] == pytest.approx(0.5)
    assert r['primary'] == pytest.approx(0.75)
    assert (r['users'], r['rows']) == (2, 4)


def test_no_discriminative_user_gauc_defaults_half():
    # neither user has 0 < positives < impressions -> GAUC falls back to 0.5
    r = evaluate(['a', 'b'], [0, 1], [0.5, 0.5])
    assert r['GAUC'] == pytest.approx(0.5)
    assert r['nDCG@5'] == pytest.approx(0.5)


def test_expected_keys():
    r = evaluate(['a', 'a'], [1, 0], [1.0, 0.0])
    assert set(r) == {'GAUC', 'nDCG@5', 'primary', 'users', 'rows'}
