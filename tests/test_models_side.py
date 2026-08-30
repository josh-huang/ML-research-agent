"""Continuous-side-feature wiring for FM / DeepFM / DIN (cont param)."""
import torch

from models import deepfm, din


def test_fm_zero_cont_equals_no_cont():
    torch.manual_seed(0)
    m = deepfm.FM(dim=20, k=8, cont_dim=6)
    x = torch.randint(0, 20, (4, 5))
    z = torch.zeros(4, 6)
    assert torch.allclose(m(x), m(x, z), atol=1e-6)   # cont_lin bias=False -> zero contribution


def test_fm_cont_changes_output():
    torch.manual_seed(0)
    m = deepfm.FM(dim=20, k=8, cont_dim=6)
    x = torch.randint(0, 20, (4, 5))
    o = torch.ones(4, 6)
    assert not torch.allclose(m(x, torch.zeros(4, 6)), m(x, o), atol=1e-6)


def test_deepfm_cont_shape_and_backcompat():
    torch.manual_seed(0)
    m0 = deepfm.DeepFM(dim=20, n_fields=5, k=8, dnn_hidden=(16,), cont_dim=0)
    x = torch.randint(0, 20, (4, 5))
    assert m0(x).shape == (4,)                              # cont=None default still works
    m = deepfm.DeepFM(dim=20, n_fields=5, k=8, dnn_hidden=(16,), cont_dim=6)
    out = m(x, torch.randn(4, 6))
    assert out.shape == (4,) and torch.isfinite(out).all()


def test_din_cont_shape():
    torch.manual_seed(0)
    m = din.DIN(dim=20, n_fields=5, k=8, hidden=(16,), cont_dim=6)
    x = torch.randint(0, 20, (4, 5))
    hist = torch.randint(0, 20, (4, 3))
    mask = torch.ones(4, 3)
    out = m(x, hist, mask, torch.randn(4, 6))
    assert out.shape == (4,) and torch.isfinite(out).all()
