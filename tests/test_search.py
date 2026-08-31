"""Config-space tests: side-feature toggles are normalize-complete and dedup-sound."""
import json

from agent import search


def test_normalize_carries_side_toggles():
    cfg = search.normalize({"model": "fm", "loss": "bce"})
    assert cfg["use_videoside"] is False and cfg["use_userside"] is False
    cfg2 = search.normalize({"model": "fm", "loss": "bce", "use_videoside": True})
    assert cfg2["use_videoside"] is True and cfg2["use_userside"] is False


def test_side_toggles_are_dedup_distinct():
    a = search.normalize({"model": "fm", "loss": "bce"})
    b = search.normalize({"model": "fm", "loss": "bce", "use_videoside": True})
    c = search.normalize({"model": "fm", "loss": "bce", "use_userside": True})
    ka = json.dumps(a, sort_keys=True)
    assert ka != json.dumps(b, sort_keys=True)
    assert ka != json.dumps(c, sort_keys=True)
    assert json.dumps(b, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_mutate_can_toggle_side():
    from agent.search import _DEFAULT
    assert "use_videoside" in _DEFAULT and "use_userside" in _DEFAULT
    cfg = search.normalize({"model": "fm", "loss": "bce", "use_videoside": False})
    from agent.search import mutate
    # mutate flips one dimension; over enough draws both toggles are reachable
    rng = __import__("random").Random(1)
    keys = set()
    for _ in range(200):
        m = mutate(cfg, rng)
        if m["use_videoside"] != cfg["use_videoside"]:
            keys.add("videoside")
        if m["use_userside"] != cfg["use_userside"]:
            keys.add("userside")
    assert "videoside" in keys
    assert "userside" in keys


def test_normalize_carries_tag_toggle():
    cfg = search.normalize({"model": "fm", "loss": "bce"})
    assert cfg["use_tagside"] is False
    cfg2 = search.normalize({"model": "fm", "loss": "bce", "use_tagside": True})
    assert cfg2["use_tagside"] is True


def test_tag_toggle_dedup_distinct():
    a = search.normalize({"model": "fm", "loss": "bce"})
    b = search.normalize({"model": "fm", "loss": "bce", "use_tagside": True})
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)


def test_mutate_can_toggle_tag():
    from agent.search import mutate
    cfg = search.normalize({"model": "fm", "loss": "bce", "use_tagside": False})
    rng = __import__("random").Random(2)
    saw_flip = False
    for _ in range(200):
        m = mutate(cfg, rng)
        if m["use_tagside"] != cfg["use_tagside"]:
            saw_flip = True
            break
    assert saw_flip
