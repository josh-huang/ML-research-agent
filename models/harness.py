"""Thin wrapper exposing the official scorer from the starter kit.

The starter kit's ``evaluate.py`` is the single source of truth for scoring:
label = ``long_view``, metrics = GAUC + nDCG@5, primary = mean(GAUC, nDCG@5).
We never modify it — we only import it here so the rest of the codebase has one
clean entry point.
"""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_KIT = os.path.join(_ROOT, "kuairand-starter-kit", "kuairand-starter-kit")
if _KIT not in sys.path:
    sys.path.insert(0, _KIT)

from evaluate import evaluate  # noqa: E402, F401
