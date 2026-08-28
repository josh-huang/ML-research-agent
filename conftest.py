"""Pytest configuration.

Presence at the repo root ensures the project root is on ``sys.path`` so tests
can ``import models.*`` and ``import agent.*``.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
