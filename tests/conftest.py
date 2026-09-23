"""Shared pytest setup: make the subsystem folders importable without packaging them."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERAS_DIR = os.path.join(REPO_ROOT, "Cameras")
if CAMERAS_DIR not in sys.path:
    sys.path.insert(0, CAMERAS_DIR)
