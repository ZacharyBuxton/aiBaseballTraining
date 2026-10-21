"""Shared pytest setup: make the subsystem folders and the dckft package importable."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERAS_DIR = os.path.join(REPO_ROOT, "Cameras")
for p in (CAMERAS_DIR, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
