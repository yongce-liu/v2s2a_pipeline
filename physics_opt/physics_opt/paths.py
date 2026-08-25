"""Shared path constants for the physics_opt package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PHYSICS_OPT_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = PHYSICS_OPT_ROOT.parent
