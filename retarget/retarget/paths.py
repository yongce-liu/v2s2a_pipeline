"""Shared path constants for the retarget package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RETARGET_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = RETARGET_ROOT.parent
