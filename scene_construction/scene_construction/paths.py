"""Shared path constants for the scene_construction package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SCENE_CONSTRUCTION_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = SCENE_CONSTRUCTION_ROOT.parent

# Repo-level robot-hand MJCF assets, untracked and managed outside git
# (``assets/hands/<robot_type>/{left,right,bimanual}.xml``).
ASSETS_HANDS_DIR = PROJECT_ROOT / "assets" / "hands"
