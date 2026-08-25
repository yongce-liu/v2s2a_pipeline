"""Shared path constants for the hand_recon package."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
HAND_RECON_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = HAND_RECON_ROOT.parent
HAWOR_SOURCE = PROJECT_ROOT / "pkgs" / "HaWoR"

CAMERA_RIGHT_UP_TO_OPENCV = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]

WRIST_FACE_ID = 0
VIS_RGB_DIR_PATTERN = "rgb"
AITVIEWER_SUBDIR = "aitviewer"
OVERLAY_FILENAME = "overlay.mp4"

DEFAULT_HAND_IDS = ("left", "right")
MANO_JOINT_COUNT = 21
MANO_FACE_COUNT = 1530 + 14
