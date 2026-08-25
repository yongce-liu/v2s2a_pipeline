"""Image, mask, and intrinsics helpers for the pose_estimation package."""

from __future__ import annotations

import os

# pyrender/PyOpenGL need a real GL context; FoundationPose itself never creates
# one (it renders via nvdiffrast). Force the EGL platform so the offscreen
# renderer works headlessly. Must be set before OpenGL/pyrender is imported,
# which is why it lives in this module header (no GL import happens here).
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB numpy array."""

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a binary mask image as a bool numpy array."""

    gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    return gray > 127


def load_intrinsics(intrinsics_path: Path) -> np.ndarray:
    """Load a 3x3 camera intrinsics matrix saved as ``.npy``."""

    matrix = np.load(intrinsics_path)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"Expected a (3, 3) intrinsics matrix, got {matrix.shape}: {intrinsics_path}"
        )
    return matrix


def render_mesh_overlay(
    frame_rgb: np.ndarray,
    pose: np.ndarray,
    mesh,
    intrinsics: np.ndarray,
    renderer,
) -> np.ndarray:
    """Alpha-blend the tracked mesh (at ``pose``) onto ``frame_rgb``.

    The mesh is the obj_recon reconstruction, rendered with FoundationPose's
    offscreen renderer (same camera convention: pose is obj_in_cvcam). Pixels
    where the mesh projects (render depth < far plane) are blended over the
    frame; everything else passes through unchanged.
    """

    h, w = frame_rgb.shape[:2]
    if renderer is None:
        from offscreen_renderer import ModelRendererOffscreen

        renderer = ModelRendererOffscreen(intrinsics, h, w)
    color, depth = renderer.render(mesh=mesh, ob_in_cvcam=pose)
    mask = depth > 0
    vis = frame_rgb.copy()
    alpha = 0.6
    vis[mask] = (
        (1.0 - alpha) * frame_rgb[mask].astype(np.float32)
        + alpha * color[mask].astype(np.float32)
    ).astype(np.uint8)
    return vis


class PoseVideoWriter:
    """Write mesh-overlay frames to a single MP4 as poses are produced.

    Tracking runs forward and backward from the init frame, so frames arrive
    out of order; VideoWriter needs monotonic frames. We therefore buffer
    overlaid frames by index and flush in order on :meth:`close`.
    """

    def __init__(self, output_path: Path, fps: float) -> None:
        self.output_path = Path(output_path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self._frames: dict[int, np.ndarray] = {}

    def add(self, index: int, vis_rgb: np.ndarray) -> None:
        self._frames[index] = vis_rgb

    def _ordered_frames(self) -> tuple[list[np.ndarray], tuple[int, int]]:
        """Frames sorted by index, all resized to a common (h, w)."""
        import cv2

        first = self._frames[sorted(self._frames)[0]]
        h, w = first.shape[:2]
        frames = []
        for index in sorted(self._frames):
            frame = self._frames[index]
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            frames.append(frame)
        return frames, (h, w)

    def _write_ffmpeg_h264(self, frames: list[np.ndarray], hw: tuple[int, int]) -> bool:
        """Encode via the system ffmpeg as H.264/yuv420p (VSCode/browser-safe).

        Returns True on success, False if ffmpeg or libx264 is unavailable.
        """
        import shutil
        import subprocess

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return False
        h, w = hw
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(self.fps),
            "-i",
            "-",  # raw frames on stdin
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",  # required by VSCode/QuickTime/browser players
            "-crf",
            "20",
            "-preset",
            "fast",
            "-movflags",
            "+faststart",
            str(self.output_path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
            return proc.wait() == 0
        except Exception as e:  # pragma: no cover - depends on system ffmpeg
            logger.warning("[pose] ffmpeg H.264 encode failed: {}; falling back", e)
            return False

    def _write_cv2(
        self, frames: list[np.ndarray], hw: tuple[int, int], fourcc: str
    ) -> bool:
        import cv2

        h, w = hw
        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*fourcc),
            self.fps,
            (w, h),
        )
        if not writer.isOpened():
            writer.release()
            return False
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return True

    def close(self) -> Path | None:
        if not self._frames:
            return None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        frames, hw = self._ordered_frames()

        # H.264 (avc1) + yuv420p is the only codec VSCode's built-in preview,
        # browsers, and QuickTime all play; mp4v (MPEG-4 Part 2) is not. Prefer
        # the system ffmpeg's libx264, then OpenCV's own avc1, and only then
        # fall back to mp4v so a video is always produced.
        if self._write_ffmpeg_h264(frames, hw):
            logger.info("[pose] vis video encoded as H.264 (libx264, yuv420p)")
        elif self._write_cv2(frames, hw, "avc1"):
            logger.info("[pose] vis video encoded as H.264 (OpenCV avc1)")
        else:
            self._write_cv2(frames, hw, "mp4v")
            logger.warning(
                "[pose] H.264 encoders unavailable; wrote mp4v (may not preview in VSCode)"
            )
        return self.output_path
