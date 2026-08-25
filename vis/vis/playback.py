"""Shared viser timeline controls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


def add_timeline(
    server, frame_count: int, fps: float, show_frame: Callable[[int], None]
):
    """Add a looping frame slider and play/pause controls."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")

    with server.gui.add_folder("Timeline"):
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=frame_count - 1, step=1, initial_value=0
        )
        playing = server.gui.add_checkbox("Playing", initial_value=False)
        fps_slider = server.gui.add_slider(
            "FPS", min=1, max=120, step=1, initial_value=max(1, round(fps))
        )

    suppress_update = False

    @frame_slider.on_update
    def _(_) -> None:
        if not suppress_update:
            show_frame(int(frame_slider.value))

    def playback_loop() -> None:
        nonlocal suppress_update
        while True:
            if playing.value:
                next_frame = (int(frame_slider.value) + 1) % frame_count
                suppress_update = True
                frame_slider.value = next_frame
                suppress_update = False
                show_frame(next_frame)
                time.sleep(1.0 / max(1.0, float(fps_slider.value)))
            else:
                time.sleep(0.05)

    threading.Thread(target=playback_loop, daemon=True).start()
    return frame_slider
