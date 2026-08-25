"""Color palette for the retargeting viewer.

Formats vary per group to match how each color is consumed: (r,g,b) 0-255 ints
for UI/trace/marker, (r,g,b,a) 0-1 floats for overlays, hex/plotly rgba strings
for plots.
"""

from __future__ import annotations

# ---- mesh base colors (0-255 ints) ------------------------------------------

FLOOR = (200, 200, 200)  # ground grid
WHITE = (255, 255, 255)  # trace gradient endpoint

# ---- kinematic reference "ghost" overlay (0-1 float RGBA) -------------------

REF_AT_REST = (0.0, 0.0, 1.0, 0.25)  # blue
REF_HELD = (1.0, 0.0, 0.0, 0.25)  # red (warmup gate held)

# ---- optimizer traces (0-255 ints) ------------------------------------------

TRACE_OBJECT = (255, 0, 0)  # actual rollout traces
TRACE_ROBOT = (0, 0, 255)
TRACE_OBJECT_REF = (0, 255, 0)  # reference/target traces
TRACE_ROBOT_REF = (255, 255, 0)

# ---- plots (plotly hex / rgba strings) --------------------------------------

# Matplotlib tab10/tab20 cycle used for reward series.
PLOT_SERIES = [
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#17becf",
    "#bcbd22",
    "#aec7e8",
    "#ffbb78",
]
PLOT_BG_FILL = "rgba(127, 127, 127, 0.20)"  # gate/warmup background band
PLOT_LABEL = "rgba(120, 120, 120, 0.9)"  # lane label text
PLOT_HIGHLIGHT = "lightgray"  # current-frame vline
GATE_HELD_FILL = "rgba(46, 204, 113, 0.30)"  # gate "held" band (green)
GATE_REST_FILL = "rgba(231, 76, 60, 0.28)"  # gate "at rest" band (red)
