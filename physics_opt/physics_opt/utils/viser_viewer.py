"""MuJoCo XML Viser visualizer."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import loguru
import mujoco
import numpy as np
import trimesh

from physics_opt.utils import palette

import logging

logging.getLogger("websockets").setLevel(logging.ERROR)

# -----------------------------
# Trace visualization defaults
# -----------------------------

DEFAULT_TRACE_RADIUS = 0.002
DEFAULT_FLOOR_COLOR = list(palette.FLOOR)


# ==============================================================================
# Module setup & shared state
# ==============================================================================


def _lazy_import_viser():
    try:
        import viser  # type: ignore

        return viser
    except ImportError as exc:
        raise ImportError(
            "viser is required for the Viser viewer. Install with `pip install viser`."
        ) from exc


@dataclass
class _ViserState:
    server: Any | None = None
    entity_root: str = "mujoco"
    body_handles: list[tuple[Any, int]] = field(default_factory=list)
    ref_body_handles: list[tuple[Any, int]] = field(default_factory=list)
    ref_geom_handles: list[Any] = field(default_factory=list)
    ref_geom_info: list[tuple[str, Any, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )
    # Per-color cache of pre-built reference handle sets. Color tuple → handles.
    # Avoids re-uploading meshes on color transitions (warmup boundary, loop wrap).
    ref_geom_handle_sets: dict[tuple, list[Any]] = field(default_factory=dict)
    ref_color: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0, 0.25], dtype=np.float32)
    )
    ref_color_history: list[np.ndarray] = field(default_factory=list)
    visual_geom_handles: list[Any] = field(default_factory=list)
    visual_geom_info: list[
        tuple[str, Any, np.ndarray, np.ndarray, np.ndarray | None]
    ] = field(default_factory=list)
    collision_geom_handles: list[Any] = field(default_factory=list)
    collision_geom_info: list[
        tuple[str, Any, np.ndarray, np.ndarray, np.ndarray | None]
    ] = field(default_factory=list)

    # Categorized subsets of body/geom handles, populated by `_recategorize_handles`
    # after each scene build/swap. The "Retargeted Data" / "Kinematic Data"
    # checkboxes drive visibility on these lists rather than re-deriving the
    # categories on every callback.
    hand_body_handles: list[tuple[Any, int]] = field(default_factory=list)
    obj_body_handles: list[tuple[Any, int]] = field(default_factory=list)
    hand_geom_handles: list[Any] = field(default_factory=list)
    obj_geom_handles: list[Any] = field(default_factory=list)
    # Single batched-mesh handle holding one cylinder instance per
    # `_PEDESTAL_SLOTS` slot. Per-task we mutate batched_positions/wxyzs/scales
    # in place (slots absent in the current XML get scale=0). Built once at
    # scene init so swap_object_subtree never re-uploads pedestal geometry.
    pedestal_batched_handle: Any | None = None
    ref_hand_body_handles: list[tuple[Any, int]] = field(default_factory=list)
    ref_obj_body_handles: list[tuple[Any, int]] = field(default_factory=list)
    ref_hand_geom_handles: list[Any] = field(default_factory=list)
    ref_obj_geom_handles: list[Any] = field(default_factory=list)

    scene_checkboxes: dict[str, Any] = field(default_factory=dict)
    trace_handle: Any | None = None
    trace_handles: dict[str, Any] = field(default_factory=dict)
    trace_lock: threading.Lock = field(default_factory=threading.Lock)
    trace_colors: np.ndarray | None = None
    trace_slider: Any | None = None
    trace_checkboxes: dict[str, Any] = field(default_factory=dict)
    last_traces: np.ndarray | None = None
    last_trace_ref: np.ndarray | None = None
    last_trace_cost: np.ndarray | None = None
    last_num_iters: int | None = None
    num_object_trace_sites: int = 1

    # Pose axes (RGB frames on hand/object bodies)
    axes_handles: list[Any] = field(default_factory=list)
    ref_axes_handles: list[Any] = field(default_factory=list)

    # Scene geom cache for fast task switching (avoids re-uploading unchanged meshes)
    _geom_handle_cache: dict[str, tuple] = field(default_factory=dict)
    _geom_cache_dirty: set[str] = field(default_factory=set)

    # Frame-change callbacks
    frame_change_callbacks: list[Callable[[int], None]] = field(default_factory=list)

    # Reward plot (matplotlib image)
    reward_plot_handle: Any | None = None
    reward_series_names: list[str] = field(default_factory=list)
    reward_series_colors: list[str] = field(default_factory=list)
    reward_frames: list[float] = field(default_factory=list)
    reward_values: list[list[float]] = field(
        default_factory=list
    )  # one list per series
    _reward_plot_last_update: float = 0.0
    # Optional execution-timeline effective-gate state + warmup-step count, used
    # to shade the reward plot so the user can see which sim steps are
    # held (green) / at rest (red) / warmup (gray). Stored as per-side lanes:
    # a list of (label, (T,) int8 state) pairs drawn as stacked half-height
    # bands so one side's band never blends into another's. A single unlabeled
    # lane is the degenerate single-object case.
    reward_gate_lanes: list[tuple[str, np.ndarray]] | None = None
    reward_warmup_steps: int = 0

    # Timeline
    frame_history: list[dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=list
    )
    trace_history: dict[
        int,
        tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None, int | None],
    ] = field(default_factory=dict)
    trace_id_counter: int = 0
    trace_last_frame_count: int = 0
    playback_slider: Any | None = None
    playback_base_fps: float = 50.0
    playback_checkbox: Any | None = None
    playback_speed: int = 0
    playback_thread: Any | None = None
    playback_folder: Any | None = None
    _playback_stop: threading.Event = field(default_factory=threading.Event)

    # Optimizer trajectory visualization
    scene_spec: Any | None = None
    scene_model: Any | None = None

    # Progress bars (optimization iterations + sim steps)
    opt_progress_bar: Any | None = None
    sim_progress_bar: Any | None = None

    # Pre-created GUI folders (created in desired display order)
    gui_folder_timeline: Any | None = None
    gui_folder_retargeted: Any | None = None
    gui_folder_kinematic: Any | None = None
    gui_folder_opt_traces: Any | None = None
    gui_folder_rewards: Any | None = None


_STATE = _ViserState()


# ==============================================================================
# Mesh & geometry helpers
# ==============================================================================


def _rgba_to_uint8(rgba: np.ndarray) -> np.ndarray:
    rgba_arr = np.asarray(rgba)
    if np.issubdtype(rgba_arr.dtype, np.floating):
        rgba_arr = np.clip(rgba_arr, 0.0, 1.0)
        rgba_arr = (rgba_arr * 255.0).astype(np.uint8)
    else:
        rgba_arr = rgba_arr.astype(np.uint8)
    if rgba_arr.size == 3:
        rgba_arr = np.concatenate([rgba_arr, np.array([255], dtype=np.uint8)])
    return rgba_arr


def _set_mesh_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial

    rgba_int = _rgba_to_uint8(rgba)
    mesh.visual = TextureVisuals(
        material=PBRMaterial(
            baseColorFactor=rgba_int,
            main_color=rgba_int,
            metallicFactor=0.5,
            roughnessFactor=1.0,
            alphaMode="BLEND" if rgba_int[-1] < 255 else "OPAQUE",
        )
    )


def _trimesh_from_primitive(
    geom_type: int, size: np.ndarray, rgba: np.ndarray | None = None
) -> trimesh.Trimesh | None:
    t = mujoco.mjtGeom
    if geom_type == t.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=float(size[0]), subdivisions=2)
    elif geom_type == t.mjGEOM_CAPSULE:
        radius = float(size[0])
        length = float(2.0 * size[1])
        mesh = trimesh.creation.capsule(radius=radius, height=length)
    elif geom_type == t.mjGEOM_CYLINDER:
        radius = float(size[0])
        height = float(2.0 * size[1])
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
    elif geom_type == t.mjGEOM_BOX:
        extents = 2.0 * np.asarray(size[:3], dtype=np.float32)
        mesh = trimesh.creation.box(extents=extents)
    elif geom_type == t.mjGEOM_PLANE:
        mesh = trimesh.creation.box(extents=[20.0, 20.0, 0.01])
    else:
        return None

    if rgba is not None:
        _set_mesh_color(mesh, rgba)
    return mesh


def _mujoco_mesh_to_trimesh(
    model: mujoco.MjModel, geom_id: int
) -> trimesh.Trimesh | None:
    mesh_id = model.geom_dataid[geom_id]
    if mesh_id < 0:
        return None

    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start : vert_start + vert_count]
    faces = model.mesh_face[face_start : face_start + face_count]

    if len(vertices) == 0 or len(faces) == 0:
        return None

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _get_mesh_file(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> Path | None:
    try:
        meshname = geom.meshname
        if not meshname:
            return None
        mesh = spec.mesh(meshname)
        mesh_dir = spec.meshdir if spec.meshdir is not None else ""
        model_dir = spec.modelfiledir if spec.modelfiledir is not None else ""
        return (Path(model_dir) / mesh_dir / mesh.file).resolve()
    except Exception:
        return None


def _get_mesh_scale(spec: mujoco.MjSpec, geom: mujoco.MjsGeom) -> np.ndarray | None:
    try:
        mesh = spec.mesh(geom.meshname)
        scale = mesh.scale
        if scale is None:
            return None
        return np.asarray(scale, dtype=np.float32)
    except Exception:
        return None


def _ensure_names(spec: mujoco.MjSpec) -> None:
    geom_placeholder_idx = 0
    body_placeholder_idx = 0
    for body in spec.bodies[1:]:
        if not body.name:
            body.name = f"VISER_BODY_{body_placeholder_idx}"
            body_placeholder_idx += 1
        for geom in body.geoms:
            if not geom.name:
                geom.name = f"VISER_GEOM_{geom_placeholder_idx}"
                geom_placeholder_idx += 1


# ==============================================================================
# Pose axes
# ==============================================================================


AXES_LENGTH = 0.03
AXES_RADIUS = 0.002


def _get_axes_body_names(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
) -> list[str]:
    """Return body names that should get pose-axes markers.

    The axes are chosen so their world transform exactly reflects the
    pos/rot reward terms in ``_weight_diff_qpos`` / ``_diff_qpos``:

    - **Robot base** (``base_pos_rew_scale`` on qpos[:3], ``base_rot_rew_scale``
      on qpos[3:6]): for hands wrapped by ``wrap_with_base_dofs`` the final
      base-chain body ``{side}_base_yaw`` has ``xpos = (tx,ty,tz)`` and
      ``xquat = R_z·R_y·R_x``, exactly the 6-DOF base pose. For robots
      without that synthetic chain, the first non-object body is the
      freejoint root and already carries the full base pose.
    - **Object** (``pos_rew_scale`` / ``rot_rew_scale`` on the trailing object
      DOFs): the ``{side}_object`` body owns all 6 object joints (freejoint
      or the slide+hinge group inserted by ``_add_object_xyzrpy_actuators``),
      so its ``xpos`` / ``xquat`` are the exact reward targets.

    Earlier base-chain bodies (``_base_tx/ty/tz/roll/pitch``) each carry
    only one DOF and are skipped. Object bodies with only the dummy
    ``_object_mass`` geom (no real geometry) are also skipped.
    """
    names: list[str] = []
    seen_sides: set[str] = set()

    for body in spec.bodies[1:]:
        bname = body.name
        if bname in ("right_object", "left_object"):
            has_real_geom = any("_object_mass" not in g.name for g in body.geoms)
            if has_real_geom:
                names.append(bname)
            continue
        for prefix in ("right_", "left_"):
            if not bname.startswith(prefix) or prefix in seen_sides:
                continue
            if bname.startswith(f"{prefix}base_") and bname != f"{prefix}base_yaw":
                break
            names.append(bname)
            seen_sides.add(prefix)
            break

    return names


def _add_pose_axes(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    server: Any,
    entity_root: str,
    body_entity_and_ids: list[tuple[Any, int]],
) -> list[Any]:
    axes_names = _get_axes_body_names(spec, model)
    if not axes_names:
        return []

    bid_to_handle: dict[int, tuple[Any, str]] = {}
    for handle, bid in body_entity_and_ids:
        body_name = model.body(bid).name
        if body_name in axes_names:
            bid_to_handle[bid] = (handle, body_name)

    axes_visible = (
        _STATE.scene_checkboxes["pose_axes"].value
        if "pose_axes" in _STATE.scene_checkboxes
        else False
    )

    handles: list[Any] = []
    for bid, (parent_handle, body_name) in bid_to_handle.items():
        path = f"{entity_root}/{body_name}/pose_axes"
        h = server.scene.add_frame(
            path,
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        h.visible = axes_visible
        handles.append(h)

    return handles


# ==============================================================================
# Server lifecycle
# ==============================================================================


def init_viser(
    app_name: str = "retargeting",
    port: int | None = None,
) -> None:
    if _STATE.server is not None:
        return
    viser = _lazy_import_viser()
    kwargs: dict[str, Any] = {"label": app_name}
    if port is not None:
        kwargs["port"] = port
    _STATE.server = viser.ViserServer(**kwargs)
    # Floor grid lives outside the MuJoCo subtree so it survives task switches
    # — it's always at z=0 and never needs rebuilding.
    try:
        _STATE.server.scene.add_grid(
            "ground_plane",
            section_color=tuple(np.array(DEFAULT_FLOOR_COLOR) / 255.0),
            cell_color=tuple(np.array(DEFAULT_FLOOR_COLOR) / 255.0),
        )
    except Exception:
        pass


def _get_server() -> Any:
    if _STATE.server is None:
        init_viser()
    return _STATE.server


def register_frame_callback(callback: Callable[[int], None]) -> None:
    """Register a callback that fires whenever the timeline frame changes."""
    _STATE.frame_change_callbacks.append(callback)


# ==============================================================================
# Pedestals
# ==============================================================================


_OBJECT_BODY_NAMES: tuple[str, ...] = ("right_object", "left_object")
# Fixed slot order for pedestal cylinders (worldbody geoms named by
# `generate_scene.py:669`). Each task populates a subset of these; slots not
# present get scale=0 in the batched handle.
_PEDESTAL_SLOTS: tuple[str, ...] = (
    "right_pedestal_start",
    "right_pedestal_end",
    "left_pedestal_start",
    "left_pedestal_end",
)


def _is_pedestal_geom_name(name: str) -> bool:
    return "pedestal" in name


def _build_pedestal_batched_handle(server: Any, entity_root: str) -> Any:
    """Upload one batched-mesh cylinder with `_PEDESTAL_SLOTS` instances.

    Initial scales=0 so all slots are invisible until the first
    `_update_pedestal_batched` call. Base mesh is a unit cylinder
    (radius=1, total height=2) so per-instance scale = (r, r, half_h)
    yields the right shape.
    """
    base_mesh = trimesh.creation.cylinder(radius=1.0, height=2.0)
    _set_mesh_color(base_mesh, np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32))
    n = len(_PEDESTAL_SLOTS)
    handle = server.scene.add_batched_meshes_trimesh(
        f"{entity_root}/pedestals",
        base_mesh,
        batched_positions=np.zeros((n, 3), dtype=np.float32),
        batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        batched_scales=np.zeros((n, 3), dtype=np.float32),
    )
    return handle


def _update_pedestal_batched(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    """Refresh the batched pedestal handle's per-instance pos/wxyz/scale.

    Reads pedestal geoms from the worldbody of the new spec and writes one
    instance per `_PEDESTAL_SLOTS` slot. Absent slots collapse to scale=0
    (invisible). The handle itself is created once at scene init.
    """
    handle = _STATE.pedestal_batched_handle
    if handle is None:
        return
    n = len(_PEDESTAL_SLOTS)
    positions = np.zeros((n, 3), dtype=np.float32)
    wxyzs = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    scales = np.zeros((n, 3), dtype=np.float32)
    worldbody = spec.bodies[0]
    geoms_by_name = {g.name: g for g in worldbody.geoms}
    for slot_idx, slot_name in enumerate(_PEDESTAL_SLOTS):
        geom = geoms_by_name.get(slot_name)
        if geom is None:
            continue
        # MuJoCo cylinder size = (radius, half_height, _).
        size = np.asarray(geom.size, dtype=np.float32)
        positions[slot_idx] = np.asarray(geom.pos, dtype=np.float32)
        try:
            q = np.asarray(geom.quat, dtype=np.float32)
            qnorm = float(np.linalg.norm(q))
            if qnorm > 0:
                wxyzs[slot_idx] = q / qnorm
        except Exception:
            pass
        scales[slot_idx] = np.array(
            [float(size[0]), float(size[0]), float(size[1])], dtype=np.float32
        )
    handle.batched_positions = positions
    handle.batched_wxyzs = wxyzs
    handle.batched_scales = scales


# ==============================================================================
# Handle categorization & visibility
# ==============================================================================


def _categorize_bodies(
    body_list: list[tuple[Any, int]], model: mujoco.MjModel
) -> tuple[list[tuple[Any, int]], list[tuple[Any, int]]]:
    """Split body handles into (hand, object); excludes worldbody (id 0)."""
    hand: list[tuple[Any, int]] = []
    obj: list[tuple[Any, int]] = []
    for handle, bid in body_list:
        if bid == 0:
            continue
        body_name = model.body(bid).name
        if body_name in _OBJECT_BODY_NAMES:
            obj.append((handle, bid))
        else:
            hand.append((handle, bid))
    return hand, obj


def _categorize_geoms(geom_list: list[Any]) -> tuple[list[Any], list[Any]]:
    """Split geom handles into (hand, object) by viser scene path.

    Pedestals are not in `geom_list` — they live on a separate batched
    handle (`_STATE.pedestal_batched_handle`) and are toggled there.
    """
    hand: list[Any] = []
    obj: list[Any] = []
    for gh in geom_list:
        name = getattr(gh, "name", "") or ""
        if "right_object" in name or "left_object" in name:
            obj.append(gh)
        else:
            hand.append(gh)
    return hand, obj


def _recategorize_handles() -> None:
    """Refresh the categorized handle subsets in `_STATE` from the live scene."""
    model = _STATE.scene_model
    if model is None:
        _STATE.hand_body_handles[:] = []
        _STATE.obj_body_handles[:] = []
        _STATE.hand_geom_handles[:] = []
        _STATE.obj_geom_handles[:] = []
        _STATE.ref_hand_body_handles[:] = []
        _STATE.ref_obj_body_handles[:] = []
        _STATE.ref_hand_geom_handles[:] = []
        _STATE.ref_obj_geom_handles[:] = []
        return
    h_b, o_b = _categorize_bodies(_STATE.body_handles, model)
    _STATE.hand_body_handles[:] = h_b
    _STATE.obj_body_handles[:] = o_b
    h_g, o_g = _categorize_geoms(_STATE.visual_geom_handles)
    _STATE.hand_geom_handles[:] = h_g
    _STATE.obj_geom_handles[:] = o_g
    rh_b, ro_b = _categorize_bodies(_STATE.ref_body_handles, model)
    _STATE.ref_hand_body_handles[:] = rh_b
    _STATE.ref_obj_body_handles[:] = ro_b
    rh_g, ro_g = _categorize_geoms(_STATE.ref_geom_handles)
    _STATE.ref_hand_geom_handles[:] = rh_g
    _STATE.ref_obj_geom_handles[:] = ro_g


def _kin_any_visible() -> bool:
    """True iff either Kinematic Hand or Kinematic Object checkbox is on.

    Used by `_apply_ref_color` to pick the visibility flag for the cached
    handle set being made active; per-handle correction is then done by
    `_reapply_scene_visibility`.
    """
    cb = _STATE.scene_checkboxes
    kh = cb.get("kin_hand", None)
    ko = cb.get("kin_obj", None)
    if kh is None and ko is None:
        return True
    return (kh is not None and kh.value) or (ko is not None and ko.value)


def _reapply_scene_visibility() -> None:
    """Re-read every scene-visibility checkbox and push its value to handles.

    Called at the end of `build_and_log_scene_from_spec` and
    `swap_object_subtree`, so checkbox state chosen for a previous task
    carries through to the new scene without any flicker. Match
    visualize.py's prior behavior: Visual/Collision act as a global
    override on top of the per-category Hand/Object/Pedestal toggles.
    """
    cb = _STATE.scene_checkboxes
    if not cb:
        return
    rt_hand = cb["rt_hand"].value if "rt_hand" in cb else True
    rt_obj = cb["rt_obj"].value if "rt_obj" in cb else True
    pedestal = cb["pedestal"].value if "pedestal" in cb else True
    visual = cb["visual_meshes"].value if "visual_meshes" in cb else True
    collision = cb["collision_meshes"].value if "collision_meshes" in cb else True
    axes = cb["pose_axes"].value if "pose_axes" in cb else False
    kin_hand = cb["kin_hand"].value if "kin_hand" in cb else True
    kin_obj = cb["kin_obj"].value if "kin_obj" in cb else True

    for h, _ in _STATE.hand_body_handles:
        h.visible = rt_hand
    for h, _ in _STATE.obj_body_handles:
        h.visible = rt_obj
    for gh in _STATE.hand_geom_handles:
        gh.visible = rt_hand
    for gh in _STATE.obj_geom_handles:
        gh.visible = rt_obj
    if _STATE.pedestal_batched_handle is not None:
        _STATE.pedestal_batched_handle.visible = pedestal
    if not visual:
        for gh in _STATE.visual_geom_handles:
            gh.visible = False
    if not collision:
        for gh in _STATE.collision_geom_handles:
            gh.visible = False
    for h, _ in _STATE.ref_hand_body_handles:
        h.visible = kin_hand
    for h, _ in _STATE.ref_obj_body_handles:
        h.visible = kin_obj
    for gh in _STATE.ref_hand_geom_handles:
        gh.visible = kin_hand
    for gh in _STATE.ref_obj_geom_handles:
        gh.visible = kin_obj
    for h in _STATE.axes_handles:
        h.visible = axes
    kin_any = kin_hand or kin_obj
    for h in _STATE.ref_axes_handles:
        h.visible = axes and kin_any
    refresh_geom_appearance(cap_at_half=axes)


# ==============================================================================
# Mesh appearance (recolor / opacity)
# ==============================================================================


def _recolor_and_reupload(
    info_list: list[tuple[str, Any, np.ndarray, np.ndarray, np.ndarray | None]],
    handle_list: list[Any],
    checkbox_key: str,
    cap_at_half: bool,
) -> None:
    """Re-color each geom in ``info_list`` and re-upload it, replacing its handle.

    Colors come from each geom's original rgba; opacity is capped at 0.5 when
    ``cap_at_half`` (pose axes shown).
    """
    server = _STATE.server
    if server is None:
        return
    cb = _STATE.scene_checkboxes
    visible = cb[checkbox_key].value if checkbox_key in cb else True
    for i, (path, tm, pos, quat, orig_rgba) in enumerate(info_list):
        if orig_rgba is not None:
            rgba = orig_rgba.copy()
        else:
            rgba = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32)
        if cap_at_half:
            rgba[3] = min(rgba[3], 0.5)
        _set_mesh_color(tm, rgba)
        try:
            h = server.scene.add_mesh_trimesh(path, tm, position=pos, wxyz=quat)
            h.visible = visible
            handle_list[i] = h
            _STATE._geom_cache_dirty.add(path)
        except Exception:
            pass


def refresh_geom_appearance(cap_at_half: bool) -> None:
    """Re-color and re-upload all visual + collision meshes, capping opacity at
    0.5 when pose axes are shown. Called on scene (re)build and when the
    pose-axes toggle flips."""
    _recolor_and_reupload(
        _STATE.visual_geom_info,
        _STATE.visual_geom_handles,
        "visual_meshes",
        cap_at_half=cap_at_half,
    )
    _recolor_and_reupload(
        _STATE.collision_geom_info,
        _STATE.collision_geom_handles,
        "collision_meshes",
        cap_at_half=cap_at_half,
    )


# ==============================================================================
# Scene construction (geoms, bodies, build, swap)
# ==============================================================================


def swap_object_subtree(
    xml_path: Path,
    entity_root: str = "mujoco",
    build_ref: bool = True,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, list[tuple[Any, int]]]:
    """Replace only the object bodies (``right_object`` / ``left_object``) in
    the active scene, leaving robot bodies/geoms/axes — which are identical
    across same-robot tasks — and their viser handles fully intact.

    Pre-requisite: a previous ``build_and_log_scene`` for the same
    ``entity_root`` must have populated ``_STATE.body_handles``. The new
    XML must share the robot section with the previous one (only the
    object body's mesh assets and joint params may differ); body IDs are
    re-resolved by name from the new model so minor body-ordering shifts
    are tolerated.

    Returns ``(spec, model, body_entity_and_ids)`` matching
    ``build_and_log_scene``.
    """
    if _STATE.server is None:
        raise RuntimeError("swap_object_subtree: server not initialized")
    if not _STATE.body_handles:
        raise RuntimeError("swap_object_subtree: no prior scene to swap into")

    server = _STATE.server
    _stop_playback_threads()

    # Load (or fetch cached) new spec/model.
    _xml_key = (str(xml_path.resolve()), xml_path.stat().st_mtime_ns)
    if _xml_key in _spec_model_cache:
        spec, model = _spec_model_cache[_xml_key]
    else:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        _ensure_names(spec)
        model = spec.compile()
        _spec_model_cache[_xml_key] = (spec, model)

    # Snapshot old object body IDs (from old model) so we can drop matching
    # handle entries before re-adding from the new spec.
    old_model = _STATE.scene_model
    old_obj_ids: set[int] = set()
    if old_model is not None:
        for name in _OBJECT_BODY_NAMES:
            try:
                old_obj_ids.add(int(old_model.body(name).id))
            except Exception:
                pass

    obj_path_prefixes = tuple(f"{entity_root}/{n}/" for n in _OBJECT_BODY_NAMES)
    obj_ref_path_prefixes = tuple(f"{entity_root}_ref/{n}/" for n in _OBJECT_BODY_NAMES)
    obj_body_paths = tuple(f"{entity_root}/{n}" for n in _OBJECT_BODY_NAMES)
    obj_ref_body_paths = tuple(f"{entity_root}_ref/{n}" for n in _OBJECT_BODY_NAMES)

    def _is_obj_path(p: str) -> bool:
        return (
            p in obj_body_paths
            or p.startswith(obj_path_prefixes)
            or p in obj_ref_body_paths
            or p.startswith(obj_ref_path_prefixes)
        )

    def _handle_path(h: Any) -> str:
        return getattr(h, "name", "") or ""

    with server.atomic():
        # Wipe object scene subtrees (this kills geom + axes children too).
        for p in obj_body_paths + obj_ref_body_paths:
            server.scene.remove_by_name(p)

        # Drop matching entries from STATE handle lists.
        _STATE.body_handles[:] = [
            (h, bid) for h, bid in _STATE.body_handles if bid not in old_obj_ids
        ]
        _STATE.ref_body_handles[:] = [
            (h, bid) for h, bid in _STATE.ref_body_handles if bid not in old_obj_ids
        ]
        _STATE.visual_geom_handles[:] = [
            h for h in _STATE.visual_geom_handles if not _is_obj_path(_handle_path(h))
        ]
        _STATE.visual_geom_info[:] = [
            t for t in _STATE.visual_geom_info if not _is_obj_path(t[0])
        ]
        _STATE.collision_geom_handles[:] = [
            h
            for h in _STATE.collision_geom_handles
            if not _is_obj_path(_handle_path(h))
        ]
        _STATE.collision_geom_info[:] = [
            t for t in _STATE.collision_geom_info if not _is_obj_path(t[0])
        ]
        _STATE.ref_geom_handles[:] = [
            h for h in _STATE.ref_geom_handles if not _is_obj_path(_handle_path(h))
        ]
        _STATE.ref_geom_info[:] = [
            t for t in _STATE.ref_geom_info if not _is_obj_path(t[0])
        ]
        # The body-subtree wipe above removed every color-suffixed object
        # scene node, so prune the now-stale object handles from each color
        # cache. Surviving entries (hand meshes) remain valid in the scene.
        active_color_key = tuple(_STATE.ref_color.tolist())
        for handle_list in _STATE.ref_geom_handle_sets.values():
            handle_list[:] = [
                h for h in handle_list if not _is_obj_path(_handle_path(h))
            ]
        _STATE.axes_handles[:] = [
            h for h in _STATE.axes_handles if not _is_obj_path(_handle_path(h))
        ]
        _STATE.ref_axes_handles[:] = [
            h for h in _STATE.ref_axes_handles if not _is_obj_path(_handle_path(h))
        ]
        for path in list(_STATE._geom_handle_cache.keys()):
            if _is_obj_path(path):
                _STATE._geom_handle_cache.pop(path, None)
                _STATE._geom_cache_dirty.discard(path)

        # Re-resolve robot body IDs against the NEW model — names are stable
        # across same-robot tasks even if joint-param edits shift indices.
        for i, (h, _bid) in enumerate(_STATE.body_handles):
            try:
                bname = _handle_path(h).split("/")[-1] or "worldbody"
                _STATE.body_handles[i] = (h, int(model.body(bname).id))
            except Exception:
                pass
        for i, (h, _bid) in enumerate(_STATE.ref_body_handles):
            try:
                bname = _handle_path(h).split("/")[-1]
                _STATE.ref_body_handles[i] = (h, int(model.body(bname).id))
            except Exception:
                pass

        # Re-add the object bodies from the new spec.
        new_obj_main: list[tuple[Any, int]] = []
        new_obj_ref: list[tuple[Any, int]] = []
        ref_color = _STATE.ref_color
        for body in spec.bodies:
            if body.name in _OBJECT_BODY_NAMES:
                new_obj_main.append(
                    _build_one_main_body(
                        spec,
                        model,
                        body,
                        server,
                        entity_root,
                        None,
                    )
                )
                if build_ref:
                    new_obj_ref.append(
                        _build_one_ref_body(
                            spec,
                            model,
                            body,
                            server,
                            entity_root,
                            ref_color,
                        )
                    )

        _STATE.body_handles.extend(new_obj_main)
        if build_ref:
            _STATE.ref_body_handles.extend(new_obj_ref)

        # New object pose-axes (only object bodies; robot axes were preserved).
        new_axes = _add_pose_axes(
            spec,
            model,
            server,
            entity_root,
            new_obj_main,
        )
        _STATE.axes_handles.extend(new_axes)
        if build_ref:
            new_ref_axes = _add_pose_axes(
                spec,
                model,
                server,
                f"{entity_root}_ref",
                new_obj_ref,
            )
            _STATE.ref_axes_handles.extend(new_ref_axes)

        # Extend every color cache with the newly-added object refs so
        # _apply_ref_color visibility toggles cover them. The active color
        # uses the bare-path handles built above; inactive colors get
        # color-suffixed copies of just the new object geom_info entries.
        if build_ref:
            existing = _STATE.ref_geom_handle_sets.get(active_color_key, [])
            for h in _STATE.ref_geom_handles:
                if h not in existing:
                    existing.append(h)
            _STATE.ref_geom_handle_sets[active_color_key] = existing
            new_obj_geom_info = [t for t in _STATE.ref_geom_info if _is_obj_path(t[0])]
            for color_key, handle_list in _STATE.ref_geom_handle_sets.items():
                if color_key == active_color_key:
                    continue
                handle_list.extend(
                    _build_ref_handle_set(
                        np.array(color_key, dtype=np.float32),
                        new_obj_geom_info,
                    )
                )

    _STATE.scene_spec = spec
    _STATE.scene_model = model

    # Pedestals (worldbody cylinders) change pose/size per task. Mutate the
    # persistent batched handle in place — no remove/re-add of scene nodes.
    _update_pedestal_batched(spec, model)

    # Per-task bookkeeping (callbacks, frame history, slider) — but keep
    # all the scene-handle lists we just maintained.
    _reset_task_state_keep_handles()

    # Refresh categorized handle subsets and push prior checkbox state to
    # every handle in the new scene (avoids a "everything visible" flash
    # when the previous task had toggles off).
    _recategorize_handles()
    _reapply_scene_visibility()

    return spec, model, list(_STATE.body_handles)


def _load_or_cache_geom(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    geom: Any,
    model_geom: Any,
    geom_path: str,
    rgba: np.ndarray | None,
    server: Any,
) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Load a geom mesh, serving unchanged geoms from `_STATE._geom_handle_cache`
    without re-loading the mesh file or re-uploading to viser."""
    # Plane geoms are redundant with the add_grid ground plane.
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
        return None

    # Build a hashable cache key from the geom's visual identity.
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_file = _get_mesh_file(spec, geom)
        mesh_scale = _get_mesh_scale(spec, geom)
        cache_key: tuple = (
            "mesh",
            str(mesh_file) if mesh_file else None,
            tuple(mesh_scale.ravel().tolist()) if mesh_scale is not None else None,
            tuple(rgba.ravel().tolist()) if rgba is not None else None,
        )
    else:
        prim_size = np.asarray(geom.size, dtype=np.float64)
        if model_geom is not None:
            try:
                ms = model.geom_size[model_geom.id]
                if np.any(prim_size == 0) or np.any(np.isnan(prim_size)):
                    prim_size = np.asarray(ms, dtype=np.float64)
            except Exception:
                pass
        cache_key = (
            "prim",
            int(geom.type),
            tuple(prim_size.ravel().tolist()),
            tuple(rgba.ravel().tolist()) if rgba is not None else None,
        )

    cached = _STATE._geom_handle_cache.get(geom_path)
    if cached is not None and cached[0] == cache_key:
        if geom_path not in _STATE._geom_cache_dirty:
            return cached[1:]  # clean cache hit

        # Dirty cache hit: restore correct colour and re-upload.
        _, handle_old, tm, geom_pos, geom_quat, rgba_copy = cached
        if rgba_copy is not None:
            _set_mesh_color(tm, rgba_copy)
        try:
            handle = server.scene.add_mesh_trimesh(
                geom_path,
                tm,
                position=geom_pos,
                wxyz=geom_quat,
            )
        except Exception:
            handle = handle_old  # fall back to stale handle
        _STATE._geom_cache_dirty.discard(geom_path)
        _STATE._geom_handle_cache[geom_path] = (
            cache_key,
            handle,
            tm,
            geom_pos,
            geom_quat,
            rgba_copy,
        )
        return (handle, tm, geom_pos, geom_quat, rgba_copy)

    # ---- Cache miss: load mesh from scratch. ----
    if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
        tm = None
        if mesh_file is not None and mesh_file.exists():
            try:
                tm = trimesh.load(str(mesh_file), force="mesh")
                if isinstance(tm, trimesh.Scene):
                    tm = tm.to_mesh()
            except Exception:
                tm = None
        if tm is None:
            try:
                geom_id = model_geom.id if model_geom is not None else -1
                tm = _mujoco_mesh_to_trimesh(model, geom_id)
            except Exception:
                tm = None
        if tm is None:
            loguru.logger.warning(f"Viser: failed to load mesh for geom '{geom.name}'")
            return None
        if mesh_scale is not None:
            try:
                tm.apply_scale(mesh_scale)
            except Exception:
                pass
        if rgba is not None:
            _set_mesh_color(tm, rgba)
    else:
        tm = _trimesh_from_primitive(geom.type, prim_size, rgba=rgba)

    if tm is None:
        return None

    # Compute local transform.
    if geom.type != mujoco.mjtGeom.mjGEOM_MESH and model_geom is not None:
        geom_pos = np.asarray(model.geom_pos[model_geom.id], dtype=np.float32)
        geom_quat = np.asarray(model.geom_quat[model_geom.id], dtype=np.float32)
    else:
        geom_pos = np.asarray(geom.pos, dtype=np.float32)
        geom_quat = np.asarray(geom.quat, dtype=np.float32)
        qnorm = np.linalg.norm(geom_quat)
        if qnorm > 0:
            geom_quat = geom_quat / qnorm

    try:
        handle = server.scene.add_mesh_trimesh(
            geom_path,
            tm,
            position=geom_pos,
            wxyz=geom_quat,
        )
    except Exception as exc:
        loguru.logger.warning(f"Viser: failed to add geom '{geom.name}': {exc}")
        return None

    rgba_copy = rgba.copy() if rgba is not None else None
    _STATE._geom_handle_cache[geom_path] = (
        cache_key,
        handle,
        tm,
        geom_pos,
        geom_quat,
        rgba_copy,
    )
    return (handle, tm, geom_pos, geom_quat, rgba_copy)


def _build_one_main_body(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    body: Any,
    server: Any,
    entity_root: str,
    collision_rgba: np.ndarray | None,
) -> tuple[Any, int]:
    body_name = body.name if body.name else "worldbody"
    body_path = f"{entity_root}/{body_name}"
    body_handle = server.scene.add_frame(body_path, show_axes=False)

    try:
        body_id = model.body(body_name).id
    except Exception:
        body_id = body.id

    for geom in body.geoms:
        geom_name = geom.name

        if "_object_mass" in geom_name:
            continue
        # Pedestals (worldbody-attached cylinders whose size/pos changes per
        # task) are rendered via the persistent batched-mesh handle managed
        # by `_update_pedestal_batched`, not as individual mesh handles.
        if _is_pedestal_geom_name(geom_name):
            continue

        try:
            gv = int(np.asarray(geom.group).ravel()[0]) if hasattr(geom, "group") else 0
            if gv >= 5:
                continue
        except Exception:
            pass

        geom_path = f"{body_path}/geom_{geom_name}"

        model_geom = None
        try:
            model_geom = model.geom(geom.name)
        except Exception:
            model_geom = None

        is_collision = False
        try:
            group_val = (
                int(np.asarray(geom.group).ravel()[0]) if hasattr(geom, "group") else 0
            )
            if group_val >= 3:
                is_collision = True
        except Exception:
            pass
        if "collision" in geom_name.lower():
            is_collision = True

        rgba = None
        if is_collision and collision_rgba is not None:
            rgba = np.asarray(collision_rgba, dtype=np.float32)
        elif model_geom is not None:
            try:
                rgba = np.asarray(model_geom.rgba, dtype=np.float32)
            except Exception:
                rgba = None
        if rgba is None:
            try:
                rgba = np.asarray(geom.rgba, dtype=np.float32)
            except Exception:
                rgba = None

        result = _load_or_cache_geom(
            spec,
            model,
            geom,
            model_geom,
            geom_path,
            rgba,
            server,
        )
        if result is None:
            continue

        handle, tm, geom_pos, geom_quat, rgba_copy = result

        if is_collision:
            handle.visible = (
                _STATE.scene_checkboxes["collision_meshes"].value
                if "collision_meshes" in _STATE.scene_checkboxes
                else True
            )
            _STATE.collision_geom_handles.append(handle)
            _STATE.collision_geom_info.append(
                (geom_path, tm, geom_pos, geom_quat, rgba_copy)
            )
        else:
            handle.visible = (
                _STATE.scene_checkboxes["visual_meshes"].value
                if "visual_meshes" in _STATE.scene_checkboxes
                else True
            )
            _STATE.visual_geom_handles.append(handle)
            _STATE.visual_geom_info.append(
                (geom_path, tm, geom_pos, geom_quat, rgba_copy)
            )

    return body_handle, body_id


def _build_one_ref_body(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    body: Any,
    server: Any,
    entity_root: str,
    ref_color: np.ndarray,
) -> tuple[Any, int]:
    body_name = body.name
    body_path = f"{entity_root}_ref/{body_name}"
    body_handle = server.scene.add_frame(body_path, show_axes=False)

    try:
        body_id = model.body(body_name).id
    except Exception:
        body_id = body.id

    for geom in body.geoms:
        geom_name = geom.name

        # Stabilizer supports are a sim-only crutch (welded to the object to
        # keep it on the pedestal); they shouldn't appear in the kinematic
        # reference visualization, which is meant to show the input motion.
        if geom_name.startswith("right_support_") or geom_name.startswith(
            "left_support_"
        ):
            continue

        try:
            gv = int(np.asarray(geom.group).ravel()[0]) if hasattr(geom, "group") else 0
            if gv >= 3:
                continue
        except Exception:
            pass

        geom_path = f"{body_path}/geom_{geom_name}"

        model_geom = None
        try:
            model_geom = model.geom(geom.name)
        except Exception:
            model_geom = None

        result = _load_or_cache_geom(
            spec,
            model,
            geom,
            model_geom,
            geom_path,
            ref_color,
            server,
        )
        if result is None:
            continue

        handle, tm, geom_pos, geom_quat, rgba_copy = result
        handle.visible = _kin_any_visible()
        _STATE.ref_geom_handles.append(handle)
        _STATE.ref_geom_info.append((geom_path, tm, geom_pos, geom_quat))

    return body_handle, body_id


def build_and_log_scene_from_spec(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    xml_path: Path | None = None,
    entity_root: str = "mujoco",
    build_ref: bool = True,
    collision_rgba: np.ndarray | None = None,
    build_gui: bool = True,
) -> list[tuple[Any, int]]:
    _ensure_names(spec)
    server = _get_server()
    _STATE.entity_root = entity_root

    if build_gui and "visual_meshes" not in _STATE.scene_checkboxes:
        # Timeline is created lazily — visualize.py creates it earlier so its
        # playback slider lands above Raw Data; optimize_physics.py creates it here.
        if _STATE.gui_folder_timeline is None:
            _STATE.gui_folder_timeline = server.gui.add_folder("Timeline")
            with _STATE.gui_folder_timeline:
                _STATE.opt_progress_bar = server.gui.add_progress_bar(
                    0, animated=True, order=100
                )
                _STATE.sim_progress_bar = server.gui.add_progress_bar(0, order=101)
        if _STATE.gui_folder_kinematic is None:
            _STATE.gui_folder_kinematic = server.gui.add_folder("Kinematic Data")
        if _STATE.gui_folder_retargeted is None:
            _STATE.gui_folder_retargeted = server.gui.add_folder("Retargeted Data")
        if _STATE.gui_folder_opt_traces is None:
            _STATE.gui_folder_opt_traces = server.gui.add_folder("Optimizer")
        if _STATE.gui_folder_rewards is None:
            _STATE.gui_folder_rewards = server.gui.add_folder("Rewards")

        with _STATE.gui_folder_retargeted:
            _STATE.scene_checkboxes["rt_hand"] = server.gui.add_checkbox(
                "Hand", initial_value=True
            )

            @_STATE.scene_checkboxes["rt_hand"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["rt_hand"].value
                for h, _ in _STATE.hand_body_handles:
                    h.visible = val
                for gh in _STATE.hand_geom_handles:
                    gh.visible = val

            _STATE.scene_checkboxes["rt_obj"] = server.gui.add_checkbox(
                "Object", initial_value=True
            )

            @_STATE.scene_checkboxes["rt_obj"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["rt_obj"].value
                for h, _ in _STATE.obj_body_handles:
                    h.visible = val
                for gh in _STATE.obj_geom_handles:
                    gh.visible = val

            _STATE.scene_checkboxes["pedestal"] = server.gui.add_checkbox(
                "Pedestal", initial_value=True
            )

            @_STATE.scene_checkboxes["pedestal"].on_update
            def _(_) -> None:
                if _STATE.pedestal_batched_handle is not None:
                    _STATE.pedestal_batched_handle.visible = _STATE.scene_checkboxes[
                        "pedestal"
                    ].value

            _STATE.scene_checkboxes["visual_meshes"] = server.gui.add_checkbox(
                "Visual", initial_value=True
            )

            @_STATE.scene_checkboxes["visual_meshes"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["visual_meshes"].value
                for h in _STATE.visual_geom_handles:
                    h.visible = val

            _STATE.scene_checkboxes["collision_meshes"] = server.gui.add_checkbox(
                "Collision", initial_value=True
            )

            @_STATE.scene_checkboxes["collision_meshes"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["collision_meshes"].value
                for h in _STATE.collision_geom_handles:
                    h.visible = val

            _STATE.scene_checkboxes["pose_axes"] = server.gui.add_checkbox(
                "Axes", initial_value=False
            )

            @_STATE.scene_checkboxes["pose_axes"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["pose_axes"].value
                for h in _STATE.axes_handles:
                    h.visible = val
                kin_any = _kin_any_visible()
                for h in _STATE.ref_axes_handles:
                    h.visible = val and kin_any
                refresh_geom_appearance(cap_at_half=val)

        with _STATE.gui_folder_kinematic:
            _STATE.scene_checkboxes["kin_hand"] = server.gui.add_checkbox(
                "Hand", initial_value=True
            )

            @_STATE.scene_checkboxes["kin_hand"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["kin_hand"].value
                for h, _ in _STATE.ref_hand_body_handles:
                    h.visible = val
                for gh in _STATE.ref_hand_geom_handles:
                    gh.visible = val
                axes_on = (
                    "pose_axes" in _STATE.scene_checkboxes
                    and _STATE.scene_checkboxes["pose_axes"].value
                )
                kin_any = _kin_any_visible()
                for h in _STATE.ref_axes_handles:
                    h.visible = axes_on and kin_any

            _STATE.scene_checkboxes["kin_obj"] = server.gui.add_checkbox(
                "Object", initial_value=True
            )

            @_STATE.scene_checkboxes["kin_obj"].on_update
            def _(_) -> None:
                val = _STATE.scene_checkboxes["kin_obj"].value
                for h, _ in _STATE.ref_obj_body_handles:
                    h.visible = val
                for gh in _STATE.ref_obj_geom_handles:
                    gh.visible = val
                axes_on = (
                    "pose_axes" in _STATE.scene_checkboxes
                    and _STATE.scene_checkboxes["pose_axes"].value
                )
                kin_any = _kin_any_visible()
                for h in _STATE.ref_axes_handles:
                    h.visible = axes_on and kin_any

    body_entity_and_ids: list[tuple[Any, int]] = []

    for body in spec.bodies:
        entry = _build_one_main_body(
            spec,
            model,
            body,
            server,
            entity_root,
            collision_rgba,
        )
        body_entity_and_ids.append(entry)

    _STATE.body_handles = body_entity_and_ids
    _STATE.scene_spec = spec
    _STATE.scene_model = model

    # One-time pedestal batched handle (4 cylinder slots). Per-task swaps
    # mutate batched_positions/wxyzs/scales in place; absent slots get scale=0.
    if _STATE.pedestal_batched_handle is None:
        _STATE.pedestal_batched_handle = _build_pedestal_batched_handle(
            server,
            entity_root,
        )
    _update_pedestal_batched(spec, model)

    _STATE.axes_handles = _add_pose_axes(
        spec, model, server, entity_root, body_entity_and_ids
    )

    ref_body_entity_and_ids: list[tuple[Any, int]] = []

    if not build_ref:
        _STATE.ref_body_handles = ref_body_entity_and_ids
        _recategorize_handles()
        _reapply_scene_visibility()
        return body_entity_and_ids

    ref_color = _STATE.ref_color

    for body in spec.bodies[1:]:
        entry = _build_one_ref_body(
            spec,
            model,
            body,
            server,
            entity_root,
            ref_color,
        )
        ref_body_entity_and_ids.append(entry)

    _STATE.ref_body_handles = ref_body_entity_and_ids

    # Cache the active-color handle set, then upload (hidden) copies for every
    # other prewarm color so _apply_ref_color is a pure visibility flip.
    active_key = tuple(np.asarray(ref_color, dtype=np.float32).tolist())
    _STATE.ref_geom_handle_sets[active_key] = list(_STATE.ref_geom_handles)
    for color in _PREWARM_REF_COLORS:
        key = tuple(color.tolist())
        if key == active_key:
            continue
        _STATE.ref_geom_handle_sets[key] = _build_ref_handle_set(
            color,
            _STATE.ref_geom_info,
        )

    _STATE.ref_axes_handles = _add_pose_axes(
        spec, model, server, f"{entity_root}_ref", ref_body_entity_and_ids
    )

    _recategorize_handles()
    _reapply_scene_visibility()

    return body_entity_and_ids


_spec_model_cache: dict[tuple, tuple] = {}


def build_and_log_scene(
    xml_path: Path,
    entity_root: str = "mujoco",
    build_ref: bool = True,
    collision_rgba: np.ndarray | None = None,
    build_gui: bool = True,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, list[tuple[Any, int]]]:
    _xml_key = (str(xml_path.resolve()), xml_path.stat().st_mtime_ns)
    if _xml_key in _spec_model_cache:
        spec, model = _spec_model_cache[_xml_key]
    else:
        spec = mujoco.MjSpec.from_file(str(xml_path))
        _ensure_names(spec)
        model = spec.compile()
        _spec_model_cache[_xml_key] = (spec, model)
    body_entity_and_ids = build_and_log_scene_from_spec(
        spec=spec,
        model=model,
        xml_path=xml_path,
        entity_root=entity_root,
        build_ref=build_ref,
        collision_rgba=collision_rgba,
        build_gui=build_gui,
    )
    return spec, model, body_entity_and_ids


REF_COLOR_BLUE = np.array(palette.REF_AT_REST, dtype=np.float32)
REF_COLOR_RED = np.array(palette.REF_HELD, dtype=np.float32)
# Colors prewarmed at scene-build time so `_apply_ref_color` is a pure
# visibility flip (no mid-run mesh upload).
_PREWARM_REF_COLORS = (REF_COLOR_BLUE, REF_COLOR_RED)


def _build_ref_handle_set(
    color: np.ndarray,
    geom_info: list[tuple[str, Any, np.ndarray, np.ndarray]],
) -> list[Any]:
    """Upload color-suffixed (initially hidden) copies of the given ref geoms."""
    server = _STATE.server
    if server is None:
        return []
    color_arr = np.asarray(color, dtype=np.float32)
    suffix = "_c{:.3f}_{:.3f}_{:.3f}_{:.3f}".format(*color_arr.tolist())
    handles: list[Any] = []
    for geom_path, tm, geom_pos, geom_quat in geom_info:
        tm_copy = tm.copy()
        _set_mesh_color(tm_copy, color_arr)
        try:
            handle = server.scene.add_mesh_trimesh(
                geom_path + suffix,
                tm_copy,
                position=geom_pos,
                wxyz=geom_quat,
            )
            handle.visible = False
            handles.append(handle)
        except Exception:
            pass
    return handles


def _apply_ref_color(color: np.ndarray) -> None:
    """Switch reference meshes to the given prewarmed color via visibility flips.

    Only ref-hand/ref-obj subset visibility is touched — the rest of the scene
    didn't change, so `_reapply_scene_visibility` would just resend identical
    flags for hundreds of handles (the source of the per-transition lag).
    """
    if _STATE.server is None or np.array_equal(_STATE.ref_color, color):
        return
    color_arr = np.asarray(color, dtype=np.float32)
    cached = _STATE.ref_geom_handle_sets[tuple(color_arr.tolist())]
    server = _STATE.server
    cb = _STATE.scene_checkboxes
    kin_hand = cb["kin_hand"].value if "kin_hand" in cb else True
    kin_obj = cb["kin_obj"].value if "kin_obj" in cb else True
    with server.atomic():
        for h in _STATE.ref_geom_handles:
            try:
                h.visible = False
            except Exception:
                pass
        _STATE.ref_geom_handles = cached
        _STATE.ref_color = color_arr.copy()
        _recategorize_handles()
        for h in _STATE.ref_hand_geom_handles:
            try:
                h.visible = kin_hand
            except Exception:
                pass
        for h in _STATE.ref_obj_geom_handles:
            try:
                h.visible = kin_obj
            except Exception:
                pass
    try:
        server.flush()
    except Exception:
        pass


# ==============================================================================
# Scene reset & task lifecycle
# ==============================================================================


def reset_mujoco_subtree() -> None:
    """Wipe the MuJoCo scene subtree but leave caller-owned handles alone.

    Removes everything under ``/{entity_root}`` and ``/{entity_root}_ref``
    (including color-suffixed ref-mesh paths from ``_build_ref_handle_set``)
    while leaving handles at unrelated paths (e.g. ``/raw/...``, ``/ik/...``)
    intact — so visualizers can keep persistent overlay handles across task
    switches.
    """
    if _STATE.server is None:
        return
    _stop_playback_threads()
    root = _STATE.entity_root
    with _STATE.server.atomic():
        _STATE.server.scene.remove_by_name(root)
        _STATE.server.scene.remove_by_name(f"{root}_ref")
    _STATE._geom_handle_cache.clear()
    _STATE._geom_cache_dirty.clear()
    _STATE.ref_geom_handle_sets.clear()
    _reset_internal_state()


def _stop_playback_threads() -> None:
    _STATE._playback_stop.set()
    _STATE._playback_stop = threading.Event()


def _reset_task_state_keep_handles() -> None:
    """Clear per-task bookkeeping (callbacks, history, slider) but leave
    scene-handle lists alone.

    Use from selective rebuilds where most viser handles persist across
    tasks. Full ``_reset_internal_state`` callers should invoke this first.
    """
    _STATE.frame_change_callbacks.clear()
    _STATE.frame_history.clear()
    _STATE.ref_color_history.clear()
    _STATE.trace_history.clear()
    _STATE.trace_last_frame_count = 0
    if _STATE.playback_slider is not None:
        _STATE.playback_slider.value = 0
        _STATE.playback_slider.max = 1
    if _STATE.opt_progress_bar is not None:
        _STATE.opt_progress_bar.value = 0
    if _STATE.sim_progress_bar is not None:
        _STATE.sim_progress_bar.value = 0
    if _STATE.playback_checkbox is not None:
        _STATE.playback_speed = 1 if _STATE.playback_checkbox.value else 0
    else:
        _STATE.playback_speed = 0
    _STATE.playback_thread = None


def _reset_internal_state() -> None:
    """Clear all non-scene-node bookkeeping AND drop scene-handle lists."""
    _reset_task_state_keep_handles()
    # Reset state lists (in-place so existing checkbox callbacks still work)
    _STATE.body_handles.clear()
    _STATE.ref_body_handles.clear()
    _STATE.ref_geom_handles.clear()
    _STATE.ref_geom_info.clear()
    _STATE.ref_geom_handle_sets.clear()
    _STATE.visual_geom_handles.clear()
    _STATE.visual_geom_info.clear()
    _STATE.collision_geom_handles.clear()
    _STATE.collision_geom_info.clear()
    _STATE.axes_handles.clear()
    _STATE.ref_axes_handles.clear()
    # Pedestal batched handle was removed by `reset_mujoco_subtree`'s wipe of
    # `/{entity_root}` — drop the dead reference so the next scene build
    # re-creates it.
    _STATE.pedestal_batched_handle = None
    _STATE.ref_color = np.array([0.0, 0.0, 1.0, 0.25], dtype=np.float32)


# ==============================================================================
# Frame playback & rendering
# ==============================================================================


def log_frame(
    data: mujoco.MjData | None,
    sim_time: float,
    viewer_body_entity_and_ids: list[tuple[Any, int]] = [],
    data_ref: mujoco.MjData | None = None,
    show_ui: bool = True,
    playback_fps: float = 50.0,
    ref_color: np.ndarray | None = None,
    record: bool = True,
) -> None:
    del sim_time
    if _STATE.server is None:
        return

    server = _STATE.server

    frame_state = {}
    for handle, bid in viewer_body_entity_and_ids:
        pos = np.asarray(data.xpos[bid], dtype=np.float32)
        quat = np.asarray(data.xquat[bid], dtype=np.float32)
        frame_state[bid] = (pos, quat)

    if data_ref is not None:
        for handle, bid in _STATE.ref_body_handles:
            # We index by the handle instance to avoid overlapping with identical body_ids
            # from the main scene, since both scenes share the exact same MJCF ID map.
            pos = np.asarray(data_ref.xpos[bid], dtype=np.float32)
            quat = np.asarray(data_ref.xquat[bid], dtype=np.float32)
            frame_state[handle.name] = (pos, quat)

    if not record:
        if ref_color is not None:
            _apply_ref_color(ref_color)
        with server.atomic():
            for handle, bid in viewer_body_entity_and_ids:
                if bid in frame_state:
                    pos, quat = frame_state[bid]
                    try:
                        handle.position = tuple(pos)
                        handle.wxyz = tuple(quat)
                    except Exception:
                        pass
            for handle, bid in _STATE.ref_body_handles:
                if handle.name in frame_state:
                    pos, quat = frame_state[handle.name]
                    try:
                        handle.position = tuple(pos)
                        handle.wxyz = tuple(quat)
                    except Exception:
                        pass
        return

    _STATE.frame_history.append(frame_state)
    # Store the intended color for this frame in history without updating _STATE.ref_color
    # (that is handled by _apply_ref_color when the frame is actually displayed)
    frame_ref_color = ref_color if ref_color is not None else _STATE.ref_color
    _STATE.ref_color_history.append(frame_ref_color.copy())
    current_frame = len(_STATE.frame_history) - 1

    # Only apply ref color visually if the user is viewing the latest frame
    if ref_color is not None:
        if (
            _STATE.playback_slider is None
            or int(_STATE.playback_slider.value) == current_frame - 1
        ):
            _apply_ref_color(ref_color)

    if show_ui and _STATE.playback_slider is None:
        if _STATE.playback_folder is None:
            _STATE.playback_folder = (
                _STATE.gui_folder_timeline or server.gui.add_folder("Timeline")
            )
        folder = _STATE.playback_folder
        with folder:
            _STATE.playback_slider = server.gui.add_slider(
                "Frame", min=0, max=max(1, current_frame), step=1, initial_value=0
            )
            _STATE.playback_base_fps = float(playback_fps)

            _STATE.playback_checkbox = server.gui.add_checkbox(
                "Playing", initial_value=False
            )

            @_STATE.playback_checkbox.on_update
            def _(event) -> None:
                _STATE.playback_speed = 1 if _STATE.playback_checkbox.value else 0

            @_STATE.playback_slider.on_update
            def _on_playback_update(_) -> None:
                render_current_frame()

        # Render frame 0 so meshes are positioned correctly on startup
        render_current_frame()

    # Start playback thread if not running (e.g. after a task switch)
    if (
        show_ui
        and _STATE.playback_slider is not None
        and (_STATE.playback_thread is None or not _STATE.playback_thread.is_alive())
    ):
        _stop = _STATE._playback_stop

        def playback_loop():
            while not _stop.is_set():
                fps = max(1.0, _STATE.playback_base_fps)
                sleep_time = 1.0 / fps

                if _STATE.playback_speed != 0 and _STATE.playback_slider is not None:
                    new_val = int(_STATE.playback_slider.value) + _STATE.playback_speed
                    slider_max = int(_STATE.playback_slider.max)
                    new_val = min(max(new_val, 0), slider_max)
                    _STATE.playback_slider.value = new_val
                time.sleep(sleep_time)

        _STATE.playback_thread = threading.Thread(target=playback_loop, daemon=True)
        _STATE.playback_thread.start()

    if _STATE.playback_slider is not None:
        _STATE.playback_slider.max = max(1, current_frame)

    # After a reset, render frame 0 so meshes appear immediately
    if current_frame == 0 and _STATE.playback_slider is not None:
        render_current_frame()


def render_current_frame(
    viewer_body_entity_and_ids: list[tuple[Any, int]] | None = None,
) -> None:
    if _STATE.server is None or _STATE.playback_slider is None:
        return

    frame_idx = int(_STATE.playback_slider.value)
    if frame_idx >= len(_STATE.frame_history) or frame_idx < 0:
        return

    frame_state = _STATE.frame_history[frame_idx]

    if viewer_body_entity_and_ids is None:
        viewer_body_entity_and_ids = _STATE.body_handles

    server = _STATE.server
    with server.atomic():
        for handle, bid in viewer_body_entity_and_ids:
            if bid not in frame_state:
                continue
            pos, quat = frame_state[bid]
            try:
                handle.position = tuple(pos)
                handle.wxyz = tuple(quat)
            except Exception:
                try:
                    handle.position = pos
                    handle.wxyz = quat
                except Exception:
                    pass

        for handle, bid in _STATE.ref_body_handles:
            if handle.name not in frame_state:
                continue
            pos, quat = frame_state[handle.name]
            try:
                handle.position = tuple(pos)
                handle.wxyz = tuple(quat)
            except Exception:
                try:
                    handle.position = pos
                    handle.wxyz = quat
                except Exception:
                    pass

    if frame_idx < len(_STATE.ref_color_history):
        _apply_ref_color(_STATE.ref_color_history[frame_idx])

    for cb in _STATE.frame_change_callbacks:
        try:
            cb(frame_idx)
        except Exception as exc:
            loguru.logger.warning(f"Frame callback error: {exc}")

    # Find newest trace log <= current restricted frame_idx
    valid_trace_idxs = [i for i in _STATE.trace_history.keys() if i <= frame_idx]
    if valid_trace_idxs:
        newest_trace_idx = max(valid_trace_idxs)
        trace_id, traces, trace_ref, _trace_cost, num_iters = _STATE.trace_history[
            newest_trace_idx
        ]

        if getattr(_STATE, "_active_trace_id", -1) != trace_id:
            _STATE._active_trace_id = trace_id
            _STATE.last_traces = traces
            _STATE.last_trace_ref = trace_ref
            _STATE.last_num_iters = num_iters
            _render_traces()

    try:
        server.flush()
    except Exception:
        pass


# ==============================================================================
# Optimizer traces
# ==============================================================================


def _compute_trace_colors(
    I: int, N: int, K: int, num_obj: int = 1, ref_colors: bool = False
) -> np.ndarray:
    colors = np.zeros([I, N, K, 3])
    white = np.array(palette.WHITE)

    if ref_colors:
        object_color = np.array(palette.TRACE_OBJECT_REF)  # green
        robot_color = np.array(palette.TRACE_ROBOT_REF)  # yellow
    else:
        object_color = np.array(palette.TRACE_OBJECT)  # red
        robot_color = np.array(palette.TRACE_ROBOT)  # blue

    for i in range(I):
        for k in range(K):
            is_object = k < num_obj
            color = object_color if is_object else robot_color
            if I == 1:
                colors[i, :, k, :] = color
            else:
                colors[i, :, k, :] = (1 - i / (I - 1)) * white + (i / (I - 1)) * color
    return colors.reshape(I * N * K, 3).astype(np.uint8)


def _build_single_iter_geometry(selected_i: int) -> None:
    """Build scene geometry for only the selected iteration (and refs). Removes old handles first."""
    if _STATE.server is None or _STATE.last_traces is None:
        return

    with _STATE.trace_lock:
        _build_single_iter_geometry_locked(selected_i)


def _build_single_iter_geometry_locked(selected_i: int) -> None:
    a = _STATE.last_traces
    trace_ref = _STATE.last_trace_ref
    num_iters = _STATE.last_num_iters

    I, N, P, K, _ = a.shape
    num_obj = _STATE.num_object_trace_sites
    selected_i = max(0, min(selected_i, I - 1))
    # Rearrange to (I, N, K, P, 3)
    a = a.transpose(0, 1, 3, 2, 4)
    # Use actual iteration count for color gradient (optimizer may pad with zeros)
    color_I = num_iters if num_iters is not None and num_iters <= I else I
    colors_all = _compute_trace_colors(
        color_I, N, K, num_obj=num_obj, ref_colors=False
    ).reshape(color_I, N, K, 3)
    # Pad colors to full I by repeating the last (fully saturated) color
    if color_I < I:
        pad = np.tile(colors_all[-1:], (I - color_I, 1, 1, 1))
        colors_all = np.concatenate([colors_all, pad], axis=0)
    server = _STATE.server

    cbs = _STATE.trace_checkboxes

    def vis(key):
        return cbs[key].value if cbs and key in cbs else True

    show_obj, show_rob = vis("object"), vis("robot")
    show_obj_ref, show_rob_ref = vis("object_ref"), vis("robot_ref")

    with server.atomic():
        for handle in _STATE.trace_handles.values():
            try:
                handle.remove()
            except Exception:
                pass
        _STATE.trace_handles.clear()

        i = selected_i
        for k in range(K):
            is_obj = k < num_obj
            group_name = f"object/site_{k}" if is_obj else f"robot/site_{k}"

            k_strips = a[i, :, k, :, :].reshape(-1, P, 3)
            k_segments = np.stack(
                [k_strips[:, :-1, :], k_strips[:, 1:, :]], axis=2
            ).reshape(-1, 2, 3)

            k_colors = colors_all[i, :, k, :].reshape(-1, 3)
            k_colors = np.repeat(k_colors, repeats=P - 1, axis=0)
            k_colors = np.repeat(k_colors[:, None, :], repeats=2, axis=1)

            visible = show_obj if is_obj else show_rob

            _STATE.trace_handles[(group_name, i)] = server.scene.add_line_segments(
                f"{_STATE.entity_root}/traces/{group_name}/iter_{i}",
                k_segments,
                k_colors,
                line_width=2.0,
                visible=visible,
            )

        if trace_ref is not None:
            ref_a = np.asarray(trace_ref, dtype=np.float32)
            if ref_a.ndim == 5 and ref_a.shape[-1] == 3:
                # (1, 1, H, K, 3) -> (1, 1, K, H, 3)
                ref_a = ref_a.transpose(0, 1, 3, 2, 4)
                ref_K = ref_a.shape[2]
                ref_P = ref_a.shape[3]

                ref_colors_all = _compute_trace_colors(
                    1, 1, ref_K, num_obj=num_obj, ref_colors=True
                ).reshape(1, 1, ref_K, 3)

                for k in range(ref_K):
                    is_obj_ref = k < num_obj
                    group_name = (
                        f"object_ref/site_{k}" if is_obj_ref else f"robot_ref/site_{k}"
                    )

                    k_strips = ref_a[:, :, k, :, :].reshape(-1, ref_P, 3)
                    k_segments = np.stack(
                        [k_strips[:, :-1, :], k_strips[:, 1:, :]], axis=2
                    ).reshape(-1, 2, 3)

                    k_colors = ref_colors_all[:, :, k, :].reshape(-1, 3)
                    k_colors = np.repeat(k_colors, repeats=ref_P - 1, axis=0)
                    k_colors = np.repeat(k_colors[:, None, :], repeats=2, axis=1)

                    ref_visible = show_obj_ref if is_obj_ref else show_rob_ref

                    _STATE.trace_handles[(group_name, 0)] = (
                        server.scene.add_line_segments(
                            f"{_STATE.entity_root}/traces/ref/{group_name}",
                            k_segments,
                            k_colors,
                            line_width=2.0,
                            visible=ref_visible,
                        )
                    )

    try:
        server.flush()
    except Exception:
        pass


def _render_traces() -> None:
    if _STATE.server is None or _STATE.last_traces is None:
        return

    I = _STATE.last_traces.shape[0]

    if _STATE.trace_slider is not None:
        selected_i = int(_STATE.trace_slider.value)
    else:
        selected_i = I - 1

    _build_single_iter_geometry(selected_i)


def log_traces_from_info(
    traces: np.ndarray,
    trace_ref: np.ndarray | None = None,
    sim_time: float = 0.0,
    show_ui: bool = True,
    num_iters: int | None = None,
    num_object_trace_sites: int = 1,
) -> None:
    del sim_time
    if _STATE.server is None:
        return

    _STATE.num_object_trace_sites = max(1, num_object_trace_sites)

    a = np.asarray(traces, dtype=np.float32)
    if a.ndim != 5 or a.shape[-1] != 3:
        loguru.logger.warning(
            f"Viser: skip trace logging with incompatible shape {a.shape}"
        )
        return

    I, N, P, K, _ = a.shape
    if P < 2:
        return

    # Cache into global timeline, keyed to the first frame logged since the last trace
    start_frame = _STATE.trace_last_frame_count
    _STATE.trace_last_frame_count = len(_STATE.frame_history)
    trace_id = _STATE.trace_id_counter
    _STATE.trace_id_counter += 1
    _STATE.trace_history[start_frame] = (
        trace_id,
        a,
        trace_ref,
        None,
        num_iters,
    )

    if show_ui and _STATE.trace_slider is None:
        folder = _STATE.gui_folder_opt_traces or _STATE.server.gui.add_folder(
            "Optimizer"
        )
        with folder:
            max_i = max(0, I - 1)
            _STATE.trace_slider = _STATE.server.gui.add_slider(
                "Trace Iter",
                min=0,
                max=max_i,
                step=1,
                initial_value=max_i,
            )

            _STATE.trace_checkboxes["robot"] = _STATE.server.gui.add_checkbox(
                "Robot Trace", initial_value=False
            )
            _STATE.trace_checkboxes["robot_ref"] = _STATE.server.gui.add_checkbox(
                "Robot Ref Trace", initial_value=False
            )
            _STATE.trace_checkboxes["object"] = _STATE.server.gui.add_checkbox(
                "Object Trace", initial_value=False
            )
            _STATE.trace_checkboxes["object_ref"] = _STATE.server.gui.add_checkbox(
                "Object Ref Trace", initial_value=False
            )

            @_STATE.trace_slider.on_update
            def _on_slider_update(_) -> None:
                _render_traces()

            for _cb in _STATE.trace_checkboxes.values():

                @_cb.on_update
                def _on_trace_toggled(_) -> None:
                    _render_traces()
    else:
        max_i = max(0, I - 1)
        if _STATE.trace_slider.max != max_i:
            _STATE.trace_slider.max = max_i

    # Render immediately if the timeline slider is within this trace's frame range
    if (
        _STATE.playback_slider is not None
        and int(_STATE.playback_slider.value) >= start_frame
    ):
        _STATE._active_trace_id = trace_id
        _STATE.last_traces = a
        _STATE.last_trace_ref = trace_ref
        _STATE.last_num_iters = num_iters
        _render_traces()


# ==============================================================================
# Reward / gate plots
# ==============================================================================


_REWARD_COLORS = palette.PLOT_SERIES


def _normalize_gate_lanes(gate) -> list[tuple[str, np.ndarray]] | None:
    """Coerce a gate spec into per-side lanes: list of (label, (T,) int8 state).

    Accepts a {label: state} dict (one lane per side), a plain 1-D state array
    (a single unlabeled lane), an already-built list of (label, state) pairs,
    or None. Empty lanes are dropped; an all-empty result collapses to None.
    """
    if gate is None:
        return None
    if isinstance(gate, dict):
        items = list(gate.items())
    elif isinstance(gate, (list, tuple)) and gate and isinstance(gate[0], tuple):
        items = list(gate)
    else:
        items = [("", gate)]
    lanes: list[tuple[str, np.ndarray]] = []
    for label, mask in items:
        m = np.asarray(mask).astype(np.int8).flatten()
        if m.size:
            lanes.append((str(label), m))
    return lanes or None


# Effective-gate state → fill color. 1: held (in-hand & far from rest surface),
# 2: at rest (near pedestal/floor & not in-hand). State 0 is left unshaded.
_GATE_FILL = {
    1: palette.GATE_HELD_FILL,  # green
    2: palette.GATE_REST_FILL,  # red
}


def _gate_rect(fig, x0: float, x1: float, y0: float, y1: float, color: str) -> None:
    """Shade band over x∈[x0,x1) confined to paper-y∈[y0,y1] in ``color``."""
    fig.add_shape(
        type="rect",
        xref="x",
        yref="paper",
        x0=x0 - 0.5,
        x1=x1 - 0.5,
        y0=y0,
        y1=y1,
        fillcolor=color,
        line_width=0,
        layer="below",
    )


def _add_gate_lane_shading(fig, lanes) -> None:
    """Shade effective-gate runs as stacked half-height lanes (one per side).

    Each lane is a (T,) int8 state array (see ``_GATE_FILL``): green where the
    object is held, red where it rests on a surface, unshaded elsewhere. The
    plot height is split into equal abutting stripes — top lane first — so
    different sides sit in their own band and never blend. Within a lane,
    contiguous runs of the same state collapse into a single rectangle; a small
    left-edge label tags each lane. A single lane spans the full height with no
    label (identical to the pre-per-side single-object band).
    """
    if not lanes:
        return
    L = len(lanes)
    for li, (label, mask) in enumerate(lanes):
        m = np.asarray(mask).astype(np.int8).flatten()
        y_hi = 1.0 - li / L
        y_lo = 1.0 - (li + 1) / L
        i = 0
        while i < m.shape[0]:
            v = int(m[i])
            if v == 0:
                i += 1
                continue
            j = i
            while j < m.shape[0] and int(m[j]) == v:
                j += 1
            _gate_rect(fig, i, j, y_lo, y_hi, _GATE_FILL.get(v, _GATE_FILL[1]))
            i = j
        if label and L > 1:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.0,
                y=(y_lo + y_hi) / 2,
                xanchor="right",
                yanchor="middle",
                showarrow=False,
                text=label,
                font=dict(size=9, color=palette.PLOT_LABEL),
            )


def _add_timeline_shading(fig) -> None:
    """Add warmup band (gray, full height) + per-side effective-gate lanes
    (green = held, red = at rest).

    See [[_add_gate_lane_shading]] for the stacked-lane layout.
    """
    ws = int(_STATE.reward_warmup_steps)
    if ws > 0:
        fig.add_vrect(
            x0=-0.5,
            x1=ws - 0.5,
            fillcolor=palette.PLOT_BG_FILL,
            line_width=0,
            layer="below",
        )
    _add_gate_lane_shading(fig, _STATE.reward_gate_lanes)


def _build_reward_figure(highlight_frame: int | None = None):
    import plotly.graph_objects as go

    x = _STATE.reward_frames
    fig = go.Figure()
    total_frames = max(len(_STATE.frame_history) - 1, max(x) if x else 0)
    _add_timeline_shading(fig)
    for idx, name in enumerate(_STATE.reward_series_names):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=_STATE.reward_values[idx],
                name=name,
                line=dict(color=_STATE.reward_series_colors[idx], width=1.5),
                mode="lines",
            )
        )
    if highlight_frame is not None:
        fig.add_vline(
            x=highlight_frame,
            line_dash="dash",
            line_color=palette.PLOT_HIGHLIGHT,
            line_width=2,
        )
    fig.update_layout(
        xaxis=dict(range=[0, total_frames]),
        margin=dict(l=50, r=10, t=10, b=30),
        showlegend=False,
        font=dict(size=10),
        template="plotly_white",
    )
    return fig


def log_reward_gate(
    gate_mask,
    warmup_steps: int = 0,
) -> None:
    """Set the execution-timeline gate state + warmup length for the plot.

    gate_mask: per-side effective-gate state, drawn as stacked half-height
        bands on the reward plot. Accepts a {label: (T,) state} dict (one
        lane per side), a plain (T,) state array (single lane), or None to
        clear. Per-frame values: 1=held (green), 2=at rest (red), 0=neither.
        Used purely for background shading — does not affect plotted values.
    """
    if _STATE.server is None:
        return
    _STATE.reward_gate_lanes = _normalize_gate_lanes(gate_mask)
    _STATE.reward_warmup_steps = int(max(0, warmup_steps))
    # Refresh now in case the plot is already up.
    if _STATE.reward_plot_handle is not None:
        try:
            _STATE.reward_plot_handle.figure = _build_reward_figure()
        except Exception:
            pass


def _update_reward_plot(highlight_frame: int | None = None) -> None:
    if _STATE.server is None or not _STATE.reward_frames:
        return
    fig = _build_reward_figure(highlight_frame)
    if _STATE.reward_plot_handle is None:
        folder = _STATE.gui_folder_rewards or _STATE.server.gui.add_folder("Rewards")
        with folder:
            _STATE.reward_plot_handle = _STATE.server.gui.add_plotly(fig, aspect=0.7)
        register_frame_callback(_on_frame_change_reward)
    else:
        _STATE.reward_plot_handle.figure = fig


def _on_frame_change_reward(frame_idx: int) -> None:
    now = time.time()
    if now - _STATE._reward_plot_last_update < 0.2:
        return
    _STATE._reward_plot_last_update = now
    _update_reward_plot(frame_idx)


def log_reward_step(values: dict[str, float]) -> None:
    """Log one frame of executed reward terms. Call once per sim step."""
    if _STATE.server is None:
        return

    frame = float(max(0, len(_STATE.frame_history) - 1))

    # First call: discover series from keys
    if not _STATE.reward_series_names:
        for idx, name in enumerate(values.keys()):
            _STATE.reward_series_names.append(name)
            _STATE.reward_series_colors.append(
                _REWARD_COLORS[idx % len(_REWARD_COLORS)]
            )
            _STATE.reward_values.append([])

    _STATE.reward_frames.append(frame)
    for idx, name in enumerate(_STATE.reward_series_names):
        _STATE.reward_values[idx].append(values.get(name, float("nan")))

    _update_reward_plot(frame)


# ==============================================================================
# Progress bars
# ==============================================================================


def update_opt_progress(iteration: int, max_iterations: int) -> None:
    if _STATE.opt_progress_bar is None:
        return
    pct = int(100 * iteration / max(max_iterations, 1))
    _STATE.opt_progress_bar.value = min(pct, 100)


def update_sim_progress(sim_step: int, max_sim_steps: int) -> None:
    if _STATE.sim_progress_bar is None:
        return
    pct = int(100 * sim_step / max(max_sim_steps, 1))
    _STATE.sim_progress_bar.value = min(pct, 100)
