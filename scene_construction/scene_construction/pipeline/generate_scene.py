import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import loguru
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

from scene_construction.io import (
    get_processed_data_dir,
    nfs_safe_lock,
    resolve_auto_embodiment,
)
from scene_construction.paths import ASSETS_HANDS_DIR, PROJECT_ROOT

# Pedestal/support placement and related helpers live in resolve_pedestal.py;
# this file only builds the pose-independent structural scene.


# Forearm direction in each hand's palm-body local frame, per (robot, side).
# The cylinder's long axis (default +Z when quat is identity) is aligned with
# this vector and the geom is placed at ``half_height * forearm_dir`` so its
# near face touches the palm body origin (the wrist mount in every bundled
# hand XML). Hand MJCFs come from different vendors with no shared frame
# convention, so we hardcode rather than infer — fingertip averaging picks up
# thumb-pose bias and was visibly off-axis on sharpa.
_UR3_FOREARM_DIRS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("sharpa", "right"): (0.0, 0.0, -1.0),
    ("sharpa", "left"): (0.0, 0.0, -1.0),
    ("mano", "right"): (1.0, 0.0, 0.0),
    ("mano", "left"): (-1.0, 0.0, 0.0),
}


def _quat_align_z_to(target: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) that rotates +Z onto ``target`` (assumed unit)."""
    z = np.array([0.0, 0.0, 1.0])
    cos_a = float(np.clip(np.dot(z, target), -1.0, 1.0))
    if cos_a > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if cos_a < -1.0 + 1e-9:
        # 180° rotation about any axis perpendicular to Z; pick X.
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, target)
    axis /= np.linalg.norm(axis)
    angle = float(np.arccos(cos_a))
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _add_ur3_arm_cylinders(
    mj_spec: mujoco.MjSpec,
    embodiment_type: str,
    robot_type: str,
    radius: float = 0.032,
    half_height: float = 0.046,
) -> list[str]:
    """Add a massless cylinder behind each palm to approximate the UR3e wrist
    housing; without it the simulator lets that volume pass through objects/table.

    Defaults: 32 mm radius matches every UR3e wrist segment in
    ``mujoco-env/universal_robots_ur3e/ur3e.xml``; 92 mm length is the straight
    section from the hand mount back to the wrist_2-wrist_3 L-bend (the L-bent
    section is not modeled — it depends on ``wrist_3_joint``, which retargeting
    does not track). Geom name ``collision_hand_{side}_arm_cyl`` auto-joins
    ``hand_collision_names``; it is excluded from hand↔object pairs (forearm
    must not manipulate the object) and, not matching ``_is_distal_or_middle``,
    from finger self-collision pairs.
    """
    sides: list[str] = []
    if embodiment_type in ("right", "bimanual"):
        sides.append("right")
    if embodiment_type in ("left", "bimanual"):
        sides.append("left")

    snap_model = mj_spec.compile()
    added: list[str] = []
    for side in sides:
        if (robot_type, side) not in _UR3_FOREARM_DIRS:
            loguru.logger.warning(
                f"{robot_type}/{side}: no forearm direction registered in "
                f"_UR3_FOREARM_DIRS; skipping UR3 arm cylinder."
            )
            continue
        forearm_dir = np.asarray(_UR3_FOREARM_DIRS[(robot_type, side)], dtype=float)
        forearm_dir /= np.linalg.norm(forearm_dir)

        palm_geom_name: str | None = None
        for cand in (
            f"collision_hand_{side}_palm_0",
            f"collision_hand_{side}_palm",
        ):
            if mujoco.mj_name2id(snap_model, mujoco.mjtObj.mjOBJ_GEOM, cand) != -1:
                palm_geom_name = cand
                break
        if palm_geom_name is None:
            loguru.logger.warning(
                f"{side}: no palm collision geom found; skipping UR3 arm cylinder."
            )
            continue
        gid = mujoco.mj_name2id(snap_model, mujoco.mjtObj.mjOBJ_GEOM, palm_geom_name)
        bid = int(snap_model.geom_bodyid[gid])
        palm_body_name = mujoco.mj_id2name(snap_model, mujoco.mjtObj.mjOBJ_BODY, bid)

        quat = _quat_align_z_to(forearm_dir)
        # Cylinder near face flush with the palm body origin, extending one
        # full length away from the fingers along the forearm direction.
        center = forearm_dir * half_height

        geom_name = f"collision_hand_{side}_arm_cyl"
        palm_body = mj_spec.body(palm_body_name)
        palm_body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, half_height, 0.0],
            pos=center.tolist(),
            quat=quat.tolist(),
            density=0.0,
            contype=0,
            conaffinity=0,
            group=3,
            rgba=[0.5, 0.5, 0.5, 0.8],
        )
        added.append(geom_name)
        loguru.logger.info(
            f"{side}: added UR3 arm cylinder '{geom_name}' on body "
            f"'{palm_body_name}' (radius={radius:.3f} m, half_h={half_height:.3f} m, "
            f"forearm_dir_local=[{forearm_dir[0]:+.3f}, {forearm_dir[1]:+.3f}, {forearm_dir[2]:+.3f}])"
        )
    return added


def _robot_src_signature(src_dir: str) -> str:
    """Hash over each file's relpath/size/mtime, so the asset copy runs only
    when the source actually changed (see main)."""
    entries = []
    for root, _, files in os.walk(src_dir):
        for name in files:
            p = os.path.join(root, name)
            st = os.stat(p)
            entries.append(
                f"{os.path.relpath(p, src_dir)}:{st.st_size}:{st.st_mtime_ns}"
            )
    return hashlib.sha1("\n".join(sorted(entries)).encode()).hexdigest()


def main(
    output_root_dir: str = str(PROJECT_ROOT / "outputs"),
    dataset_name: str = "do_as_i_do",
    robot_type: str = "sharpa",
    embodiment_type: str = "bimanual",
    task: str = "",
    data_id: int = 0,
    object_object_collision: bool = True,
    object_density: float = 1000,
    use_visual_mesh_as_collision: bool = False,
    object_floor_collision: bool = False,
    hand_floor_collision: bool = False,
    use_pedestal: bool = False,
    use_support: bool = False,
    object_armature: float = 1e-4,
    object_damping: float = 1e-2,
    object_frictionloss: float = 1e-4,
    friction_scale: float = 1.0,
    show_viewer: bool = True,
    force: bool = False,
    add_ur3_arm: bool = False,
    ur3_arm_radius: float = 0.032,
    ur3_arm_half_height: float = 0.046,
):
    output_root_dir = os.path.abspath(output_root_dir)
    if embodiment_type == "auto":
        embodiment_type = resolve_auto_embodiment(dataset_name, output_root_dir, task)

    processed_dir = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    os.makedirs(processed_dir, exist_ok=True)
    scene_path = f"{processed_dir}/scene_ik.xml"
    if not force and os.path.exists(scene_path):
        loguru.logger.info(f"Skipping generate_scene.py (output exists: {scene_path})")
        return

    src_robot_dir = str(ASSETS_HANDS_DIR / robot_type)
    robots_assets_dir = f"{output_root_dir}/assets/robots/{robot_type}"
    # The robot assets dir is shared across jobs. copytree truncates each dest
    # file before rewriting, so a copy racing a sibling's mesh read yields an
    # empty .STL. Copy at most once per source revision: the sentinel holds a
    # fingerprint of the source tree, so exactly one copytree runs per run,
    # under the lock, before any reader is released. (No longer keyed on `force`.)
    os.makedirs(os.path.dirname(robots_assets_dir), exist_ok=True)
    sentinel = Path(f"{robots_assets_dir}/.copied")
    with nfs_safe_lock(f"{robots_assets_dir}.lock", timeout=120):
        src_sig = _robot_src_signature(src_robot_dir)
        cached_sig = sentinel.read_text().strip() if sentinel.exists() else None
        if cached_sig == src_sig:
            loguru.logger.info(
                f"Robot assets up to date at {robots_assets_dir}; reusing."
            )
        else:
            shutil.copytree(src_robot_dir, robots_assets_dir, dirs_exist_ok=True)
            sentinel.write_text(src_sig)
            loguru.logger.info(
                f"Copied robot assets from {src_robot_dir} to {robots_assets_dir}"
            )

    robot_xml_name = (
        "bimanual.xml" if embodiment_type == "bimanual" else f"{embodiment_type}.xml"
    )
    robot_xml_path = f"{robots_assets_dir}/{robot_xml_name}"
    if not os.path.exists(robot_xml_path):
        raise FileNotFoundError(f"Robot XML not found: {robot_xml_path}")

    mj_spec = mujoco.MjSpec.from_file(robot_xml_path)

    # ``mj_spec.meshdir`` has two consumers with different base dirs: compile()
    # resolves it relative to the loaded XML's dir (robots_dir_abs); the saved
    # scene_ik.xml embeds it verbatim and downstream loaders resolve it relative
    # to that file's dir (processed_dir). Use the compile-time value now, rewrite
    # to save_meshdir just before to_xml() below.
    assets_root_dir = f"{output_root_dir}/assets"
    robots_dir_abs = f"{assets_root_dir}/robots/{robot_type}"
    compile_meshdir = os.path.relpath(assets_root_dir, robots_dir_abs)
    save_meshdir = os.path.relpath(assets_root_dir, processed_dir)
    original_meshdir = mj_spec.meshdir
    mj_spec.meshdir = compile_meshdir

    # Collect all mesh subdirs: bimanual included XMLs have different meshdirs
    # (meshes_right/, meshes_left/).
    mesh_subdirs = [original_meshdir] if original_meshdir else []
    for subdir in os.listdir(robots_dir_abs):
        subdir_path = os.path.join(robots_dir_abs, subdir)
        if os.path.isdir(subdir_path) and subdir not in mesh_subdirs:
            mesh_subdirs.append(subdir)
    for mesh in getattr(mj_spec, "meshes", []):
        original = mesh.file
        if os.path.isabs(original):
            candidate_abs = original
        else:
            candidate_abs = os.path.normpath(
                os.path.join(robots_dir_abs, original_meshdir, original)
            )
        # Not found: search mesh subdirs (bimanual right.xml/left.xml use
        # separate meshes_right/, meshes_left/).
        if not os.path.exists(candidate_abs):
            basename = os.path.basename(original)
            for subdir in mesh_subdirs:
                alt = os.path.normpath(os.path.join(robots_dir_abs, subdir, basename))
                if os.path.exists(alt):
                    candidate_abs = alt
                    break
        try:
            file_rel_to_assets = os.path.relpath(candidate_abs, assets_root_dir)
        except ValueError:
            # different drives, etc.
            file_rel_to_assets = original
        mesh.file = file_rel_to_assets

    keypoint_data_dir = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    contact_npz_path = f"{keypoint_data_dir}/trajectory_keypoints.npz"
    # Read-only: this file is a preprocess artifact and is not modified here.
    with np.load(contact_npz_path) as _npz:
        loaded_data = {k: _npz[k] for k in _npz.files}
    try:
        contact_pos_left = loaded_data["contact_pos_left"]
        contact_pos_right = loaded_data["contact_pos_right"]
    except KeyError:
        loguru.logger.warning(
            f"No contact data found at {contact_npz_path}; falling back to zeros"
        )
        contact_pos_left = np.zeros((5, 3))
        contact_pos_right = np.zeros((5, 3))
    finger_names = [
        "thumb_tip",
        "index_tip",
        "middle_tip",
        "ring_tip",
        "pinky_tip",
    ]

    mj_spec.add_texture(
        name="skybox",
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0, 0, 0],
        width=512,
        height=3072,
    )
    # NOTE: floor/pedestal geom is added after mesh paths are resolved (see below).

    # Load object convex meshes from task_info.json
    task_info_path = f"{keypoint_data_dir}/../task_info.json"
    task_info = {}
    with open(task_info_path) as f:
        task_info = json.load(f)
    right_convex_dir = task_info.get("right_object_convex_dir")
    if right_convex_dir and not os.path.isabs(right_convex_dir):
        right_convex_dir = f"{output_root_dir}/{right_convex_dir}"
    left_convex_dir = task_info.get("left_object_convex_dir")
    if left_convex_dir and not os.path.isabs(left_convex_dir):
        left_convex_dir = f"{output_root_dir}/{left_convex_dir}"
    # Fallback: look for convex/ subdir inside the mesh dir
    if not right_convex_dir or not os.path.isdir(right_convex_dir):
        right_mesh_dir_raw = task_info.get("right_object_mesh_dir")
        if right_mesh_dir_raw:
            candidate = (
                f"{output_root_dir}/{right_mesh_dir_raw}/convex"
                if not os.path.isabs(right_mesh_dir_raw)
                else f"{right_mesh_dir_raw}/convex"
            )
            if os.path.isdir(candidate):
                right_convex_dir = candidate
    if not left_convex_dir or not os.path.isdir(left_convex_dir):
        left_mesh_dir_raw = task_info.get("left_object_mesh_dir")
        if left_mesh_dir_raw:
            candidate = (
                f"{output_root_dir}/{left_mesh_dir_raw}/convex"
                if not os.path.isabs(left_mesh_dir_raw)
                else f"{left_mesh_dir_raw}/convex"
            )
            if os.path.isdir(candidate):
                left_convex_dir = candidate
    right_mesh_dir = task_info.get("right_object_mesh_dir")
    if right_mesh_dir and not os.path.isabs(right_mesh_dir):
        right_mesh_dir = f"{output_root_dir}/{right_mesh_dir}"
    left_mesh_dir = task_info.get("left_object_mesh_dir")
    if left_mesh_dir and not os.path.isabs(left_mesh_dir):
        left_mesh_dir = f"{output_root_dir}/{left_mesh_dir}"

    # Visual meshes (non-colliding)
    right_visual_file = f"{right_mesh_dir}/visual.obj" if right_mesh_dir else None
    left_visual_file = f"{left_mesh_dir}/visual.obj" if left_mesh_dir else None
    if (
        embodiment_type in ["right", "bimanual"]
        and right_visual_file
        and os.path.exists(right_visual_file)
    ):
        file_rel_to_meshdir = os.path.relpath(right_visual_file, assets_root_dir)
        mj_spec.add_mesh(name="right_visual", file=file_rel_to_meshdir)
    if (
        embodiment_type in ["left", "bimanual"]
        and left_visual_file
        and os.path.exists(left_visual_file)
    ):
        file_rel_to_meshdir = os.path.relpath(left_visual_file, assets_root_dir)
        mj_spec.add_mesh(name="left_visual", file=file_rel_to_meshdir)

    # Right object meshes
    right_object_files = []
    if embodiment_type in ["right", "bimanual"]:
        if use_visual_mesh_as_collision and right_visual_file:
            if os.path.exists(right_visual_file):
                # Reuse the visual mesh for collision (no extra collision mesh).
                right_object_files = ["visual"]
        elif right_convex_dir and os.path.isdir(right_convex_dir):
            right_object_files = sorted(
                [f for f in os.listdir(right_convex_dir) if f.endswith(".obj")]
            )
            for f in right_object_files:
                suffix = f.split(".")[0]
                file_abs = f"{right_convex_dir}/{f}"
                file_rel_to_meshdir = os.path.relpath(file_abs, assets_root_dir)
                mj_spec.add_mesh(name=f"right_{suffix}", file=file_rel_to_meshdir)

    # Left object meshes
    left_object_files = []
    if embodiment_type in ["left", "bimanual"]:
        if use_visual_mesh_as_collision and left_visual_file:
            if os.path.exists(left_visual_file):
                # Reuse the visual mesh for collision (no extra collision mesh).
                left_object_files = ["visual"]
        elif left_convex_dir and os.path.isdir(left_convex_dir):
            left_object_files = sorted(
                [f for f in os.listdir(left_convex_dir) if f.endswith(".obj")]
            )
            for f in left_object_files:
                suffix = f.split(".")[0]
                file_abs = f"{left_convex_dir}/{f}"
                file_rel_to_meshdir = os.path.relpath(file_abs, assets_root_dir)
                mj_spec.add_mesh(name=f"left_{suffix}", file=file_rel_to_meshdir)

    # Pedestal/welded-support geoms are emitted later by resolve_pedestal.py
    # (they need the IK-output pose). ``use_pedestal``/``use_support`` are no-ops
    # here, kept in the signature for caller compatibility.
    #
    # Single floor plane at z=0. Collisions are driven by the explicit pairs
    # below (gated on object_floor_collision / hand_floor_collision); the default
    # contype=0 conaffinity=0 means no dynamic pairs.
    floor_names: list[str] = []
    if object_floor_collision or hand_floor_collision:
        mj_spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
        )
        floor_names.append("floor")

    right_object_collision_names = []
    if embodiment_type in ["right", "bimanual"]:
        right_object_handle = mj_spec.worldbody.add_body(
            name="right_object",
            mocap=False,
        )
        right_object_joint_handle = right_object_handle.add_joint(
            name="right_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=object_armature,
            damping=object_damping,
            frictionloss=object_frictionloss,
        )
        for obj_file in right_object_files:
            suffix = obj_file.split(".")[0]
            is_visual_collision = use_visual_mesh_as_collision and suffix == "visual"
            geom_name = f"right_object_{suffix}"
            if suffix.isdigit() or is_visual_collision:
                rgba = [0, 1, 0, 1]
                density = object_density
                if is_visual_collision:
                    geom_name = "right_object_collision_visual"
                right_object_collision_names.append(geom_name)
                group = 3
            else:
                rgba = [1, 1, 1, 1]
                density = 0
                group = 0
            right_object_handle.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"right_{suffix}",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=rgba,
                density=density,
                group=group,
            )
        # No object: add a dummy mass so the free joint stays well-conditioned.
        if len(right_object_files) == 0:
            right_object_handle.add_geom(
                name="right_object_mass",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0.5, 0.5, 0.5],  # put it far away to avoid collision
                size=[0.1, 0.1, 0.1],
                density=10,
                group=5,
            )
            right_object_joint_handle.frictionloss = 1.0
            right_object_joint_handle.armature = 1.0
        right_object_handle.add_site(
            name="right_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            rgba=[1, 0, 0, 1],
            group=3,
        )
        if "right_visual" in [m.name for m in mj_spec.meshes]:
            right_object_handle.add_geom(
                name="right_object_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="right_visual",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=[1, 1, 1, 1],
                density=0,
                group=0,
            )
        right_object_handle.add_site(
            name="trace_right_object",
            pos=[0, 0, 0],
            size=[0.01, 0.01, 0.01],
            rgba=[0, 1, 0, 1],
            group=4,
        )
        # Contact sites for the virtual tracking constraint.
        for i, finger_name in enumerate(finger_names):
            right_object_handle.add_site(
                name=f"track_object_right_{finger_name}",
                pos=contact_pos_right[i],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                group=4,
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_object_right_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_object_right_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_hand_right_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_hand_right_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )

    left_object_collision_names = []
    if embodiment_type in ["left", "bimanual"]:
        left_object_handle = mj_spec.worldbody.add_body(
            name="left_object",
            mocap=False,
            gravcomp=(
                1 if len(left_object_files) == 0 else 0
            ),  # if left object is not present, set gravcomp to 1 to avoid gravity
        )
        left_joint_handle = left_object_handle.add_joint(
            name="left_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=object_armature,
            damping=object_damping,
            frictionloss=object_frictionloss,
        )
        for obj_file in left_object_files:
            suffix = obj_file.split(".")[0]
            is_visual_collision = use_visual_mesh_as_collision and suffix == "visual"
            geom_name = f"left_object_{suffix}"
            if suffix.isdigit() or is_visual_collision:
                rgba = [0, 1, 0, 1]
                density = object_density
                if is_visual_collision:
                    geom_name = "left_object_collision_visual"
                left_object_collision_names.append(geom_name)
                group = 3
            else:
                rgba = [1, 1, 1, 1]
                density = 0
                group = 0
            left_object_handle.add_geom(
                name=geom_name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"left_{suffix}",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=rgba,
                density=density,
                group=group,
            )
        # No object: add a dummy mass so the free joint stays well-conditioned.
        if len(left_object_files) == 0:
            left_object_handle.add_geom(
                name="left_object_mass",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0.5, 0.5, 0.5],  # put it far away to avoid collision
                size=[0.1, 0.1, 0.1],
                density=10,
                group=5,
            )
            left_joint_handle.frictionloss = 1.0
            left_joint_handle.armature = 1.0
        left_object_handle.add_site(
            name="left_object",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.02, 0.03],
            pos=[0, 0, 0],
            rgba=[1, 0, 0, 1],
            group=3,
        )
        if "left_visual" in [m.name for m in mj_spec.meshes]:
            left_object_handle.add_geom(
                name="left_object_visual",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname="left_visual",
                pos=[0, 0, 0],
                conaffinity=0,
                contype=0,
                rgba=[1, 1, 1, 1],
                density=0,
                group=0,
            )
        left_object_handle.add_site(
            name="trace_left_object",
            pos=[0, 0, 0],
            size=[0.01, 0.01, 0.01],
            rgba=[0, 1, 0, 1],
            group=4,
        )
        for i, finger_name in enumerate(finger_names):
            # if left object is not present, add contact site to the right object
            if len(left_object_files) == 0:
                handle = right_object_handle
            else:
                handle = left_object_handle
            handle.add_site(
                name=f"track_object_left_{finger_name}",
                pos=contact_pos_left[i],
                size=[0.01, 0.01, 0.01],
                rgba=[0, 1, 0, 1],
                group=4,
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_object_left_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_object_left_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )
            mocap_handle = mj_spec.worldbody.add_body(
                name=f"ref_hand_left_{finger_name}",
                pos=[0, 0, 0],
                quat=[1, 0, 0, 0],
                mocap=True,
            )
            mocap_handle.add_site(
                name=f"ref_hand_left_{finger_name}",
                pos=[0, 0, 0],
                size=[0.02, 0.02, 0.02],
                group=4,
                rgba=[0, 1, 0, 1],
            )

    # Add inactive weld constraints for warmup (pin object to world frame)
    for side in ("right", "left"):
        if embodiment_type in [side, "bimanual"]:
            e = mj_spec.add_equality(
                name=f"{side}_object_weld",
                type=mujoco.mjtEq.mjEQ_WELD,
                name1=f"{side}_object",
                objtype=mujoco.mjtObj.mjOBJ_BODY,
            )
            e.active = False
            e.solref = np.array([0.01, 1.0])
            e.solimp = np.array([0.9995, 0.9995, 0.001, 0.5, 2.0])

    # Welded stabilizer supports are now emitted by the post-IK step in
    # ``retargeting/pipeline/resolve_pedestal.py`` (they need the IK-output object pose).

    object_collision_names = right_object_collision_names + left_object_collision_names
    loguru.logger.info(f"Added {len(object_collision_names)} objects to model")

    default_solref = [0.01, 1]
    # Stiff constraint reserved for hand<->object contacts where we want very
    # little interpenetration during manipulation. All other pairs (object<->
    # ground, hand<->ground, hand self, etc.) use MuJoCo's default solimp,
    # which is softer and produces more stable resting contact.
    hand_object_solimp = [0.998, 0.998, 0.001, 0.5, 2]
    default_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    default_friction = [
        1.0 * friction_scale,
        1.0 * friction_scale,
        0.1 * friction_scale,
        0.0,
        0.0,
    ]
    small_friction = [
        0.01 * friction_scale,
        0.01 * friction_scale,
        0.0001 * friction_scale,
        0.0,
        0.0,
    ]
    if add_ur3_arm:
        _add_ur3_arm_cylinders(
            mj_spec,
            embodiment_type,
            robot_type,
            radius=ur3_arm_radius,
            half_height=ur3_arm_half_height,
        )

    # [thumb, index intermediate, index, middle, ring, pinky + floor] <-> object
    hand_collision_names = []
    for geom_id in range(len(mj_spec.geoms)):
        geom = mj_spec.geoms[geom_id]
        if geom.name.startswith("collision_hand_"):
            hand_collision_names.append(geom.name)
    # No pedestals yet: ground_names is just the optional floor plane.
    # resolve_pedestal.py re-emits the hand↔pedestal / object↔pedestal pairs.
    ground_names = list(floor_names) if object_floor_collision else []
    hand_collision_names_for_object = hand_collision_names + ground_names

    object_names = []
    if embodiment_type in ["left", "bimanual"]:
        object_names.append("left_object")
    if embodiment_type in ["right", "bimanual"]:
        object_names.append("right_object")

    contact_cnt = 0

    # hand/floor/pedestal <-> object collision.
    # All pairs use condim=6 (normal + tangential + torsional + rolling). On
    # the object<->ground pair the torsional/rolling terms are what damp out
    # the rocking of an elongated object on the small cylindrical pedestal
    # without having to add joint-level damping (which would also slow the
    # object in free flight).
    ground_names_set = set(ground_names)
    for object_collision_name in object_collision_names:
        for hand_collision_name in hand_collision_names_for_object:
            # The UR3 arm cylinder approximates the wrist housing for
            # floor/pedestal collision only; it must not collide with the
            # object (the hand, not the forearm, is what manipulates it).
            if hand_collision_name.endswith("_arm_cyl"):
                continue
            is_hand_object = hand_collision_name not in ground_names_set
            mj_spec.add_pair(
                name=f"{hand_collision_name}_{object_collision_name}",
                geomname1=hand_collision_name,
                geomname2=object_collision_name,
                solref=default_solref,
                solimp=hand_object_solimp if is_hand_object else default_solimp,
                friction=default_friction,
                condim=6,
            )
            contact_cnt += 1

    # support <-> pedestal pairs are emitted by ``retargeting/pipeline/resolve_pedestal.py``
    # (alongside the supports themselves).

    if (
        object_object_collision
        and embodiment_type == "bimanual"
        and len(right_object_collision_names) > 0
        and len(left_object_collision_names) > 0
    ):
        for right_object_collision_name in right_object_collision_names:
            for left_object_collision_name in left_object_collision_names:
                mj_spec.add_pair(
                    name=f"{right_object_collision_name}_{left_object_collision_name}",
                    geomname1=right_object_collision_name,
                    geomname2=left_object_collision_name,
                    solref=default_solref,
                    solimp=default_solimp,
                    friction=small_friction,
                    condim=3,
                )
    if hand_floor_collision:
        for hand_collision_name in hand_collision_names:
            for floor_name in floor_names:
                mj_spec.add_pair(
                    name=f"{hand_collision_name}_{floor_name}",
                    geomname1=hand_collision_name,
                    geomname2=floor_name,
                    solref=default_solref,
                    solimp=default_solimp,
                    friction=default_friction,
                    condim=3,
                )
                contact_cnt += 1

    # hand self-collision pairs (same-hand different fingers, plus cross-hand)
    def _is_fingertip(name: str) -> bool:
        return name[-1].isdigit() and name.rsplit("_", 1)[-1] in ("0", "3")

    def _is_distal_or_middle(name: str) -> bool:
        return (
            name[-1].isdigit()
            and name.rsplit("_", 1)[-1]
            in (
                "0",
                "1",
                "2",
                "3",
            )
            and "palm" not in name
        )

    def _get_finger_name(name: str) -> str:
        # 'collision_hand_right_thumb_0' -> 'thumb'
        parts = name.split("_")
        return "_".join(parts[3:-1])

    hand_collision_pairs = []
    for collision_name in hand_collision_names:
        if _is_distal_or_middle(collision_name):
            hand_side = collision_name.split("_")[2]
            # hand <-> hand collision (bimanual cross-hand)
            if embodiment_type == "bimanual":
                another_hand_side = "right" if hand_side == "left" else "left"
                for another_collision_name in hand_collision_names:
                    if (
                        another_hand_side in another_collision_name
                        and another_collision_name != collision_name
                        and _is_distal_or_middle(another_collision_name)
                        and (collision_name, another_collision_name)
                        not in hand_collision_pairs
                        and (another_collision_name, collision_name)
                        not in hand_collision_pairs
                    ):
                        mj_spec.add_pair(
                            name=f"{collision_name}_{another_collision_name}",
                            geomname1=collision_name,
                            geomname2=another_collision_name,
                            solref=default_solref,
                            solimp=default_solimp,
                            friction=default_friction,
                            condim=3,
                        )
                        hand_collision_pairs.append(
                            (collision_name, another_collision_name)
                        )
                        contact_cnt += 1
            # hand self collision (same-hand, different fingers)
            for another_collision_name in hand_collision_names:
                if (
                    hand_side in another_collision_name
                    and another_collision_name != collision_name
                    and _is_distal_or_middle(another_collision_name)
                    and _get_finger_name(collision_name)
                    != _get_finger_name(another_collision_name)
                    and (collision_name, another_collision_name)
                    not in hand_collision_pairs
                    and (another_collision_name, collision_name)
                    not in hand_collision_pairs
                ):
                    mj_spec.add_pair(
                        name=f"{collision_name}_{another_collision_name}",
                        geomname1=collision_name,
                        geomname2=another_collision_name,
                        solref=default_solref,
                        solimp=default_solimp,
                        friction=default_friction,
                        condim=3,
                    )
                    hand_collision_pairs.append(
                        (collision_name, another_collision_name)
                    )
                    contact_cnt += 1

    loguru.logger.info(f"Added {contact_cnt} contact pairs")

    mj_spec.worldbody.add_camera(
        name="front",
        pos=[0.031, 0.941, 0.844],
        xyaxes=[-0.999, 0.033, -0.000, -0.022, -0.667, 0.745],
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    )

    mj_model = mj_spec.compile()
    mj_data = mujoco.MjData(mj_model)

    # to_xml() re-validates mesh files against mj_spec.meshdir (relative to the
    # loaded robot XML's dir), so keep compile_meshdir set during serialization,
    # then patch the saved <compiler meshdir="..."> to save_meshdir (what
    # downstream consumers, based at the saved XML's location, need).
    xml_file = mj_spec.to_xml()
    # MuJoCo may normalize the meshdir string (e.g. append a trailing slash),
    # so do a regex replacement on the attribute itself rather than expecting
    # ``compile_meshdir`` to appear verbatim.
    xml_file = re.sub(
        r'meshdir="[^"]*"', f'meshdir="{save_meshdir}"', xml_file, count=1
    )
    export_file_path = f"{processed_dir}/scene_ik.xml"
    with open(export_file_path, "w") as f:
        f.write(xml_file)
    loguru.logger.info(f"Saved structural scene to {export_file_path}")

    # save task info (task_info.json lives at the task level — shared across data_ids)
    task_info["robot_type"] = robot_type
    with open(f"{processed_dir}/../task_info.json", "w") as f:
        json.dump(task_info, f, indent=2)

    if show_viewer:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            rate_limiter = RateLimiter(1 / mj_model.opt.timestep)
            while viewer.is_running():
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                rate_limiter.sleep()
