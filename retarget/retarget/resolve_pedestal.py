"""Post-IK pedestal placement: ``scene_ik.xml`` + IK output → ``scene.xml``.

generate_scene.py writes a pedestal-free structural scene; this runs the in-hand
endpoint check to decide which endpoints need a stabilizing pedestal, then
injects pedestal + welded-support geoms (and collision pairs) using the actual
IK-output object pose from ``trajectory_kinematic.npz``. Produces scene.xml plus
the scene_eq.xml / scene_act.xml variants. Runs after IK (which is purely
kinematic mink and needs only the structural scene).
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import loguru
import mujoco
import numpy as np
import trimesh
from scene_construction.in_hand import DEFAULT_DISTANCE_THRESH, in_hand_at_endpoint
from scene_construction.io import get_processed_data_dir, resolve_auto_embodiment
from scipy.spatial.transform import Rotation

from retarget.paths import RETARGET_ROOT as ROOT

# Contact-pair parameters. Kept in sync with the same constants in
# ``generate_scene.py`` (which still emits the floor pairs and the
# hand↔object / object↔object pairs). Pedestal/support pairs added here use
# the *ground* params (default_solimp, default_friction) — never the stiffer
# hand_object_solimp.
DEFAULT_SOLREF = [0.01, 1]
DEFAULT_SOLIMP = [0.9, 0.95, 0.001, 0.5, 2]

# Pedestal geometry. Same values as the originals in ``generate_scene.py``.
PEDESTAL_HALF_H = 0.005
SUPPORT_HALF_H = 0.002
# Drop the pedestal a hair below the object's bottom vertex so warmup —
# which holds the object in its reference pose with a weld + zero gravity —
# never sits in a tangential/penetrating contact with the pedestal.
PEDESTAL_OBJ_GAP = 0.0001


def _default_friction(robot_type: str, friction_scale: float) -> list[float]:
    """Match ``generate_scene.py``'s default_friction selection for ground pairs."""
    if robot_type == "mano":
        base = [2.0, 2.0, 0.3, 0.01, 0.01]
    else:
        base = [1.0, 1.0, 0.1, 0.0, 0.0]
    return [
        base[0] * friction_scale,
        base[1] * friction_scale,
        base[2] * friction_scale,
        base[3] * friction_scale,
        base[4] * friction_scale,
    ]


def _load_object_vertices(
    obj_files: list[str],
    convex_dir: str | None,
    visual_file: str | None,
    use_visual_mesh_as_collision: bool,
) -> np.ndarray | None:
    all_verts: list[np.ndarray] = []
    for f in obj_files:
        suffix = f.split(".")[0]
        if use_visual_mesh_as_collision and suffix == "visual":
            if visual_file and os.path.exists(visual_file):
                mesh = trimesh.load(visual_file, force="mesh")
                all_verts.append(np.asarray(mesh.vertices))
        elif convex_dir and suffix.isdigit():
            path = os.path.join(convex_dir, f)
            if os.path.exists(path):
                mesh = trimesh.load(path, force="mesh")
                all_verts.append(np.asarray(mesh.vertices))
    if not all_verts:
        return None
    return np.concatenate(all_verts, axis=0)


def _load_object_com_local(
    obj_files: list[str],
    convex_dir: str | None,
    visual_file: str | None,
    use_visual_mesh_as_collision: bool,
) -> np.ndarray | None:
    """Volume-weighted COM in mesh-local frame (matches MuJoCo's ``xipos``)."""
    coms: list[np.ndarray] = []
    vols: list[float] = []
    for f in obj_files:
        suffix = f.split(".")[0]
        path: str | None = None
        if use_visual_mesh_as_collision and suffix == "visual":
            if visual_file and os.path.exists(visual_file):
                path = visual_file
        elif convex_dir and suffix.isdigit():
            cand = os.path.join(convex_dir, f)
            if os.path.exists(cand):
                path = cand
        if path is None:
            continue
        mesh = trimesh.load(path, force="mesh")
        vol = float(mesh.volume)
        if vol <= 0.0:
            continue
        coms.append(np.asarray(mesh.center_mass, dtype=float))
        vols.append(vol)
    if not coms:
        return None
    coms_arr = np.stack(coms, axis=0)
    vols_arr = np.asarray(vols, dtype=float)
    return (coms_arr * vols_arr[:, None]).sum(axis=0) / vols_arr.sum()


def _object_bottom_footprint(
    verts: np.ndarray,
    qpos: np.ndarray,
    com_local: np.ndarray | None = None,
    slice_thickness: float = 0.01,
    margin: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """Footprint (center_xy, radius, bottom_z) of the object's resting region.

    Upright (xy bbox center inside the bottom-slice circle): hug the bottom-slice
    contact patch. Slanted: center under the COM proxy, sized to cover every
    vertex's xy projection.
    """
    pos = qpos[:3]
    quat_xyzw = qpos[[4, 5, 6, 3]]
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    world_verts = verts @ R.T + pos
    z_world = world_verts[:, 2]
    bottom_z = float(z_world.min())
    mask = z_world <= bottom_z + slice_thickness
    bottom_xy = world_verts[mask, :2]
    xy_min = bottom_xy.min(axis=0)
    xy_max = bottom_xy.max(axis=0)
    bottom_center = (xy_min + xy_max) / 2.0
    bottom_radius = float(np.linalg.norm(bottom_xy - bottom_center, axis=1).max())
    all_xy = world_verts[:, :2]
    if com_local is not None:
        com_xy = (R @ com_local + pos)[:2]
    else:
        com_xy = (all_xy.min(axis=0) + all_xy.max(axis=0)) / 2.0
    com_offset = float(np.linalg.norm(com_xy - bottom_center))
    if com_offset > bottom_radius:
        center_xy = com_xy
        radius = float(np.linalg.norm(all_xy - com_xy, axis=1).max())
    else:
        center_xy = bottom_center
        radius = bottom_radius
    return center_xy, radius * (1.0 + margin), bottom_z


def _resolve_mesh_dirs(task_info: dict, output_root_dir: str) -> dict[str, dict]:
    """Resolve absolute object mesh/convex dirs per side from ``task_info.json``."""
    out: dict[str, dict] = {}
    for side in ("right", "left"):
        convex_dir = task_info.get(f"{side}_object_convex_dir")
        if convex_dir and not os.path.isabs(convex_dir):
            convex_dir = f"{output_root_dir}/{convex_dir}"
        mesh_dir = task_info.get(f"{side}_object_mesh_dir")
        if mesh_dir and not os.path.isabs(mesh_dir):
            mesh_dir = f"{output_root_dir}/{mesh_dir}"
        # Fallback: convex/ subdir inside the mesh dir.
        if (not convex_dir or not os.path.isdir(convex_dir)) and mesh_dir:
            cand = f"{mesh_dir}/convex"
            if os.path.isdir(cand):
                convex_dir = cand
        visual_file = f"{mesh_dir}/visual.obj" if mesh_dir else None
        out[side] = {
            "convex_dir": convex_dir,
            "mesh_dir": mesh_dir,
            "visual_file": visual_file,
        }
    return out


def _list_object_files(
    side_info: dict, use_visual_mesh_as_collision: bool
) -> list[str]:
    # Same selection as generate_scene.py: convex parts, or the visual mesh
    # in visual-as-collision mode.
    convex_dir = side_info["convex_dir"]
    visual_file = side_info["visual_file"]
    if use_visual_mesh_as_collision and visual_file and os.path.exists(visual_file):
        return ["visual"]
    if convex_dir and os.path.isdir(convex_dir):
        return sorted(f for f in os.listdir(convex_dir) if f.endswith(".obj"))
    return []


def _object_qpos_endpoints(
    qpos: np.ndarray, embodiment_type: str
) -> dict[str, dict[int, np.ndarray]]:
    """Endpoint (frame 0 and -1) object 7-DOF poses, ``{side: {0: q, -1: q}}``.

    The object free joint lives in the last 7 (single-side) or 14 (bimanual)
    qpos columns.
    """
    out: dict[str, dict[int, np.ndarray]] = {}
    if embodiment_type == "bimanual":
        right_slice = slice(-14, -7)
        left_slice = slice(-7, None)
        out["right"] = {
            0: qpos[0, right_slice].copy(),
            -1: qpos[-1, right_slice].copy(),
        }
        out["left"] = {0: qpos[0, left_slice].copy(), -1: qpos[-1, left_slice].copy()}
    elif embodiment_type == "right":
        out["right"] = {0: qpos[0, -7:].copy(), -1: qpos[-1, -7:].copy()}
    elif embodiment_type == "left":
        out["left"] = {0: qpos[0, -7:].copy(), -1: qpos[-1, -7:].copy()}
    return out


def _add_object_xyzrpy_actuators(
    xml_text: str,
    object_armature: float,
    object_frictionloss: float,
    object_pos_kp: float,
    object_pos_kd: float,
    object_rot_kp: float,
    object_rot_kd: float,
) -> str:
    """Replace each ``{side}_object`` free joint with six slide/hinge joints
    plus position actuators (pure text transform → scene_act.xml variant)."""

    def _fmt(v: float) -> str:
        return f"{v:.6g}"

    root = ET.fromstring(xml_text)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml_text

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    existing_actuators = {
        elem.get("name")
        for elem in actuator.findall("*")
        if elem.get("name") is not None
    }

    joint_defs = [
        ("pos_x", "slide", "1 0 0", "pos"),
        ("pos_y", "slide", "0 1 0", "pos"),
        ("pos_z", "slide", "0 0 1", "pos"),
        ("rot_x", "hinge", "1 0 0", "rot"),
        ("rot_y", "hinge", "0 1 0", "rot"),
        ("rot_z", "hinge", "0 0 1", "rot"),
    ]

    for side in ("right", "left"):
        body = worldbody.find(f".//body[@name='{side}_object']")
        if body is None:
            continue
        free_joint_name = f"{side}_object_joint"
        free_joint = None
        for joint in body.findall("joint"):
            if joint.get("name") == free_joint_name:
                free_joint = joint
                break
        if free_joint is None:
            continue

        body_children = list(body)
        insert_index = body_children.index(free_joint)
        free_joint_damping = float(free_joint.get("damping", "0"))
        body.remove(free_joint)

        for offset, (suffix, joint_type, axis, group) in enumerate(joint_defs):
            joint_name = f"{side}_object_{suffix}"
            joint_attrs = {
                "name": joint_name,
                "type": joint_type,
                "axis": axis,
                "armature": _fmt(object_armature),
                "damping": _fmt(free_joint_damping),
                "frictionloss": _fmt(object_frictionloss),
            }
            body.insert(insert_index + offset, ET.Element("joint", joint_attrs))

            actuator_name = joint_name
            if actuator_name not in existing_actuators:
                kp = object_pos_kp if group == "pos" else object_rot_kp
                kd = object_pos_kd if group == "pos" else object_rot_kd
                actuator_attrs = {
                    "name": actuator_name,
                    "joint": joint_name,
                    "kp": _fmt(kp),
                    "kv": _fmt(kd),
                }
                actuator.append(ET.Element("position", actuator_attrs))
                existing_actuators.add(actuator_name)

    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    return ET.tostring(root, encoding="unicode")


def _inject_pedestals_supports(
    spec: mujoco.MjSpec,
    placements: list[dict],
    object_collision_names_by_side: dict[str, list[str]],
    robot_type: str,
    friction_scale: float,
    use_support: bool,
) -> None:
    """Add pedestal worldbody geoms, welded support geoms, and the matching
    collision pairs (object↔pedestal, hand↔pedestal, support↔pedestal)."""
    if not placements:
        return

    friction = _default_friction(robot_type, friction_scale)
    hand_collision_names = [
        g.name for g in spec.geoms if g.name.startswith("collision_hand_")
    ]

    pedestal_geoms: list[tuple[str, str]] = []  # (pedestal_name, side)
    support_specs: list[dict] = []
    for p in placements:
        side = p["side"]
        ep_name = p["ep_name"]
        radius = p["radius"]
        center_xy = p["center_xy"]
        top_z = p["top_z"]
        qpos_frame = p["qpos_frame"]

        material = "right_groundplane" if side == "right" else "left_groundplane"
        pedestal_name = f"{side}_pedestal_{ep_name}"
        spec.worldbody.add_geom(
            name=pedestal_name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, PEDESTAL_HALF_H, 0],
            pos=[
                float(center_xy[0]),
                float(center_xy[1]),
                top_z - PEDESTAL_HALF_H - PEDESTAL_OBJ_GAP,
            ],
            material=material,
        )
        pedestal_geoms.append((pedestal_name, side))
        loguru.logger.info(
            f"{side} {ep_name} pedestal: radius={radius:.4f} m, "
            f"pos=({center_xy[0]:.4f}, {center_xy[1]:.4f}, top_z={top_z:.4f})"
        )

        if not use_support:
            continue
        # Support: a thin horizontal cylinder welded to the object body so a
        # slanted object does not tip over. Placed in the object body's
        # local frame using the IK-output qpos (same pose the simulator
        # initializes from).
        R_body = Rotation.from_quat(qpos_frame[[4, 5, 6, 3]]).as_matrix()
        world_support_pos = np.array(
            [float(center_xy[0]), float(center_xy[1]), top_z + SUPPORT_HALF_H]
        )
        support_local_pos = (R_body.T @ (world_support_pos - qpos_frame[:3])).tolist()
        qw, qx, qy, qz = qpos_frame[3:7]
        support_local_quat = [float(qw), float(-qx), float(-qy), float(-qz)]
        support_name = f"{side}_support_{ep_name}"
        obj_body = spec.body(f"{side}_object")
        obj_body.add_geom(
            name=support_name,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, SUPPORT_HALF_H, 0],
            pos=support_local_pos,
            quat=support_local_quat,
            contype=0,
            conaffinity=0,
            density=0,
            group=3,
            rgba=[0.4, 0.4, 0.8, 0.4],
        )
        support_specs.append(
            {
                "support_name": support_name,
                "pedestal_name": pedestal_name,
                "side": side,
            }
        )

    # Collision pairs.
    # 1. object↔pedestal: every object collision geom × every pedestal. condim=6,
    #    DEFAULT_SOLIMP (ground pair, not hand_object_solimp).
    for pedestal_name, _side in pedestal_geoms:
        for side, names in object_collision_names_by_side.items():
            for ocn in names:
                spec.add_pair(
                    name=f"{ocn}_{pedestal_name}",
                    geomname1=ocn,
                    geomname2=pedestal_name,
                    solref=DEFAULT_SOLREF,
                    solimp=DEFAULT_SOLIMP,
                    friction=friction,
                    condim=6,
                )
    # 2. hand↔pedestal: every hand collision × every pedestal. condim=3,
    #    DEFAULT_SOLIMP.
    for pedestal_name, _side in pedestal_geoms:
        for hcn in hand_collision_names:
            spec.add_pair(
                name=f"{hcn}_{pedestal_name}",
                geomname1=hcn,
                geomname2=pedestal_name,
                solref=DEFAULT_SOLREF,
                solimp=DEFAULT_SOLIMP,
                friction=friction,
                condim=3,
            )
    # 3. support↔pedestal (one pair per support, matched to its own pedestal).
    for sp in support_specs:
        spec.add_pair(
            name=f"{sp['support_name']}_{sp['pedestal_name']}",
            geomname1=sp["support_name"],
            geomname2=sp["pedestal_name"],
            solref=DEFAULT_SOLREF,
            solimp=DEFAULT_SOLIMP,
            friction=friction,
            condim=3,
        )


def _add_track_ref_equality(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    """Add connect equality constraints between ``*track*`` and ``*ref*`` sites."""
    for sid in range(model.nsite):
        site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if "track" in site_name:
            ref_site_name = site_name.replace("track", "ref")
            e = spec.add_equality(
                name=f"{site_name}_equality_constraint",
                type=mujoco.mjtEq.mjEQ_CONNECT,
                name1=site_name,
                name2=ref_site_name,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                data=np.zeros(11),
            )
            e.solref = [0.02, 1.0]
            e.solimp = [0.0, 1.0, 100.0, 0.5, 2.0]


def resolve_scene_pedestal(
    output_root_dir: str,
    dataset_name: str,
    robot_type: str,
    embodiment_type: str,
    task: str,
    data_id: int,
    use_pedestal: bool = True,
    use_support: bool = True,
    use_visual_mesh_as_collision: bool = False,
    object_armature: float = 1e-4,
    object_frictionloss: float = 1e-4,
    friction_scale: float = 1.5,
    hand_object_distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    act_scene: bool = False,
    force: bool = False,
) -> None:
    """Resolve ``scene_ik.xml`` → ``scene.xml`` (+ eq/act variants).

    Runs the in-hand endpoint check against the raw keypoint reference and places
    pedestals using the IK-output object pose. With ``use_pedestal=False`` (or no
    endpoint needing stabilization) the outputs are copies of the structural input.
    """
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
    out_path = (
        f"{processed_dir}/scene_act.xml" if act_scene else f"{processed_dir}/scene.xml"
    )
    if not force and os.path.exists(out_path):
        loguru.logger.info(
            f"Skipping resolve_scene_pedestal (output exists: {out_path})"
        )
        return

    scene_ik_path = f"{processed_dir}/scene_ik.xml"
    if not os.path.exists(scene_ik_path):
        raise FileNotFoundError(
            f"Structural scene not found: {scene_ik_path}. Did generate_scene run?"
        )

    keypoint_dir = get_processed_data_dir(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type="mano",
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
    )
    task_info_path = f"{keypoint_dir}/../task_info.json"
    with open(task_info_path) as f:
        task_info = json.load(f)
    side_dirs = _resolve_mesh_dirs(task_info, output_root_dir)

    with np.load(f"{keypoint_dir}/trajectory_keypoints.npz") as _npz:
        keypoints = {k: _npz[k] for k in _npz.files}
    with np.load(f"{processed_dir}/trajectory_kinematic.npz") as _npz:
        ik_qpos = _npz["qpos"]  # (T, nq)

    ik_endpoints = _object_qpos_endpoints(ik_qpos, embodiment_type)

    placements: list[dict] = []
    object_collision_names_by_side: dict[str, list[str]] = {}
    if use_pedestal:
        for side in ("right", "left"):
            if embodiment_type not in (side, "bimanual"):
                continue
            if f"qpos_obj_{side}" not in keypoints:
                continue
            obj_files = _list_object_files(
                side_dirs[side], use_visual_mesh_as_collision
            )
            if not obj_files:
                continue
            mano_verts_key = f"mano_verts_{side}"
            if mano_verts_key not in keypoints:
                continue
            hand_verts = keypoints[mano_verts_key]
            if hand_verts.shape[0] == 0:
                # The processor emits an empty (0,0,3) array on the unprocessed side.
                continue
            obj_verts = _load_object_vertices(
                obj_files,
                side_dirs[side]["convex_dir"],
                side_dirs[side]["visual_file"],
                use_visual_mesh_as_collision,
            )
            obj_com = _load_object_com_local(
                obj_files,
                side_dirs[side]["convex_dir"],
                side_dirs[side]["visual_file"],
                use_visual_mesh_as_collision,
            )
            ref_qpos_obj = keypoints[f"qpos_obj_{side}"]
            placed_for_side: list[tuple[np.ndarray, float, float]] = []
            for ep_name, frame_idx in (("start", 0), ("end", -1)):
                in_hand, min_dist, _hand_idx, *_ = in_hand_at_endpoint(
                    hand_verts_world=hand_verts,
                    qpos_obj=ref_qpos_obj,
                    obj_verts=obj_verts,
                    frame=frame_idx,
                    hand_object_distance_thresh=hand_object_distance_thresh,
                )
                pass_fail = "PASS" if in_hand else "FAIL"
                cmp_op = "<" if in_hand else ">="
                loguru.logger.info(
                    f"{side} object ({ep_name}): in_hand={in_hand} "
                    f"(min hand-vert→object dist={min_dist:.4f}m {cmp_op} "
                    f"{hand_object_distance_thresh:.4f}m [{pass_fail}]) → "
                    + (
                        "stabilize (add pedestal)"
                        if not in_hand
                        else "skip (no pedestal)"
                    )
                )
                if in_hand:
                    continue
                # Place using actual IK-output pose at this endpoint.
                qpos_frame = ik_endpoints[side][frame_idx]
                center_xy, radius, top_z = _object_bottom_footprint(
                    obj_verts,
                    qpos_frame,
                    com_local=obj_com,
                )
                # De-duplicate near-overlapping pedestals on the same side.
                if any(
                    float(np.linalg.norm(center_xy - prev_xy)) < min(radius, prev_r)
                    and abs(top_z - prev_z) < 2 * PEDESTAL_HALF_H
                    for prev_xy, prev_z, prev_r in placed_for_side
                ):
                    loguru.logger.info(
                        f"{side} {ep_name}: skipping (overlaps existing)"
                    )
                    continue
                placed_for_side.append(
                    (np.asarray(center_xy, dtype=float), top_z, radius)
                )
                placements.append(
                    {
                        "side": side,
                        "ep_name": ep_name,
                        "center_xy": np.asarray(center_xy, dtype=float),
                        "radius": float(radius),
                        "top_z": float(top_z),
                        "qpos_frame": np.asarray(qpos_frame, dtype=float),
                    }
                )

    spec = mujoco.MjSpec.from_file(scene_ik_path)
    if placements:
        # generate_scene emits object collision geoms as "{side}_object_{N}" or
        # "{side}_object_collision_visual", all group=3.
        for side in ("right", "left"):
            names: list[str] = []
            for g in spec.geoms:
                if not g.name.startswith(f"{side}_object_"):
                    continue
                if getattr(g, "group", 0) != 3:
                    continue
                names.append(g.name)
            object_collision_names_by_side[side] = names
        _inject_pedestals_supports(
            spec,
            placements,
            object_collision_names_by_side,
            robot_type=robot_type,
            friction_scale=friction_scale,
            use_support=use_support,
        )

    # Compile to validate before writing.
    model = spec.compile()
    xml_text = spec.to_xml()
    if act_scene:
        xml_act = _add_object_xyzrpy_actuators(
            xml_text,
            object_armature=object_armature,
            object_frictionloss=object_frictionloss,
            object_pos_kp=0,
            object_pos_kd=0,
            object_rot_kp=0,
            object_rot_kd=0,
        )
        with open(out_path, "w") as f:
            f.write(xml_act)
        loguru.logger.info(f"Wrote {out_path}")
    else:
        with open(out_path, "w") as f:
            f.write(xml_text)
        loguru.logger.info(f"Wrote {out_path}")
        # Eq variant: layer site connect constraints on top of the resolved spec.
        _add_track_ref_equality(spec, model)
        spec.compile()
        xml_eq = spec.to_xml()
        eq_path = f"{processed_dir}/scene_eq.xml"
        with open(eq_path, "w") as f:
            f.write(xml_eq)
        loguru.logger.info(f"Wrote {eq_path}")


def main(
    output_root_dir: str = f"{ROOT}/../outputs",
    dataset_name: str = "do_as_i_do",
    robot_type: str = "sharpa",
    embodiment_type: str = "auto",
    task: str = "",
    data_id: int = 0,
    use_pedestal: bool = True,
    use_support: bool = True,
    use_visual_mesh_as_collision: bool = False,
    object_armature: float = 1e-4,
    object_frictionloss: float = 1e-4,
    friction_scale: float = 1.5,
    hand_object_distance_thresh: float = DEFAULT_DISTANCE_THRESH,
    act_scene: bool = False,
    force: bool = False,
):
    resolve_scene_pedestal(
        output_root_dir=output_root_dir,
        dataset_name=dataset_name,
        robot_type=robot_type,
        embodiment_type=embodiment_type,
        task=task,
        data_id=data_id,
        use_pedestal=use_pedestal,
        use_support=use_support,
        use_visual_mesh_as_collision=use_visual_mesh_as_collision,
        object_armature=object_armature,
        object_frictionloss=object_frictionloss,
        friction_scale=friction_scale,
        hand_object_distance_thresh=hand_object_distance_thresh,
        act_scene=act_scene,
        force=force,
    )
