"""MuJoCo trajectory replay with IK and MANO references."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
from loguru import logger

from vis.data import load_qpos
from vis.overlays import SceneReference, load_trimesh
from vis.playback import add_timeline


def run_scene_viewer(
    *,
    scene_path: Path,
    trajectory_path: Path,
    ik_path: Path | None,
    mano_path: Path | None,
    object_mesh_path: Path | None,
    config: dict,
    port: int,
    fps: float,
    skip_warmup: bool,
) -> None:
    from physics_opt.utils import viser_viewer

    viser_viewer.init_viser(app_name="v2s2a scene replay", port=port)
    server = viser_viewer._get_server()
    spec, model, body_handles = viser_viewer.build_and_log_scene(
        scene_path, build_ref=ik_path is not None, build_gui=False
    )
    del spec
    data = mujoco.MjData(model)
    qpos = load_qpos(trajectory_path, model.nq)

    ik_qpos = None
    data_ref = None
    if ik_path is not None:
        try:
            ik_qpos = load_qpos(ik_path, model.nq)
            data_ref = mujoco.MjData(model)
        except ValueError as error:
            logger.warning("[vis] IK layer disabled: {}", error)

    object_mesh = load_trimesh(object_mesh_path) if object_mesh_path else None
    mano = (
        SceneReference(mano_path, object_mesh)
        if mano_path is not None and mano_path.exists()
        else None
    )
    if mano is not None:
        mano.build(server)

    warmup_steps = max(0, int(config.get("warmup_steps", 0) or 0))
    sim_dt = float(config.get("sim_dt", 0.005) or 0.005)
    ref_dt = float(config.get("ref_dt", 0.0333) or 0.0333)
    ref_steps = max(1, round(ref_dt / sim_dt))
    start = min(warmup_steps if skip_warmup else 0, len(qpos) - 1)
    frame_count = len(qpos) - start

    def reference_frame(display_index: int) -> int:
        return max(0, display_index + start - warmup_steps) // ref_steps

    with server.gui.add_folder("Layers"):
        robot_visible = server.gui.add_checkbox("Retargeted robot", initial_value=True)
        ik_visible = (
            server.gui.add_checkbox("IK reference (blue)", initial_value=True)
            if ik_qpos is not None
            else None
        )
        mano_visible = (
            server.gui.add_checkbox("MANO reference (orange)", initial_value=True)
            if mano is not None
            else None
        )

    def show_frame(display_index: int) -> None:
        source_index = min(max(display_index + start, 0), len(qpos) - 1)
        data.qpos[:] = qpos[source_index]
        mujoco.mj_kinematics(model, data)
        ref_index = reference_frame(display_index)
        if ik_qpos is not None and data_ref is not None:
            data_ref.qpos[:] = ik_qpos[min(ref_index, len(ik_qpos) - 1)]
            mujoco.mj_kinematics(model, data_ref)
        viser_viewer.log_frame(
            data,
            sim_time=source_index * sim_dt,
            viewer_body_entity_and_ids=body_handles,
            data_ref=data_ref,
            record=False,
        )
        if mano is not None:
            mano.show_frame(ref_index)

    slider = add_timeline(server, frame_count, fps, show_frame)

    @robot_visible.on_update
    def _(_) -> None:
        for handle, body_id in body_handles:
            if body_id != 0:
                handle.visible = robot_visible.value

    if ik_visible is not None:

        @ik_visible.on_update
        def _(_) -> None:
            for handle, _body_id in viser_viewer._STATE.ref_body_handles:
                handle.visible = ik_visible.value
            for handle in viser_viewer._STATE.ref_geom_handles:
                handle.visible = ik_visible.value

    if mano_visible is not None:

        @mano_visible.on_update
        def _(_) -> None:
            mano.set_visible(mano_visible.value, reference_frame(int(slider.value)))

    show_frame(0)
    logger.info(
        "[vis] Scene viewer: http://localhost:{} ({} frames, skipped {} warmup)",
        port,
        frame_count,
        start,
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("[vis] Viewer stopped")
