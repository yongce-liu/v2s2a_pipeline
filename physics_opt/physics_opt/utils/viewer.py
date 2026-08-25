"""Define the viewers for the retargeting."""

from contextlib import contextmanager
from pathlib import Path

import cv2
import loguru
import mujoco
import mujoco.viewer
import numpy as np

from physics_opt.config import Config
from physics_opt.utils import viser_viewer as viser_viewer


def setup_viewer(config: Config, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
    viewer_str = (config.viewer or "").lower()
    if not config.show_viewer:
        viewer_str = ""
    use_viser = "viser" in viewer_str

    if use_viser:
        viser_viewer.init_viser(app_name="retargeting")
        if mj_model is not None and config.model_path is not None:
            _, _, config.viewer_body_entity_and_ids = viser_viewer.build_and_log_scene(
                Path(config.model_path)
            )
            loguru.logger.info(
                "viewer is set to viser, build and log scene from xml file"
            )
        else:
            loguru.logger.warning(
                "Viser enabled but 3D scene not available (no model_path). Trajectory logging only."
            )
    if "mujoco" in viewer_str:

        def run_viewer():
            return mujoco.viewer.launch_passive(mj_model, mj_data)

        loguru.logger.info("viewer is set to mujoco, launch passive viewer")
    else:

        @contextmanager
        def run_viewer():
            yield type(
                "DummyViewer",
                (),
                {"is_running": lambda: True, "sync": lambda: None, "user_scn": None},
            )

        loguru.logger.info("viewer is disabled, launch dummy viewer")

    return run_viewer


def update_viewer(
    config: Config,
    viewer: mujoco.viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mj_data_ref: mujoco.MjData,
    info: dict,
):
    viewer_str = (config.viewer or "").lower()
    if not config.show_viewer:
        viewer_str = ""
    use_viser = "viser" in viewer_str

    if "mujoco" in viewer_str:
        mujoco.mj_kinematics(mj_model, mj_data)
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC
        mujoco.mj_forward(mj_model, mj_data_ref)
        mujoco.mjv_updateScene(
            mj_model,
            mj_data_ref,
            vopt,
            pert,
            viewer.cam,
            catmask,
            getattr(viewer, "user_scn", None),
        )
        if hasattr(viewer, "sync"):
            viewer.sync()

    if use_viser:
        if "trace_sample" in info:
            viser_viewer.log_traces_from_info(
                info["trace_sample"],
                trace_ref=info.get("trace_ref"),
                sim_time=mj_data.time,
                num_iters=int(info["opt_steps"][0]) if "opt_steps" in info else None,
                num_object_trace_sites=config.num_object_trace_sites,
            )
        if "sim_step" in info:
            viser_viewer.update_sim_progress(info["sim_step"], config.max_sim_steps)


REF_COLOR_BLUE = viser_viewer.REF_COLOR_BLUE
REF_COLOR_RED = viser_viewer.REF_COLOR_RED


def log_frame(
    data: mujoco.MjData,
    sim_time: float,
    viewer_body_entity_and_ids: list,
    data_ref: mujoco.MjData | None = None,
    ref_color: np.ndarray | None = None,
    record: bool = True,
    playback_fps: float = 50.0,
) -> None:
    if not viewer_body_entity_and_ids:
        return
    viser_viewer.log_frame(
        data,
        sim_time=sim_time,
        viewer_body_entity_and_ids=viewer_body_entity_and_ids,
        data_ref=data_ref,
        ref_color=ref_color,
        record=record,
        playback_fps=playback_fps,
    )


def setup_renderer(config: Config, mj_model: mujoco.MjModel):
    mj_model.vis.global_.offwidth = 720
    mj_model.vis.global_.offheight = 480
    renderer = (
        mujoco.Renderer(mj_model, height=480, width=720) if config.save_video else None
    )
    return renderer


def render_image(
    config: Config,
    renderer: mujoco.Renderer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    mj_data_ref: mujoco.MjData,
):
    options = mujoco.MjvOption()
    mujoco.mjv_defaultOption(options)
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # render sim
    mujoco.mj_forward(mj_model, mj_data)
    try:
        renderer.update_scene(mj_data, "front", options)
    except Exception:
        renderer.update_scene(mj_data, 0, options)
    sim_image = renderer.render()
    cv2.putText(
        sim_image,
        "sim",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (128, 128, 128),
        2,
    )
    mujoco.mj_forward(mj_model, mj_data_ref)
    try:
        renderer.update_scene(mj_data_ref, "front")
    except Exception:
        renderer.update_scene(mj_data_ref, 0)
    ref_image = renderer.render()
    cv2.putText(
        ref_image,
        "ref",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (128, 128, 128),
        2,
    )
    image = np.concatenate([ref_image, sim_image], axis=1)
    return image
