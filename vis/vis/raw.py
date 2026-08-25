"""Camera-frame visualization of reconstruction-stage artifacts."""

from __future__ import annotations

import math
import time

import numpy as np
from loguru import logger

from vis.data import load_json, load_raw_frames, subsample_point_map
from vis.overlays import DeformingHands, load_trimesh, rotation_to_wxyz
from vis.paths import RawInputs
from vis.playback import add_timeline


def run_raw_viewer(inputs: RawInputs, port: int, fps: float, max_points: int) -> None:
    import viser
    from PIL import Image

    frames = load_raw_frames(inputs.geometry_json, inputs.poses_json)
    hands = DeformingHands(inputs.meshes_npz) if inputs.meshes_npz else None
    frame_count = max(len(frames), hands.frame_count if hands else 0)
    if frame_count == 0:
        raise FileNotFoundError(
            "Raw mode found no frames. Expected geometry.json, poses.json, or meshes.npz."
        )

    pose_manifest = load_json(inputs.poses_json)
    mesh_scale = float(pose_manifest.get("mesh_scale") or inputs.mesh_scale)
    object_mesh = (
        load_trimesh(inputs.mesh_path, mesh_scale) if inputs.mesh_path else None
    )

    server = viser.ViserServer(port=port, label="v2s2a raw replay")
    server.scene.set_up_direction("-y")
    server.scene.add_frame("/camera/axes", axes_length=0.1, axes_radius=0.004)
    if hands:
        hands.build(server)

    object_handle = None
    if object_mesh is not None:
        object_handle = server.scene.add_mesh_trimesh("/object/mesh", object_mesh)

    layer_handles = {}
    with server.gui.add_folder("Layers"):
        if hands:
            layer_handles["hands"] = server.gui.add_checkbox(
                "Hands", initial_value=True
            )
        if object_handle:
            layer_handles["object"] = server.gui.add_checkbox(
                "Tracked object", initial_value=True
            )
        if inputs.geometry_json:
            layer_handles["points"] = server.gui.add_checkbox(
                "Point cloud", initial_value=True
            )
        layer_handles["camera"] = server.gui.add_checkbox(
            "Camera image", initial_value=True
        )

    def show_frame(display_index: int) -> None:
        frame = frames[min(display_index, len(frames) - 1)] if frames else None
        with server.atomic():
            if hands:
                hands.show_frame(display_index)
            if frame and frame.pose_path and frame.pose_path.exists() and object_handle:
                pose = np.loadtxt(frame.pose_path, dtype=np.float64).reshape(4, 4)
                object_handle.position = tuple(pose[:3, 3])
                object_handle.wxyz = rotation_to_wxyz(pose[:3, :3])
                object_handle.visible = layer_handles["object"].value
                server.scene.add_frame(
                    "/object/axes",
                    position=tuple(pose[:3, 3]),
                    wxyz=rotation_to_wxyz(pose[:3, :3]),
                    axes_length=0.08,
                    axes_radius=0.003,
                )
            elif object_handle:
                object_handle.visible = False

            if frame and frame.points_path and frame.points_path.exists():
                image = None
                if frame.image_path and frame.image_path.exists():
                    image = np.asarray(Image.open(frame.image_path).convert("RGB"))
                points, colors = subsample_point_map(
                    np.load(frame.points_path), image, max_points
                )
                point_handle = server.scene.add_point_cloud(
                    "/geometry/points", points=points, colors=colors, point_size=0.002
                )
                point_handle.visible = layer_handles["points"].value

            if (
                frame
                and frame.image_path
                and frame.image_path.exists()
                and frame.intrinsics_path
                and frame.intrinsics_path.exists()
            ):
                image = np.asarray(Image.open(frame.image_path).convert("RGB"))
                intrinsics = np.load(frame.intrinsics_path)
                height, width = image.shape[:2]
                fov = 2.0 * math.atan(height / (2.0 * float(intrinsics[1, 1])))
                camera = server.scene.add_camera_frustum(
                    "/camera/frustum",
                    fov=fov,
                    aspect=width / height,
                    scale=0.2,
                    image=image,
                    color=(30, 30, 30),
                )
                camera.visible = layer_handles["camera"].value

    slider = add_timeline(server, frame_count, fps, show_frame)
    if hands:

        @layer_handles["hands"].on_update
        def _(_) -> None:
            hands.set_visible(layer_handles["hands"].value, int(slider.value))

    if object_handle:

        @layer_handles["object"].on_update
        def _(_) -> None:
            show_frame(int(slider.value))

    for name in ("points", "camera"):
        if name in layer_handles:
            layer_handles[name].on_update(lambda _: show_frame(int(slider.value)))

    show_frame(0)
    logger.info("[vis] Raw viewer: http://localhost:{} ({} frames)", port, frame_count)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("[vis] Viewer stopped")
