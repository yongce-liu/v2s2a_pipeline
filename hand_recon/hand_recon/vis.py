"""Subclasses and overrides that make HaWoR's aitviewer render work headless.

pkgs/HaWoR's ``lib.vis.run_vis2.run_vis2_on_video_cam`` always constructs an
``ARCTICViewer(interactive=True)``, which requires an X display. We never
edit pkgs/HaWoR — instead this module reimplements ``run_vis2_on_video_cam``
verbatim except for the viewer construction call sites, which are switched
to ``interactive=False``.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch


def run_video_cam_headless(
    left_dict: dict,
    right_dict: dict,
    output_pth,
    focal_length: float,
    image_names: Sequence[str],
    R_w2c: torch.Tensor | None = None,
    t_w2c: torch.Tensor | None = None,
) -> None:
    """Run HaWoR's ``run_vis2_on_video_cam`` headlessly.

    Mirrors ``lib.vis.run_vis2.run_vis2_on_video_cam`` (pkgs/HaWoR) except
    for the headless viewer construction；其它步骤直接重用同一个模块的
    geometry helpers，所以上游继续维护时此处不需要再同步。
    """
    import cv2

    import lib.vis.viewer as viewer_utils

    img0 = cv2.imread(image_names[0])
    height, width, _ = img0.shape

    world_mano = {"vertices": left_dict["vertices"], "faces": left_dict["faces"]}
    world_mano2 = {"vertices": right_dict["vertices"], "faces": right_dict["faces"]}

    vis_dict: dict = {}
    for _id, _verts in enumerate(world_mano["vertices"]):
        verts = _verts.cpu().numpy()
        body_faces = world_mano["faces"]
        vis_dict[f"hand_{_id}"] = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand_{_id}",
            "color": "director-purple",
        }

    for _id, _verts in enumerate(world_mano2["vertices"]):
        verts = _verts.cpu().numpy()
        body_faces = world_mano2["faces"]
        vis_dict[f"hand2_{_id}"] = {
            "v3d": verts,
            "f3d": body_faces,
            "vc": None,
            "name": f"hand2_{_id}",
            "color": "director-blue",
        }

    meshes = viewer_utils.construct_viewer_meshes(
        vis_dict, draw_edges=False, flat_shading=False
    )

    num_frames = len(world_mano["vertices"][_id])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :3, :3] = R_w2c[:num_frames]
    Rt[:, :3, 3] = t_w2c[:num_frames]

    cols, rows = (width, height)
    K = np.array(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ]
    )

    data = viewer_utils.ViewerData(Rt, K, cols, rows, imgnames=image_names)
    batch = (meshes, data)

    viewer = viewer_utils.ARCTICViewer(
        interactive=False, size=(height, width), render_types=["video"]
    )
    out_folder = os.path.join(str(output_pth), "aitviewer")
    existing = os.path.join(out_folder, "video_0.mp4")
    if os.path.exists(existing):
        os.remove(existing)
    viewer.render_seq(batch, out_folder=out_folder)


__all__ = ["run_video_cam_headless"]
