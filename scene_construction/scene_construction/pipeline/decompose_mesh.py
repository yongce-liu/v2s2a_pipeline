from loguru import logger

try:
    import trimesh
except ModuleNotFoundError:
    logger.error("trimesh is required. Please install with `pip install trimesh`")
    raise SystemExit(1)

try:
    import coacd
except ModuleNotFoundError:
    coacd = None

try:
    import vhacdx
except ModuleNotFoundError:
    vhacdx = None


import json
import os
from collections.abc import Iterable
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from scene_construction.io import nfs_safe_lock, resolve_auto_embodiment
from scene_construction.paths import SCENE_CONSTRUCTION_ROOT


class DecompMethod(Enum):
    VHACD = "vhacd"
    COACD = "coacd"


MeshPart = tuple[np.ndarray, np.ndarray]


def coacd_convex_decomp(
    mesh: trimesh.Trimesh,
    threshold: float = 0.05,
    max_convex_hull: int = 40,
    max_ch_vertex: int = 256,
    preprocess_resolution: int = 100,
) -> list[MeshPart]:
    coacd_mesh = coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        max_ch_vertex=max_ch_vertex,
        preprocess_resolution=preprocess_resolution,
        decimate=True,
    )
    hulls: list[MeshPart] = []
    for vertices, faces in parts:
        hulls.append((np.asarray(vertices), np.asarray(faces, dtype=int)))
    return hulls


def vhacd_convex_decomp(
    mesh: trimesh.Trimesh,
    max_convex_hull: int = 32,
    max_ch_vertex: int = 64,
    resolution: int = 400_000,
    fill_mode: str = "surface",
) -> list[MeshPart]:
    """fill_mode "surface" only marks surface voxels (best for thin-walled/hollow
    objects where flood-fill would solidify the cavity); "flood" flood-fills from
    outside (watertight solids); "raycast" uses ray-based inside/outside tests.
    """
    parts = vhacdx.compute_vhacd(
        points=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.uint32),
        maxConvexHulls=max_convex_hull,
        maxNumVerticesPerCH=max_ch_vertex,
        resolution=resolution,
        fillMode=fill_mode,
    )
    hulls: list[MeshPart] = []
    for vertices, faces in parts:
        hulls.append((np.asarray(vertices), np.asarray(faces, dtype=int)))
    return hulls


def fast_voxel_convex_decomp_from_pointcloud(
    points: np.ndarray, pitch: float = 0.1, min_points: int = 20
) -> list[MeshPart]:
    coords = np.floor(points / pitch).astype(int)
    unique_voxels, inverse = np.unique(coords, axis=0, return_inverse=True)

    hulls: list[MeshPart] = []
    for idx, _ in enumerate(unique_voxels):
        cluster_points = points[inverse == idx]
        if len(cluster_points) < min_points:
            continue

        cluster_mesh = trimesh.Trimesh(vertices=cluster_points, faces=[])
        hull = cluster_mesh.convex_hull
        vertices = np.asarray(hull.vertices)
        faces = np.asarray(hull.faces, dtype=int)
        hulls.append((vertices, faces))

    return hulls


def _local_thickness(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex thickness via inward raycasting; returns nearest hit distance."""
    origins = mesh.vertices
    directions = -mesh.vertex_normals
    hits, ray_ids, _ = mesh.ray.intersects_location(origins, directions)
    # closest hit that isn't the origin itself
    thickness = np.full(len(mesh.vertices), np.inf)
    for hit_pos, ray_id in zip(hits, ray_ids):
        dist = np.linalg.norm(hit_pos - origins[ray_id])
        if dist > 1e-6 and dist < thickness[ray_id]:
            thickness[ray_id] = dist
    return thickness


def thicken_mesh(mesh: trimesh.Trimesh, min_thickness: float) -> trimesh.Trimesh:
    """Thicken only the regions thinner than min_thickness, leaving thick parts alone."""
    local_t = _local_thickness(mesh)
    deficit = np.maximum(min_thickness - local_t, 0.0)
    half_offset = (deficit / 2)[:, None]
    normals = mesh.vertex_normals
    outer_verts = mesh.vertices + normals * half_offset
    inner_verts = mesh.vertices - normals * half_offset
    all_verts = np.concatenate([outer_verts, inner_verts], axis=0)
    n_verts = len(mesh.vertices)
    outer_faces = mesh.faces
    inner_faces = mesh.faces[:, ::-1] + n_verts
    all_faces = np.concatenate([outer_faces, inner_faces], axis=0)
    # Outer/inner copies coincide at vertices where deficit==0; leaving them
    # duplicated produces a non-manifold input that crashes CoACD on certain
    # meshes. process=True merges duplicate verts and degenerate faces.
    return trimesh.Trimesh(all_verts, all_faces, process=True)


def dilate_hulls(hulls: Iterable[MeshPart], margin: float = 0) -> list[MeshPart]:
    """Expand each convex hull outward by margin (Minkowski sum with a sphere).

    Offsets each face's vertices along that face's normal rather than the vertex
    normal: vertex-normal dilation under-expands at sharp edges (normals average
    out), which would leave seams between thin-walled pieces unsealed.
    """
    dilated: list[MeshPart] = []
    for vertices, faces in hulls:
        mesh = trimesh.Trimesh(
            np.asarray(vertices), np.asarray(faces, dtype=int), process=False
        )
        face_normals = mesh.face_normals
        new_points = []
        for fi, face in enumerate(mesh.faces):
            fn = face_normals[fi]
            for vi in face:
                new_points.append(mesh.vertices[vi] + fn * margin)
        new_points = np.array(new_points)
        hull = ConvexHull(new_points)
        hull_mesh = trimesh.Trimesh(
            new_points[hull.vertices], process=False
        ).convex_hull
        dilated.append(
            (np.asarray(hull_mesh.vertices), np.asarray(hull_mesh.faces, dtype=int))
        )
    return dilated


def flatten_base(hulls: Iterable[MeshPart], thickness: float = 0.01) -> list[MeshPart]:
    """Append a thin plate that flattens the bottom of the decomposition."""
    hull_list = list(hulls)
    if not hull_list:
        return hull_list

    all_vertices = np.vstack([vertices for vertices, _ in hull_list])
    min_x, max_x = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
    min_y, max_y = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
    min_z = np.min(all_vertices[:, 2])

    z0 = min_z
    z1 = min_z + thickness
    plate_vertices = np.array(
        [
            [min_x, min_y, z0],
            [max_x, min_y, z0],
            [max_x, max_y, z0],
            [min_x, max_y, z0],
            [min_x, min_y, z1],
            [max_x, min_y, z1],
            [max_x, max_y, z1],
            [min_x, max_y, z1],
        ]
    )
    plate_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=int,
    )

    hull_list.append((plate_vertices, plate_faces))
    return hull_list


def main(
    output_root_dir: str = str(SCENE_CONSTRUCTION_ROOT.parent / "outputs"),
    dataset_name: str = "do_as_i_do",
    robot_type: str = "sharpa",
    embodiment_type: str = "bimanual",
    task: str = "",
    data_id: int = 0,
    add_floor: bool = False,
    threshold: float = 0.05,
    max_convex_hull: int = 32,
    max_ch_vertex: int = 256,
    shrink: float = 1.0,
    thicken: float = 0,
    dilate: float = 0,
    convex_hull: bool = False,
    method: DecompMethod = DecompMethod.COACD,
    force: bool = False,
) -> None:
    dataset_path = Path(output_root_dir)

    if embodiment_type == "auto":
        embodiment_type = resolve_auto_embodiment(dataset_name, output_root_dir, task)

    convex_dir = dataset_path / "assets" / "objects" / task / "convex"
    if not force and convex_dir.exists() and any(convex_dir.glob("*.obj")):
        logger.info(f"Skipping decompose_mesh.py (output exists: {convex_dir})")
        return

    if embodiment_type == "right":
        hands = ["right"]
    elif embodiment_type == "left":
        hands = ["left"]
    elif embodiment_type == "bimanual":
        hands = ["right", "left"]
    else:
        raise ValueError(f"Invalid hand type: {embodiment_type}")

    processed_dir = dataset_path / "mano" / embodiment_type / task / str(data_id)
    task_info_path = processed_dir.parent / "task_info.json"

    if not task_info_path.exists():
        logger.error(
            "Missing task_info at {}. Run dataset preprocessing first.",
            task_info_path,
        )
        return

    with task_info_path.open("r", encoding="utf-8") as file:
        task_info = json.load(file)

    for hand in hands:
        mesh_dir_key = (
            "right_object_mesh_dir" if hand == "right" else "left_object_mesh_dir"
        )
        rel_mesh_dir = task_info.get(mesh_dir_key)
        if not rel_mesh_dir:
            logger.warning("No mesh_dir for {} hand; skipping.", hand)
            continue

        mesh_path = Path(f"{dataset_path}/{rel_mesh_dir}")
        input_file = mesh_path / "visual.obj"
        output_dir = mesh_path / "convex"

        if not input_file.exists():
            logger.warning(
                "Input mesh {} does not exist. Skipping {} hand.", input_file, hand
            )
            continue

        # Object-asset dirs are shared across concurrent runs that reference
        # the same mesh. CoACD is not safe to run concurrently on the same
        # input (segfaults on popular meshes), and the wipe+rewrite of
        # output_dir races between runs and corrupts in-flight reads. Serialize
        # per object and skip when the output is already current for the input.
        lock_path = mesh_path / "decompose.lock"
        with nfs_safe_lock(str(lock_path), timeout=600):
            existing = list(output_dir.glob("*.obj"))
            if (
                existing
                and min(p.stat().st_mtime for p in existing)
                > input_file.stat().st_mtime
            ):
                logger.info(
                    "Reusing fresh convex output at {} ({} parts).",
                    output_dir,
                    len(existing),
                )
            else:
                mesh = trimesh.load(
                    str(input_file),
                    force="mesh",
                    process=False,
                    skip_materials=True,
                )
                if shrink != 1.0:
                    center = mesh.centroid
                    mesh.vertices = (mesh.vertices - center) * shrink + center
                    logger.info(
                        "Shrunk mesh by factor {} before decomposition.", shrink
                    )
                if thicken > 0:
                    mesh = thicken_mesh(mesh, thicken)
                    logger.info(
                        "Thickened mesh to {:.1f} mm wall width before decomposition.",
                        thicken * 1000,
                    )
                if convex_hull:
                    logger.info("Using single convex hull (filling concavities).")
                    hull_mesh = mesh.convex_hull
                    hulls = [
                        (
                            np.asarray(hull_mesh.vertices),
                            np.asarray(hull_mesh.faces, dtype=int),
                        )
                    ]
                elif method == DecompMethod.VHACD:
                    if vhacdx is None:
                        raise RuntimeError(
                            "vhacdx is required for V-HACD decomposition. "
                            "Install with `pip install vhacdx`."
                        )
                    logger.info(
                        "Using V-HACD decomposition (max_hulls={}).", max_convex_hull
                    )
                    hulls = vhacd_convex_decomp(
                        mesh,
                        max_convex_hull=max_convex_hull,
                        max_ch_vertex=max_ch_vertex,
                    )
                elif method == DecompMethod.COACD:
                    if coacd is None:
                        raise RuntimeError(
                            "coacd is required for CoACD decomposition. "
                            "Install with `pip install coacd`."
                        )
                    logger.info(
                        "Using CoACD decomposition (threshold={}, max_hulls={}).",
                        threshold,
                        max_convex_hull,
                    )
                    hulls = coacd_convex_decomp(
                        mesh,
                        threshold=threshold,
                        max_convex_hull=max_convex_hull,
                        max_ch_vertex=max_ch_vertex,
                    )

                if not hulls:
                    logger.warning(
                        "No convex parts generated for {}; skipping export.", hand
                    )
                    continue

                if dilate > 0:
                    hulls = dilate_hulls(hulls, margin=dilate)
                    logger.info(
                        "Dilated {} hulls by {:.1f} mm to seal boundary gaps.",
                        len(hulls),
                        dilate * 1000,
                    )
                if add_floor:
                    hulls = flatten_base(hulls)

                output_dir.mkdir(parents=True, exist_ok=True)
                for old_file in existing:
                    old_file.unlink()
                for idx, (vertices, faces) in enumerate(hulls):
                    mesh_part = trimesh.Trimesh(vertices, faces)
                    part_path = output_dir / f"{idx}.obj"
                    mesh_part.export(part_path)

        convex_key = (
            "right_object_convex_dir" if hand == "right" else "left_object_convex_dir"
        )
        relative_path = os.path.relpath(output_dir, dataset_path)
        task_info[convex_key] = str(relative_path)

    with task_info_path.open("w", encoding="utf-8") as file:
        json.dump(task_info, file, indent=2)

    logger.info("Updated task_info with convex dirs at {}", task_info_path)
