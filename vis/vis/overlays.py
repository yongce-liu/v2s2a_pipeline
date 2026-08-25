"""Reusable hand and object overlays for viser scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

HAND_COLORS = {"left": (204, 128, 128), "right": (128, 128, 204)}
REFERENCE_COLOR = (255, 150, 40)


def rotation_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= np.linalg.norm(quat)
    return tuple(float(value) for value in quat)


class DeformingHands:
    def __init__(
        self, npz_path: Path, *, prefix: str = "/hands", reference: bool = False
    ):
        self.prefix = prefix
        self.color = REFERENCE_COLOR if reference else None
        self.opacity = 0.55 if reference else 1.0
        self.hands: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
        with np.load(npz_path, allow_pickle=False) as archive:
            key_prefix = "mano_verts" if reference else "vertices"
            for side in ("left", "right"):
                vertices_key = (
                    f"{key_prefix}_{side}" if reference else f"{side}_vertices"
                )
                faces_key = f"mano_faces_{side}" if reference else f"{side}_faces"
                valid_key = f"{side}_valid"
                if vertices_key not in archive.files or not archive[vertices_key].size:
                    continue
                valid = (
                    np.asarray(archive[valid_key]).reshape(-1)
                    if valid_key in archive.files
                    else None
                )
                self.hands[side] = (
                    np.asarray(archive[vertices_key], dtype=np.float32),
                    np.asarray(archive[faces_key], dtype=np.uint32),
                    valid,
                )
        self.frame_count = max(
            (len(value[0]) for value in self.hands.values()), default=0
        )
        self.server = None
        self.handles: dict[str, object] = {}
        self.visible = True

    def build(self, server) -> None:
        self.server = server

    def show_frame(self, frame_index: int) -> None:
        if self.server is None:
            return
        for side, (vertices, faces, valid) in self.hands.items():
            local_index = min(max(frame_index, 0), len(vertices) - 1)
            is_valid = valid is None or bool(valid[min(local_index, len(valid) - 1)])
            color = self.color or HAND_COLORS[side]
            handle = self.server.scene.add_mesh_simple(
                f"{self.prefix}/{side}",
                vertices=vertices[local_index],
                faces=faces,
                color=color,
                opacity=self.opacity,
                side="double",
            )
            handle.visible = self.visible and is_valid
            self.handles[side] = handle

    def set_visible(self, visible: bool, frame_index: int) -> None:
        self.visible = visible
        if visible:
            self.show_frame(frame_index)
        else:
            for handle in self.handles.values():
                handle.visible = False


class SceneReference(DeformingHands):
    """World-frame MANO meshes and tracked object from scene construction."""

    def __init__(self, npz_path: Path, object_mesh=None):
        super().__init__(npz_path, prefix="/mano_reference/hands", reference=True)
        self.object_mesh = object_mesh
        self.object_qpos = None
        with np.load(npz_path, allow_pickle=False) as archive:
            for side in ("right", "left"):
                key = f"qpos_obj_{side}"
                if key in archive.files and archive[key].size:
                    self.object_qpos = np.asarray(archive[key], dtype=np.float64)
                    self.frame_count = max(self.frame_count, len(self.object_qpos))
                    break
        self.object_handle = None

    def build(self, server) -> None:
        super().build(server)
        if self.object_mesh is not None:
            self.object_handle = server.scene.add_mesh_simple(
                "/mano_reference/object",
                vertices=np.asarray(self.object_mesh.vertices, dtype=np.float32),
                faces=np.asarray(self.object_mesh.faces, dtype=np.uint32),
                color=REFERENCE_COLOR,
                opacity=self.opacity,
                side="double",
            )

    def show_frame(self, frame_index: int) -> None:
        super().show_frame(frame_index)
        if self.object_handle is not None and self.object_qpos is not None:
            local_index = min(max(frame_index, 0), len(self.object_qpos) - 1)
            qpos = self.object_qpos[local_index]
            self.object_handle.position = tuple(qpos[:3])
            self.object_handle.wxyz = tuple(qpos[3:7])
            self.object_handle.visible = self.visible

    def set_visible(self, visible: bool, frame_index: int) -> None:
        super().set_visible(visible, frame_index)
        if self.object_handle is not None:
            self.object_handle.visible = visible


def load_trimesh(path: Path, scale: float = 1.0):
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if scale != 1.0:
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices) * scale
    return mesh
