"""Build the explicit hand/object/asset contract consumed by RL training.

The upstream v2s2a retarget stage stores robot joints followed by a MuJoCo free
joint in ``trajectory_kinematic.npz``. This module separates those signals and
copies no large assets: the bundle manifest stores resolved absolute paths so
Isaac Sim's MJCF converter sees the original, self-contained asset tree.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE_VERSION = 2


@dataclass(frozen=True)
class TaskBundle:
    """Validated inputs for one trajectory-conditioned RL task."""

    version: int
    name: str
    trajectory_path: str
    robot_asset_path: str
    object_asset_paths: tuple[str, ...]
    scene_asset_path: str
    scene_usd_path: str
    hand_joint_names: tuple[str, ...]
    finger_joint_names: tuple[str, ...]
    fingertip_body_names: tuple[str, ...]
    fingertip_body_paths: tuple[str, ...]
    hand_side: str
    robot_root_body_name: str
    object_joint_name: str
    object_body_name: str
    keypoints_path: str
    hand_dofs: int
    frequency: float
    num_frames: int
    trajectory_sha256: str
    scene_sha256: str

    def to_json(self, path: str | Path) -> Path:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_json(cls, path: str | Path) -> TaskBundle:
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        version = int(raw.get("version", 1))
        if version < 1 or version > BUNDLE_VERSION:
            raise ValueError(f"unsupported task bundle version {version}")
        raw["version"] = version
        raw["hand_joint_names"] = tuple(raw["hand_joint_names"])
        raw["finger_joint_names"] = tuple(
            raw.get("finger_joint_names", raw["hand_joint_names"][6:])
        )
        raw["object_asset_paths"] = tuple(raw["object_asset_paths"])
        raw.setdefault(
            "hand_side",
            raw.get("object_body_name", raw.get("object_joint_name", "")).split("_", 1)[0],
        )
        discovered_fingertips: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        if "fingertip_body_names" in raw:
            raw["fingertip_body_names"] = tuple(raw["fingertip_body_names"])
        else:
            scene_root = ET.parse(Path(raw["scene_asset_path"]).expanduser()).getroot()
            robot_root = next(
                (
                    body
                    for body in scene_root.findall(".//body")
                    if body.get("name") == raw.get("robot_root_body_name", "")
                ),
                None,
            )
            side = raw.get("object_body_name", raw.get("object_joint_name", "")).split(
                "_", 1
            )[0]
            discovered_fingertips = (
                _fingertip_bodies(robot_root, side) if robot_root is not None else ((), ())
            )
            raw["fingertip_body_names"] = discovered_fingertips[0]
        if "fingertip_body_paths" in raw:
            raw["fingertip_body_paths"] = tuple(raw["fingertip_body_paths"])
        else:
            if discovered_fingertips is None:
                scene_root = ET.parse(Path(raw["scene_asset_path"]).expanduser()).getroot()
                robot_root = next(
                    (
                        body
                        for body in scene_root.findall(".//body")
                        if body.get("name") == raw.get("robot_root_body_name", "")
                    ),
                    None,
                )
                discovered_fingertips = (
                    _fingertip_bodies(robot_root, raw["hand_side"])
                    if robot_root is not None
                    else ((), ())
                )
            raw["fingertip_body_paths"] = discovered_fingertips[1]
        raw.setdefault("scene_usd_path", "")
        raw.setdefault("robot_root_body_name", "")
        raw.setdefault("object_body_name", raw.get("object_joint_name", "").removesuffix("_joint"))
        raw.setdefault("keypoints_path", "")
        bundle = cls(**raw)
        bundle.validate_files()
        if bundle.trajectory_sha256 and _sha256(Path(bundle.trajectory_path)) != bundle.trajectory_sha256:
            raise ValueError("trajectory checksum does not match task bundle")
        if bundle.scene_sha256 and _sha256(Path(bundle.scene_asset_path)) != bundle.scene_sha256:
            raise ValueError("scene checksum does not match task bundle")
        return bundle

    def validate_files(self) -> None:
        for label, value in (
            ("trajectory", self.trajectory_path),
            ("robot asset", self.robot_asset_path),
            ("scene asset", self.scene_asset_path),
        ):
            if not Path(value).is_file():
                raise FileNotFoundError(f"{label} not found: {value}")
        if self.scene_usd_path and not Path(self.scene_usd_path).is_file():
            raise FileNotFoundError(f"converted scene USD not found: {self.scene_usd_path}")
        if self.keypoints_path and not Path(self.keypoints_path).is_file():
            raise FileNotFoundError(f"hand keypoints not found: {self.keypoints_path}")
        for value in self.object_asset_paths:
            if not Path(value).is_file():
                raise FileNotFoundError(f"object asset not found: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_meshes(scene: Path, root: ET.Element) -> tuple[Path, ...]:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_root = (scene.parent / meshdir).resolve()
    object_meshes: list[Path] = []
    for mesh in root.findall("./asset/mesh"):
        name = mesh.get("name", "")
        filename = mesh.get("file")
        if filename and ("object" in name or "/objects/" in f"/{filename}"):
            path = (mesh_root / filename).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"MJCF object mesh not found: {path}")
            object_meshes.append(path)
    return tuple(dict.fromkeys(object_meshes))


def _named_joints(root: ET.Element) -> list[ET.Element]:
    return [joint for joint in root.findall(".//joint") if joint.get("name")]


def _fingertip_bodies(
    robot_root: ET.Element, side: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names: list[str] = []
    paths: list[str] = []
    site_prefix = f"{side}_"

    def visit(body: ET.Element, parent_path: str) -> None:
        body_name = body.get("name", "")
        body_path = f"{parent_path}/{body_name}" if parent_path else body_name
        canonical_sites = [
            site.get("name", "")
            for site in body.findall("site")
            if site.get("name", "").startswith(site_prefix)
            and site.get("name", "").endswith("_tip")
        ]
        if canonical_sites:
            names.append(body_name)
            paths.append(body_path)
        for child in body.findall("body"):
            visit(child, body_path)

    visit(robot_root, "")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate fingertip bodies in {side} hand: {names}")
    return tuple(names), tuple(paths)


def inspect_inputs(trajectory: str | Path, scene: str | Path) -> dict[str, Any]:
    """Validate upstream files and infer their exact trajectory layout."""
    trajectory_path = Path(trajectory).expanduser().resolve()
    scene_path = Path(scene).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {trajectory_path}")
    if not scene_path.is_file():
        raise FileNotFoundError(f"scene MJCF not found: {scene_path}")

    root = ET.parse(scene_path).getroot()
    joints = _named_joints(root)
    free_joints = [j for j in joints if j.get("type") in {"free", "floating"}]
    if len(free_joints) != 1:
        raise ValueError(
            f"expected exactly one manipulated-object free joint, found {len(free_joints)}"
        )
    object_joint = free_joints[0]
    object_body = next(
        (body for body in root.findall(".//body") if object_joint in list(body)), None
    )
    object_body_name = object_body.get("name", "") if object_body is not None else ""
    hand_joints = [j.get("name", "") for j in joints if j is not object_joint]
    side = object_body_name.split("_", 1)[0]
    robot_root_body = next(
        (
            body
            for body in root.findall(".//body")
            if body.get("name") == f"{side}_hand_C_MC"
        ),
        None,
    )
    if robot_root_body is None or not object_body_name:
        raise ValueError("could not infer robot/object root bodies from MJCF")

    with np.load(trajectory_path, allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise ValueError(f"trajectory has no qpos array: {trajectory_path}")
        qpos = np.asarray(archive["qpos"])
        qvel = np.asarray(archive["qvel"]) if "qvel" in archive else None
        frequency = float(archive["frequency"]) if "frequency" in archive else 50.0
    if qpos.ndim != 2 or qpos.shape[0] < 2:
        raise ValueError(f"qpos must be (T, D), T>=2; got {qpos.shape}")
    expected_qpos = len(hand_joints) + 7
    if qpos.shape[1] != expected_qpos:
        raise ValueError(
            f"qpos width {qpos.shape[1]} != {len(hand_joints)} hand joints + 7 object pose"
        )
    if qvel is not None and qvel.shape != (qpos.shape[0], len(hand_joints) + 6):
        raise ValueError(f"qvel has incompatible shape {qvel.shape}")
    if not np.isfinite(qpos).all() or (qvel is not None and not np.isfinite(qvel).all()):
        raise ValueError("trajectory contains non-finite values")
    if frequency <= 0:
        raise ValueError("trajectory has an invalid frequency")

    # Use the original robot asset if the generated scene identifies one. The
    # full scene remains authoritative for actual training and asset conversion.
    robot_assets = tuple(
        path
        for path in ((scene_path.parent / "../../../../assets" / "robots").resolve(),)
        if path.exists()
    )
    object_assets = _resolve_meshes(scene_path, root)
    fingertip_names, fingertip_paths = _fingertip_bodies(robot_root_body, side)
    return {
        "trajectory_path": trajectory_path,
        "scene_path": scene_path,
        "hand_joint_names": tuple(hand_joints),
        "finger_joint_names": tuple(hand_joints[6:]),
        "fingertip_body_names": fingertip_names,
        "fingertip_body_paths": fingertip_paths,
        "hand_side": side,
        "robot_root_body_name": robot_root_body.get("name", ""),
        "object_joint_name": object_joint.get("name", object_body_name),
        "object_body_name": object_body_name,
        "object_asset_paths": object_assets,
        "robot_asset_path": robot_assets[0] if robot_assets else scene_path,
        "frequency": frequency,
        "num_frames": qpos.shape[0],
        "hand_dofs": len(hand_joints),
    }


def prepare_task_bundle(
    trajectory: str | Path,
    scene: str | Path,
    output: str | Path,
    name: str | None = None,
    scene_usd: str | Path | None = None,
    keypoints: str | Path | None = None,
) -> TaskBundle:
    """Create a reproducible task manifest from retarget/physics outputs."""
    info = inspect_inputs(trajectory, scene)
    trajectory_path: Path = info["trajectory_path"]
    scene_path: Path = info["scene_path"]
    robot_asset = info["robot_asset_path"]
    # A directory is useful provenance but TaskBundle promises a concrete asset.
    if robot_asset.is_dir():
        candidates = sorted(robot_asset.rglob("*.xml")) + sorted(robot_asset.rglob("*.urdf"))
        robot_asset = candidates[0] if candidates else scene_path
    bundle = TaskBundle(
        version=BUNDLE_VERSION,
        name=name or scene_path.parent.parent.name,
        trajectory_path=str(trajectory_path),
        robot_asset_path=str(robot_asset),
        object_asset_paths=tuple(str(path) for path in info["object_asset_paths"]),
        scene_asset_path=str(scene_path),
        scene_usd_path=str(Path(scene_usd).expanduser().resolve()) if scene_usd else "",
        hand_joint_names=info["hand_joint_names"],
        finger_joint_names=info["finger_joint_names"],
        fingertip_body_names=info["fingertip_body_names"],
        fingertip_body_paths=info["fingertip_body_paths"],
        hand_side=info["hand_side"],
        robot_root_body_name=info["robot_root_body_name"],
        object_joint_name=info["object_joint_name"],
        object_body_name=info["object_body_name"],
        keypoints_path=str(Path(keypoints).expanduser().resolve()) if keypoints else "",
        hand_dofs=info["hand_dofs"],
        frequency=info["frequency"],
        num_frames=info["num_frames"],
        trajectory_sha256=_sha256(trajectory_path),
        scene_sha256=_sha256(scene_path),
    )
    bundle.to_json(output)
    return bundle
