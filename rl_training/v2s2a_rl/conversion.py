"""Offline MJCF-to-USD conversion for Isaac Sim pip installations."""

from __future__ import annotations

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_scene_mjcf(scene: str | Path, output_dir: str | Path, force: bool = False) -> Path:
    """Convert scene MJCF using NVIDIA's open-source ``mujoco-usd-converter``.

    Isaac Sim 6 pip packages currently omit the Kit MJCF importer extension
    expected by IsaacLab beta2. The converter already bundled by Isaac Sim is
    deterministic and preserves the articulation, rigid object and mesh assets.
    Conversion happens once during ``prepare``, never in every training worker.
    """
    scene_path = Path(scene).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # mujoco-usd-converter 0.2 does not support explicit contact pairs. Bake
    # the pair allow-list into participating geoms before conversion. Remove
    # visual-only sites and inactive equalities, which otherwise become invalid
    # static-body fixed joints in PhysX.
    root = ET.parse(scene_path).getroot()
    pair_geom_names = {
        name
        for pair in root.findall("./contact/pair")
        for name in (pair.get("geom1"), pair.get("geom2"))
        if name
    }
    for geom in root.findall(".//geom"):
        if geom.get("name") in pair_geom_names:
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
    contact = root.find("contact")
    if contact is not None:
        root.remove(contact)
    for parent in root.iter():
        for site in list(parent.findall("site")):
            parent.remove(site)
    for body in list(worldbody.findall("body")) if (worldbody := root.find("worldbody")) is not None else []:
        if body.get("mocap", "false").lower() == "true":
            worldbody.remove(body)
    equality = root.find("equality")
    if equality is not None:
        for constraint in list(equality):
            if constraint.get("active", "true").lower() == "false":
                equality.remove(constraint)

    # Replace the six nested Cartesian wrist joints by a floating articulation
    # root. Their qpos convention is [xyz, roll, pitch, yaw]; controlling the
    # palm root directly is both equivalent and avoids PhysX dropping the first
    # world-to-body joint from an articulation.
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    wrist_names = [
        "left_pos_x", "left_pos_y", "left_pos_z", "left_rot_x", "left_rot_y", "left_rot_z",
        "right_pos_x", "right_pos_y", "right_pos_z", "right_rot_x", "right_rot_y", "right_rot_z",
    ]
    removed_wrist_names: set[str] = set()
    unwrapped_sides: set[str] = set()
    for side in ("left", "right"):
        outer = next((b for b in worldbody.findall("body") if b.get("name") == f"{side}_base_tx"), None)
        if outer is None:
            continue
        chain = outer
        palm = None
        while chain is not None:
            joint = chain.find("joint")
            if joint is not None and joint.get("name") in wrist_names:
                removed_wrist_names.add(joint.get("name", ""))
            children = chain.findall("body")
            candidate = next((b for b in children if b.get("name") == f"{side}_hand_C_MC"), None)
            if candidate is not None:
                palm = candidate
                break
            chain = children[0] if children else None
        if palm is None:
            raise ValueError(f"could not unwrap {side} wrist chain")
        chain.remove(palm)
        worldbody.remove(outer)
        worldbody.append(palm)
        unwrapped_sides.add(side)
    present_sides = {
        body.get("name", "").split("_", 1)[0]
        for body in root.findall(".//body")
        if body.get("name", "").endswith("_hand_C_MC")
    }
    missing_unwrap = present_sides - unwrapped_sides
    if missing_unwrap:
        raise ValueError(
            "could not unwrap Cartesian wrist chain for: "
            + ", ".join(sorted(missing_unwrap))
        )
    actuator = root.find("actuator")
    if actuator is not None:
        for motor in list(actuator):
            if motor.get("joint") in removed_wrist_names:
                actuator.remove(motor)

    with tempfile.TemporaryDirectory(prefix="v2s2a_mjcf_") as temp:
        sanitized = Path(temp) / scene_path.name
        # Mesh paths remain relative to the original MJCF, so make the
        # temporary compiler meshdir absolute.
        compiler = root.find("compiler")
        if compiler is not None and compiler.get("meshdir"):
            compiler.set("meshdir", str((scene_path.parent / compiler.get("meshdir", "")).resolve()))
        ET.ElementTree(root).write(sanitized, encoding="unicode")

        # Import lazily so data-only tests and `prepare --no-convert` remain light.
        from mujoco_usd_converter import Converter

        asset_path = Converter(layer_structure=True, scene=False).convert(str(sanitized), str(output))
    usd = Path(str(asset_path).strip("@")).resolve()
    if not usd.is_file():
        raise RuntimeError(f"MJCF converter did not produce an asset: {usd}")
    return usd
