from pathlib import Path

from vis.paths import resolve_raw_inputs, resolve_robot


def test_resolve_robot_from_pipeline_layout(tmp_path: Path) -> None:
    clip = tmp_path / "demo"
    scene = (
        clip / "scene_construction" / "sharpa" / "bimanual" / "demo" / "0" / "scene.xml"
    )
    scene.parent.mkdir(parents=True)
    scene.touch()

    assert resolve_robot(clip, "bimanual", "demo", "0") == "sharpa"


def test_resolve_raw_inputs_uses_manifests(tmp_path: Path) -> None:
    clip = tmp_path / "demo"
    for relative in (
        "hand_recon/meshes.npz",
        "hand_recon/hands.json",
        "pose_estimation/poses.json",
        "geometry/geometry.json",
    ):
        path = clip / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "")

    inputs = resolve_raw_inputs(clip, "demo")

    assert inputs.poses_json == clip / "pose_estimation" / "poses.json"
    assert inputs.geometry_json == clip / "geometry" / "geometry.json"
    assert inputs.hands_json == clip / "hand_recon" / "hands.json"
