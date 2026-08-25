"""Smoke tests for the workflow argument dataclass.

These don't run the heavy HaWoR pipeline — they exercise the manifest/JSON
helpers that pure-python and the argument shapes handed to HaWoR.
"""

from __future__ import annotations

import json
from pathlib import Path

from hand_recon.hawor_stage import _hawor_argparse_args
from hand_recon.workflow import _load_manifest, _write_json


def test_write_json_roundtrip(tmp_path: Path) -> None:
    payload = {"k": [1, 2, 3], "nested": {"a": None}}
    out = tmp_path / "out.json"
    _write_json(out, payload)
    assert json.loads(out.read_text()) == payload


def test_load_manifest(tmp_path: Path) -> None:
    data = {"source_video": "v.mp4", "fps": 30, "width": 1920}
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(data))
    assert _load_manifest(path) == data


def test_hawor_namespace_shim() -> None:
    from hand_recon.hawor_stage import HandReconHaworArgs

    args = HandReconHaworArgs(
        video_path=Path("/real/clip.mp4"),
        checkpoint=Path("/ckpt/hawor.ckpt"),
        infiller_weight=Path("/ckpt/infiller.pt"),
        img_focal=612.0,
        static_camera=True,
        vis_mode="cam",
    )
    ns = _hawor_argparse_args(args, Path("/tmp/redirect"))
    assert ns.video_path == "/tmp/redirect"
    assert ns.checkpoint == "/ckpt/hawor.ckpt"
    assert ns.infiller_weight == "/ckpt/infiller.pt"
    assert ns.img_focal == 612.0
    assert ns.static_camera is True
    assert ns.input_type == "file"
