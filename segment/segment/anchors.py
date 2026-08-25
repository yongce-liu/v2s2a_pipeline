"""Generic per-frame geometric anchor manifest support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Anchor:
    """One frame's point and/or bounding-box anchor in source-image pixels."""

    frame_index: int
    box_xyxy: tuple[float, float, float, float] | None
    point_xy: tuple[float, float] | None
    confidence: float


@dataclass(frozen=True)
class AnchorTarget:
    """How one output target should consume its per-frame anchors."""

    prompt_id: str
    anchor_type: str


@dataclass(frozen=True)
class AnchorManifest:
    """Parsed anchor JSON ready for the segmentation workflow."""

    source_path: Path
    targets: tuple[AnchorTarget, ...]
    anchors: dict[str, dict[int, Anchor]]


def _validate_prompt_id(prompt_id: str) -> str:
    prompt_id = prompt_id.strip()
    if not prompt_id:
        raise ValueError("Anchor target prompt_id cannot be empty.")
    if prompt_id in {".", ".."} or Path(prompt_id).name != prompt_id:
        raise ValueError(
            "Anchor target prompt_id is used as an output directory and cannot "
            f"contain path separators: {prompt_id!r}"
        )
    return prompt_id


def load_anchor_manifest(anchors_json: Path) -> AnchorManifest:
    """Load a producer-independent point/box anchor JSON artifact."""

    path = anchors_json.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Anchor JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coordinate_space") != "pixel":
        raise ValueError(f"Anchor JSON must use pixel coordinates: {path}")
    if payload.get("box_format") != "xyxy":
        raise ValueError(f"Anchor JSON must use xyxy boxes: {path}")

    targets = tuple(
        AnchorTarget(
            prompt_id=_validate_prompt_id(str(raw["prompt_id"])),
            anchor_type=str(raw["anchor_type"]),
        )
        for raw in payload.get("targets", [])
    )
    if not targets:
        raise ValueError(f"Anchor JSON contains no targets: {path}")
    prompt_ids = tuple(target.prompt_id for target in targets)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Anchor target prompt_id values must be unique.")
    invalid_types = [
        target.anchor_type
        for target in targets
        if target.anchor_type not in {"box", "point"}
    ]
    if invalid_types:
        raise ValueError(f"Unsupported anchor types: {invalid_types}")

    anchors: dict[str, dict[int, Anchor]] = {target.prompt_id: {} for target in targets}
    target_by_id = {target.prompt_id: target for target in targets}
    for entry in payload.get("entries", []):
        frame_index = int(entry["index"])
        raw_anchors = entry.get("anchors", {})
        for prompt_id, target in target_by_id.items():
            raw = raw_anchors.get(prompt_id)
            if raw is None:
                continue
            raw_box = raw.get("box_xyxy")
            box_xyxy = (
                tuple(float(value) for value in raw_box)
                if raw_box is not None
                else None
            )
            if box_xyxy is not None and len(box_xyxy) != 4:
                raise ValueError(
                    f"Anchor box for {prompt_id!r} frame {frame_index} must have 4 values."
                )
            raw_point = raw.get("point_xy")
            point_xy = (
                tuple(float(value) for value in raw_point)
                if raw_point is not None
                else None
            )
            if point_xy is not None and len(point_xy) != 2:
                raise ValueError(
                    f"Anchor point for {prompt_id!r} frame {frame_index} must have 2 values."
                )
            if target.anchor_type == "box" and box_xyxy is None:
                raise ValueError(
                    f"Missing box anchor for {prompt_id!r} frame {frame_index}."
                )
            if target.anchor_type == "point" and point_xy is None:
                raise ValueError(
                    f"Missing point anchor for {prompt_id!r} frame {frame_index}."
                )
            anchors[prompt_id][frame_index] = Anchor(
                frame_index=frame_index,
                box_xyxy=box_xyxy,
                point_xy=point_xy,
                confidence=float(raw.get("confidence", 1.0)),
            )

    return AnchorManifest(source_path=path, targets=targets, anchors=anchors)
