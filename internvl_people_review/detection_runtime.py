"""Standalone YOLO + Depth-Anything runtime for grounding/scoring InternVL's people-observation output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
PERSON_CLASS_ID = 0

ZONES = ("left", "center_left", "center", "center_right", "right")
ZONE_BOUNDARIES_PCT = (20.0, 40.0, 60.0, 80.0)

_YOLO_CACHE: dict[tuple, Any] = {}
_DEPTH_CACHE: dict[tuple, Any] = {}


def _detect_device() -> tuple[str, "int | str"]:
    """Return (torch device string, transformers pipeline device arg)."""
    import torch

    if torch.cuda.is_available():
        return "cuda", 0
    if torch.backends.mps.is_available():
        return "mps", "mps"
    return "cpu", -1


def load_yolo(model_id: str = DEFAULT_YOLO_MODEL):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Missing detection runtime packages. Install ultralytics in this app's Python environment."
        ) from exc

    key = (model_id,)
    if key in _YOLO_CACHE:
        return _YOLO_CACHE[key]

    model = YOLO(model_id)
    _YOLO_CACHE[key] = model
    return model


def load_depth(model_id: str = DEFAULT_DEPTH_MODEL):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Missing detection runtime packages. Install transformers in this app's Python environment."
        ) from exc

    _, pipeline_device = _detect_device()
    key = (model_id, pipeline_device)
    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]

    depth_pipe = pipeline("depth-estimation", model=model_id, device=pipeline_device)
    _DEPTH_CACHE[key] = depth_pipe
    return depth_pipe


def _zone_for(bbox_center_x_pct: float) -> str:
    for boundary, zone in zip(ZONE_BOUNDARIES_PCT, ZONES):
        if bbox_center_x_pct < boundary:
            return zone
    return ZONES[-1]


def _median_depth(depth_map, x1: float, y1: float, x2: float, y2: float) -> float:
    import torch

    height, width = depth_map.shape[-2:]
    ix1 = max(0, min(width - 1, int(x1)))
    iy1 = max(0, min(height - 1, int(y1)))
    ix2 = max(ix1 + 1, min(width, int(x2)))
    iy2 = max(iy1 + 1, min(height, int(y2)))
    patch = depth_map[iy1:iy2, ix1:ix2]
    return float(torch.median(patch))


def _assign_depth_ranks(detections: list[dict[str, Any]]) -> None:
    """Rank by depth_value descending (larger = closer, rank 1 = closest); tie within 2% of range."""
    if not detections:
        return
    values = [d["depth_value"] for d in detections]
    value_range = max(values) - min(values)
    tie_epsilon = value_range * 0.02 if value_range > 0 else 0.0

    ordered = sorted(detections, key=lambda d: -d["depth_value"])
    rank = 1
    ordered[0]["depth_rank"] = rank
    for previous, current in zip(ordered, ordered[1:]):
        if previous["depth_value"] - current["depth_value"] > tie_epsilon:
            rank += 1
        current["depth_rank"] = rank


def _assign_depth_buckets(
    detections: list[dict[str, Any]], depth_low: float, depth_high: float
) -> None:
    """Label each person near/mid/far from its depth_value, normalized against the
    FULL depth map's range (depth_low..depth_high) — the same range the depth
    heatmap is colorized with, so a person's label matches where they sit in the
    heatmap. Normalizing against only the detected people (the old behavior) made a
    lone person always "near" and the nearest of any group always "near" regardless
    of true distance. This keeps the near/mid/far judgment out of the VLM."""
    span = depth_high - depth_low
    for d in detections:
        normalized = (d["depth_value"] - depth_low) / span if span > 0 else 0.5
        d["depth"] = "near" if normalized >= 0.66 else "mid" if normalized >= 0.33 else "far"


def analyze_image(
    image_path: Path,
    conf_threshold: float = 0.35,
    yolo_model_id: str = DEFAULT_YOLO_MODEL,
    depth_model_id: str = DEFAULT_DEPTH_MODEL,
) -> tuple[list[dict[str, Any]], Any, int, int]:
    """Run YOLO + Depth-Anything on an image, returning (detections, depth_map,
    image_width, image_height). Depth always runs (even with zero people) since
    the depth map itself is useful on its own, e.g. for a heatmap visualization.

    Zone boundaries and depth-rank convention match prompt3.txt's costmap and
    depth_rank contract exactly (see internvl_people_review/prompt3.txt).
    """
    from PIL import Image

    yolo = load_yolo(yolo_model_id)
    results = yolo(str(image_path), verbose=False)
    boxes = results[0].boxes
    person_boxes = [
        box.xyxy[0].tolist() + [float(box.conf[0])]
        for box in boxes
        if int(box.cls[0]) == PERSON_CLASS_ID and float(box.conf[0]) >= conf_threshold
    ]

    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    depth_pipe = load_depth(depth_model_id)
    depth_output = depth_pipe(image)
    depth_map = depth_output["predicted_depth"]
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)

    if not person_boxes:
        return [], depth_map, image_width, image_height

    # Depth-Anything's map resolution may not match the source image's pixel
    # dimensions, so bbox coordinates are scaled into depth-map space before cropping.
    depth_height, depth_width = depth_map.shape[-2:]
    scale_x = depth_width / image_width
    scale_y = depth_height / image_height

    detections: list[dict[str, Any]] = []
    for index, (x1, y1, x2, y2, confidence) in enumerate(person_boxes, start=1):
        bbox_center_x_pct = round(((x1 + x2) / 2) / image_width * 100, 1)
        depth_value = _median_depth(
            depth_map, x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y
        )
        detections.append(
            {
                "id": f"detected_{index}",
                "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                "bbox_pct": [
                    round(x1 / image_width * 100, 1),
                    round(y1 / image_height * 100, 1),
                    round(x2 / image_width * 100, 1),
                    round(y2 / image_height * 100, 1),
                ],
                "confidence": round(confidence, 3),
                "zone": _zone_for(bbox_center_x_pct),
                "bbox_center_x_pct": bbox_center_x_pct,
                "depth_value": depth_value,
            }
        )

    _assign_depth_ranks(detections)
    _assign_depth_buckets(detections, float(depth_map.min()), float(depth_map.max()))
    return detections, depth_map, image_width, image_height


def detect_people(
    image_path: Path,
    conf_threshold: float = 0.35,
    yolo_model_id: str = DEFAULT_YOLO_MODEL,
    depth_model_id: str = DEFAULT_DEPTH_MODEL,
) -> list[dict[str, Any]]:
    """Detect people in an image, returning per-person bbox, zone, and depth rank."""
    detections, _, _, _ = analyze_image(
        image_path,
        conf_threshold=conf_threshold,
        yolo_model_id=yolo_model_id,
        depth_model_id=depth_model_id,
    )
    return detections


def render_depth_heatmap(depth_map, colormap: str = "inferno") -> bytes:
    """Render a Depth-Anything depth map as a colorized PNG (warmer = closer).

    Returns raw PNG bytes rather than a file, since this is used to build a
    base64 data URI for the live review UI, not persisted to disk.
    """
    import io

    import matplotlib
    import numpy as np
    from PIL import Image

    values = depth_map.detach().cpu().numpy().astype("float32")
    low, high = float(values.min()), float(values.max())
    normalized = (values - low) / (high - low) if high > low else np.zeros_like(values)
    colored = (matplotlib.colormaps[colormap](normalized)[:, :, :3] * 255).astype("uint8")

    buffer = io.BytesIO()
    Image.fromarray(colored).save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    import json
    import sys

    images_dir = Path(__file__).resolve().parent / "evaluation_set" / "images"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = next(images_dir.glob("21_crowd*"), next(images_dir.iterdir()))

    print(f"Running analyze_image on {target}")
    people, depth_map, width, height = analyze_image(target)
    print(f"image size: {width}x{height}, depth map shape: {tuple(depth_map.shape)}")
    print(json.dumps(people, indent=2))
