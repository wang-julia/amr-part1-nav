"""Standalone YOLO + Depth-Anything runtime for grounding/scoring InternVL's people-observation output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_YOLO_MODEL = "yolov8n.pt"
DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
PERSON_CLASS_ID = 0

ZONES = ("left", "center_left", "center", "center_right", "right")
ZONE_BOUNDARIES_PCT = (20.0, 40.0, 60.0, 80.0)

# Representative bearing (degrees) at each zone's midpoint across the robot's
# -90..90 forward arc.  Negative = left of forward, 0 = straight, positive = right.
ZONE_DEGREES: dict[str, int] = {
    "left": -72,
    "center_left": -36,
    "center": 0,
    "center_right": 36,
    "right": 72,
}


def zone_to_angle(zone_name: str) -> int:
    """Return the canonical bearing (degrees) for a named zone.

    Negative = left, 0 = straight ahead, positive = right.
    Unknown zone names return 0.
    """
    return ZONE_DEGREES.get(zone_name, 0)

# 2-D people cost grid.
GRID_ROWS = 5       # depth bins: row 0 = nearest, row GRID_ROWS-1 = farthest
BASE_WEIGHT = 100.0 # cost contributed by the nearest possible detection
DECAY_RATE = 0.5    # exponential decay per row; cost = BASE_WEIGHT * exp(-DECAY_RATE * row)

# Obstacle (wall/door) detection from the depth map.
# Sample rows 15%–60% of the image: above floor, below ceiling.
OBSTACLE_ROW_BAND = (0.15, 0.60)
# Normalized depth (0=far, 1=near) above which a zone is treated as walled.
OBSTACLE_DEPTH_THRESHOLD = 0.70
# Cost added to a zone where a close obstacle is detected.
OBSTACLE_ZONE_COST = 75

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
    """Label each person near/mid/far and store depth_normalized (0=far, 1=near).

    Normalizes against the FULL depth map range so labels match the heatmap and
    are absolute (a lone far person stays "far", not "near"). depth_normalized is
    stored so build_cost_grid can place detections in the correct grid row without
    relying on the relative depth_rank.
    """
    span = depth_high - depth_low
    for d in detections:
        normalized = (d["depth_value"] - depth_low) / span if span > 0 else 0.5
        d["depth_normalized"] = round(float(normalized), 4)
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


# Fallback row for each depth label when depth_normalized is not present.
_DEPTH_LABEL_ROW = {"near": 0, "mid": 2, "far": GRID_ROWS - 1}


def build_cost_grid(detected: list[dict]) -> list[list[float]]:
    """Build a GRID_ROWS × 5 cost grid from YOLO+depth detections.

    Row 0 = nearest, row GRID_ROWS-1 = farthest. Each detection is placed using
    depth_normalized (0=far, 1=near) so that the row reflects absolute scene
    distance rather than rank among detected people. This means a crowd of people
    all labeled "far" correctly cluster in the far rows instead of being spread
    across near rows by their relative rank.

    Falls back to the coarse depth label ("near"/"mid"/"far") for detections that
    pre-date the depth_normalized field.
    """
    import math

    zone_col = {zone: i for i, zone in enumerate(ZONES)}
    grid = [[0.0] * len(ZONES) for _ in range(GRID_ROWS)]
    for d in detected or []:
        zone = d.get("zone")
        if zone not in zone_col:
            continue
        col = zone_col[zone]

        depth_norm = d.get("depth_normalized")
        if depth_norm is not None:
            # depth_normalized: 1=nearest, 0=farthest → row 0=nearest, GRID_ROWS-1=farthest
            row = int((1.0 - float(depth_norm)) * GRID_ROWS)
        else:
            row = _DEPTH_LABEL_ROW.get(d.get("depth", "far"), GRID_ROWS - 1)
        row = max(0, min(row, GRID_ROWS - 1))

        grid[row][col] += BASE_WEIGHT * math.exp(-DECAY_RATE * row)
    return grid


def reduce_zone_costs(grid: list[list[float]]) -> dict:
    """Sum each zone column across all depth rows.

    Returns {"left": float, ..., "right": float, "lowest_cost_zone": str}.
    Ties break toward center (smallest column distance from center index 2).
    """
    totals = {zone: 0.0 for zone in ZONES}
    for row in grid:
        for col, zone in enumerate(ZONES):
            if col < len(row):
                totals[zone] += row[col]
    lowest = min(ZONES, key=lambda z: (totals[z], abs(ZONES.index(z) - 2)))
    return {**totals, "lowest_cost_zone": lowest}


def _smoke_test_cost_grid() -> None:
    """Verify build_cost_grid + reduce_zone_costs with synthetic detections."""
    import math

    # Case 1: far-center person (depth_normalized=0.10 → row 4, the far row)
    far_center = [{"zone": "center", "depth_normalized": 0.10, "depth": "far"}]
    grid1 = build_cost_grid(far_center)
    costs1 = reduce_zone_costs(grid1)
    far_row = min(int((1.0 - 0.10) * GRID_ROWS), GRID_ROWS - 1)  # = 4
    expected_center = BASE_WEIGHT * math.exp(-DECAY_RATE * far_row)
    assert abs(costs1["center"] - expected_center) < 1e-6, f"Case 1 center cost wrong: {costs1}"
    assert all(costs1[z] == 0.0 for z in ZONES if z != "center"), f"Case 1 non-center nonzero: {costs1}"

    # Case 2: near-right person (depth_normalized=0.90 → row 0, the near row)
    near_right = [{"zone": "right", "depth_normalized": 0.90, "depth": "near"}]
    grid2 = build_cost_grid(near_right)
    costs2 = reduce_zone_costs(grid2)
    assert costs2["right"] == BASE_WEIGHT, f"Case 2 right cost wrong: {costs2}"

    # Case 3: far-center vs near-right — far person must be cheaper than near person
    both = far_center + near_right
    grid3 = build_cost_grid(both)
    costs3 = reduce_zone_costs(grid3)
    assert costs3["center"] < costs3["right"], (
        f"Case 3: center ({costs3['center']:.2f}) should be cheaper than right ({costs3['right']:.2f})"
    )
    assert costs3["lowest_cost_zone"] != "right", f"Case 3 lowest_cost_zone is 'right', expected center-ish"

    # Case 4: crowd all labeled "far" — they must all land in far rows, not near rows
    crowd_far = [
        {"zone": "left",   "depth_normalized": 0.12, "depth": "far", "depth_rank": 1},
        {"zone": "center", "depth_normalized": 0.08, "depth": "far", "depth_rank": 2},
        {"zone": "right",  "depth_normalized": 0.15, "depth": "far", "depth_rank": 3},
    ]
    grid4 = build_cost_grid(crowd_far)
    near_rows_sum = sum(grid4[r][c] for r in range(2) for c in range(5))
    assert near_rows_sum == 0.0, f"Case 4: far crowd should not appear in near rows, got {grid4}"

    print("Smoke test passed.")
    print(f"  far-center cost: {costs3['center']:.2f}")
    print(f"  near-right cost: {costs3['right']:.2f}")
    print(f"  lowest_cost_zone: {costs3['lowest_cost_zone']}")


def detect_zone_obstacles(
    depth_map,
    image_width: int,
    image_height: int,
    person_zones: "set[str] | None" = None,
) -> dict[str, int]:
    """Return per-zone obstacle cost derived purely from the depth map.

    Scans a mid-image row band (skipping floor and ceiling) in each horizontal
    zone. If the median depth there is close (normalized >= OBSTACLE_DEPTH_THRESHOLD),
    the zone likely contains a wall or hard obstacle and gets OBSTACLE_ZONE_COST.

    person_zones: set of zones already occupied by YOLO-detected people; those
    zones are skipped so people are not mistaken for walls.
    """
    import torch

    depth_height, depth_width = depth_map.shape[-2:]
    depth_low = float(depth_map.min())
    depth_high = float(depth_map.max())
    span = depth_high - depth_low

    row_start = int(OBSTACLE_ROW_BAND[0] * depth_height)
    row_end = max(row_start + 1, int(OBSTACLE_ROW_BAND[1] * depth_height))

    col_boundaries = [0.0] + [b / 100.0 for b in ZONE_BOUNDARIES_PCT] + [1.0]
    obstacles: dict[str, int] = {}
    for zone, col_lo, col_hi in zip(ZONES, col_boundaries, col_boundaries[1:]):
        if person_zones and zone in person_zones:
            obstacles[zone] = 0
            continue
        col_start = int(col_lo * depth_width)
        col_end = max(col_start + 1, int(col_hi * depth_width))
        region = depth_map[row_start:row_end, col_start:col_end]
        if region.numel() == 0:
            obstacles[zone] = 0
            continue
        median_val = float(torch.median(region))
        normalized = (median_val - depth_low) / span if span > 0 else 0.5
        obstacles[zone] = OBSTACLE_ZONE_COST if normalized >= OBSTACLE_DEPTH_THRESHOLD else 0
    return obstacles


if __name__ == "__main__":
    import json
    import sys

    images_dir = Path(__file__).resolve().parent / "evaluation_set" / "images"
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = next(images_dir.glob("21_crowd*"), next(images_dir.iterdir()))

    _smoke_test_cost_grid()

    print(f"Running analyze_image on {target}")
    people, depth_map, width, height = analyze_image(target)
    print(f"image size: {width}x{height}, depth map shape: {tuple(depth_map.shape)}")
    print(json.dumps(people, indent=2))
    grid = build_cost_grid(people)
    costs = reduce_zone_costs(grid)
    print("Zone costs:", {k: round(v, 1) if isinstance(v, float) else v for k, v in costs.items()})
