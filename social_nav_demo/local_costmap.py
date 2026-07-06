from typing import Dict, Tuple

import numpy as np


GridPoint = Tuple[int, int]
LOCAL_SIZE = 40


DIRECTION_VECTORS = {
    "forward": (0.0, -1.0),
    "backward": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "forward-left": (-0.707, -0.707),
    "forward-right": (0.707, -0.707),
    "backward-left": (-0.707, 0.707),
    "backward-right": (0.707, 0.707),
}


def create_local_costmap(size: int = LOCAL_SIZE) -> np.ndarray:
    """Create an empty local semantic layer."""
    return np.zeros((size, size), dtype=np.uint8)


def image_position_to_local_grid(position: str) -> GridPoint:
    """Map coarse VLM image positions to cells in a 40x40 robot-local grid."""
    mapping = {
        "top-left": (10, 7),
        "top-center": (20, 7),
        "top-right": (30, 7),
        "center-left": (10, 20),
        "center": (20, 20),
        "center-right": (30, 20),
        "bottom-left": (10, 32),
        "bottom-center": (20, 32),
        "bottom-right": (30, 32),
    }
    return mapping.get(position, mapping["center"])


def add_circular_semantic_cost(
    costmap: np.ndarray,
    center: GridPoint,
    radius: int,
    cost: int,
) -> np.ndarray:
    """Add a soft circular semantic zone without using 255 for non-obstacles."""
    updated = costmap.copy()
    cx, cy = center
    radius = max(1, int(radius))
    cost = max(0, min(int(cost), 245))

    for y in range(updated.shape[0]):
        for x in range(updated.shape[1]):
            distance = np.hypot(x - cx, y - cy)
            if distance <= radius:
                falloff = 0.55 + 0.45 * (1.0 - distance / radius)
                cell_cost = int(cost * falloff)
                updated[y, x] = max(int(updated[y, x]), cell_cost)

    return updated


def add_directional_semantic_cost(
    costmap: np.ndarray,
    origin: GridPoint,
    direction: str,
    length: int,
    width: int,
    cost: int,
) -> np.ndarray:
    """Add an elongated soft zone in a person's likely walking direction."""
    updated = costmap.copy()
    ox, oy = origin
    dx, dy = DIRECTION_VECTORS.get(direction, DIRECTION_VECTORS["forward"])
    norm = max(np.hypot(dx, dy), 1e-6)
    dx /= norm
    dy /= norm
    length = max(1, int(length))
    width = max(1, int(width))
    cost = max(0, min(int(cost), 245))

    for y in range(updated.shape[0]):
        for x in range(updated.shape[1]):
            rel_x = x - ox
            rel_y = y - oy
            forward_distance = rel_x * dx + rel_y * dy
            if forward_distance < 0 or forward_distance > length:
                continue

            lateral_distance = abs(rel_x * -dy + rel_y * dx)
            tapered_width = width * (0.78 + 0.42 * (forward_distance / length))
            if lateral_distance > tapered_width:
                continue

            forward_falloff = 1.0 - 0.42 * (forward_distance / length)
            lateral_falloff = 1.0 - 0.36 * (lateral_distance / tapered_width)
            cell_cost = int(cost * forward_falloff * lateral_falloff)
            updated[y, x] = max(int(updated[y, x]), cell_cost)

    return updated


def semantic_json_to_local_costmap(
    semantic_json: Dict,
    size: int = LOCAL_SIZE,
) -> np.ndarray:
    """Convert VLM semantic JSON intent into a deterministic local costmap."""
    costmap = create_local_costmap(size)
    for item in semantic_json.get("local_costmap_intent", []):
        shape = item.get("shape")
        center = image_position_to_local_grid(item.get("center", "center"))
        cost = int(item.get("cost", 160))

        if shape == "circle":
            radius = int(item.get("radius", 10))
            costmap = add_circular_semantic_cost(costmap, center, radius, cost)
        elif shape == "directional_ellipse":
            costmap = add_directional_semantic_cost(
                costmap,
                center,
                item.get("direction", "forward"),
                int(item.get("length", 16)),
                int(item.get("width", 7)),
                cost,
            )
    return costmap
