from typing import Dict, Tuple

import numpy as np


GridPoint = Tuple[int, int]
LOCAL_SIZE = 40


MOBILITY_COST_BONUS = {
    "static": 0,
    "possibly_moving": 25,
    "moving": 0,
}

MOBILITY_RADIUS_BONUS = {
    "static": 0,
    "possibly_moving": 0,
    "moving": 0,
}


def create_local_costmap(size: int = LOCAL_SIZE) -> np.ndarray:
    """Create an empty local semantic layer."""
    return np.zeros((size, size), dtype=np.uint8)


def image_position_to_local_grid(position: str) -> GridPoint:
    """Map coarse VLM image positions to cells in a 40x40 robot-local grid."""
    aliases = {
        "left": "center-left",
        "right": "center-right",
        "middle": "center",
        "bottom": "bottom-center",
        "top": "top-center",
    }
    position = aliases.get(position, position)
    mapping = {
        "top-left": (13, 7),
        "top-center": (20, 7),
        "top-right": (27, 7),
        "center-left": (14, 20),
        "center": (20, 20),
        "center-right": (26, 20),
        "bottom-left": (14, 32),
        "bottom-center": (20, 32),
        "bottom-right": (26, 32),
    }
    return mapping.get(position, mapping["center"])


def apply_distance_estimate(center: GridPoint, distance_estimate: str | None) -> GridPoint:
    """Shift a coarse image-position anchor forward/backward by depth estimate."""
    if not distance_estimate:
        return center

    x, y = center
    y_offsets = {
        "near": 8,
        "medium": -2,
        "far": -9,
    }
    y += y_offsets.get(distance_estimate, 0)
    return x, max(4, min(35, y))


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


def semantic_json_to_local_costmap(
    semantic_json: Dict,
    size: int = LOCAL_SIZE,
) -> np.ndarray:
    """Convert VLM semantic JSON intent into a deterministic local costmap."""
    costmap = create_local_costmap(size)
    for item in semantic_json.get("local_costmap_intent", []):
        shape = item.get("shape")
        if shape != "circle":
            continue
        center = image_position_to_local_grid(item.get("center", "center"))
        center = apply_distance_estimate(center, item.get("distance_estimate"))
        mobility_class = item.get("mobility_class", "static")
        cost = int(item.get("cost", 160))

        radius = int(item.get("radius", 4))
        radius += MOBILITY_RADIUS_BONUS.get(mobility_class, 0)
        cost += MOBILITY_COST_BONUS.get(mobility_class, 0)
        costmap = add_circular_semantic_cost(costmap, center, radius, cost)
    return costmap
