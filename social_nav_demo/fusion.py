from typing import Dict, Tuple

import numpy as np


GridPoint = Tuple[int, int]
OBSTACLE = 255


HEADING_VECTORS = {
    "north": (0.0, -1.0),
    "north-east": (0.707, -0.707),
    "east": (1.0, 0.0),
    "south-east": (0.707, 0.707),
    "south": (0.0, 1.0),
    "south-west": (-0.707, 0.707),
    "west": (-1.0, 0.0),
    "north-west": (-0.707, -0.707),
}


def local_to_global(
    local_cell: GridPoint,
    robot_pose: Dict,
    local_size: int = 40,
) -> GridPoint:
    """
    Project a robot-local cell into the global demo map.

    The robot sits near the bottom-center of the local view. Lower local y
    values are farther ahead. Robot heading rotates that local forward axis
    into the global T-junction map.
    """
    local_x, local_y = local_cell
    robot_x = int(robot_pose["x"])
    robot_y = int(robot_pose["y"])
    robot_local_x = local_size // 2
    robot_local_y = int(local_size * 0.9)
    lateral_distance = local_x - robot_local_x
    forward_distance = robot_local_y - local_y

    heading = robot_pose.get("heading", "north")
    forward_x, forward_y = HEADING_VECTORS.get(heading, HEADING_VECTORS["north"])
    right_x, right_y = -forward_y, forward_x

    global_x = robot_x + lateral_distance * right_x + forward_distance * forward_x
    global_y = robot_y + lateral_distance * right_y + forward_distance * forward_y
    return int(round(global_x)), int(round(global_y))


def fuse_local_semantic_costmap(
    global_costmap: np.ndarray,
    local_costmap: np.ndarray,
    robot_pose: Dict,
) -> np.ndarray:
    """Fuse semantic cost with max(existing, semantic), preserving obstacles."""
    fused = global_costmap.copy()
    height, width = fused.shape

    for local_y in range(local_costmap.shape[0]):
        for local_x in range(local_costmap.shape[1]):
            semantic_cost = int(local_costmap[local_y, local_x])
            if semantic_cost <= 0:
                continue

            global_x, global_y = local_to_global(
                (local_x, local_y),
                robot_pose,
                local_costmap.shape[0],
            )
            if not (0 <= global_x < width and 0 <= global_y < height):
                continue
            if int(fused[global_y, global_x]) >= OBSTACLE:
                continue

            fused[global_y, global_x] = max(int(fused[global_y, global_x]), semantic_cost)

    return fused
