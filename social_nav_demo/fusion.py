from typing import Dict, Tuple

import numpy as np


GridPoint = Tuple[int, int]
OBSTACLE = 255


def local_to_global(
    local_cell: GridPoint,
    robot_pose: Dict,
    local_size: int = 40,
) -> GridPoint:
    """
    Project a robot-local cell into the global demo map.

    The robot sits near the bottom-center of the local view. Lower local y
    values are farther ahead, which maps to decreasing global y in this demo.
    """
    local_x, local_y = local_cell
    robot_x = int(robot_pose["x"])
    robot_y = int(robot_pose["y"])
    robot_local_x = local_size // 2
    robot_local_y = int(local_size * 0.9)
    global_x = robot_x + (local_x - robot_local_x)
    global_y = robot_y - (robot_local_y - local_y)
    return global_x, global_y


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
