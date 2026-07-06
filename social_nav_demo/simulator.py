import heapq
import math
from typing import Dict, List, Tuple

import numpy as np

from fusion import fuse_local_semantic_costmap
from local_costmap import semantic_json_to_local_costmap


GridPoint = Tuple[int, int]

GRID_SIZE = 100
OBSTACLE = 255


def create_base_costmap() -> np.ndarray:
    """Create a 100x100 indoor demo map: 0 free, 255 hard obstacle."""
    costmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    costmap[0, :] = OBSTACLE
    costmap[-1, :] = OBSTACLE
    costmap[:, 0] = OBSTACLE
    costmap[:, -1] = OBSTACLE

    # Office-like walls and furniture leave a clean diagonal hallway route.
    costmap[8:38, 8:14] = OBSTACLE
    costmap[8:14, 8:48] = OBSTACLE
    costmap[28:34, 8:34] = OBSTACLE
    costmap[10:36, 68:74] = OBSTACLE
    costmap[10:16, 62:92] = OBSTACLE
    costmap[46:52, 10:38] = OBSTACLE
    costmap[66:92, 84:90] = OBSTACLE
    costmap[86:92, 54:90] = OBSTACLE

    # A few low, deterministic navigation priors make the map read like a costmap.
    costmap[58:63, 18:45] = 18
    costmap[38:43, 54:82] = 22
    costmap[20:26, 44:58] = 14

    return costmap


def _heuristic(a: GridPoint, b: GridPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reconstruct_path(
    came_from: Dict[GridPoint, GridPoint],
    current: GridPoint,
) -> List[GridPoint]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(
    costmap: np.ndarray,
    start: GridPoint,
    goal: GridPoint,
) -> List[GridPoint]:
    """Plan over hard obstacles while treating semantic costs as soft penalties."""
    width = costmap.shape[1]
    height = costmap.shape[0]
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    open_set = [(0.0, start)]
    came_from: Dict[GridPoint, GridPoint] = {}
    g_score = {start: 0.0}
    visited = set()

    neighbors = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return _reconstruct_path(came_from, current)

        for dx, dy in neighbors:
            nx = current[0] + dx
            ny = current[1] + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            cell_cost = int(costmap[ny, nx])
            if cell_cost >= OBSTACLE:
                continue

            movement_cost = 1.414 if dx and dy else 1.0
            semantic_penalty = cell_cost / 40.0
            tentative = g_score[current] + movement_cost + semantic_penalty
            neighbor = (nx, ny)

            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heapq.heappush(
                    open_set,
                    (tentative + _heuristic(neighbor, goal), neighbor),
                )

    return []


def calculate_path_metrics(costmap: np.ndarray, path: List[GridPoint]) -> Dict:
    """Return path length, semantic exposure, social-zone hits, and risk score."""
    if not path:
        return {
            "path_length": 0.0,
            "average_cost": 0.0,
            "social_violations": 0,
            "risk_score": 100,
        }

    total_cost = 0
    social_violations = 0
    distance = 0.0
    previous = None

    for x, y in path:
        cell_cost = int(costmap[y, x])
        total_cost += cell_cost
        if 160 <= cell_cost < OBSTACLE:
            social_violations += 1
        if previous:
            distance += math.hypot(x - previous[0], y - previous[1])
        previous = (x, y)

    average_cost = total_cost / len(path)
    risk_score = min(100, int(average_cost * 0.5 + social_violations * 1.8))

    return {
        "path_length": round(distance, 1),
        "average_cost": round(average_cost, 2),
        "social_violations": social_violations,
        "risk_score": risk_score,
    }


def build_demo_result(
    semantic_json: Dict,
    navigation_context: Dict,
    start: GridPoint = (18, 82),
    goal: GridPoint = (84, 18),
) -> Dict:
    """Run the demo pipeline from semantic JSON through planning metrics."""
    base_costmap = create_base_costmap()
    local_costmap = semantic_json_to_local_costmap(semantic_json)
    enhanced_costmap = fuse_local_semantic_costmap(
        base_costmap,
        local_costmap,
        navigation_context["robot_pose"],
    )

    baseline_path = astar(base_costmap, start, goal)
    enhanced_path = astar(enhanced_costmap, start, goal)

    baseline_metrics = calculate_path_metrics(enhanced_costmap, baseline_path)
    enhanced_metrics = calculate_path_metrics(enhanced_costmap, enhanced_path)

    return {
        "start": start,
        "goal": goal,
        "base_costmap": base_costmap,
        "local_costmap": local_costmap,
        "enhanced_costmap": enhanced_costmap,
        "baseline_path": baseline_path,
        "enhanced_path": enhanced_path,
        "baseline_metrics": baseline_metrics,
        "enhanced_metrics": enhanced_metrics,
    }
