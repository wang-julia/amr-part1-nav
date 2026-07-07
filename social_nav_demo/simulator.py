import heapq
import math
from typing import Dict, List, Tuple

import numpy as np

from fusion import fuse_local_semantic_costmap
from local_costmap import semantic_json_to_local_costmap


GridPoint = Tuple[int, int]

GRID_SIZE = 100
OBSTACLE = 255


def add_wall_clearance_cost(
    costmap: np.ndarray,
    radius: int = 3,
    max_cost: int = 34,
) -> np.ndarray:
    """Add low soft cost around walls so paths do not hug corridor edges."""
    inflated = costmap.copy()
    obstacle_cells = np.argwhere(costmap >= OBSTACLE)
    free_cells = np.argwhere(costmap < OBSTACLE)

    for y, x in free_cells:
        nearby = obstacle_cells[
            (np.abs(obstacle_cells[:, 0] - y) <= radius)
            & (np.abs(obstacle_cells[:, 1] - x) <= radius)
        ]
        if nearby.size == 0:
            continue
        distances = np.hypot(nearby[:, 1] - x, nearby[:, 0] - y)
        min_distance = float(distances.min())
        if min_distance > radius:
            continue
        wall_cost = int(max_cost * (1.0 - min_distance / (radius + 1)))
        inflated[y, x] = max(int(inflated[y, x]), wall_cost)

    return inflated


def create_base_costmap() -> np.ndarray:
    """Create a 100x100 T-junction corridor map: 0 free, 255 hard obstacle."""
    costmap = np.full((GRID_SIZE, GRID_SIZE), OBSTACLE, dtype=np.uint8)

    # T-junction corridor scaled from the reference floor plan.
    # Corridor width is about 2.10 m; top arms are 5.20 m left, 2.40 m
    # around the stem, and 5.30 m right. The vertical stem is about 9.80 m.
    top_y0, top_y1 = 16, 31
    stem_x0, stem_x1 = 45, 60
    left_x0, right_x1 = 8, 92
    stem_y0, stem_y1 = top_y0, 92

    costmap[top_y0:top_y1, left_x0:right_x1] = 0
    costmap[stem_y0:stem_y1, stem_x0:stem_x1] = 0

    # Light deterministic travel priors keep paths centered in the corridors.
    costmap[22:25, 12:88] = 10
    costmap[34:88, 51:54] = 10

    # Door thresholds at corridor ends are traversable but visible as mild cost.
    costmap[18:29, 8:10] = 22
    costmap[18:29, 90:92] = 22
    costmap[90:92, 47:58] = 22

    return add_wall_clearance_cost(costmap)


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


def interpolate_waypoint_path(waypoints: List[GridPoint]) -> List[GridPoint]:
    """Expand integer waypoints into a connected grid path."""
    if not waypoints:
        return []

    expanded = [(int(waypoints[0][0]), int(waypoints[0][1]))]
    for start, end in zip(waypoints, waypoints[1:]):
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        steps = max(abs(x1 - x0), abs(y1 - y0))
        if steps == 0:
            continue
        for step in range(1, steps + 1):
            t = step / steps
            point = (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
            if point != expanded[-1]:
                expanded.append(point)
    return expanded


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
    start: GridPoint | None = None,
    goal: GridPoint = (13, 23),
) -> Dict:
    """Run the demo pipeline from semantic JSON through planning metrics."""
    if start is None:
        robot_pose = navigation_context["robot_pose"]
        start = (int(robot_pose["x"]), int(robot_pose["y"]))

    base_costmap = create_base_costmap()
    local_costmap = semantic_json_to_local_costmap(semantic_json)
    enhanced_costmap = fuse_local_semantic_costmap(
        base_costmap,
        local_costmap,
        navigation_context["robot_pose"],
    )

    baseline_waypoints = navigation_context.get("baseline_path_waypoints")
    if baseline_waypoints:
        baseline_path = interpolate_waypoint_path([tuple(point) for point in baseline_waypoints])
    else:
        baseline_path = astar(base_costmap, start, goal)
    enhanced_path = astar(enhanced_costmap, start, goal)

    baseline_metrics = calculate_path_metrics(enhanced_costmap, baseline_path)
    enhanced_metrics = calculate_path_metrics(enhanced_costmap, enhanced_path)

    return {
        "start": start,
        "goal": goal,
        "robot_pose": navigation_context["robot_pose"],
        "base_costmap": base_costmap,
        "local_costmap": local_costmap,
        "enhanced_costmap": enhanced_costmap,
        "baseline_path": baseline_path,
        "enhanced_path": enhanced_path,
        "baseline_metrics": baseline_metrics,
        "enhanced_metrics": enhanced_metrics,
    }
