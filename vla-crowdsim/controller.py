"""
controller.py — Stage 5
Waypoint frame transform + sampling-based local planner.
"""

import math

import numpy as np

from costmap import _RES, _NX, _NY


def waypoint_to_world(robot_x, robot_y, robot_theta, dx, dy):
    """
    Convert (dx=forward, dy=lateral/left) in robot frame to world (wx, wy).
    """
    cos_t = math.cos(robot_theta)
    sin_t = math.sin(robot_theta)
    return (
        robot_x + dx * cos_t - dy * sin_t,
        robot_y + dx * sin_t + dy * cos_t,
    )


def select_action(robot, costmap, target_wx, target_wy,
                  horizon=1.0, dt=0.1):
    """
    Evaluate a 5x5 grid of (v, omega) candidates.

    Each candidate is rolled out kinematically for `horizon` seconds and
    scored as:
        path_cost * 2.0  +  distance_to_target_at_end * 1.5

    Returns the (v, omega) with the lowest score.
    """
    n_steps = max(1, int(horizon / dt))
    cmap    = costmap.combined()        # compute once; 250 index ops are fast

    def _cost(x, y):
        col = max(0, min(_NX - 1, int(x / _RES)))
        row = max(0, min(_NY - 1, int(y / _RES)))
        return float(cmap[row, col])

    vs     = np.linspace(0.0, 0.5, 5)
    omegas = np.linspace(-1.2, 1.2, 5)

    # Angular error to waypoint — used to break v=0 ties geometrically
    target_angle = math.atan2(target_wy - robot.y, target_wx - robot.x)
    angle_err    = (target_angle - robot.theta + math.pi) % (2 * math.pi) - math.pi

    best_score     = float("inf")
    best_v, best_w = 0.0, 0.0          # safe fallback: stop

    for v in vs:
        for omega in omegas:
            x, y, theta = robot.x, robot.y, robot.theta
            path_cost   = 0.0
            for _ in range(n_steps):
                x     += v * math.cos(theta) * dt
                y     += v * math.sin(theta) * dt
                theta += omega * dt
                path_cost += _cost(x, y)
            score = path_cost * 2.0 + math.hypot(x - target_wx, y - target_wy) * 1.5

            if v < 1e-9:
                # Bias against standing still so the robot prefers any forward motion
                score += 0.5
                # Tiebreaker among v=0 candidates: prefer the omega that minimises
                # remaining angle error after horizon seconds.
                remaining = abs(((angle_err - omega * horizon) + math.pi) % (2 * math.pi) - math.pi)
                score += 1e-4 * remaining

            if score < best_score:
                best_score     = score
                best_v, best_w = float(v), float(omega)

    return best_v, best_w
