"""
scenarios.py — Stage 6
Headless benchmark: EMPTY / SPARSE / DENSE scenarios.

Usage:
    python scenarios.py
    python scenarios.py --episodes 5 --log-dir runs/
"""

import argparse
import math
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from world import World, Robot, raycast_lidar, sample_valid_pos
from pedestrians import spawn_pedestrians
from costmap import Costmap
from vla_node import get_subgoal_nav
from controller import waypoint_to_world, select_action
from metrics import MetricsLogger

# ── constants ─────────────────────────────────────────────────────────────────

START_X, START_Y = 0.5, 4.5
GOAL_RADIUS      = 0.5
MAX_SIM_TIME     = 60.0      # seconds per episode
DT               = 1.0 / 30.0

SCENARIOS = {
    "EMPTY":  {"n_peds":  0, "seed": 10},
    "SPARSE": {"n_peds":  5, "seed": 20},
    "DENSE":  {"n_peds": 20, "seed": 30},
}


# ── episode runner ─────────────────────────────────────────────────────────────

def run_episode(world, n_peds, seed, log_path=None):
    """Run one headless episode with a randomly sampled goal. Returns a stats dict."""
    rng            = np.random.default_rng(seed)
    goal_x, goal_y = sample_valid_pos(world, rng)
    straight_line  = math.hypot(goal_x - START_X, goal_y - START_Y)

    robot   = Robot(x=START_X, y=START_Y, theta=0.0, world=world)
    peds    = spawn_pedestrians(world, n=n_peds, seed=seed)
    costmap = Costmap()
    costmap.build_static(world)
    logger  = MetricsLogger(log_path)

    t, frame     = 0.0, 0
    wp_wx, wp_wy = goal_x, goal_y

    while t < MAX_SIM_TIME:
        for ped in peds:
            ped.step(world, peds, DT, robot=robot)

        ranges = raycast_lidar(robot, world)
        costmap.update_obstacle(ranges, robot)
        costmap.update_social(peds)

        if frame % 10 == 0:
            dx, dy = get_subgoal_nav(robot.x, robot.y, robot.theta,
                                     goal_x, goal_y, costmap)
            for _s in (1.0, 0.8, 0.6, 0.4, 0.2):
                wp_wx, wp_wy = waypoint_to_world(robot.x, robot.y, robot.theta,
                                                 dx * _s, dy * _s)
                if costmap.cost_at(wp_wx, wp_wy) < 0.85:
                    break
            if math.hypot(wp_wx - robot.x, wp_wy - robot.y) < 0.4:
                wp_wx, wp_wy = waypoint_to_world(robot.x, robot.y, robot.theta,
                                                 dx, dy)
        costmap.set_attractor(wp_wx, wp_wy)

        v, omega = select_action(robot, costmap, wp_wx, wp_wy)
        robot.step(v, omega, DT, pedestrians=peds)
        logger.update(t, robot, peds, v)

        if math.hypot(robot.x - goal_x, robot.y - goal_y) < GOAL_RADIUS:
            logger.close()
            s = logger.summary()
            return {
                "success":       True,
                "sim_time":      round(t, 1),
                "path_ratio":    round(s["path_length_m"] / straight_line, 3),
                "freeze_count":  s["freeze_count"],
                "ps_violations": s["ps_violations"],
            }

        t     += DT
        frame += 1

    logger.close()
    s = logger.summary()
    return {
        "success":       False,
        "sim_time":      MAX_SIM_TIME,
        "path_ratio":    round(s["path_length_m"] / straight_line, 3),
        "freeze_count":  s["freeze_count"],
        "ps_violations": s["ps_violations"],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Headless scenario benchmarks")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per scenario (default: 3)")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for per-episode CSV logs (default: logs/)")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_mode((1, 1))
    world = World()

    col = "{:<8} {:>3}  {:>4}  {:>7}  {:>9}  {:>7}  {:>6}"
    hdr = col.format("Scenario", "Ep", "OK?", "Time", "PathRatio", "Freezes", "PSViol")
    print(hdr)
    print("-" * len(hdr))

    for name, cfg in SCENARIOS.items():
        results = []
        for ep in range(args.episodes):
            log_path = os.path.join(args.log_dir, f"{name}_ep{ep + 1}.csv")
            print(f"  {name} ep {ep + 1}/{args.episodes} ...", end=" ", flush=True)
            r = run_episode(world, cfg["n_peds"], cfg["seed"] + ep, log_path)
            results.append(r)
            ok = "YES" if r["success"] else "NO"
            print(col.format("", "", ok, f"{r['sim_time']:.1f}s",
                             f"{r['path_ratio']:.2f}x",
                             r["freeze_count"], r["ps_violations"]))

        n       = len(results)
        rate    = sum(r["success"]        for r in results) / n * 100
        avg_t   = sum(r["sim_time"]       for r in results) / n
        avg_pr  = sum(r["path_ratio"]     for r in results) / n
        avg_frz = sum(r["freeze_count"]   for r in results) / n
        avg_psv = sum(r["ps_violations"]  for r in results) / n
        print(f"  {name} SUMMARY  "
              f"success={rate:.0f}%  time={avg_t:.1f}s  "
              f"ratio={avg_pr:.2f}x  freezes={avg_frz:.1f}  psv={avg_psv:.1f}")
        print()

    pygame.quit()


if __name__ == "__main__":
    main()
