"""
main.py — Stage 5
Autonomous navigation: VLA waypoints + costmap-based controller.

Usage:
    python main.py                      # stub VLA (default)
    python main.py --model <ckpt_dir>   # real NoMaD checkpoint
    python main.py --stub               # force stub
"""

import argparse
import math
import sys

import numpy as np
import pygame

from world import (World, Robot, raycast_lidar,
                   draw_world, draw_robot, draw_lidar, w2s,
                   render_robot_camera, render_goal_image,
                   sample_valid_pos)
from pedestrians import spawn_pedestrians, draw_pedestrians
from costmap import Costmap
from vla_node import get_subgoal_nav, load_model, get_subgoal
from controller import waypoint_to_world, select_action
from metrics import MetricsLogger

# ── constants ─────────────────────────────────────────────────────────────────

WIN_W, WIN_H     = 800, 600
START_X, START_Y = 0.5, 4.5
N_PEDS           = 15
GOAL_RADIUS      = 0.5      # metres — success threshold

_CAM_PX = 144               # camera preview size on screen
_CAM_X  = WIN_W - _CAM_PX - 8
_CAM_Y  = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to NoMaD checkpoint directory or .pth file")
    parser.add_argument("--stub", action="store_true",
                        help="Force stub VLA even if --model is provided")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Write metrics CSV to FILE (e.g. --log metrics.csv)")
    args = parser.parse_args()

    # ── VLA setup ─────────────────────────────────────────────────────────────
    model    = None
    use_real = False
    if args.model and not args.stub and load_model is not None:
        try:
            model    = load_model(args.model)
            use_real = True
            print("[main] Real ViNT/NoMaD model active.")
        except Exception as e:
            print(f"[main] Model load failed ({e}) — falling back to stub.")
    if not use_real:
        print("[main] Stub VLA active.")

    # ── pygame + sim setup ────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    world   = World()
    robot   = Robot(x=START_X, y=START_Y, theta=0.0, world=world)
    peds    = spawn_pedestrians(world, n=N_PEDS, seed=42)
    costmap = Costmap()
    costmap.build_static(world)

    _goal_rng      = np.random.default_rng()
    goal_x, goal_y = sample_valid_pos(world, _goal_rng)
    goal_img       = render_goal_image(world, goal_x, goal_y)
    wp_wx, wp_wy   = goal_x, goal_y

    logger   = MetricsLogger(args.log)
    sim_time = 0.0
    episode  = 1

    show_costmap = True
    show_camera  = True
    frame        = 0
    running      = True

    while running:
        dt = min(clock.tick(30) / 1000.0, 0.05)

        # ── events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:  running      = False
                if event.key == pygame.K_c:  show_costmap = not show_costmap
                if event.key == pygame.K_v:  show_camera  = not show_camera

        # ── simulation step ───────────────────────────────────────────────────
        for ped in peds:
            ped.step(world, peds, dt, robot=robot)

        ranges = raycast_lidar(robot, world)
        costmap.update_obstacle(ranges, robot)
        costmap.update_social(peds)

        # Render camera only when needed (preview on, or real VLA refresh due)
        need_cam = show_camera or (use_real and frame % 10 == 0)
        cam_arr  = render_robot_camera(world, peds, robot) if need_cam else None

        # VLA subgoal refresh every 10 frames (~3 Hz)
        if frame % 10 == 0:
            if use_real and cam_arr is not None and get_subgoal is not None:
                dx, dy = get_subgoal(cam_arr, goal_img, model)
            else:
                dx, dy = get_subgoal_nav(robot.x, robot.y, robot.theta,
                                         goal_x, goal_y, costmap)
            for _s in (1.0, 0.8, 0.6, 0.4, 0.2):
                wp_wx, wp_wy = waypoint_to_world(robot.x, robot.y, robot.theta,
                                                 dx * _s, dy * _s)
                if costmap.cost_at(wp_wx, wp_wy) < 0.85:
                    break
            # If projection collapsed the waypoint too close, restore the full target
            if math.hypot(wp_wx - robot.x, wp_wy - robot.y) < 0.4:
                wp_wx, wp_wy = waypoint_to_world(robot.x, robot.y, robot.theta,
                                                 dx, dy)

        # Attractor at waypoint — must come after update_social
        costmap.set_attractor(wp_wx, wp_wy)

        # Select best action and apply
        v, omega = select_action(robot, costmap, wp_wx, wp_wy)
        robot.step(v, omega, dt, pedestrians=peds)

        sim_time += dt
        logger.update(sim_time, robot, peds, v)

        # Goal check → reset
        if math.hypot(robot.x - goal_x, robot.y - goal_y) < GOAL_RADIUS:
            s = logger.summary()
            print(f"[ep {episode} frame {frame:5d}] Goal reached! "
                  f"path={s['path_length_m']:.1f}m  "
                  f"freezes={s['freeze_count']}  psv={s['ps_violations']}")
            logger.close()
            episode       += 1
            sim_time       = 0.0
            logger         = MetricsLogger(args.log)
            robot.reset(START_X, START_Y, 0.0)
            goal_x, goal_y = sample_valid_pos(world, _goal_rng)
            goal_img       = render_goal_image(world, goal_x, goal_y)
            wp_wx, wp_wy   = goal_x, goal_y

        frame += 1

        # ── render ────────────────────────────────────────────────────────────
        screen.fill((15, 15, 20))
        draw_world(screen, world)

        if show_costmap:
            costmap.render(screen)

        draw_lidar(screen, robot, ranges)
        draw_pedestrians(screen, peds)
        draw_robot(screen, robot)

        # Goal marker — yellow
        gsx, gsy = w2s(goal_x, goal_y)
        pygame.draw.circle(screen, (255, 220,   0), (gsx, gsy), 10)
        pygame.draw.circle(screen, (255, 255, 150), (gsx, gsy), 10, 2)
        screen.blit(font.render("goal", True, (200, 180, 50)), (gsx + 12, gsy - 7))

        # VLA waypoint marker — green
        wsx, wsy = w2s(wp_wx, wp_wy)
        pygame.draw.circle(screen, ( 50, 220,  80), (wsx, wsy), 7)
        pygame.draw.circle(screen, (150, 255, 180), (wsx, wsy), 7, 2)

        # Camera preview — top-right corner
        if show_camera and cam_arr is not None:
            cam_surf = pygame.surfarray.make_surface(cam_arr.swapaxes(0, 1))
            cam_surf = pygame.transform.scale(cam_surf, (_CAM_PX, _CAM_PX))
            pygame.draw.rect(screen, (80, 80, 80),
                             (_CAM_X - 2, _CAM_Y - 2, _CAM_PX + 4, _CAM_PX + 4), 1)
            screen.blit(cam_surf, (_CAM_X, _CAM_Y))
            screen.blit(font.render("camera (V)", True, (120, 120, 120)),
                        (_CAM_X, _CAM_Y + _CAM_PX + 2))

        # HUD
        vla_lbl = "VLA" if use_real else "stub"
        screen.blit(font.render(
            f"  ({robot.x:5.2f},{robot.y:5.2f})  "
            f"v={v:+.2f} ω={omega:+.2f}  "
            f"wp=({wp_wx:.1f},{wp_wy:.1f})  |  "
            f"{vla_lbl}  C=costmap  V=camera  Q=quit",
            True, (150, 150, 150)), (0, WIN_H - 20))

        pygame.display.set_caption(
            f"VLA CrowdSim — Stage 5 | {clock.get_fps():.0f} FPS")
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
