"""
world.py — Stage 1
2D top-down simulation: world geometry, robot kinematics, lidar raycast.
"""

import math
import sys

import numpy as np
import pygame

ROBOT_RADIUS = 0.2  # collision radius in meters

# Rendering constants
_SCALE = 60         # pixels per meter
_OFF_X = 40         # screen x offset (pixels)
_OFF_Y = 60         # screen y offset (pixels)
_WIN_W = 800
_WIN_H = 600


class World:
    WIDTH  = 12.0   # meters
    HEIGHT = 8.0

    def __init__(self):
        self.walls = [
            ((0.0,        0.0),         (self.WIDTH,  0.0)),
            ((self.WIDTH,  0.0),         (self.WIDTH,  self.HEIGHT)),
            ((self.WIDTH,  self.HEIGHT),  (0.0,         self.HEIGHT)),
            ((0.0,         self.HEIGHT),  (0.0,         0.0)),
        ]
        # Classroom layout: 3 rows x 3 columns of desks stored as (x, y, w, h)
        self.desks = [
            (1.0, 1.0, 1.2, 0.6),
            (4.5, 1.0, 1.2, 0.6),
            (8.0, 1.0, 1.2, 0.6),
            (1.0, 3.5, 1.2, 0.6),
            (4.5, 3.5, 1.2, 0.6),
            (8.0, 3.5, 1.2, 0.6),
            (1.0, 6.0, 1.2, 0.6),
            (4.5, 6.0, 1.2, 0.6),
            (8.0, 6.0, 1.2, 0.6),
        ]

    def get_all_segments(self):
        """Return every wall and desk edge as ((x1,y1),(x2,y2)) pairs."""
        segs = list(self.walls)
        for (x, y, w, h) in self.desks:
            segs += [
                ((x,     y    ), (x + w, y    )),
                ((x + w, y    ), (x + w, y + h)),
                ((x + w, y + h), (x,     y + h)),
                ((x,     y + h), (x,     y    )),
            ]
        return segs


class Robot:
    def __init__(self, x=0.5, y=4.0, theta=0.0, world=None):
        self.x      = x
        self.y      = y
        self.theta  = theta
        self._world = world

    def step(self, v, omega, dt, pedestrians=None):
        """Differential-drive step; rejects the move on collision.

        Sub-steps so fast moves can't tunnel through thin walls or pedestrians.
        """
        dist = abs(v) * dt
        n_sub = max(1, int(math.ceil(dist / (ROBOT_RADIUS * 0.5))))
        sub_dt = dt / n_sub

        for _ in range(n_sub):
            nx = self.x + v * math.cos(self.theta) * sub_dt
            ny = self.y + v * math.sin(self.theta) * sub_dt
            nt = self.theta + omega * sub_dt

            if self._world is not None:
                for seg in self._world.get_all_segments():
                    if _pt_seg_dist(nx, ny, *seg[0], *seg[1]) < ROBOT_RADIUS:
                        return

            if pedestrians is not None:
                for ped in pedestrians:
                    if math.hypot(nx - ped.x, ny - ped.y) < ROBOT_RADIUS + ped.radius:
                        return

            self.x, self.y, self.theta = nx, ny, nt

    def reset(self, x, y, theta):
        self.x, self.y, self.theta = x, y, theta


# ── geometry helpers ──────────────────────────────────────────────────────────

def _pt_seg_dist(px, py, ax, ay, bx, by):
    """Shortest distance from point to line segment."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _ray_seg_hit(px, py, dx, dy, ax, ay, bx, by):
    """Return t >= 0 where ray (px + t*dx, py + t*dy) hits segment, else None."""
    ex, ey = bx - ax, by - ay
    rx, ry = ax - px, ay - py
    det = ex * dy - dx * ey
    if abs(det) < 1e-10:
        return None
    t = (ex * ry - ey * rx) / det
    s = (dx * ry - dy * rx) / det
    if t >= 0.0 and 0.0 <= s <= 1.0:
        return t
    return None


def sample_valid_pos(world, rng, clearance=0.5, max_tries=1000):
    """Sample a random (x, y) at least `clearance` m from every wall and desk edge.
    rng must have a .uniform(low, high) method (numpy Generator or random.Random).
    """
    for _ in range(max_tries):
        x = float(rng.uniform(clearance, World.WIDTH  - clearance))
        y = float(rng.uniform(clearance, World.HEIGHT - clearance))
        if all(_pt_seg_dist(x, y, *s[0], *s[1]) >= clearance
               for s in world.get_all_segments()):
            return x, y
    raise RuntimeError(f"sample_valid_pos: no valid position in {max_tries} tries")


# ── lidar ─────────────────────────────────────────────────────────────────────

def raycast_lidar(robot, world, n_rays=36, max_range=5.0):
    """Cast n_rays evenly around the robot; return float32 range array."""
    segs   = world.get_all_segments()
    ranges = np.full(n_rays, max_range, dtype=np.float32)
    for i in range(n_rays):
        angle    = robot.theta + 2.0 * math.pi * i / n_rays
        rdx, rdy = math.cos(angle), math.sin(angle)
        for seg in segs:
            t = _ray_seg_hit(robot.x, robot.y, rdx, rdy, *seg[0], *seg[1])
            if t is not None and t < ranges[i]:
                ranges[i] = t
    return ranges


# ── rendering helpers (used by later stages too) ──────────────────────────────

def w2s(wx, wy):
    """World coordinates (meters, y-up) → screen pixel coordinates (y-down)."""
    return (
        int(_OFF_X + wx * _SCALE),
        int(_OFF_Y + (World.HEIGHT - wy) * _SCALE),
    )


def draw_world(surface, world):
    for seg in world.walls:
        pygame.draw.line(surface, (220, 220, 220), w2s(*seg[0]), w2s(*seg[1]), 2)
    for (x, y, ww, h) in world.desks:
        pts = [w2s(x, y), w2s(x + ww, y), w2s(x + ww, y + h), w2s(x, y + h)]
        pygame.draw.polygon(surface, (70, 50, 30), pts)
        pygame.draw.polygon(surface, (160, 120, 60), pts, 1)


def draw_robot(surface, robot):
    R     = 0.25
    tip   = w2s(robot.x + R * math.cos(robot.theta),
                robot.y + R * math.sin(robot.theta))
    left  = w2s(robot.x + R * math.cos(robot.theta + 2.4),
                robot.y + R * math.sin(robot.theta + 2.4))
    right = w2s(robot.x + R * math.cos(robot.theta - 2.4),
                robot.y + R * math.sin(robot.theta - 2.4))
    pygame.draw.polygon(surface, (220, 80, 80), [tip, left, right])
    pygame.draw.polygon(surface, (255, 200, 200), [tip, left, right], 1)


def draw_lidar(surface, robot, ranges, n_rays=36):
    for i, r in enumerate(ranges):
        angle = robot.theta + 2.0 * math.pi * i / n_rays
        end   = w2s(robot.x + r * math.cos(angle),
                    robot.y + r * math.sin(angle))
        pygame.draw.line(surface, (180, 160, 0), w2s(robot.x, robot.y), end, 1)


# ── synthetic camera ──────────────────────────────────────────────────────────

def render_robot_camera(world, pedestrians, robot, image_size=96):
    """
    Top-down crop centred on the robot, rotated so the robot's heading points up.
    Covers a 3 m x 3 m area. Returns (image_size, image_size, 3) uint8 numpy array.
    """
    crop_m = 3.0
    scale  = image_size / crop_m
    half   = image_size / 2.0

    cos_t = math.cos(robot.theta)
    sin_t = math.sin(robot.theta)

    def _w2i(wx, wy):
        dx, dy = wx - robot.x, wy - robot.y
        fwd = dx * cos_t + dy * sin_t       # forward component  → top of image
        lat = dx * sin_t - dy * cos_t       # lateral component  → right of image
        return (int(half + lat * scale),
                int(half - fwd * scale))

    surf = pygame.Surface((image_size, image_size))
    surf.fill((30, 30, 30))

    # Walls
    for (ax, ay), (bx, by) in world.get_all_segments():
        pygame.draw.line(surf, (200, 200, 200), _w2i(ax, ay), _w2i(bx, by), 2)

    # Desks
    for (x, y, w, h) in world.desks:
        pts = [_w2i(x, y), _w2i(x + w, y), _w2i(x + w, y + h), _w2i(x, y + h)]
        pygame.draw.polygon(surf, (90, 70, 40), pts)
        pygame.draw.polygon(surf, (140, 110, 60), pts, 1)

    # Pedestrians
    ped_r = max(1, int(0.3 * scale))
    for ped in pedestrians:
        pygame.draw.circle(surf, (70, 130, 210), _w2i(ped.x, ped.y), ped_r)

    # Robot: red dot + forward tick
    rx, ry = _w2i(robot.x, robot.y)
    rob_r  = max(2, int(ROBOT_RADIUS * scale))
    pygame.draw.circle(surf, (220, 80, 80), (rx, ry), rob_r)
    fx, fy = _w2i(robot.x + 0.3 * cos_t, robot.y + 0.3 * sin_t)
    pygame.draw.line(surf, (255, 160, 160), (rx, ry), (fx, fy), 2)

    # surfarray returns (W, H, 3); swap to standard (H, W, 3)
    return pygame.surfarray.array3d(surf).swapaxes(0, 1)


def render_goal_image(world, gx, gy, image_size=96):
    """
    Top-down crop centred on (gx, gy) with no pedestrians, facing east (theta=0).
    Returns (image_size, image_size, 3) uint8 numpy array.
    """
    class _Pose:
        pass
    p = _Pose()
    p.x, p.y, p.theta = gx, gy, 0.0
    return render_robot_camera(world, [], p, image_size)


# ── demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pedestrians import spawn_pedestrians, draw_pedestrians
    from costmap import Costmap

    pygame.init()
    screen = pygame.display.set_mode((_WIN_W, _WIN_H))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    world   = World()
    robot   = Robot(x=0.5, y=4.0, theta=0.0, world=world)
    peds    = spawn_pedestrians(world, n=15, seed=42)
    costmap = Costmap()
    costmap.build_static(world)

    show_costmap  = True
    show_camera   = True
    _CAM_PX       = 144          # preview size on screen (pixels)
    _CAM_X        = _WIN_W - _CAM_PX - 8   # top-right corner
    _CAM_Y        = 8

    running = True
    while running:
        dt = min(clock.tick(30) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_c:
                    show_costmap = not show_costmap
                if event.key == pygame.K_v:
                    show_camera = not show_camera

        keys  = pygame.key.get_pressed()
        v     = 1.5 if keys[pygame.K_UP]   else (-0.8 if keys[pygame.K_DOWN]  else 0.0)
        omega = 1.5 if keys[pygame.K_LEFT] else (-1.5 if keys[pygame.K_RIGHT] else 0.0)

        robot.step(v, omega, dt, pedestrians=peds)
        for ped in peds:
            ped.step(world, peds, dt, robot=robot)

        ranges = raycast_lidar(robot, world)
        costmap.update_obstacle(ranges, robot)
        costmap.update_social(peds)

        screen.fill((15, 15, 20))
        draw_world(screen, world)

        if show_costmap:
            costmap.render(screen)

        draw_lidar(screen, robot, ranges)
        draw_pedestrians(screen, peds)
        draw_robot(screen, robot)

        if show_camera:
            cam_arr  = render_robot_camera(world, peds, robot)
            # surfarray expects (W, H, 3) so swap axes back
            cam_surf = pygame.surfarray.make_surface(cam_arr.swapaxes(0, 1))
            cam_surf = pygame.transform.scale(cam_surf, (_CAM_PX, _CAM_PX))
            pygame.draw.rect(screen, (80, 80, 80),
                             (_CAM_X - 2, _CAM_Y - 2, _CAM_PX + 4, _CAM_PX + 4), 1)
            screen.blit(cam_surf, (_CAM_X, _CAM_Y))
            label = font.render("camera (V)", True, (120, 120, 120))
            screen.blit(label, (_CAM_X, _CAM_Y + _CAM_PX + 2))

        hud = font.render(
            f"  pos ({robot.x:5.2f}, {robot.y:5.2f})  "
            f"theta {math.degrees(robot.theta):6.1f} deg  |  "
            f"peds {len(peds)}  |  arrows=drive  C=costmap  V=camera  Q=quit",
            True, (150, 150, 150),
        )
        screen.blit(hud, (0, _WIN_H - 20))

        pygame.display.set_caption(
            f"VLA CrowdSim — Stage 3 | {clock.get_fps():.0f} FPS"
        )
        pygame.display.flip()

    pygame.quit()
    sys.exit()
