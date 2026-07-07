"""
pedestrians.py — Stage 2
Social force model pedestrian simulation.
"""

import math

import numpy as np
import pygame

from world import w2s, World, ROBOT_RADIUS

# Social force parameters (Helbing 1995)
_A_PED  = 1.5   # pedestrian repulsion strength
_B_PED  = 0.4   # pedestrian repulsion range (m)
_A_WALL = 2.0   # wall repulsion strength
_B_WALL = 0.25  # wall repulsion range (m)
_TAU    = 0.5   # velocity relaxation time (s)
_R_PED  = 2.0   # pedestrian interaction radius (m)
_R_WALL = 1.0   # wall interaction radius (m)
_MAX_F  = 5.0   # force clamp — prevents explosions

_MARGIN    = 0.5   # min distance from room edge when spawning
_CLEARANCE = 0.6   # min distance from desk edges when spawning
_GOAL_R    = 0.4   # goal-reached threshold (m)

_rng = np.random.default_rng()   # module-level rng for mid-run goal reassignment


class Pedestrian:
    def __init__(self, x, y, gx, gy, speed=1.0):
        self.x      = x
        self.y      = y
        self.vx     = 0.0
        self.vy     = 0.0
        self.gx     = gx
        self.gy     = gy
        self.speed  = speed
        self.radius = 0.3

    def step(self, world, others, dt, robot=None):
        # Re-assign goal when current one is reached
        if math.hypot(self.gx - self.x, self.gy - self.y) < _GOAL_R:
            self.gx, self.gy = _random_valid_pos(world)

        # Desired velocity direction
        gd = math.hypot(self.gx - self.x, self.gy - self.y)
        if gd > 1e-6:
            ex, ey = (self.gx - self.x) / gd, (self.gy - self.y) / gd
        else:
            ex, ey = 0.0, 0.0

        # Driving force: relax current velocity toward desired velocity
        fx = (self.speed * ex - self.vx) / _TAU
        fy = (self.speed * ey - self.vy) / _TAU

        # Pedestrian repulsion
        for other in others:
            if other is self:
                continue
            ddx = self.x - other.x
            ddy = self.y - other.y
            d   = math.hypot(ddx, ddy)
            if 1e-6 < d < _R_PED:
                strength = _A_PED * math.exp((self.radius + other.radius - d) / _B_PED)
                fx += strength * ddx / d
                fy += strength * ddy / d

        # Wall repulsion: find nearest point on each segment
        for (ax, ay), (bx, by) in world.get_all_segments():
            sdx, sdy = bx - ax, by - ay
            len_sq   = sdx * sdx + sdy * sdy
            if len_sq < 1e-12:
                continue
            t   = max(0.0, min(1.0, ((self.x - ax) * sdx + (self.y - ay) * sdy) / len_sq))
            npx = ax + t * sdx
            npy = ay + t * sdy
            ddx = self.x - npx
            ddy = self.y - npy
            d   = math.hypot(ddx, ddy)
            if 1e-6 < d < _R_WALL:
                strength = _A_WALL * math.exp((self.radius - d) / _B_WALL)
                fx += strength * ddx / d
                fy += strength * ddy / d

        # Robot repulsion — robot won't yield, so use double strength
        if robot is not None:
            ddx = self.x - robot.x
            ddy = self.y - robot.y
            d   = math.hypot(ddx, ddy)
            if 1e-6 < d < _R_PED:
                strength = _A_PED * 2.0 * math.exp((self.radius + ROBOT_RADIUS - d) / _B_PED)
                fx += strength * ddx / d
                fy += strength * ddy / d

        # Clamp forces, integrate velocity
        self.vx += max(-_MAX_F, min(_MAX_F, fx)) * dt
        self.vy += max(-_MAX_F, min(_MAX_F, fy)) * dt

        # Clamp speed
        spd = math.hypot(self.vx, self.vy)
        if spd > self.speed * 1.5:
            self.vx = self.vx / spd * self.speed * 1.5
            self.vy = self.vy / spd * self.speed * 1.5

        # Move — hard-clamp to room boundary as a safety net
        self.x = max(self.radius, min(world.WIDTH  - self.radius, self.x + self.vx * dt))
        self.y = max(self.radius, min(world.HEIGHT - self.radius, self.y + self.vy * dt))


def spawn_pedestrians(world, n=15, seed=None):
    """Place n pedestrians at random valid positions with random goals."""
    global _rng
    _rng = np.random.default_rng(seed)
    peds = []
    for _ in range(n):
        x, y   = _random_valid_pos(world)
        gx, gy = _random_valid_pos(world)
        speed  = float(_rng.uniform(0.3, 0.5))
        peds.append(Pedestrian(x, y, gx, gy, speed=speed))
    return peds


# ── helpers ───────────────────────────────────────────────────────────────────

def _random_valid_pos(world):
    """Sample a random position not inside or adjacent to any desk."""
    for _ in range(2000):
        x = float(_rng.uniform(_MARGIN, world.WIDTH  - _MARGIN))
        y = float(_rng.uniform(_MARGIN, world.HEIGHT - _MARGIN))
        if _is_clear(x, y, world):
            return x, y
    raise RuntimeError("Could not find a valid position — room too crowded?")


def _is_clear(x, y, world):
    for (dx, dy, dw, dh) in world.desks:
        if (dx - _CLEARANCE < x < dx + dw + _CLEARANCE and
                dy - _CLEARANCE < y < dy + dh + _CLEARANCE):
            return False
    return True


# ── rendering ─────────────────────────────────────────────────────────────────

def draw_pedestrians(surface, pedestrians):
    for ped in pedestrians:
        sx, sy = w2s(ped.x, ped.y)
        gx, gy = w2s(ped.gx, ped.gy)
        pygame.draw.line(surface, (30, 60, 120), (sx, sy), (gx, gy), 1)
        pygame.draw.circle(surface, (70, 130, 210), (sx, sy), 8)
        pygame.draw.circle(surface, (160, 210, 255), (sx, sy), 8, 1)
