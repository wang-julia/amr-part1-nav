"""
costmap.py — Stage 3
2D occupancy costmap: static (walls/desks), obstacle (lidar), and social (pedestrians) layers.
"""

import math

import numpy as np
import pygame
from scipy.ndimage import distance_transform_edt

from world import World, w2s

_RES = 0.05                          # metres per cell
_NX  = int(World.WIDTH  / _RES)      # 240 cells wide
_NY  = int(World.HEIGHT / _RES)      # 160 cells tall

# Rendering layout — derived from world.py's SCALE / offsets via w2s
_CELL_PX   = 3                                 # screen pixels per cell (60 px/m × 0.05 m)
_ROOM_PX_W = _NX * _CELL_PX                   # 720 px
_ROOM_PX_H = _NY * _CELL_PX                   # 480 px
_ROOM_X, _ROOM_Y = w2s(0, World.HEIGHT)        # top-left of room on screen = (40, 60)


class Costmap:
    def __init__(self):
        self.static   = np.zeros((_NY, _NX), dtype=np.float32)
        self.obstacle = np.zeros((_NY, _NX), dtype=np.float32)
        self.social   = np.zeros((_NY, _NX), dtype=np.float32)

        # Precomputed coordinate grids (ny, nx) — reused every frame
        gi = np.arange(_NX, dtype=np.float32)
        gj = np.arange(_NY, dtype=np.float32)
        self._GI, self._GJ = np.meshgrid(gi, gj)

        # Pre-allocated render buffer (nx, ny, 3) — avoids per-frame allocation
        self._rgb_buf = np.zeros((_NX, _NY, 3), dtype=np.uint8)
        self._surf    = None   # pygame.Surface created lazily on first render()

    # ── public API ────────────────────────────────────────────────────────────

    def build_static(self, world):
        """Rasterize walls + desks and apply 0.6m inflation decay. Call once."""
        raw = np.zeros((_NY, _NX), dtype=np.float32)
        for (ax, ay), (bx, by) in world.get_all_segments():
            _draw_segment(raw, ax, ay, bx, by)
        for (x, y, w, h) in world.desks:
            _fill_rect(raw, x, y, w, h)

        dist            = distance_transform_edt(raw < 0.5)  # cells to nearest obstacle
        sigma           = 0.2 / _RES                          # e-fold decay at 0.2 m
        inflated        = np.exp(-dist / sigma)
        inflated[raw > 0.5] = 1.0
        self.static     = np.clip(inflated, 0.0, 1.0).astype(np.float32)

    def update_obstacle(self, ranges, robot, n_rays=36, max_range=5.0):
        """Mark lidar hit points and apply 0.1m inflation. Call every frame."""
        raw = np.zeros((_NY, _NX), dtype=np.float32)
        for i, r in enumerate(ranges):
            if r >= max_range - 0.01:
                continue
            angle = robot.theta + 2.0 * math.pi * i / n_rays
            col   = int((robot.x + r * math.cos(angle)) / _RES)
            row   = int((robot.y + r * math.sin(angle)) / _RES)
            if 0 <= row < _NY and 0 <= col < _NX:
                raw[row, col] = 1.0

        dist              = distance_transform_edt(raw < 0.5)
        sigma             = 0.1 / _RES
        inflated          = np.exp(-dist / sigma)
        inflated[raw > 0.5] = 1.0
        self.obstacle     = np.clip(inflated, 0.0, 1.0).astype(np.float32)

    def update_social(self, pedestrians):
        """Add elliptical Gaussian cost blob per pedestrian. Call every frame."""
        self.social[:] = 0.0
        sigma_f = 0.8 / _RES   # forward sigma (16 cells = 0.8 m)
        sigma_r = 0.4 / _RES   # lateral sigma  (8 cells = 0.4 m)

        for ped in pedestrians:
            cx, cy = ped.x / _RES, ped.y / _RES
            spd    = math.hypot(ped.vx, ped.vy)
            if spd > 0.1:
                fx, fy = ped.vx / spd, ped.vy / spd
            else:
                fx, fy = 1.0, 0.0
            rx, ry = -fy, fx   # lateral unit vector

            ddx = self._GI - cx
            ddy = self._GJ - cy
            df  = ddx * fx + ddy * fy   # forward component
            dr  = ddx * rx + ddy * ry   # lateral component

            cost = 0.8 * np.exp(-0.5 * ((df / sigma_f) ** 2 + (dr / sigma_r) ** 2))
            np.maximum(self.social, cost, out=self.social)

    def set_attractor(self, wx, wy, strength=0.3):
        """Subtract a Gaussian at (wx, wy) from the social layer.
        Call after update_social so it isn't cleared next frame.
        """
        cx, cy = wx / _RES, wy / _RES
        sigma  = 1.0 / _RES
        ddx    = self._GI - cx
        ddy    = self._GJ - cy
        self.social -= strength * np.exp(-0.5 * ((ddx / sigma) ** 2 + (ddy / sigma) ** 2))

    def combined(self):
        """Sum all layers, clipped to [0, 1]."""
        return np.clip(self.static + self.obstacle + self.social, 0.0, 1.0)

    def cost_at(self, wx, wy):
        """Return combined cost at world coordinate (wx, wy)."""
        col = max(0, min(_NX - 1, int(wx / _RES)))
        row = max(0, min(_NY - 1, int(wy / _RES)))
        return float(self.combined()[row, col])

    def render(self, surface, alpha=150):
        """Blit heatmap overlay onto surface. Blue = low cost, red = high cost."""
        if self._surf is None:
            self._surf = pygame.Surface((_NX, _NY))

        # Flip y (world y-up → screen y-down), then transpose to (nx, ny)
        c = self.combined()[::-1, :].T

        self._rgb_buf[:, :, 0] = (c * 255).astype(np.uint8)           # R
        self._rgb_buf[:, :, 1] = 0                                      # G
        self._rgb_buf[:, :, 2] = ((1.0 - c) * 255).astype(np.uint8)   # B

        pygame.surfarray.blit_array(self._surf, self._rgb_buf)
        self._surf.set_alpha(alpha)
        scaled = pygame.transform.scale(self._surf, (_ROOM_PX_W, _ROOM_PX_H))
        surface.blit(scaled, (_ROOM_X, _ROOM_Y))


# ── rasterisation helpers ─────────────────────────────────────────────────────

def _draw_segment(grid, ax, ay, bx, by):
    length = math.hypot(bx - ax, by - ay)
    steps  = max(2, int(length / _RES * 2))
    for k in range(steps + 1):
        t   = k / steps
        col = max(0, min(_NX - 1, int((ax + t * (bx - ax)) / _RES)))
        row = max(0, min(_NY - 1, int((ay + t * (by - ay)) / _RES)))
        grid[row, col] = 1.0


def _fill_rect(grid, x, y, w, h):
    col0 = max(0, int(x / _RES))
    col1 = min(_NX, int((x + w) / _RES) + 1)
    row0 = max(0, int(y / _RES))
    row1 = min(_NY, int((y + h) / _RES) + 1)
    grid[row0:row1, col0:col1] = 1.0
