"""
metrics.py — Stage 6
10 Hz CSV logger: sim time, pose, velocity, ped proximity, freeze/PSV flags.
"""

import csv
import math
import os

_LOG_HZ       = 10
_FREEZE_SPEED = 0.05   # m/s; below this counts as frozen
_PERSONAL_D   = 0.5    # m; closer than this = personal-space violation


class MetricsLogger:
    def __init__(self, path=None):
        """path=None → collect stats in memory only (no CSV written)."""
        self._writer = None
        self._file   = None
        if path:
            dirn = os.path.dirname(os.path.abspath(path))
            os.makedirs(dirn, exist_ok=True)
            self._file   = open(path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(
                ["t", "x", "y", "theta", "v", "nearest_ped", "freeze", "ps_viol"]
            )

        self._t_next        = 0.0
        self.path_length    = 0.0
        self.freeze_count   = 0
        self.ps_violations  = 0
        self._freeze_active = False
        self._prev_x        = None
        self._prev_y        = None

    def update(self, t, robot, peds, v):
        """Call every sim frame. Logs to CSV at _LOG_HZ; always updates stats."""
        if self._prev_x is not None:
            self.path_length += math.hypot(robot.x - self._prev_x,
                                           robot.y - self._prev_y)
        self._prev_x, self._prev_y = robot.x, robot.y

        nearest = min(
            (math.hypot(robot.x - p.x, robot.y - p.y) for p in peds),
            default=float("inf"),
        )

        frozen = abs(v) < _FREEZE_SPEED
        if frozen and not self._freeze_active:
            self.freeze_count += 1
        self._freeze_active = frozen

        ps = nearest < _PERSONAL_D
        if ps:
            self.ps_violations += 1

        if t >= self._t_next and self._writer is not None:
            self._writer.writerow([
                f"{t:.3f}",
                f"{robot.x:.4f}", f"{robot.y:.4f}", f"{robot.theta:.4f}",
                f"{v:.4f}", f"{nearest:.4f}",
                int(frozen), int(ps),
            ])
            self._t_next = t + 1.0 / _LOG_HZ

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()

    def summary(self):
        return {
            "path_length_m": round(self.path_length, 3),
            "freeze_count":  self.freeze_count,
            "ps_violations": self.ps_violations,
        }
