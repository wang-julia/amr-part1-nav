#!/usr/bin/env python3
"""Generate static_demo/assets/path_data.js with path arrays for every trial/frame combo.

Run from social_nav_demo/:
    python gen_path_data.py

Output: static_demo/assets/path_data.js
  window.PATH_DATA["frame_001_trial_00"] = {
    baseline: [[x,y], ...],
    enhanced: [[x,y], ...],
    start: [x, y],
    goal:  [x, y],
  }
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np

from local_costmap import semantic_json_to_local_costmap
from mock_context import get_mock_context, get_timeline
from simulator import astar, build_demo_result

OUT = ROOT / "static_demo" / "assets" / "path_data.js"

# trial_00 in the HTML asset names = vlm_outputs (the original curated JSONs).
# trial_01 and trial_02 = the saved Qwen outputs in trial_outputs/.
TRIAL_MAP = {
    "trial_00": ROOT / "vlm_outputs",
    "trial_01": ROOT / "trial_outputs" / "trial_01",
    "trial_02": ROOT / "trial_outputs" / "trial_02",
}

_HEADING_DIRS = {
    "north":      ((0, -1), (1,  0)),
    "north-east": ((1, -1), (1,  1)),
    "east":       ((1,  0), (0,  1)),
    "south-east": ((1,  1), (-1, 1)),
    "south":      ((0,  1), (-1, 0)),
    "south-west": ((-1, 1), (-1,-1)),
    "west":       ((-1, 0), (0, -1)),
    "north-west": ((-1,-1), (1, -1)),
}


def _fuse_with_heading(base: np.ndarray, local_cm: np.ndarray, robot_pose: dict) -> np.ndarray:
    heading = robot_pose.get("heading", "north")
    (fwd_dx, fwd_dy), (right_dx, right_dy) = _HEADING_DIRS.get(heading, _HEADING_DIRS["north"])
    fwd_len   = (fwd_dx**2   + fwd_dy**2)   ** 0.5
    right_len = (right_dx**2 + right_dy**2) ** 0.5
    fwd_dx   /= fwd_len;   fwd_dy   /= fwd_len
    right_dx /= right_len; right_dy /= right_len

    robot_x, robot_y = int(robot_pose["x"]), int(robot_pose["y"])
    sz = local_cm.shape[0]
    rlx, rly = sz // 2, int(sz * 0.9)

    fused = base.copy()
    h, w = fused.shape
    for ly in range(sz):
        for lx in range(sz):
            cost = int(local_cm[ly, lx])
            if cost <= 0:
                continue
            gx = int(round(robot_x + (rly - ly) * fwd_dx + (lx - rlx) * right_dx))
            gy = int(round(robot_y + (rly - ly) * fwd_dy + (lx - rlx) * right_dy))
            if not (0 <= gx < w and 0 <= gy < h):
                continue
            if int(fused[gy, gx]) >= 255:
                continue
            fused[gy, gx] = max(int(fused[gy, gx]), cost)
    return fused


def compute_paths(sem: dict, ctx: dict):
    res = build_demo_result(sem, ctx)
    heading = ctx["robot_pose"].get("heading", "north")
    if heading not in ("north", "north-east"):
        local_cm = semantic_json_to_local_costmap(sem)
        enhanced = _fuse_with_heading(res["base_costmap"], local_cm, ctx["robot_pose"])
        baseline_path = astar(res["base_costmap"], res["start"], res["goal"])
        enhanced_path = astar(enhanced, res["start"], res["goal"])
    else:
        baseline_path = res["baseline_path"]
        enhanced_path = res["enhanced_path"]
    return {
        "baseline": [list(p) for p in baseline_path],
        "enhanced": [list(p) for p in enhanced_path],
        "start": list(res["start"]),
        "goal":  list(res["goal"]),
    }


def main():
    data = {}
    for item in get_timeline():
        fid = item["frame_id"]
        ctx = get_mock_context(fid)
        for trial_key, trial_dir in TRIAL_MAP.items():
            json_path = trial_dir / f"{fid}.json"
            if not json_path.exists():
                print(f"  SKIP missing: {json_path}")
                continue
            with json_path.open(encoding="utf-8") as f:
                sem = json.load(f)
            key = f"{fid}_{trial_key}"
            paths = compute_paths(sem, ctx)
            data[key] = paths
            print(f"  {key}: baseline {len(paths['baseline'])} pts, enhanced {len(paths['enhanced'])} pts")

    js_lines = ["window.PATH_DATA = " + json.dumps(data, separators=(",", ":")) + ";"]
    OUT.write_text("\n".join(js_lines), encoding="utf-8")
    size_kb = OUT.stat().st_size // 1024
    print(f"\nWritten: {OUT}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
