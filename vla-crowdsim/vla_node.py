"""
vla_node.py — Stage 4
VLA interface: stub (always works) + optional real ViNT/NoMaD.

Stub:       works out of the box — no extra dependencies.
Real model: requires cloning https://github.com/robodhruv/visualnav-transformer
            installing it (cd visualnav-transformer/train && pip install -e .)
            and downloading a NoMaD or ViNT checkpoint.
"""

import math
import os
import sys

import numpy as np

# ── stub (primary interface) ──────────────────────────────────────────────────

def get_subgoal_stub(robot_x, robot_y, robot_theta, goal_wx, goal_wy):
    """
    Return (dx, dy) in the robot's local frame pointing toward (goal_wx, goal_wy),
    clipped to at most 1.0 m so the controller gets a reachable short-horizon target.
    """
    dx_w = goal_wx - robot_x
    dy_w = goal_wy - robot_y
    dist = math.hypot(dx_w, dy_w)
    if dist < 1e-6:
        return 0.0, 0.0

    scale = min(1.0 / dist, 1.0)      # clip to 1 m
    cos_t = math.cos(robot_theta)
    sin_t = math.sin(robot_theta)

    # Rotate world-frame delta into robot frame
    dx_r =  cos_t * dx_w + sin_t * dy_w   # forward component
    dy_r = -sin_t * dx_w + cos_t * dy_w   # lateral component

    return dx_r * scale, dy_r * scale


def get_subgoal_nav(robot_x, robot_y, robot_theta, goal_wx, goal_wy, costmap,
                    reach=1.0, n_dirs=13, spread=math.radians(60)):
    """
    Direction-scan waypoint: sample n_dirs directions within ±spread of the
    goal bearing, score each by average costmap cost + angular deviation,
    return (dx_r, dy_r) in robot frame for the best direction.

    Avoids placing the waypoint inside walls or desks by scoring the full
    path to the candidate point, not just the endpoint.
    """
    dist_to_goal = math.hypot(goal_wx - robot_x, goal_wy - robot_y)
    if dist_to_goal < 1e-6:
        return 0.0, 0.0

    goal_bearing = math.atan2(goal_wy - robot_y, goal_wx - robot_x)
    r            = min(reach, dist_to_goal)   # don't overshoot a close goal

    best_score = float('inf')
    best_angle = goal_bearing

    for i in range(n_dirs):
        t     = i / (n_dirs - 1)                          # 0 → 1
        angle = goal_bearing + (t - 0.5) * 2.0 * spread  # bearing ± spread

        # Average costmap cost along 4 evenly-spaced samples toward 1 m
        cost_sum = sum(
            costmap.cost_at(robot_x + s * r * math.cos(angle),
                            robot_y + s * r * math.sin(angle))
            for s in (0.25, 0.5, 0.75, 1.0)
        )
        avg_cost  = cost_sum / 4.0
        angle_dev = abs(angle - goal_bearing)   # 0 at centre, ≤ spread at edges

        score = avg_cost * 2.0 + angle_dev * 0.5
        if score < best_score:
            best_score = score
            best_angle = angle

    # Convert winning world-frame direction → robot frame
    dx_w  =  r * math.cos(best_angle)
    dy_w  =  r * math.sin(best_angle)
    cos_t =  math.cos(robot_theta)
    sin_t =  math.sin(robot_theta)
    return (cos_t * dx_w + sin_t * dy_w,
           -sin_t * dx_w + cos_t * dy_w)


# ── real model (optional) ─────────────────────────────────────────────────────
# load_model and get_subgoal are set to None if the repo / checkpoint isn't found.
# The rest of the pipeline always calls get_subgoal_stub as a fallback.

load_model  = None
get_subgoal = None

_VNT_TRAIN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "visualnav-transformer", "train",
)

try:
    import torch
    import torchvision.transforms.functional as _TF
    from PIL import Image as _PIL
    import yaml as _yaml

    if not os.path.isdir(_VNT_TRAIN):
        raise ImportError(
            f"visualnav-transformer repo not found at "
            f"{os.path.dirname(_VNT_TRAIN)}\n"
            f"  Clone it: git clone https://github.com/robodhruv/visualnav-transformer\n"
            f"  Install:  cd visualnav-transformer/train && pip install -e ."
        )

    if _VNT_TRAIN not in sys.path:
        sys.path.insert(0, _VNT_TRAIN)

    from vint_train.models.nomad.nomad import NoMaD, DenseNetwork  # noqa: E402

    def load_model(checkpoint_path, model_type="nomad"):
        """
        Load a NoMaD checkpoint.
        checkpoint_path: path to a .pth file, or a directory containing
                         config.yaml + latest.pth.
        Returns (model, device).
        """
        if os.path.isdir(checkpoint_path):
            cfg_path  = os.path.join(checkpoint_path, "config.yaml")
            ckpt_file = os.path.join(checkpoint_path, "latest.pth")
        else:
            cfg_path  = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
            ckpt_file = checkpoint_path

        with open(cfg_path) as f:
            cfg = _yaml.safe_load(f)

        vision_encoder = DenseNetwork(
            obs_encoding_size      = cfg.get("obs_encoding_size", 1024),
            context_size           = cfg.get("context_size", 5),
            mha_num_attention_heads  = cfg.get("mha_num_attention_heads", 4),
            mha_num_attention_layers = cfg.get("mha_num_attention_layers", 4),
            mha_ff_dim_factor        = cfg.get("mha_ff_dim_factor", 4),
        )
        model = NoMaD(
            vision_encoder     = vision_encoder,
            noise_scheduler    = None,
            num_diffusion_iters = cfg.get("num_diffusion_iters", 10),
        )

        raw = torch.load(ckpt_file, map_location="cpu")
        state = raw.get("model_state_dict", raw.get("state_dict", raw))
        model.load_state_dict(state, strict=False)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[vla_node] Model loaded on {device}")
        return model.to(device), device

    def get_subgoal(obs_img, goal_img, model_and_device):
        """
        Run a no_grad forward pass through the loaded model.

        obs_img, goal_img: (H, W, 3) uint8 numpy arrays (96×96 recommended).
        Returns (dx, dy) in the robot's local frame in metres —
        the first predicted waypoint (NoMaD convention: dx=forward, dy=lateral).
        """
        model, device = model_and_device

        def _prep(arr):
            img = _PIL.fromarray(arr.astype(np.uint8))
            t   = _TF.to_tensor(img)                     # (3, H, W) float32
            return _TF.resize(t, [96, 96]).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(_prep(obs_img), _prep(goal_img))

        # NoMaD returns (waypoints, dist_pred); waypoints: (B, N, 2)
        waypoints = out[0] if isinstance(out, (tuple, list)) else out
        wp = waypoints[0, 0, :].cpu().numpy()   # first batch, first waypoint
        return float(wp[0]), float(wp[1])

    print("[vla_node] ViNT/NoMaD imports OK — call load_model(checkpoint_path) to activate.")

except Exception as _e:
    print(f"[vla_node] Real model not available: {_e}")
    print( "[vla_node] Stub is active. Pipeline works end-to-end without a checkpoint.")


# ── test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to NoMaD checkpoint dir or .pth file")
    args = parser.parse_args()

    # Need pygame for render_robot_camera; use dummy driver if no display
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    pygame.init()

    from world import World, Robot, render_robot_camera, render_goal_image

    world = World()
    robot = Robot(x=1.0, y=4.0, theta=0.0, world=world)

    obs_img  = render_robot_camera(world, [], robot)
    goal_img = render_goal_image(world, 11.0, 4.0)

    print(f"\n--- Rendered images ---")
    print(f"  obs_img  shape={obs_img.shape}  dtype={obs_img.dtype}  "
          f"min={obs_img.min()}  max={obs_img.max()}")
    print(f"  goal_img shape={goal_img.shape}  dtype={goal_img.dtype}  "
          f"min={goal_img.min()}  max={goal_img.max()}")

    print(f"\n--- Stub output ---")
    dx, dy = get_subgoal_stub(robot.x, robot.y, robot.theta, 11.0, 4.0)
    print(f"  get_subgoal_stub → dx={dx:.4f} m (fwd)  dy={dy:.4f} m (lat)")

    if args.model and load_model is not None:
        print(f"\n--- Real model ({args.model}) ---")
        try:
            mdl = load_model(args.model)
            dx, dy = get_subgoal(obs_img, goal_img, mdl)
            print(f"  get_subgoal → dx={dx:.4f} m (fwd)  dy={dy:.4f} m (lat)")
        except Exception as e:
            print(f"  ERROR: {e}")
    elif args.model:
        print("\n--- Real model ---")
        print("  torch/vint_train not importable — check installation.")
    else:
        print("\n  (pass --model <path> to test the real checkpoint)")

    pygame.quit()
