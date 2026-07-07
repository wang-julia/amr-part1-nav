# Local VLM Semantic Costmap Layer

This is a polished static demo for a local VLM-style semantic costmap module.
It shows how camera understanding and mocked navigation context become a
40x40 local semantic costmap, then fuse into a 100x100 global demo costmap for
deterministic A* path visualization.

The VLM does not control the robot. It converts visual and navigation context
into semantic costmap updates. A deterministic planner still makes the final
path decision.

## Run

```bash
cd ~/vlm/social_nav_demo
source venv/bin/activate
python generate_demo.py
cd static_demo
python -m http.server 8000
```

Open:

```text
http://localhost:8000
```

## Files

- `generate_demo.py` builds `static_demo/index.html` and image assets.
- `simulator.py` creates the 100x100 base map, runs A*, and computes metrics.
- `mock_context.py` provides replayed robot pose, waypoints, LiDAR sectors, and depth hints.
- `vlm_inference.py` defaults to replay mode and includes an optional Qwen2.5-VL hook.
- `local_costmap.py` converts semantic JSON intent into a 40x40 local costmap.
- Objects are classified as `static`, `possibly_moving`, or `moving` before shaping semantic costs.
- Static objects such as parked bikes add only circular clearance cost.
- Possibly moving objects or spaces get larger/stronger circular caution zones.
- Walking people can add a directional predicted-motion cost zone in their facing direction.
- `fusion.py` fuses local semantic costs into the global map without overwriting obstacles.
- `prompts/qwen_semantic_costmap.txt` contains the strict JSON prompt.
- `vlm_outputs/` contains prepared VLM-style JSON for the four-frame timeline.
- `frames/` contains generated synthetic camera frames after running the generator.
- `static_demo/` contains the presentation-ready dashboard.

## Dependencies

The demo uses lightweight Python packages:

- numpy
- matplotlib
- pillow
- json, pathlib, and standard library modules

No Streamlit, torch, transformers, or Qwen installation is required for the
static replay demo.

## Optional Live Qwen Mode

Replay mode is still the default. To run live Qwen2.5-VL inference, install the
Qwen runtime packages into the environment that runs `generate_demo.py`, then run:

```bash
python generate_demo.py --vlm-mode qwen
```

Use a local model path or another model id with:

```bash
python generate_demo.py --vlm-mode qwen --qwen-model /path/to/Qwen2.5-VL-3B-Instruct
```

Qwen mode expects:

- torch
- transformers
- accelerate
- qwen-vl-utils

If Qwen mode fails and fallback is enabled, the demo uses replay JSON and writes
the attempted outputs/status to `qwen_outputs/`. Add `--no-qwen-fallback` if you
want missing packages or model-loading errors to fail loudly.
