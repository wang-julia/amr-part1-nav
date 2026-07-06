import argparse
import html
import json
import os
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from mock_context import get_mock_context, get_timeline, write_timeline_file
from simulator import build_demo_result
from vlm_inference import infer_semantic_json


ROOT = Path(__file__).parent
FRAMES_DIR = ROOT / "frames"
STATIC_DIR = ROOT / "static_demo"
ASSETS_DIR = STATIC_DIR / "assets"
SCENARIO_PATH = ROOT / "scenarios" / "office_hallway_timeline.json"
PROMPT_PATH = ROOT / "prompts" / "qwen_semantic_costmap.txt"
QWEN_OUTPUT_DIR = ROOT / "qwen_outputs"


def ensure_dirs() -> None:
    for directory in [FRAMES_DIR, ASSETS_DIR, ROOT / "vlm_outputs", ROOT / "prompts", QWEN_OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the VLM semantic costmap static demo.")
    parser.add_argument(
        "--vlm-mode",
        choices=["replay", "qwen"],
        default=os.environ.get("VLM_MODE", "replay"),
        help="Use prepared JSON replay or run local Qwen2.5-VL inference.",
    )
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"),
        help="Hugging Face model ID or local path for Qwen2.5-VL.",
    )
    parser.add_argument(
        "--no-qwen-fallback",
        action="store_true",
        help="Fail instead of falling back to replay JSON if Qwen inference errors.",
    )
    return parser.parse_args()


def image_direction_delta(direction: str, length: int = 86) -> tuple[int, int]:
    mapping = {
        "forward": (0, -length),
        "backward": (0, length),
        "left": (-length, 0),
        "right": (length, 0),
        "forward-left": (-int(length * 0.72), -int(length * 0.72)),
        "forward-right": (int(length * 0.72), -int(length * 0.72)),
        "backward-left": (-int(length * 0.72), int(length * 0.72)),
        "backward-right": (int(length * 0.72), int(length * 0.72)),
    }
    return mapping.get(direction, (0, -length))


def draw_motion_arrow(draw: ImageDraw.ImageDraw, x: int, y: int, direction: str) -> None:
    dx, dy = image_direction_delta(direction)
    end_x = x + dx
    end_y = y + dy
    draw.line((x, y, end_x, end_y), fill=(94, 234, 212, 230), width=8)
    angle = np.arctan2(dy, dx)
    for offset in (2.45, -2.45):
        head_x = end_x + int(np.cos(angle + offset) * 22)
        head_y = end_y + int(np.sin(angle + offset) * 22)
        draw.line((end_x, end_y, head_x, head_y), fill=(94, 234, 212, 230), width=8)
    draw.rounded_rectangle((end_x - 54, end_y - 33, end_x + 54, end_y - 9), radius=8, fill=(8, 13, 22, 210))
    draw.text((end_x - 43, end_y - 29), "walking", fill=(209, 250, 229, 255))


def draw_synthetic_camera_frame(frame_id: str, semantic_json: dict, output_path: Path) -> None:
    image = Image.new("RGB", (960, 540), "#131923")
    draw = ImageDraw.Draw(image, "RGBA")

    draw.polygon([(0, 540), (330, 260), (630, 260), (960, 540)], fill=(48, 58, 68, 255))
    draw.polygon([(0, 0), (330, 260), (630, 260), (960, 0)], fill=(28, 35, 45, 255))
    draw.polygon([(0, 0), (330, 260), (0, 540)], fill=(34, 43, 54, 255))
    draw.polygon([(960, 0), (630, 260), (960, 540)], fill=(31, 40, 51, 255))
    draw.line([(330, 260), (630, 260)], fill=(125, 151, 168, 150), width=3)

    for i in range(8):
        y = 520 - i * 35
        shrink = i * 23
        draw.line([(80 + shrink, y), (880 - shrink, y)], fill=(255, 255, 255, 25), width=2)

    draw.line([(480, 540), (480, 262)], fill=(120, 205, 255, 80), width=3)
    draw.line([(400, 540), (455, 262)], fill=(120, 205, 255, 35), width=2)
    draw.line([(560, 540), (505, 262)], fill=(120, 205, 255, 35), width=2)

    position_pixels = {
        "top-left": (340, 210),
        "top-center": (480, 210),
        "top-right": (620, 210),
        "center-left": (335, 320),
        "center": (480, 320),
        "center-right": (625, 320),
        "bottom-left": (260, 430),
        "bottom-center": (480, 430),
        "bottom-right": (700, 430),
    }

    for obj in semantic_json.get("objects", []):
        x, y = position_pixels[obj["image_position"]]
        obj_type = obj["type"]
        if obj_type == "person":
            draw.ellipse((x - 24, y - 92, x + 24, y - 44), fill=(240, 210, 160, 255))
            draw.rounded_rectangle((x - 34, y - 45, x + 34, y + 55), radius=20, fill=(53, 139, 255, 255))
            draw.line((x - 18, y + 54, x - 32, y + 105), fill=(30, 36, 46, 255), width=12)
            draw.line((x + 18, y + 54, x + 32, y + 105), fill=(30, 36, 46, 255), width=12)
            draw.ellipse((x - 86, y - 96, x + 86, y + 112), outline=(255, 99, 99, 150), width=6)
            if obj.get("motion_state") == "walking":
                draw_motion_arrow(draw, x, y - 6, obj.get("facing_direction", "forward"))
        elif obj_type == "group":
            for offset, color in [(-48, "#3b82f6"), (0, "#f59e0b"), (48, "#10b981")]:
                draw.ellipse((x + offset - 22, y - 84, x + offset + 22, y - 40), fill=(235, 203, 162, 255))
                draw.rounded_rectangle((x + offset - 30, y - 42, x + offset + 30, y + 48), radius=18, fill=color)
            draw.ellipse((x - 132, y - 110, x + 132, y + 110), outline=(255, 99, 99, 165), width=8)
        elif obj_type == "doorway":
            draw.rectangle((x - 58, y - 120, x + 58, y + 75), outline=(125, 211, 252, 210), width=8)
            draw.rectangle((x - 44, y - 104, x + 44, y + 72), fill=(12, 18, 27, 180))
            draw.ellipse((x - 68, y - 72, x + 68, y + 90), outline=(252, 211, 77, 150), width=5)
        elif obj_type == "blind_corner":
            draw.polygon([(x - 85, y - 70), (x + 70, y - 125), (x + 92, y + 78), (x - 60, y + 95)], fill=(16, 24, 35, 230))
            draw.line([(x - 65, y + 78), (x + 75, y - 110)], fill=(252, 211, 77, 210), width=7)
            draw.ellipse((x - 78, y - 85, x + 88, y + 102), outline=(252, 211, 77, 145), width=5)

    draw.rounded_rectangle((24, 24, 308, 82), radius=12, fill=(8, 13, 22, 185), outline=(94, 234, 212, 120), width=2)
    draw.text((44, 43), frame_id.replace("_", " ").upper(), fill=(222, 247, 255, 255))
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
    image.save(output_path)


def draw_local_costmap(local_costmap: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0b1020")
    ax.imshow(local_costmap, cmap="magma", vmin=0, vmax=245, origin="upper")
    ax.scatter([20], [36], marker="^", s=110, c="#67e8f9", edgecolors="#07111f", linewidths=1.5)
    ax.text(20, 38.5, "robot", color="#e5f6ff", ha="center", va="top", fontsize=8)
    ax.set_title("40x40 local semantic costmap", color="#f8fafc", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_global_costmap(result: dict, output_path: Path) -> None:
    enhanced = result["enhanced_costmap"]
    base = result["base_costmap"]
    obstacle_mask = base == 255

    fig, ax = plt.subplots(figsize=(6.2, 6.2), facecolor="#0b1020")
    ax.imshow(enhanced.astype(float), cmap="magma", vmin=0, vmax=255, origin="upper")
    ax.imshow(np.ma.masked_where(~obstacle_mask, obstacle_mask), cmap="gray_r", alpha=0.95, origin="upper")

    baseline = result["baseline_path"]
    enhanced_path = result["enhanced_path"]
    if baseline:
        ax.plot([p[0] for p in baseline], [p[1] for p in baseline], "--", color="#38bdf8", linewidth=2.4, label="Baseline")
    if enhanced_path:
        ax.plot([p[0] for p in enhanced_path], [p[1] for p in enhanced_path], color="#f8fafc", linewidth=3.1, label="VLM-enhanced")

    sx, sy = result["start"]
    gx, gy = result["goal"]
    ax.scatter([sx], [sy], s=90, c="#22c55e", edgecolors="#04111f", linewidths=1.5, zorder=5)
    ax.scatter([gx], [gy], s=145, marker="*", c="#facc15", edgecolors="#04111f", linewidths=1.2, zorder=5)
    ax.text(sx + 2, sy + 2, "START", color="#bbf7d0", fontsize=8, weight="bold")
    ax.text(gx + 2, gy + 2, "GOAL", color="#fef08a", fontsize=8, weight="bold")
    ax.set_title("Global costmap fusion and A* visualization", color="#f8fafc", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def metric_block(metrics: dict) -> str:
    return f"""
    <dl class="metrics">
      <div><dt>Path length</dt><dd>{metrics['path_length']}</dd></div>
      <div><dt>Average cost</dt><dd>{metrics['average_cost']}</dd></div>
      <div><dt>Social violations</dt><dd>{metrics['social_violations']}</dd></div>
      <div><dt>Risk score</dt><dd>{metrics['risk_score']}/100</dd></div>
    </dl>
    """


def build_html(frame_sections: list[dict], vlm_mode: str) -> str:
    sections_html = []
    for index, section in enumerate(frame_sections, start=1):
        context_text = html.escape(json.dumps(section["context"], indent=2))
        semantic_text = html.escape(json.dumps(section["semantic_json"], indent=2))
        baseline = section["result"]["baseline_metrics"]
        enhanced = section["result"]["enhanced_metrics"]
        delta = baseline["risk_score"] - enhanced["risk_score"]
        demo_name = html.escape(section["title"].split(" - ", 1)[1].title())
        sections_html.append(
            f"""
            <section class="demo-screen" id="{section['frame_id']}">
              <div class="demo-heading">
                <span class="step">0{index}</span>
                <div>
                  <h2>{demo_name}</h2>
                  <p>{html.escape(section['summary'])}</p>
                </div>
                <div class="risk-chip">Risk {baseline['risk_score']} -> {enhanced['risk_score']}</div>
              </div>
              <div class="demo-grid">
                <div class="visual-stack">
                  <article class="panel camera-panel">
                    <h3>Robot Camera Frame</h3>
                    <img src="assets/{section['camera_asset']}" alt="{html.escape(section['title'])} synthetic camera frame" />
                  </article>
                  <article class="panel local-panel">
                    <h3>40x40 Local Semantic Costmap</h3>
                    <img src="assets/{section['local_asset']}" alt="Local semantic costmap heatmap" />
                  </article>
                </div>
                <article class="panel map-panel">
                  <h3>Global Costmap with Baseline and Enhanced Paths</h3>
                  <img src="assets/{section['global_asset']}" alt="Global costmap path comparison" />
                </article>
                <aside class="panel insight-panel">
                  <h3>Metrics Comparison</h3>
                  <div class="metric-columns">
                    <div>
                      <h4>Baseline Planner</h4>
                      {metric_block(section['result']['baseline_metrics'])}
                    </div>
                    <div>
                      <h4>VLM-Enhanced Planner</h4>
                      {metric_block(section['result']['enhanced_metrics'])}
                    </div>
                  </div>
                  <p class="summary">Risk reduction: {delta} points. Slightly longer path, lower semantic risk.</p>
                  <div class="code-pair">
                    <div>
                      <h3>VLM Semantic JSON</h3>
                      <pre>{semantic_text}</pre>
                    </div>
                    <div>
                      <h3>Mock Navigation Context</h3>
                      <pre>{context_text}</pre>
                    </div>
                  </div>
                </aside>
              </div>
            </section>
            """
        )

    timeline_links = "".join(
        f"<a class='nav-button' href='#{section['frame_id']}'><span>{section['frame_id'][-3:]}</span>{html.escape(section['title'].split(' - ', 1)[1])}</a>"
        for section in frame_sections
    )

    return dedent(
        f"""
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>Local VLM Semantic Costmap Layer</title>
          <style>
            :root {{
              color-scheme: dark;
              --bg: #070b14;
              --surface: #0e1626;
              --surface-2: #121d31;
              --line: #26364d;
              --text: #e6edf7;
              --muted: #9fb0c5;
              --cyan: #67e8f9;
              --green: #5eead4;
              --amber: #facc15;
            }}
            * {{ box-sizing: border-box; }}
            html {{ scroll-behavior: smooth; scroll-snap-type: y proximity; }}
            body {{
              margin: 0;
              background:
                radial-gradient(circle at 18% 0%, rgba(56, 189, 248, 0.16), transparent 28%),
                linear-gradient(180deg, #08111f 0%, var(--bg) 42%, #050812 100%);
              color: var(--text);
              font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              line-height: 1.5;
            }}
            main {{ width: min(1560px, calc(100% - 32px)); margin: 0 auto; padding: 20px 0 46px; }}
            .hero {{
              min-height: 92vh;
              display: grid;
              align-items: center;
              padding: 34px 0 28px;
              border-bottom: 1px solid rgba(148, 163, 184, 0.22);
              scroll-snap-align: start;
            }}
            .hero-grid {{ display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.72fr); gap: 28px; align-items: center; }}
            .eyebrow {{ color: var(--green); text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.78rem; font-weight: 700; }}
            h1 {{ margin: 12px 0 10px; font-size: clamp(2.5rem, 5.6vw, 5.9rem); line-height: 0.94; letter-spacing: 0; max-width: 980px; }}
            .subtitle {{ color: #cbd5e1; font-size: clamp(1.05rem, 2vw, 1.45rem); max-width: 830px; margin: 0 0 28px; }}
            .principle {{
              display: inline-block;
              max-width: 980px;
              padding: 14px 16px;
              background: rgba(14, 22, 38, 0.86);
              border: 1px solid rgba(103, 232, 249, 0.32);
              border-radius: 8px;
              color: #dff8ff;
              font-weight: 650;
            }}
            .overview-panel {{
              background: rgba(14, 22, 38, 0.82);
              border: 1px solid rgba(148, 163, 184, 0.22);
              border-radius: 8px;
              padding: 18px;
            }}
            .overview-panel h2 {{ font-size: 1.1rem; margin-bottom: 10px; color: #dbeafe; }}
            .overview-list {{ display: grid; gap: 10px; margin: 0; padding: 0; list-style: none; }}
            .overview-list li {{ padding: 10px 12px; background: rgba(18, 29, 49, 0.72); border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 8px; color: #d7e2f0; }}
            .architecture {{
              margin: 22px 0 18px;
              display: grid;
              grid-template-columns: repeat(3, minmax(0, 1fr));
              gap: 10px;
            }}
            .arch-node {{
              min-height: 66px;
              display: grid;
              place-items: center;
              text-align: center;
              padding: 12px;
              border: 1px solid var(--line);
              border-radius: 8px;
              background: linear-gradient(180deg, rgba(18, 29, 49, 0.95), rgba(10, 17, 30, 0.92));
              color: #eaf6ff;
              font-weight: 700;
            }}
            .timeline {{
              display: grid;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 10px;
              margin: 18px 0 0;
            }}
            .demo-nav {{
              position: sticky;
              top: 0;
              z-index: 10;
              padding: 8px 0;
              background: linear-gradient(180deg, rgba(7, 11, 20, 0.96), rgba(7, 11, 20, 0.78));
              backdrop-filter: blur(14px);
            }}
            .demo-nav .architecture {{ display: none; }}
            .demo-nav .timeline {{ margin: 0; }}
            .nav-button {{
              text-decoration: none;
              color: var(--text);
              background: var(--surface);
              border: 1px solid var(--line);
              border-radius: 8px;
              padding: 12px;
              display: flex;
              gap: 10px;
              align-items: center;
              min-height: 58px;
              transition: border-color 140ms ease, transform 140ms ease, background 140ms ease;
            }}
            .nav-button:hover {{ border-color: rgba(103, 232, 249, 0.58); background: #122035; transform: translateY(-1px); }}
            .nav-button span {{
              width: 34px;
              height: 34px;
              border-radius: 50%;
              display: grid;
              place-items: center;
              background: rgba(103, 232, 249, 0.12);
              color: var(--cyan);
              font-weight: 800;
              flex: 0 0 auto;
            }}
            .demo-screen {{
              min-height: calc(100vh - 76px);
              margin: 14px 0 28px;
              padding: 14px;
              border: 1px solid rgba(148, 163, 184, 0.22);
              border-radius: 8px;
              background: rgba(8, 13, 24, 0.72);
              scroll-snap-align: start;
              scroll-margin-top: 76px;
            }}
            .demo-heading {{ display: flex; align-items: flex-start; gap: 14px; margin-bottom: 12px; }}
            .step {{ color: var(--amber); font-size: 1.9rem; font-weight: 900; font-variant-numeric: tabular-nums; }}
            h2, h3, h4 {{ margin: 0; letter-spacing: 0; }}
            h2 {{ font-size: clamp(1.25rem, 2.2vw, 1.72rem); }}
            .demo-heading p {{ margin: 3px 0 0; color: var(--muted); }}
            .risk-chip {{ margin-left: auto; white-space: nowrap; padding: 8px 10px; border-radius: 999px; color: #fef3c7; border: 1px solid rgba(250, 204, 21, 0.32); background: rgba(250, 204, 21, 0.08); font-weight: 800; }}
            .demo-grid {{
              display: grid;
              grid-template-columns: minmax(220px, 0.86fr) minmax(420px, 1.42fr) minmax(360px, 1fr);
              gap: 12px;
              align-items: stretch;
              height: calc(100vh - 190px);
              min-height: 520px;
            }}
            .visual-stack {{ display: grid; grid-template-rows: minmax(0, 0.68fr) minmax(0, 0.62fr); gap: 12px; min-height: 0; }}
            .camera-panel, .local-panel, .map-panel, .insight-panel {{ min-height: 0; overflow: hidden; }}
            .camera-panel img, .local-panel img, .map-panel img {{
              width: 100%;
              height: calc(100% - 32px);
              object-fit: contain;
            }}
            .panel {{
              background: var(--surface);
              border: 1px solid var(--line);
              border-radius: 8px;
              padding: 14px;
              min-width: 0;
            }}
            .panel.wide {{ grid-column: span 2; }}
            .panel h3 {{ font-size: 0.95rem; color: #dbeafe; margin-bottom: 10px; }}
            .panel img {{
              width: 100%;
              display: block;
              border-radius: 6px;
              border: 1px solid rgba(148, 163, 184, 0.18);
              background: #030712;
            }}
            .map-panel h3 {{ margin-bottom: 8px; }}
            .insight-panel {{ display: flex; flex-direction: column; gap: 10px; }}
            pre {{
              margin: 0;
              max-height: 100%;
              overflow: auto;
              white-space: pre-wrap;
              word-break: break-word;
              color: #d6e4ff;
              font: 0.72rem/1.34 "SFMono-Regular", Consolas, "Liberation Mono", monospace;
            }}
            .metric-columns {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
            .metric-columns h4 {{ color: var(--green); font-size: 0.86rem; margin-bottom: 8px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; margin: 0; }}
            .metrics div {{ padding: 8px; background: var(--surface-2); border-radius: 6px; border: 1px solid rgba(148, 163, 184, 0.14); }}
            dt {{ color: var(--muted); font-size: 0.68rem; }}
            dd {{ margin: 1px 0 0; color: #f8fafc; font-size: 0.98rem; font-weight: 800; }}
            .summary {{ margin: 0; color: #fef3c7; font-weight: 750; }}
            .code-pair {{ display: grid; grid-template-rows: minmax(0, 1.25fr) minmax(0, 0.85fr); gap: 10px; min-height: 0; flex: 1; }}
            .code-pair > div {{ min-height: 0; padding-top: 8px; border-top: 1px solid rgba(148, 163, 184, 0.18); overflow: hidden; display: flex; flex-direction: column; }}
            .code-pair h3 {{ font-size: 0.82rem; margin-bottom: 7px; }}
            footer {{ color: var(--muted); padding-top: 24px; border-top: 1px solid rgba(148, 163, 184, 0.18); }}
            @media (max-width: 1180px) {{
              .hero-grid {{ grid-template-columns: 1fr; }}
              .architecture {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
              .demo-grid {{ grid-template-columns: 1fr; height: auto; min-height: 0; }}
              .visual-stack {{ grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: auto; }}
              .camera-panel img, .local-panel img, .map-panel img {{ height: auto; }}
              .demo-screen {{ min-height: auto; }}
            }}
            @media (max-width: 760px) {{
              main {{ width: min(100% - 24px, 760px); }}
              .architecture, .timeline, .metric-columns, .visual-stack {{ grid-template-columns: 1fr; }}
              .panel.wide {{ grid-column: auto; }}
              .hero {{ min-height: auto; }}
              .demo-heading {{ flex-wrap: wrap; }}
              .risk-chip {{ margin-left: 0; }}
            }}
          </style>
        </head>
        <body>
          <main>
            <section class="hero">
              <div class="hero-grid">
                <div>
                  <div class="eyebrow">Qwen2.5-VL-style semantic understanding for safer social robot navigation</div>
                  <h1>Local VLM Semantic Costmap Layer</h1>
                  <p class="subtitle">A local VLM-style module interprets robot camera frames, combines them with mocked navigation context, and contributes semantic costs for safer social navigation.</p>
                  <p class="principle">The VLM does not control the robot. It converts visual and navigation context into semantic costmap updates. A deterministic planner still makes the final path decision.</p>
                </div>
                <aside class="overview-panel">
                  <h2>Project Overview</h2>
                  <ul class="overview-list">
                    <li>Semantic intent becomes a 40x40 local costmap, not a robot command.</li>
                    <li>Walking people produce a circular personal-space zone plus a directional predicted-motion zone.</li>
                    <li>Local semantic costs fuse into a 100x100 global demo costmap while hard obstacles stay fixed.</li>
                    <li>A* visualizes baseline versus VLM-enhanced paths for four office hallway moments.</li>
                  </ul>
                </aside>
              </div>
            </section>

            <section class="demo-nav">
              <div class="architecture" aria-label="Architecture diagram">
                <div class="arch-node">Robot Camera Frame</div>
                <div class="arch-node">Local VLM Semantic JSON</div>
                <div class="arch-node">Mock Waypoints / LiDAR / Depth</div>
                <div class="arch-node">40x40 Local Semantic Costmap</div>
                <div class="arch-node">Global Costmap Fusion</div>
                <div class="arch-node">A* Path Visualization</div>
              </div>
              <nav class="timeline" aria-label="Office hallway navigation buttons">
                {timeline_links}
              </nav>
            </section>

            {''.join(sections_html)}

            <footer>
              Generated by <code>generate_demo.py</code>. Replay mode is the default; real Qwen2.5-VL inference can be connected through <code>vlm_inference.py</code>.
            </footer>
          </main>
        </body>
        </html>
        """
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()
    write_timeline_file(SCENARIO_PATH)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    frame_sections = []
    for item in get_timeline():
        frame_id = item["frame_id"]
        context = get_mock_context(frame_id)
        camera_asset = f"{frame_id}_camera.png"
        local_asset = f"{frame_id}_local_costmap.png"
        global_asset = f"{frame_id}_global_costmap.png"

        seed_semantic_json = infer_semantic_json(frame_id)
        draw_synthetic_camera_frame(frame_id, seed_semantic_json, FRAMES_DIR / f"{frame_id}.png")
        image_path = FRAMES_DIR / f"{frame_id}.png"
        semantic_json = infer_semantic_json(
            frame_id,
            mode=args.vlm_mode,
            image_path=image_path,
            prompt=prompt,
            context=context,
            model_id=args.qwen_model,
            fallback_to_replay=not args.no_qwen_fallback,
            save_output_dir=QWEN_OUTPUT_DIR if args.vlm_mode == "qwen" else None,
        )
        mode_label = args.vlm_mode
        if semantic_json.get("inference_mode") == "replay_fallback_after_qwen_error":
            mode_label = "qwen requested, replay fallback used"
        result = build_demo_result(semantic_json, context)

        draw_synthetic_camera_frame(frame_id, semantic_json, ASSETS_DIR / camera_asset)
        draw_local_costmap(result["local_costmap"], ASSETS_DIR / local_asset)
        draw_global_costmap(result, ASSETS_DIR / global_asset)

        frame_sections.append(
            {
                **item,
                "context": context,
                "semantic_json": semantic_json,
                "result": result,
                "mode_label": mode_label,
                "camera_asset": camera_asset,
                "local_asset": local_asset,
                "global_asset": global_asset,
            }
        )

    html_text = build_html(frame_sections, args.vlm_mode)
    (STATIC_DIR / "index.html").write_text(html_text, encoding="utf-8")
    print(f"Wrote {STATIC_DIR / 'index.html'}")
    print(f"Wrote assets to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
