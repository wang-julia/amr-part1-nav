#!/usr/bin/env python3
"""Generate live_demo.html — self-contained animated conference presentation.

Run from the social_nav_demo directory:
    python make_live_demo.py

Writes live_demo.html alongside this script. Open it in any browser.
No server, no internet connection required.
"""

import base64
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from mock_context import get_mock_context, get_timeline
from simulator import build_demo_result

OUT = ROOT / "live_demo.html"
CAMERA_DIR = ROOT / "camera_photos"
TRIAL_DIR  = ROOT / "trial_outputs" / "trial_03"

# Extensions for real camera photos (frame_005 is .png, rest are .jpg)
_EXTS = {"frame_005": ".png"}

def _photo_b64(fid: str) -> str:
    ext = _EXTS.get(fid, ".jpg")
    path = CAMERA_DIR / f"{fid}{ext}"
    return base64.b64encode(path.read_bytes()).decode()

def _load_trial(fid: str) -> dict:
    path = TRIAL_DIR / f"{fid}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def collect_frames() -> list:
    frames = []
    for item in get_timeline():
        fid = item["frame_id"]
        ctx = get_mock_context(fid)
        sem = _load_trial(fid)
        res = build_demo_result(sem, ctx)
        label = item["title"].split(" - ", 1)[1].title() if " - " in item["title"] else item["title"]
        frames.append({
            "frame_id": fid,
            "title": label,
            "summary": item["summary"],
            "camera_b64": _photo_b64(fid),
            "local_costmap": res["local_costmap"].astype(int).tolist(),
            "enhanced_costmap": res["enhanced_costmap"].astype(int).tolist(),
            "base_costmap": res["base_costmap"].astype(int).tolist(),
            "baseline_path": [list(p) for p in res["baseline_path"]],
            "enhanced_path": [list(p) for p in res["enhanced_path"]],
            "start": list(res["start"]),
            "goal": list(res["goal"]),
            "baseline_metrics": res["baseline_metrics"],
            "enhanced_metrics": res["enhanced_metrics"],
            "nav_advice": sem.get("navigation_advice", ""),
            "sem_json": json.dumps(sem, indent=2),
            "objects": sem.get("objects", []),
            "local_costmap_intent": sem.get("local_costmap_intent", []),
            "robot_pose": ctx["robot_pose"],
        })
    return frames


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VLM Social Costmap — Live Demo</title>
<style>
:root {
  --bg: #070b14; --surface: #0e1626; --surface2: #121d31; --line: #26364d;
  --text: #e6edf7; --muted: #9fb0c5;
  --cyan: #67e8f9; --green: #5eead4; --amber: #facc15; --red: #f87171;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; background: var(--bg); color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, sans-serif; overflow: hidden; }
#app { display: grid; grid-template-rows: auto 1fr auto; height: 100vh;
  padding: 10px 14px; gap: 8px; }

/* ── Header ── */
#header { display: flex; align-items: center; gap: 12px; }
#header h1 { font-size: 1rem; color: var(--cyan); letter-spacing: 0.06em;
  text-transform: uppercase; font-weight: 800; }
#dots { display: flex; gap: 7px; margin-left: auto; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--line);
  transition: background 0.3s, transform 0.3s; }
.dot.active { background: var(--cyan); transform: scale(1.3); }
.hbtn { background: transparent; border: 1px solid var(--line); color: var(--muted);
  padding: 3px 12px; border-radius: 6px; cursor: pointer; font-size: 0.78rem; }
.hbtn:hover { border-color: var(--cyan); color: var(--cyan); }

/* ── Main three-column grid ── */
#main { display: grid; grid-template-columns: 1fr 1fr 1.55fr; gap: 10px; min-height: 0; }

.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  padding: 10px 12px; display: flex; flex-direction: column; gap: 7px; min-height: 0; overflow: hidden; }
.ptitle { font-size: 0.68rem; color: var(--cyan); text-transform: uppercase;
  letter-spacing: 0.1em; font-weight: 700; flex-shrink: 0; }

/* Left column: camera stacked above JSON */
#leftCol { display: grid; grid-template-rows: 1fr 1fr; gap: 10px; min-height: 0; }
#cameraImg { width: 100%; flex: 1; min-height: 0; object-fit: contain;
  border-radius: 6px; opacity: 0; transition: opacity 0.5s; display: block; }
#jsonText { flex: 1; min-height: 0; overflow: hidden; font-size: 0.7rem; line-height: 1.42;
  color: #93c5fd; font-family: "SFMono-Regular", Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; }
#navAdvice { font-size: 0.78rem; color: var(--amber); font-style: italic;
  padding: 5px 8px; background: rgba(250,204,21,0.07);
  border: 1px solid rgba(250,204,21,0.2); border-radius: 6px;
  flex-shrink: 0; min-height: 38px; }

/* Middle column: local costmap */
.dir-top { text-align: center; font-size: 0.64rem; color: var(--muted);
  letter-spacing: 0.08em; flex-shrink: 0; }
.local-row { display: flex; align-items: center; gap: 5px; min-height: 0; }
.dir-side { font-size: 0.62rem; color: var(--muted); writing-mode: vertical-lr;
  flex-shrink: 0; white-space: nowrap; }
.dir-side.lft { transform: rotate(180deg); }
#localCanvas { flex: 1; min-width: 0; aspect-ratio: 1; display: block;
  border-radius: 4px; }
.robot-label { font-size: 0.7rem; color: var(--muted); text-align: center; flex-shrink: 0; }

/* Right column: global costmap + metrics */
#globalCanvas { width: 100%; aspect-ratio: 1; display: block;
  image-rendering: pixelated; border-radius: 4px; flex-shrink: 0; }
#metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; flex-shrink: 0; }
.mg h4 { font-size: 0.68rem; color: var(--muted); margin-bottom: 5px; font-weight: 600; }
.mrow { display: flex; justify-content: space-between; align-items: baseline;
  padding: 3px 0; border-bottom: 1px solid var(--line); }
.mrow:last-of-type { border-bottom: none; }
.mlabel { font-size: 0.63rem; color: var(--muted); }
.mval { font-size: 0.88rem; font-weight: 800; }
.mval.base { color: var(--red); }
.mval.enh  { color: var(--green); }
#riskDelta { grid-column: span 2; text-align: center; font-size: 0.82rem; font-weight: 700;
  color: var(--amber); padding: 5px; background: rgba(250,204,21,0.05);
  border: 1px solid rgba(250,204,21,0.16); border-radius: 6px; }

/* Legend chips */
#legend { display: flex; gap: 10px; flex-shrink: 0; }
.lchip { display: flex; align-items: center; gap: 5px; font-size: 0.68rem; color: var(--muted); }
.lline { width: 18px; height: 2px; display: inline-block; }

/* ── Footer ── */
#footer { display: flex; align-items: center; gap: 14px; padding: 0 2px; }
#scenTitle { font-size: 1rem; font-weight: 800; white-space: nowrap; }
#scenSummary { font-size: 0.8rem; color: var(--muted); flex: 1;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#progressWrap { width: 160px; flex-shrink: 0; }
#progressBar { height: 3px; background: var(--line); border-radius: 2px; }
#progressFill { height: 100%; background: var(--cyan); border-radius: 2px;
  width: 0%; transition: width 0.08s linear; }
#kbhints { font-size: 0.68rem; color: var(--line); white-space: nowrap; }
</style>
</head>
<body>
<div id="app">
  <div id="header">
    <h1>VLM Semantic Costmap Layer</h1>
    <div id="dots"></div>
    <button class="hbtn" id="pauseBtn">⏸ Pause</button>
    <button class="hbtn" id="fsBtn">⛶ Fullscreen</button>
  </div>

  <div id="main">
    <!-- Left: camera + JSON -->
    <div id="leftCol">
      <div class="panel">
        <div class="ptitle">Robot Camera Frame</div>
        <img id="cameraImg" src="" alt="camera frame">
      </div>
      <div class="panel">
        <div class="ptitle">VLM Semantic JSON</div>
        <div id="jsonText"></div>
        <div id="navAdvice"></div>
      </div>
    </div>

    <!-- Middle: local costmap -->
    <div class="panel">
      <div class="ptitle">40×40 Local Semantic Costmap</div>
      <div class="dir-top">▲ FORWARD</div>
      <div class="local-row">
        <div class="dir-side lft">◀ LEFT</div>
        <canvas id="localCanvas"></canvas>
        <div class="dir-side">RIGHT ▶</div>
      </div>
      <div class="robot-label">▲ robot position (bottom-center)</div>
    </div>

    <!-- Right: global costmap + metrics -->
    <div class="panel">
      <div class="ptitle">Global Costmap · A* Path Comparison</div>
      <canvas id="globalCanvas"></canvas>
      <div id="legend">
        <div class="lchip"><span class="lline" style="background:#38bdf8;border-top:2px dashed #38bdf8;"></span>Baseline path</div>
        <div class="lchip"><span class="lline" style="background:#f8fafc;"></span>VLM-enhanced path</div>
        <div class="lchip"><span style="color:#22c55e;font-size:0.8rem;">●</span>Start</div>
        <div class="lchip"><span style="color:#facc15;font-size:0.8rem;">★</span>Goal</div>
      </div>
      <div id="metrics"></div>
    </div>
  </div>

  <div id="footer">
    <div id="scenTitle"></div>
    <div id="scenSummary"></div>
    <div id="progressWrap"><div id="progressBar"><div id="progressFill"></div></div></div>
    <div id="kbhints">← → navigate &nbsp;·&nbsp; Space pause &nbsp;·&nbsp; F fullscreen</div>
  </div>
</div>

<script>
const FRAMES = __FRAMES_DATA__;

// ── Magma colormap ──────────────────────────────────────────────────────────
const MAGMA_STOPS = [
  [0,0,4],[20,11,52],[58,9,99],[96,19,110],[134,36,108],
  [168,56,96],[200,78,78],[228,107,50],[246,148,18],[252,200,74],[252,253,191]
];
function magmaRgb(val, maxVal) {
  if (val <= 0) return [7, 11, 20];
  const t = Math.min(1, val / maxVal);
  const i = Math.min(Math.floor(t * 10), 9);
  const f = t * 10 - i;
  const a = MAGMA_STOPS[i], b = MAGMA_STOPS[i + 1];
  return [0,1,2].map(j => Math.round(a[j] + f * (b[j] - a[j])));
}

// ── Canvas renderers ────────────────────────────────────────────────────────
function paintCostmap(canvas, data, basemap, revealFrac) {
  const H = data.length, W = data[0].length;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  const total = H * W;
  const revealed = Math.floor(total * revealFrac);
  for (let i = 0; i < total; i++) {
    const r = Math.floor(i / W), c = i % W;
    let rgb;
    if (i > revealed) {
      rgb = [7, 11, 20];
    } else if (basemap && basemap[r][c] >= 255) {
      rgb = [230, 235, 255];
    } else {
      rgb = magmaRgb(data[r][c], 245);
    }
    const p = i * 4;
    img.data[p] = rgb[0]; img.data[p+1] = rgb[1]; img.data[p+2] = rgb[2]; img.data[p+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawPath(ctx, path, frac, style, dash) {
  if (!path || path.length < 2) return;
  const n = Math.max(2, Math.ceil(path.length * frac));
  ctx.beginPath();
  ctx.strokeStyle = style;
  ctx.lineWidth = dash ? 1.5 : 2.2;
  if (dash) ctx.setLineDash([3, 2]); else ctx.setLineDash([]);
  ctx.moveTo(path[0][0], path[0][1]);
  for (let k = 1; k < n; k++) ctx.lineTo(path[k][0], path[k][1]);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawMarker(ctx, x, y, color, size, star) {
  ctx.fillStyle = color;
  if (star) {
    ctx.font = `${size * 2}px sans-serif`;
    ctx.fillText("★", x - size, y + size);
  } else {
    ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI * 2); ctx.fill();
  }
}

function renderGlobal(frame, globalFrac, baseFrac, enhFrac) {
  const canvas = document.getElementById("globalCanvas");
  paintCostmap(canvas, frame.enhanced_costmap, frame.base_costmap, globalFrac);
  if (globalFrac < 0.05) return;
  const ctx = canvas.getContext("2d");
  if (baseFrac > 0) drawPath(ctx, frame.baseline_path, baseFrac, "rgba(56,189,248,0.85)", true);
  if (enhFrac > 0)  drawPath(ctx, frame.enhanced_path, enhFrac, "rgba(248,250,252,0.95)", false);
  drawMarker(ctx, frame.start[0], frame.start[1], "#22c55e", 2.2, false);
  drawMarker(ctx, frame.goal[0],  frame.goal[1],  "#facc15", 1.8, true);
  // Draw the local costmap footprint so audiences can see how it maps onto the global
  if (frame.robot_pose && globalFrac > 0.1) {
    const rx = frame.robot_pose.x, ry = frame.robot_pose.y;
    const lx = rx - 20, ly = ry - 36; // top-left of the 40x40 local region
    ctx.strokeStyle = "rgba(103,232,249,0.55)";
    ctx.lineWidth = 0.7;
    ctx.setLineDash([2, 2]);
    ctx.strokeRect(lx, ly, 40, 40);
    ctx.setLineDash([]);
    // robot triangle at its actual pose
    ctx.fillStyle = "#67e8f9";
    ctx.beginPath();
    ctx.moveTo(rx, ry - 3); ctx.lineTo(rx - 2, ry + 2); ctx.lineTo(rx + 2, ry + 2);
    ctx.closePath(); ctx.fill();
  }
}

function renderLocal(frame, frac) {
  const canvas = document.getElementById("localCanvas");
  const data = frame.local_costmap;
  const H = data.length, W = data[0].length;
  const S = 10; // render each cell as SxS pixels for crisp text
  canvas.width = W * S; canvas.height = H * S;
  const ctx = canvas.getContext("2d");
  const total = H * W;
  const revealed = Math.floor(total * frac);
  let i = 0;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const rgb = i > revealed ? [7,11,20] : magmaRgb(data[r][c], 245);
      ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      ctx.fillRect(c * S, r * S, S, S);
      i++;
    }
  }
  // robot triangle at local position (20, 36), scaled
  if (frac > 0.85) {
    ctx.fillStyle = "#67e8f9";
    ctx.beginPath();
    ctx.moveTo(20*S, 32*S); ctx.lineTo(17*S, 37*S); ctx.lineTo(23*S, 37*S);
    ctx.closePath(); ctx.fill();
  }
  // C: ring + readable object type label
  if (frac > 0.6 && frame.local_costmap_intent && frame.local_costmap_intent.length > 0) {
    const ringT = Math.min(1, (frac - 0.6) / 0.4);
    const posMap = {
      "top-left":[10,7],"top-center":[20,7],"top-right":[30,7],
      "center-left":[10,20],"center":[20,20],"center-right":[30,20],
      "bottom-left":[10,32],"bottom-center":[20,32],"bottom-right":[30,32]
    };
    const typeAt = {};
    (frame.objects || []).forEach(o => { typeAt[o.image_position] = o.type; });

    frame.local_costmap_intent.forEach(item => {
      const pos = posMap[item.center];
      if (!pos) return;
      const cx = pos[0] * S, cy = pos[1] * S;
      const r = Math.min((item.radius || 10), 17) * S;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(248,113,113,${0.9 * ringT})`;
      ctx.lineWidth = 4;
      ctx.stroke();
      const label = typeAt[item.center] || item.shape;
      ctx.font = `bold 14px Inter, sans-serif`;
      ctx.fillStyle = `rgba(255,255,255,${ringT})`;
      ctx.textAlign = "center";
      ctx.fillText(label, cx, cy - r - 6);
    });
  }
}

// ── Animation ────────────────────────────────────────────────────────────────
const DURATION = 20000; // ms per scenario
// Phase start times (ms)
const PH = {
  cam:      0,
  json:   1000,
  local:  5000,
  glob:   8000,
  base:   9200,
  enh:   11000,
  met:   13000,
  hold:  16000,
};

function ease(t) { return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }
function clamp(t) { return Math.max(0, Math.min(1, t)); }
function pt(e, s, end) { return ease(clamp((e - s) / Math.max(1, end - s))); }
function lerp(a, b, t) { return Math.round(a + (b - a) * t); }

let curFrame = 0;
let tStart = null;
let paused = false;
let lastFrame = -1;

function renderMetrics(frame, t) {
  const bm = frame.baseline_metrics, em = frame.enhanced_metrics;
  const rows = [
    ["path_length",       "Path length"],
    ["average_cost",      "Avg cost"],
    ["social_violations", "Violations"],
    ["risk_score",        "Risk score"],
  ];
  let h = '<div class="mg"><h4>Baseline</h4>';
  rows.forEach(([k, lab]) => {
    const isRisk = k === "risk_score";
    const raw = bm[k];
    const disp = isRisk ? lerp(0, Math.round(raw), t) : raw;
    h += `<div class="mrow"><span class="mlabel">${lab}</span><span class="mval${isRisk?" base":""}">${disp}</span></div>`;
  });
  h += '</div><div class="mg"><h4>VLM-Enhanced</h4>';
  rows.forEach(([k, lab]) => {
    const isRisk = k === "risk_score";
    const raw = em[k];
    const disp = isRisk ? lerp(0, Math.round(raw), t) : raw;
    h += `<div class="mrow"><span class="mlabel">${lab}</span><span class="mval${isRisk?" enh":""}">${disp}</span></div>`;
  });
  h += '</div>';
  const delta = bm.risk_score - em.risk_score;
  h += `<div id="riskDelta">${delta > 0 ? "↓ " + delta + " point risk reduction" : "Paths identical — no social cost"}</div>`;
  document.getElementById("metrics").innerHTML = h;
}

function tick(ts) {
  if (paused) { requestAnimationFrame(tick); return; }
  if (tStart === null) tStart = ts;
  const elapsed = ts - tStart;
  const frame = FRAMES[curFrame];

  // Camera fade
  const camT = pt(elapsed, PH.cam, PH.cam + 600);
  const img = document.getElementById("cameraImg");
  if (curFrame !== lastFrame) {
    img.src = "data:image/png;base64," + frame.camera_b64;
    lastFrame = curFrame;
    document.getElementById("metrics").innerHTML = "";
    document.getElementById("jsonText").textContent = "";
    document.getElementById("navAdvice").textContent = "";
  }
  img.style.opacity = camT;

  // JSON typewriter
  const jsonT = pt(elapsed, PH.json, PH.local);
  const full = frame.sem_json;
  document.getElementById("jsonText").textContent = full.slice(0, Math.floor(full.length * jsonT));
  if (jsonT > 0.88) document.getElementById("navAdvice").textContent = frame.nav_advice;

  // Local costmap sweep
  renderLocal(frame, pt(elapsed, PH.local, PH.glob));

  // Global costmap + paths
  renderGlobal(
    frame,
    pt(elapsed, PH.glob,  PH.glob + 800),
    pt(elapsed, PH.base,  PH.enh),
    pt(elapsed, PH.enh,   PH.met)
  );

  // Metrics count-up
  const metT = pt(elapsed, PH.met, PH.hold);
  if (metT > 0) renderMetrics(frame, metT);

  // Progress bar + footer
  document.getElementById("progressFill").style.width = (elapsed / DURATION * 100) + "%";
  document.getElementById("scenTitle").textContent = frame.title;
  document.getElementById("scenSummary").textContent = frame.summary;

  // Auto-advance
  if (elapsed >= DURATION) {
    curFrame = (curFrame + 1) % FRAMES.length;
    tStart = ts;
    updateDots();
  }

  requestAnimationFrame(tick);
}

// ── Controls ─────────────────────────────────────────────────────────────────
function updateDots() {
  document.querySelectorAll(".dot").forEach((d, i) => d.classList.toggle("active", i === curFrame));
}

function jumpTo(idx) {
  curFrame = (idx + FRAMES.length) % FRAMES.length;
  tStart = null;
  lastFrame = -1;
  updateDots();
}

document.getElementById("pauseBtn").addEventListener("click", () => {
  paused = !paused;
  document.getElementById("pauseBtn").textContent = paused ? "▶ Resume" : "⏸ Pause";
});

document.getElementById("fsBtn").addEventListener("click", () => {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen();
  else document.exitFullscreen();
});

document.addEventListener("keydown", e => {
  if (e.key === " ")           { paused = !paused; document.getElementById("pauseBtn").textContent = paused ? "▶ Resume" : "⏸ Pause"; }
  else if (e.key === "ArrowRight") jumpTo(curFrame + 1);
  else if (e.key === "ArrowLeft")  jumpTo(curFrame - 1);
  else if (e.key === "f" || e.key === "F") {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  }
});

// ── Init ─────────────────────────────────────────────────────────────────────
const dotsEl = document.getElementById("dots");
FRAMES.forEach((_, i) => {
  const d = document.createElement("div");
  d.className = "dot" + (i === 0 ? " active" : "");
  dotsEl.appendChild(d);
});

requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main() -> None:
    print("Collecting frame data (replay mode — no Qwen inference needed)...")
    frames = collect_frames()
    print(f"  {len(frames)} frames ready.")

    frames_json = json.dumps(frames, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__FRAMES_DATA__", frames_json)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size // 1024
    print(f"Written: {OUT}  ({size_kb} KB)")
    print("Open live_demo.html in any browser. Controls: ← → Space F")


if __name__ == "__main__":
    main()
