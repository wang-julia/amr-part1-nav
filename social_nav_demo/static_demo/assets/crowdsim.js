/**
 * crowdsim.js — JS port of vla-crowdsim/main.py + costmap.py
 *
 * Robot navigates autonomously via A* on the live 3-layer costmap.
 * C = costmap  V = camera  L = lidar  (buttons only)
 *
 * 3-layer costmap (matches costmap.py):
 *   static   — Chamfer DT on walls+desks, sigma=4 cells (0.2 m)
 *   obstacle — Gaussian per lidar hit, sigma=2 cells (0.1 m)
 *   social   — elliptical Gaussian per pedestrian, sigF=16, sigR=8 cells
 *   render   — R=c*255 G=0 B=(1-c)*255 A=150 (blue=safe, red=danger)
 */
(function (global) {
  'use strict';

  // ── World constants ────────────────────────────────────────────────────────
  const W = 12.0, H = 8.0;
  const ROBOT_R = 0.25, PED_R = 0.3;
  const N_PEDS = 15, N_LIDAR = 36, LIDAR_RANGE = 5.0;

  // Social Force Model (Helbing 1995)
  const A_PED = 1.5, B_PED = 0.4, A_WALL = 2.0, B_WALL = 0.25;
  const TAU = 0.5, R_PED = 2.0, R_WALL = 1.0, MAX_F = 5.0;
  const GOAL_R = 0.4, SPAWN_M = 0.5, DESK_C = 0.6;

  // Desk layout — matches world.py: [x, y, w, h] meters
  const DESKS = [
    [1.0, 1.0, 1.2, 0.6], [4.5, 1.0, 1.2, 0.6], [8.0, 1.0, 1.2, 0.6],
    [1.0, 3.5, 1.2, 0.6], [4.5, 3.5, 1.2, 0.6], [8.0, 3.5, 1.2, 0.6],
    [1.0, 6.0, 1.2, 0.6], [4.5, 6.0, 1.2, 0.6], [8.0, 6.0, 1.2, 0.6],
  ];

  // All collision segments: room walls + desk edges
  const SEGS = (function () {
    const s = [[0, 0, W, 0], [W, 0, W, H], [W, H, 0, H], [0, H, 0, 0]];
    for (const [x, y, w, h] of DESKS) {
      s.push([x, y, x + w, y], [x + w, y, x + w, y + h],
             [x + w, y + h, x, y + h], [x, y + h, x, y]);
    }
    return s;
  })();

  // ── Costmap grid (RES=0.05 m, NX=240, NY=160) ────────────────────────────
  const RES = 0.05;
  const NX = Math.round(W / RES);   // 240
  const NY = Math.round(H / RES);   // 160
  const SIG_STATIC = 4;             // cells
  const SIG_OBS    = 2;             // cells
  const SIG_FWD    = 16;            // cells — social forward
  const SIG_LAT    = 8;             // cells — social lateral

  // ── Pseudo-RNG (Mulberry32) ───────────────────────────────────────────────
  function mkRng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ── Geometry helpers ──────────────────────────────────────────────────────
  function ptSegDist(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    if (l2 < 1e-12) return Math.hypot(px - ax, py - ay);
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  function nearestOnSeg(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
    if (l2 < 1e-12) return [ax, ay];
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
    return [ax + t * dx, ay + t * dy];
  }

  function rayHit(px, py, rdx, rdy, ax, ay, bx, by) {
    const ex = bx - ax, ey = by - ay, det = ex * rdy - rdx * ey;
    if (Math.abs(det) < 1e-10) return null;
    const rx = ax - px, ry = ay - py;
    const t = (ex * ry - ey * rx) / det;
    const s = (rdx * ry - rdy * rx) / det;
    return (t >= 0 && s >= 0 && s <= 1) ? t : null;
  }

  // ── Valid spawn position ──────────────────────────────────────────────────
  function validPos(x, y) {
    if (x < SPAWN_M || x > W - SPAWN_M || y < SPAWN_M || y > H - SPAWN_M) return false;
    for (const [dx, dy, dw, dh] of DESKS) {
      if (x > dx - DESK_C && x < dx + dw + DESK_C &&
          y > dy - DESK_C && y < dy + dh + DESK_C) return false;
    }
    return true;
  }

  function samplePos(rng) {
    for (let i = 0; i < 3000; i++) {
      const x = SPAWN_M + rng() * (W - 2 * SPAWN_M);
      const y = SPAWN_M + rng() * (H - 2 * SPAWN_M);
      if (validPos(x, y)) return [x, y];
    }
    return [6.0, 4.0];
  }

  // ── Static costmap layer (built once at startup) ──────────────────────────
  // Chamfer DT on all wall+desk cells, then exp(-d/sigma).
  const staticLayer = new Float32Array(NX * NY);

  (function buildStaticLayer() {
    const INF = 1e7;
    const dist = new Float32Array(NX * NY).fill(INF);
    for (let gy = 0; gy < NY; gy++) {
      for (let gx = 0; gx < NX; gx++) {
        const wx = gx * RES, wy = gy * RES;
        let occ = gx === 0 || gx === NX - 1 || gy === 0 || gy === NY - 1;
        if (!occ) {
          for (const [dx, dy, dw, dh] of DESKS) {
            if (wx >= dx && wx <= dx + dw && wy >= dy && wy <= dy + dh) { occ = true; break; }
          }
        }
        if (occ) dist[gy * NX + gx] = 0;
      }
    }
    const S2 = Math.SQRT2;
    for (let r = 1; r < NY; r++) {
      for (let c = 1; c < NX - 1; c++) {
        const i = r * NX + c;
        dist[i] = Math.min(dist[i],
          dist[(r - 1) * NX + c] + 1, dist[r * NX + c - 1] + 1,
          dist[(r - 1) * NX + c - 1] + S2, dist[(r - 1) * NX + c + 1] + S2);
      }
    }
    for (let r = NY - 2; r >= 0; r--) {
      for (let c = NX - 2; c >= 1; c--) {
        const i = r * NX + c;
        dist[i] = Math.min(dist[i],
          dist[(r + 1) * NX + c] + 1, dist[r * NX + c + 1] + 1,
          dist[(r + 1) * NX + c + 1] + S2, dist[(r + 1) * NX + c - 1] + S2);
      }
    }
    for (let i = 0; i < NX * NY; i++) staticLayer[i] = Math.exp(-dist[i] / SIG_STATIC);
  })();

  // ── Dynamic layers ────────────────────────────────────────────────────────
  const obsLayer    = new Float32Array(NX * NY);
  const socialLayer = new Float32Array(NX * NY);
  const combined    = new Float32Array(NX * NY);

  function computeCostmap(peds, hits) {
    // Obstacle layer — lidar hits inflated with Gaussian
    obsLayer.fill(0);
    const RAD = Math.ceil(3 * SIG_OBS);
    for (const [hx, hy] of hits) {
      const cgx = Math.round(hx / RES), cgy = Math.round(hy / RES);
      for (let dr = -RAD; dr <= RAD; dr++) {
        for (let dc = -RAD; dc <= RAD; dc++) {
          const gr = cgy + dr, gc = cgx + dc;
          if (gr < 0 || gr >= NY || gc < 0 || gc >= NX) continue;
          const cost = Math.exp(-0.5 * (dr * dr + dc * dc) / (SIG_OBS * SIG_OBS));
          const idx = gr * NX + gc;
          if (cost > obsLayer[idx]) obsLayer[idx] = cost;
        }
      }
    }
    // Social layer — elliptical Gaussian per pedestrian
    socialLayer.fill(0);
    const MAXD = 40;
    for (const p of peds) {
      const cgx = p.x / RES, cgy = p.y / RES;
      const spd = Math.hypot(p.vx, p.vy);
      const fwdX = spd > 0.05 ? p.vx / spd : 1, fwdY = spd > 0.05 ? p.vy / spd : 0;
      const r0 = Math.max(0, Math.floor(cgy - MAXD)), r1 = Math.min(NY - 1, Math.ceil(cgy + MAXD));
      const c0 = Math.max(0, Math.floor(cgx - MAXD)), c1 = Math.min(NX - 1, Math.ceil(cgx + MAXD));
      for (let gr = r0; gr <= r1; gr++) {
        for (let gc = c0; gc <= c1; gc++) {
          const dx = gc - cgx, dy = gr - cgy;
          const df = dx * fwdX + dy * fwdY;   // forward
          const dr = dx * fwdY - dy * fwdX;   // lateral
          const cost = 0.8 * Math.exp(-0.5 * ((df / SIG_FWD) ** 2 + (dr / SIG_LAT) ** 2));
          const idx = gr * NX + gc;
          if (cost > socialLayer[idx]) socialLayer[idx] = cost;
        }
      }
    }
    // Combined = clip(static + obs + social, 0, 1)
    for (let i = 0; i < NX * NY; i++) {
      combined[i] = Math.min(1, staticLayer[i] + obsLayer[i] + socialLayer[i]);
    }
  }

  // ── A* planner on the combined costmap ───────────────────────────────────
  // Edge cost = step_length * (1 + COST_WEIGHT * combined_cost)
  // Cells with combined >= WALL_THRESH are treated as impassable.
  const COST_WEIGHT = 6;
  const WALL_THRESH = 0.94;
  const DIRS8 = [[1,0,1],[-1,0,1],[0,1,1],[0,-1,1],[1,1,1.414],[1,-1,1.414],[-1,1,1.414],[-1,-1,1.414]];

  function astar(startGx, startGy, goalGx, goalGy) {
    const N = NX * NY;
    const gScore = new Float32Array(N).fill(Infinity);
    const parent = new Int32Array(N).fill(-1);
    const closed = new Uint8Array(N);
    const heap = [];   // [fScore, key]

    function push(f, k) {
      heap.push([f, k]);
      let i = heap.length - 1;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (heap[p][0] <= heap[i][0]) break;
        const tmp = heap[p]; heap[p] = heap[i]; heap[i] = tmp; i = p;
      }
    }
    function pop() {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          let m = i, l = 2 * i + 1, r = 2 * i + 2;
          if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
          if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
          if (m === i) break;
          const tmp = heap[m]; heap[m] = heap[i]; heap[i] = tmp; i = m;
        }
      }
      return top;
    }

    const startK = startGy * NX + startGx;
    const goalK  = goalGy  * NX + goalGx;
    gScore[startK] = 0;
    push(Math.hypot(goalGx - startGx, goalGy - startGy), startK);

    while (heap.length > 0) {
      const [, ck] = pop();
      if (closed[ck]) continue;
      closed[ck] = 1;
      if (ck === goalK) {
        const path = [];
        let k = ck;
        while (k !== -1) {
          path.push([(k % NX) * RES + RES * 0.5, ((k / NX) | 0) * RES + RES * 0.5]);
          k = parent[k];
        }
        path.reverse();
        return path;
      }
      const cx = ck % NX, cy = (ck / NX) | 0;
      for (const [dx, dy, mc] of DIRS8) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || nx >= NX || ny < 0 || ny >= NY) continue;
        const nk = ny * NX + nx;
        if (closed[nk]) continue;
        if (combined[nk] >= WALL_THRESH) continue;
        const tentG = gScore[ck] + mc * (1 + COST_WEIGHT * combined[nk]);
        if (tentG < gScore[nk]) {
          gScore[nk] = tentG;
          parent[nk] = ck;
          push(tentG + Math.hypot(goalGx - nx, goalGy - ny), nk);
        }
      }
    }
    return [];  // no path found
  }

  function worldToGrid(wx, wy) {
    return [
      Math.max(1, Math.min(NX - 2, Math.round(wx / RES))),
      Math.max(1, Math.min(NY - 2, Math.round(wy / RES))),
    ];
  }

  // ── Offscreen costmap canvas (ImageData → scaled to world area) ───────────
  let cmCanvas, cmCtx, cmImgData;

  function initCmCanvas() {
    cmCanvas = document.createElement('canvas');
    cmCanvas.width = NX; cmCanvas.height = NY;
    cmCtx = cmCanvas.getContext('2d');
    cmImgData = cmCtx.createImageData(NX, NY);
  }

  function renderCostmapImage() {
    const d = cmImgData.data;
    for (let gy = 0; gy < NY; gy++) {
      const row = NY - 1 - gy;  // y-flip: row 0 = world top
      for (let gx = 0; gx < NX; gx++) {
        const c = combined[gy * NX + gx];
        const p = (row * NX + gx) * 4;
        d[p]     = (c * 255) | 0;
        d[p + 1] = 0;
        d[p + 2] = ((1 - c) * 255) | 0;
        d[p + 3] = 150;
      }
    }
    cmCtx.putImageData(cmImgData, 0, 0);
  }

  // ── Camera preview (robot-POV rotated top-down) ───────────────────────────
  let camCanvas, camCtx;
  const CAM_PX = 96, CAM_FOV = 3.0;

  function initCamCanvas() {
    camCanvas = document.createElement('canvas');
    camCanvas.width = CAM_PX; camCanvas.height = CAM_PX;
    camCtx = camCanvas.getContext('2d');
  }

  function renderCam() {
    const cc = camCtx;
    cc.fillStyle = '#0f0f14'; cc.fillRect(0, 0, CAM_PX, CAM_PX);
    const sc2 = CAM_PX / (CAM_FOV * 2);
    const cx = CAM_PX / 2, cy = CAM_PX * 0.65;
    const cosT = Math.cos(robot.theta), sinT = Math.sin(robot.theta);
    function wToC(wx, wy) {
      const dx = wx - robot.x, dy = wy - robot.y;
      return [cx + (sinT * dx - cosT * dy) * sc2, cy - (cosT * dx + sinT * dy) * sc2];
    }
    for (const [x, y, w, h] of DESKS) {
      const corners = [[x,y],[x+w,y],[x+w,y+h],[x,y+h]];
      const [fx, fy] = wToC(...corners[0]);
      cc.beginPath(); cc.moveTo(fx, fy);
      for (let i = 1; i < 4; i++) { const [px, py] = wToC(...corners[i]); cc.lineTo(px, py); }
      cc.closePath(); cc.fillStyle = '#46311a'; cc.fill();
      cc.strokeStyle = '#a07a3c'; cc.lineWidth = 0.8; cc.stroke();
    }
    cc.strokeStyle = '#c8c8c8'; cc.lineWidth = 1.2;
    for (const [ax, ay, bx, by] of [[0,0,W,0],[W,0,W,H],[W,H,0,H],[0,H,0,0]]) {
      const [sax, say] = wToC(ax, ay), [sbx, sby] = wToC(bx, by);
      cc.beginPath(); cc.moveTo(sax, say); cc.lineTo(sbx, sby); cc.stroke();
    }
    for (const p of peds) {
      const [px, py] = wToC(p.x, p.y);
      cc.beginPath(); cc.arc(px, py, p.radius * sc2, 0, Math.PI * 2);
      cc.fillStyle = '#4682d2'; cc.fill();
    }
    const R = ROBOT_R * sc2;
    cc.beginPath(); cc.moveTo(cx, cy - R); cc.lineTo(cx - R * 0.8, cy + R * 0.6); cc.lineTo(cx + R * 0.8, cy + R * 0.6);
    cc.closePath(); cc.fillStyle = '#dc5050'; cc.fill(); cc.strokeStyle = '#ffc8c8'; cc.lineWidth = 0.7; cc.stroke();
  }

  // ── Simulation state ──────────────────────────────────────────────────────
  let rng, robot, goal, peds, lidar, lidarHits;
  let showCostmap = true, showCamera = false, showLidar = true;
  let psViol = 0, goalsDone = 0;
  let robotPath = null, robotPathIdx = 0, replanTimer = 0;
  const REPLAN_EVERY = 45;  // frames (~0.75 s at 60 fps)

  function reset(seed) {
    rng = mkRng(seed !== undefined ? seed : 42);
    psViol = 0; goalsDone = 0;
    lidar = new Array(N_LIDAR).fill(LIDAR_RANGE); lidarHits = [];
    robotPath = null; robotPathIdx = 0; replanTimer = 0;

    robot = { x: 0.5, y: 4.5, theta: 0.0 };
    const [gx, gy] = samplePos(rng);
    goal = { x: gx, y: gy };

    peds = [];
    for (let i = 0; i < N_PEDS; i++) {
      const [px, py] = samplePos(rng), [gpx, gpy] = samplePos(rng);
      peds.push({ x: px, y: py, vx: 0, vy: 0, gx: gpx, gy: gpy,
                  speed: 0.3 + rng() * 0.2, radius: PED_R });
    }
    // Seed combined with static layer so first A* has valid costs
    for (let i = 0; i < NX * NY; i++) combined[i] = staticLayer[i];
  }

  // ── Physics ───────────────────────────────────────────────────────────────
  function stepPed(p, dt) {
    if (Math.hypot(p.gx - p.x, p.gy - p.y) < GOAL_R) [p.gx, p.gy] = samplePos(rng);
    const gd = Math.hypot(p.gx - p.x, p.gy - p.y);
    const ex = gd > 1e-6 ? (p.gx - p.x) / gd : 0, ey = gd > 1e-6 ? (p.gy - p.y) / gd : 0;
    let fx = (p.speed * ex - p.vx) / TAU, fy = (p.speed * ey - p.vy) / TAU;
    for (const q of peds) {
      if (q === p) continue;
      const ddx = p.x - q.x, ddy = p.y - q.y, d = Math.hypot(ddx, ddy);
      if (d > 1e-6 && d < R_PED) { const s = A_PED * Math.exp((p.radius + q.radius - d) / B_PED); fx += s * ddx / d; fy += s * ddy / d; }
    }
    for (const [ax, ay, bx, by] of SEGS) {
      const [npx, npy] = nearestOnSeg(p.x, p.y, ax, ay, bx, by);
      const ddx = p.x - npx, ddy = p.y - npy, d = Math.hypot(ddx, ddy);
      if (d > 1e-6 && d < R_WALL) { const s = A_WALL * Math.exp((p.radius - d) / B_WALL); fx += s * ddx / d; fy += s * ddy / d; }
    }
    { const ddx = p.x - robot.x, ddy = p.y - robot.y, d = Math.hypot(ddx, ddy);
      if (d > 1e-6 && d < R_PED) { const s = A_PED * 2.0 * Math.exp((p.radius + ROBOT_R - d) / B_PED); fx += s * ddx / d; fy += s * ddy / d; } }
    p.vx += Math.max(-MAX_F, Math.min(MAX_F, fx)) * dt;
    p.vy += Math.max(-MAX_F, Math.min(MAX_F, fy)) * dt;
    const spd = Math.hypot(p.vx, p.vy);
    if (spd > p.speed * 1.5) { p.vx *= p.speed * 1.5 / spd; p.vy *= p.speed * 1.5 / spd; }
    p.x = Math.max(p.radius, Math.min(W - p.radius, p.x + p.vx * dt));
    p.y = Math.max(p.radius, Math.min(H - p.radius, p.y + p.vy * dt));
  }

  function planPath() {
    const [sgx, sgy] = worldToGrid(robot.x, robot.y);
    const [ggx, ggy] = worldToGrid(goal.x, goal.y);
    const raw = astar(sgx, sgy, ggx, ggy);
    if (raw.length < 2) { robotPath = null; return; }
    // Subsample: keep every 4th waypoint + final goal
    const path = [];
    for (let i = 0; i < raw.length; i += 4) path.push(raw[i]);
    if (path[path.length - 1] !== raw[raw.length - 1]) path.push(raw[raw.length - 1]);
    robotPath = path; robotPathIdx = 0;
  }

  function stepRobot(dt) {
    // Goal reached → celebrate and pick new goal
    if (Math.hypot(goal.x - robot.x, goal.y - robot.y) < GOAL_R) {
      goalsDone++; [goal.x, goal.y] = samplePos(rng);
      robotPath = null; replanTimer = 0; return;
    }
    // Replan periodically so robot reacts to moving pedestrians
    replanTimer++;
    if (robotPath === null || robotPathIdx >= robotPath.length || replanTimer >= REPLAN_EVERY) {
      planPath(); replanTimer = 0;
    }
    if (!robotPath || robotPathIdx >= robotPath.length) return;

    const [wx, wy] = robotPath[robotPathIdx];
    const dx = wx - robot.x, dy = wy - robot.y;
    const dist = Math.hypot(dx, dy);
    if (dist < 0.28) { robotPathIdx++; return; }

    const ta = Math.atan2(dy, dx);
    let da = ta - robot.theta;
    while (da > Math.PI)  da -= 2 * Math.PI;
    while (da < -Math.PI) da += 2 * Math.PI;

    const v     = Math.min(1.5, dist * 0.9) * Math.max(0.15, 1 - Math.abs(da) * 0.55);
    const omega = Math.max(-2.2, Math.min(2.2, da * 3.5));

    const nSub = Math.max(1, Math.ceil(Math.abs(v) * dt / (ROBOT_R * 0.5)));
    const sub = dt / nSub;
    for (let s = 0; s < nSub; s++) {
      const nx = robot.x + v * Math.cos(robot.theta) * sub;
      const ny = robot.y + v * Math.sin(robot.theta) * sub;
      let ok = true;
      for (const [ax, ay, bx, by] of SEGS) {
        if (ptSegDist(nx, ny, ax, ay, bx, by) < ROBOT_R) { ok = false; break; }
      }
      if (ok) { robot.x = nx; robot.y = ny; }
      robot.theta += omega * sub;
    }
  }

  function updateLidar() {
    lidarHits = [];
    for (let i = 0; i < N_LIDAR; i++) {
      const ang = robot.theta + 2 * Math.PI * i / N_LIDAR;
      const rdx = Math.cos(ang), rdy = Math.sin(ang);
      let minT = LIDAR_RANGE;
      for (const [ax, ay, bx, by] of SEGS) {
        const t = rayHit(robot.x, robot.y, rdx, rdy, ax, ay, bx, by);
        if (t !== null && t < minT) minT = t;
      }
      lidar[i] = minT;
      if (minT < LIDAR_RANGE) lidarHits.push([robot.x + rdx * minT, robot.y + rdy * minT]);
    }
  }

  function checkViol() {
    for (const p of peds) {
      if (Math.hypot(robot.x - p.x, robot.y - p.y) < 0.5) { psViol++; return; }
    }
  }

  // ── Rendering ─────────────────────────────────────────────────────────────
  let canvas, ctx, sc, offX, offY;

  function resize() {
    if (!canvas) return;
    const cw = canvas.parentElement.clientWidth;
    sc = Math.max(28, Math.min(62, (cw - 80) / W));
    canvas.width  = Math.round(W * sc + 80);
    canvas.height = Math.round(H * sc + 70);
    offX = 40; offY = 24;
  }

  function sx(wx) { return offX + wx * sc; }
  function sy(wy) { return offY + (H - wy) * sc; }

  function draw() {
    const cw = canvas.width, ch = canvas.height;
    ctx.fillStyle = '#0f0f14'; ctx.fillRect(0, 0, cw, ch);

    // ── Costmap heatmap ──────────────────────────────────────────────────────
    if (showCostmap) {
      renderCostmapImage();
      ctx.drawImage(cmCanvas, offX, offY, W * sc, H * sc);
    }

    // ── Boundary walls ───────────────────────────────────────────────────────
    ctx.strokeStyle = '#dcdcdc'; ctx.lineWidth = 2;
    for (const [ax, ay, bx, by] of [[0,0,W,0],[W,0,W,H],[W,H,0,H],[0,H,0,0]]) {
      ctx.beginPath(); ctx.moveTo(sx(ax), sy(ay)); ctx.lineTo(sx(bx), sy(by)); ctx.stroke();
    }

    // ── Desks ────────────────────────────────────────────────────────────────
    for (const [x, y, w, h] of DESKS) {
      ctx.beginPath();
      ctx.moveTo(sx(x), sy(y)); ctx.lineTo(sx(x + w), sy(y));
      ctx.lineTo(sx(x + w), sy(y + h)); ctx.lineTo(sx(x), sy(y + h));
      ctx.closePath(); ctx.fillStyle = '#46311a'; ctx.fill();
      ctx.strokeStyle = '#a07a3c'; ctx.lineWidth = 1.5; ctx.stroke();
    }

    // ── Lidar rays ───────────────────────────────────────────────────────────
    if (showLidar) {
      ctx.strokeStyle = 'rgba(200,185,30,0.22)'; ctx.lineWidth = 0.8;
      for (let i = 0; i < N_LIDAR; i++) {
        const ang = robot.theta + 2 * Math.PI * i / N_LIDAR;
        const ex = robot.x + lidar[i] * Math.cos(ang), ey = robot.y + lidar[i] * Math.sin(ang);
        ctx.beginPath(); ctx.moveTo(sx(robot.x), sy(robot.y)); ctx.lineTo(sx(ex), sy(ey)); ctx.stroke();
      }
    }

    // ── Planned path ─────────────────────────────────────────────────────────
    if (robotPath && robotPathIdx < robotPath.length) {
      ctx.strokeStyle = 'rgba(100,220,255,0.55)'; ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 5]);
      ctx.beginPath(); ctx.moveTo(sx(robot.x), sy(robot.y));
      for (let i = robotPathIdx; i < robotPath.length; i++) {
        ctx.lineTo(sx(robotPath[i][0]), sy(robotPath[i][1]));
      }
      ctx.stroke(); ctx.setLineDash([]);
    }

    // ── Pedestrians ──────────────────────────────────────────────────────────
    for (const p of peds) {
      const [pcx, pcy] = [sx(p.x), sy(p.y)], r = p.radius * sc;
      const spd = Math.hypot(p.vx, p.vy);
      if (spd > 0.08) {
        ctx.strokeStyle = 'rgba(90,150,255,0.4)'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(pcx, pcy);
        ctx.lineTo(sx(p.x + p.vx * 0.7), sy(p.y + p.vy * 0.7)); ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(pcx, pcy, r, 0, Math.PI * 2);
      ctx.fillStyle = '#4682d2'; ctx.fill(); ctx.strokeStyle = '#a0d2ff'; ctx.lineWidth = 1; ctx.stroke();
    }

    // ── Goal marker (pulsing ring + star) ────────────────────────────────────
    const [gcx, gcy] = [sx(goal.x), sy(goal.y)];
    const pulse = 0.7 + 0.3 * Math.sin(performance.now() / 350);
    ctx.strokeStyle = `rgba(250,204,21,${(pulse * 0.85).toFixed(2)})`; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(gcx, gcy, 12 * pulse, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#facc15';
    ctx.font = `bold ${Math.max(11, Math.round(sc * 0.32))}px sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('★', gcx, gcy);

    // ── Robot triangle (matches draw_robot in world.py) ───────────────────────
    const [rcx, rcy] = [sx(robot.x), sy(robot.y)];
    const triR = ROBOT_R * sc, th = robot.theta;
    const tip = [rcx + triR * Math.cos(th),       rcy - triR * Math.sin(th)];
    const lft = [rcx + triR * Math.cos(th + 2.4), rcy - triR * Math.sin(th + 2.4)];
    const rgt = [rcx + triR * Math.cos(th - 2.4), rcy - triR * Math.sin(th - 2.4)];
    ctx.beginPath(); ctx.moveTo(tip[0], tip[1]); ctx.lineTo(lft[0], lft[1]); ctx.lineTo(rgt[0], rgt[1]);
    ctx.closePath(); ctx.fillStyle = '#dc5050'; ctx.fill(); ctx.strokeStyle = '#ffc8c8'; ctx.lineWidth = 1.2; ctx.stroke();

    // ── Camera preview (top-right) ────────────────────────────────────────────
    if (showCamera) {
      renderCam();
      const previewW = 144, px = cw - previewW - 8, py = 8;
      ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1; ctx.strokeRect(px, py, previewW, previewW);
      ctx.drawImage(camCanvas, px, py, previewW, previewW);
      ctx.fillStyle = 'rgba(0,0,0,0.5)'; ctx.fillRect(px, py + previewW - 16, previewW, 16);
      ctx.fillStyle = '#8ca9c4'; ctx.font = '9px monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('ROBOT CAM', px + previewW / 2, py + previewW - 8);
    }

    // ── HUD bar ───────────────────────────────────────────────────────────────
    const nearest = Math.min(...peds.map(p => Math.hypot(robot.x - p.x, robot.y - p.y)));
    ctx.fillStyle = 'rgba(7,11,20,0.74)'; ctx.fillRect(0, ch - 22, cw, 22);
    ctx.fillStyle = '#6a8fa8'; ctx.font = '10px monospace';
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText(
      `pos(${robot.x.toFixed(2)},${robot.y.toFixed(2)})  ·  nearest_ped ${nearest.toFixed(2)} m  ·  goals ${goalsDone}  ·  ps_viol ${psViol}`,
      8, ch - 11
    );
  }

  // ── Main loop ──────────────────────────────────────────────────────────────
  let rafId = null, running = false, lastTs = null;
  let fpsFrames = 0, fpsLast = 0;

  function tick(ts) {
    if (!running) return;
    if (lastTs === null) lastTs = ts;
    const dt = Math.min((ts - lastTs) / 1000, 0.05);
    lastTs = ts;

    fpsFrames++;
    if (ts - fpsLast > 1000) {
      const el = document.getElementById('cs-fps');
      if (el) el.textContent = Math.round(fpsFrames * 1000 / (ts - fpsLast)) + ' fps';
      fpsFrames = 0; fpsLast = ts;
    }

    for (const p of peds) stepPed(p, dt);
    updateLidar();
    computeCostmap(peds, lidarHits);  // update combined before A*
    stepRobot(dt);
    checkViol();
    draw();
    rafId = requestAnimationFrame(tick);
  }

  function start() {
    if (running) return;
    running = true; lastTs = null; fpsLast = performance.now(); fpsFrames = 0;
    rafId = requestAnimationFrame(tick);
    const b = document.getElementById('cs-play');
    if (b) b.textContent = '⏸ Pause';
  }

  function stop() {
    running = false;
    if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
    const b = document.getElementById('cs-play');
    if (b) b.textContent = '▶ Play';
  }

  // ── Public API ─────────────────────────────────────────────────────────────
  global.CrowdSim = {
    init(el) {
      canvas = el; ctx = canvas.getContext('2d');
      initCmCanvas(); initCamCanvas();
      resize();
      new ResizeObserver(() => { resize(); if (!running) draw(); }).observe(canvas.parentElement);
      reset(42); draw();
      new IntersectionObserver((entries) => {
        entries.forEach(e => { if (e.isIntersecting) start(); else stop(); });
      }, { threshold: 0.15 }).observe(canvas);
    },
    toggle()  { running ? stop() : start(); },
    restart() { stop(); reset(42); start(); },
    toggleCostmap() {
      showCostmap = !showCostmap;
      const b = document.getElementById('cs-cm');
      if (b) _styleBtn(b, showCostmap);
    },
    toggleCamera() {
      showCamera = !showCamera;
      const b = document.getElementById('cs-cam');
      if (b) _styleBtn(b, showCamera);
    },
    toggleLidar() {
      showLidar = !showLidar;
      const b = document.getElementById('cs-lid');
      if (b) _styleBtn(b, showLidar);
    },
  };

  function _styleBtn(b, on) {
    b.style.borderColor = on ? 'var(--cyan)' : '';
    b.style.color = on ? 'var(--cyan)' : '';
  }
})(window);
