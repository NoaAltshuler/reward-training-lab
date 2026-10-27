// Reward Training Lab viewer.
"use strict";

const RUN_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d"];
const CLASS_COLORS = ["--c0", "--c1", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#84cc16"];

const S = {
  data: null, run: 0, frame: 0, playing: false, follow: true, hover: -1, dims: [0, 1],
  opts: { halo: true, arrows: true, maxmargin: true, neurons: true, predicted: true, gradarrow: true, normalize: false },
  chartSeek: {},
};

const $ = id => document.getElementById(id);
const run = () => S.data.runs[S.run];
const snaps = () => run().snapshots;
const snap = () => snaps()[Math.min(S.frame, snaps().length - 1)];
const classColor = k => { const c = CLASS_COLORS[k % CLASS_COLORS.length]; return c.startsWith("--") ? cssVar(c) : c; };
const hasTheory = r => r.theory && r.theory.theta_hat;
const isBinaryLinear = r => r.linear && r.K === 2;

// ------------------------------------------------------------------ loading
async function poll() {
  try {
    const d = await (await fetch("data.json", { cache: "no-store" })).json();
    accept(d);
    if (d.runs.some(r => r.status === "running")) setTimeout(poll, 1000);
    else setTimeout(poll, 4000);   // keep watching for new runs started later in the same process
  } catch (e) {
    $("status").textContent = "viewer server not reachable";
    setTimeout(poll, 3000);
  }
}

function accept(d) {
  const first = !S.data;
  S.data = d;
  if (!d.runs.length) { $("status").textContent = "waiting for an experiment…"; return; }
  S.run = Math.min(S.run, d.runs.length - 1);
  if (first) S.frame = 0;
  const n = snaps().length;
  $("frame").max = Math.max(0, n - 1);
  if (S.follow && !first) S.frame = n - 1;
  if (first && run().status !== "running") S.frame = n - 1;
  buildRuns();
  render();
}

// ------------------------------------------------------------------ header / timeline
function buildRuns() {
  const box = $("runs");
  const key = S.data.runs.map(r => r.name).join("|") + S.run;
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = "";
    S.data.runs.forEach((r, i) => {
      const b = document.createElement("button");
      b.className = i === S.run ? "on" : "";
      b.innerHTML = `<span class="sw" style="background:${RUN_COLORS[i % RUN_COLORS.length]}"></span>${r.name}`;
      b.onclick = () => selectRun(i);
      box.appendChild(b);
    });
  }
  const r = run();
  const pct = r.steps ? Math.round(100 * r.progress / r.steps) : 0;
  $("status").textContent = r.status === "running" ? `training… ${pct}%  (${r.progress.toLocaleString()} / ${r.steps.toLocaleString()} steps)`
    : `${r.status} · ${r.steps.toLocaleString()} steps · N=${r.N} · K=${r.K}` + (r.lr ? ` · η=${r.lr}` : "") +
      (r.batch_size ? ` · batch ${r.batch_size}` : " · full batch");
}

function selectRun(i) {
  const t = snap() ? snap().t : 0;
  S.run = i;
  S.dims = [0, 1];
  $("frame").max = Math.max(0, snaps().length - 1);
  S.frame = nearestFrame(t);
  buildRuns();
  render();
}

function nearestFrame(t) {
  const s = snaps();
  let best = 0;
  for (let i = 0; i < s.length; i++) if (Math.abs(Math.log1p(s[i].t) - Math.log1p(t)) < Math.abs(Math.log1p(s[best].t) - Math.log1p(t))) best = i;
  return best;
}

function setFrame(f) {
  S.frame = Math.max(0, Math.min(snaps().length - 1, f));
  S.follow = S.frame === snaps().length - 1;
  render();
}

// ------------------------------------------------------------------ input space
function inputView(canvas, r) {
  const { ctx, w, h } = setupCanvas(canvas);
  const [bx0, bx1, by0, by1] = r.bounds;
  const size = Math.min(w, h);
  const ox = (w - size) / 2, oy = (h - size) / 2;
  const X = x => ox + (x - bx0) / (bx1 - bx0) * size;
  const Y = y => oy + size - (y - by0) / (by1 - by0) * size;
  const invX = px => bx0 + (px - ox) / size * (bx1 - bx0);
  const invY = py => by0 + (oy + size - py) / size * (by1 - by0);
  return { ctx, w, h, X, Y, invX, invY, size, ox, oy };
}

function hexToRgb(c) {
  if (c.startsWith("#")) {
    const v = c.length === 4 ? c.slice(1).split("").map(x => x + x).join("") : c.slice(1);
    return [0, 2, 4].map(i => parseInt(v.slice(i, i + 2), 16));
  }
  const m = c.match(/\d+/g);
  return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
}

function drawHeatmap(v, r, s) {
  const G = r.grid_res, img = new ImageData(G, G);
  const rgb = [...Array(r.K)].map((_, k) => hexToRgb(classColor(k)));
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const idx = j * G + i, o = ((G - 1 - j) * G + i) * 4;
    let c, a;
    if (r.K === 2) {
      const p = s.grid[idx];
      c = p >= 0.5 ? rgb[1] : rgb[0];
      a = Math.abs(p - 0.5) * 2;
    } else {
      c = rgb[s.grid_class[idx]];
      a = (s.grid[idx] - 1 / r.K) / (1 - 1 / r.K);
    }
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2];
    img.data[o + 3] = Math.round(255 * (0.08 + 0.32 * Math.sqrt(Math.max(0, a))));
  }
  const off = document.createElement("canvas");
  off.width = off.height = G;
  off.getContext("2d").putImageData(img, 0, 0);
  v.ctx.imageSmoothingEnabled = true;
  // grid points are cell centres at the bounds, so extend by half a cell
  const half = v.size / (G - 1) / 2;
  v.ctx.drawImage(off, v.ox - half, v.oy - half, v.size + 2 * half, v.size + 2 * half);
}

// Marching squares on a scalar grid; calls seg(x0,y0,x1,y1) in world coords.
function contour(r, f, level, seg) {
  const G = r.grid_res, [bx0, bx1, by0, by1] = r.bounds;
  const gx = i => bx0 + i / (G - 1) * (bx1 - bx0), gy = j => by0 + j / (G - 1) * (by1 - by0);
  for (let j = 0; j < G - 1; j++) for (let i = 0; i < G - 1; i++) {
    const v = [f(j * G + i), f(j * G + i + 1), f((j + 1) * G + i + 1), f((j + 1) * G + i)];
    const p = [[gx(i), gy(j)], [gx(i + 1), gy(j)], [gx(i + 1), gy(j + 1)], [gx(i), gy(j + 1)]];
    const pts = [];
    for (let e = 0; e < 4; e++) {
      const a = v[e] - level, b = v[(e + 1) % 4] - level;
      if ((a < 0) !== (b < 0)) {
        const t = a / (a - b), P = p[e], Q = p[(e + 1) % 4];
        pts.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]);
      }
    }
    if (pts.length >= 2) seg(pts[0][0], pts[0][1], pts[1][0], pts[1][1]);
    if (pts.length === 4) seg(pts[2][0], pts[2][1], pts[3][0], pts[3][1]);
  }
}

// Line {x : a x0 + b x1 + c = 0} clipped to the view.
function drawLine(v, r, a, b, c, style, width, dash) {
  const [bx0, bx1, by0, by1] = r.bounds, pts = [];
  if (Math.abs(b) > 1e-12) for (const x of [bx0, bx1]) pts.push([x, -(a * x + c) / b]);
  if (Math.abs(a) > 1e-12) for (const y of [by0, by1]) pts.push([-(b * y + c) / a, y]);
  const inside = pts.filter(([x, y]) => x >= bx0 - 1e-9 && x <= bx1 + 1e-9 && y >= by0 - 1e-9 && y <= by1 + 1e-9);
  if (inside.length < 2) return;
  const ctx = v.ctx;
  ctx.strokeStyle = style; ctx.lineWidth = width; ctx.setLineDash(dash || []);
  ctx.beginPath(); ctx.moveTo(v.X(inside[0][0]), v.Y(inside[0][1]));
  ctx.lineTo(v.X(inside[inside.length - 1][0]), v.Y(inside[inside.length - 1][1])); ctx.stroke();
  ctx.setLineDash([]);
}

function arrow(ctx, x0, y0, x1, y1, color, width, head) {
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  if (L < 0.5) return;
  head = Math.min(head || 7, L * 0.5);
  const ux = dx / L, uy = dy / L;
  ctx.strokeStyle = ctx.fillStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - ux * head * 0.8, y1 - uy * head * 0.8); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * head - uy * head * 0.45, y1 - uy * head + ux * head * 0.45);
  ctx.lineTo(x1 - ux * head + uy * head * 0.45, y1 - uy * head - ux * head * 0.45);
  ctx.closePath(); ctx.fill();
}

function drawInput() {
  const r = run(), s = snap(), canvas = $("c-input");
  const v = inputView(canvas, r), ctx = v.ctx;
  S.inputView = v;
  if (!s) return;
  drawHeatmap(v, r, s);

  // axes through the origin
  ctx.strokeStyle = cssVar("--line"); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(v.X(0), v.oy); ctx.lineTo(v.X(0), v.oy + v.size);
  ctx.moveTo(v.ox, v.Y(0)); ctx.lineTo(v.ox + v.size, v.Y(0)); ctx.stroke();

  const th = r.theory || {}, hat = th.theta_hat;
  if (S.opts.maxmargin && isBinaryLinear(r) && hat) {
    const c = r.bias ? hat[2] : 0;
    drawLine(v, r, hat[0], hat[1], c, cssVar("--hat"), 1.5, [6, 5]);
    drawLine(v, r, hat[0], hat[1], c - 1, cssVar("--muted"), 1, [2, 4]);
    drawLine(v, r, hat[0], hat[1], c + 1, cssVar("--muted"), 1, [2, 4]);
  }
  // current decision boundary
  ctx.strokeStyle = cssVar("--fg"); ctx.lineWidth = 2.2;
  if (isBinaryLinear(r) && s.theta && Math.hypot(s.theta[0], s.theta[1]) > 0) {
    drawLine(v, r, s.theta[0], s.theta[1], r.bias ? s.theta[2] : 0, cssVar("--fg"), 2.2);
  } else {
    ctx.beginPath();
    const seg = (a, b, c, d) => { ctx.moveTo(v.X(a), v.Y(b)); ctx.lineTo(v.X(c), v.Y(d)); };
    if (r.K === 2) contour(r, i => s.grid[i], 0.5, seg);
    else if (r.linear && s.theta) {
      // exact scores z_k = <w_k, phi(x)> give smooth borders: contour of z_k - max_{j != k} z_j at 0
      const G = r.grid_res, [bx0, bx1, by0, by1] = r.bounds, D = s.theta.length / r.K;
      const z = idx => {
        const phi = [bx0 + (idx % G) / (G - 1) * (bx1 - bx0), by0 + Math.floor(idx / G) / (G - 1) * (by1 - by0), 1];
        return [...Array(r.K)].map((_, k) => { let v = 0; for (let q = 0; q < D; q++) v += s.theta[k * D + q] * phi[q]; return v; });
      };
      for (let k = 0; k < r.K; k++) contour(r, i => { const v = z(i); return v[k] - Math.max(...v.filter((_, j) => j !== k)); }, 0, seg);
    }
    else for (let k = 0; k < r.K; k++) contour(r, i => (s.grid_class[i] === k ? 1 : 0), 0.5, seg);
    ctx.stroke();
  }

  // MLP neurons: direction w_j from the origin, weighted by outgoing norm
  if (S.opts.neurons && s.neurons) {
    const a = s.neurons.a, amax = Math.max(...a) || 1;
    const scale = 0.35 * v.size;
    s.neurons.w.forEach((wj, j) => {
      const n = Math.hypot(wj[0], wj[1]) || 1;
      ctx.globalAlpha = 0.15 + 0.6 * a[j] / amax;
      arrow(ctx, v.X(0), v.Y(0), v.X(0) + scale * wj[0] / n * a[j] / amax, v.Y(0) - scale * wj[1] / n * a[j] / amax,
        cssVar("--muted"), 1.2, 5);
    });
    ctx.globalAlpha = 1;
  }

  const support = new Set(th.support || []);
  const push = s.push.map(Math.abs), pmax = Math.max(...push) || 1;
  const gmax = Math.max(...r.gaps.map(Math.abs)) || 1;

  if (S.opts.halo) {
    r.X.forEach((x, n) => {
      const a = push[n] / pmax;
      if (a < 0.01) return;
      ctx.fillStyle = cssVar("--grad"); ctx.globalAlpha = 0.55 * Math.sqrt(a);
      ctx.beginPath(); ctx.arc(v.X(x[0]), v.Y(x[1]), 6 + 12 * Math.sqrt(a), 0, 7); ctx.fill();
    });
    ctx.globalAlpha = 1;
  }
  if (S.opts.arrows && isBinaryLinear(r) && s.contrib) {
    const cmax = Math.max(...s.contrib.map(c => Math.hypot(c[0], c[1]))) || 1;
    r.X.forEach((x, n) => {
      const c = s.contrib[n], L = Math.hypot(c[0], c[1]) / cmax;
      if (L < 0.02) return;
      const len = 0.12 * v.size * L;
      const n0 = Math.hypot(c[0], c[1]);
      arrow(ctx, v.X(x[0]), v.Y(x[1]), v.X(x[0]) + len * c[0] / n0, v.Y(x[1]) - len * c[1] / n0, cssVar("--grad"), 1.4, 6);
    });
  }
  r.X.forEach((x, n) => {
    const rad = 3 + 3.5 * Math.sqrt(Math.abs(r.gaps[n]) / gmax);
    ctx.fillStyle = classColor(r.labels[n]);
    ctx.beginPath(); ctx.arc(v.X(x[0]), v.Y(x[1]), rad, 0, 7); ctx.fill();
    ctx.lineWidth = support.has(n) ? 2.2 : 1;
    ctx.strokeStyle = support.has(n) ? cssVar("--hat") : cssVar("--panel");
    if (r.gaps[n] <= 0) { ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 2.2; }
    ctx.stroke();
    if (n === S.hover) { ctx.strokeStyle = cssVar("--accent"); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(v.X(x[0]), v.Y(x[1]), rad + 4, 0, 7); ctx.stroke(); }
  });

  // theta direction (normal of the separator) and descent direction at the origin
  if (isBinaryLinear(r) && s.theta && S.opts.arrows) {
    const L = 0.16 * v.size;
    const nt = Math.hypot(s.theta[0], s.theta[1]);
    if (nt > 0) arrow(ctx, v.X(0), v.Y(0), v.X(0) + L * s.theta[0] / nt, v.Y(0) - L * s.theta[1] / nt, cssVar("--fg"), 2.4, 9);
    const g = s.neg_grad && s.neg_grad.theta, ng = g ? Math.hypot(g[0], g[1]) : 0;
    if (ng > 0) arrow(ctx, v.X(0), v.Y(0), v.X(0) + L * g[0] / ng, v.Y(0) - L * g[1] / ng, cssVar("--grad"), 2.4, 9);
  }
  $("input-sub").textContent = isBinaryLinear(r)
    ? "— solid: current separator · dashed: max-margin · black arrow: θ/‖θ‖ · green: −∇L / pull of each sample · halo: |∂L/∂u_n|"
    : "— line: decision boundary · halo: per-sample gradient magnitude";
}

// ------------------------------------------------------------------ weight space
function drawWeights() {
  const r = run(), canvas = $("c-weights");
  const { ctx, w, h } = setupCanvas(canvas);
  const s = snaps();
  if (!s.length) return;
  const cur = Math.min(S.frame, s.length - 1);
  const [i, j] = S.dims;
  const norm = S.opts.normalize;
  const muted = cssVar("--muted");
  const paths = [];      // {pts, color, width, dash}
  const marks = [];      // {p, color, r}
  const arrows = [];     // {p, d, color}
  let sub = "";

  if (r.linear) {
    const K = r.K, D = r.K === 2 ? s[0].theta.length : s[0].theta.length / K;
    const vec = (theta, k) => K === 2 ? theta : theta.slice(k * D, (k + 1) * D);
    const nrm = a => Math.hypot(...a) || 1;
    const proj = (a, k) => { const v = vec(a, k), n = norm ? nrm(a) : 1; return [v[i] / n, v[j] / n]; };
    const hat = r.theory && r.theory.theta_hat ? r.theory.theta_hat.flat() : null;
    for (let k = 0; k < (K === 2 ? 1 : K); k++) {
      const color = K === 2 ? cssVar("--accent") : classColor(k);
      paths.push({ pts: s.map(x => proj(x.theta, k)), color, width: 1, alpha: 0.35 });
      paths.push({ pts: s.slice(0, cur + 1).map(x => proj(x.theta, k)), color, width: 2.2 });
      marks.push({ p: proj(s[cur].theta, k), color, r: 5 });
      if (hat) {
        const hk = vec(hat, k), hn = norm ? nrm(hat) : 1;
        const far = norm ? 1 : Math.max(...s.map(x => nrm(x.theta))) / nrm(hat);
        paths.push({ pts: [[0, 0], [far * hk[i] / hn, far * hk[j] / hn]], color: cssVar("--hat"), width: 1.3, dash: [6, 5] });
      }
      if (S.opts.gradarrow && s[cur].neg_grad) {
        const g = vec(s[cur].neg_grad.theta, k), u = s[cur].update ? vec(s[cur].update, k) : null;
        arrows.push({ p: proj(s[cur].theta, k), d: [g[i], g[j]], color: cssVar("--grad") });
        if (u && r.batch_size) arrows.push({ p: proj(s[cur].theta, k), d: [u[i], u[j]], color: "#f59e0b" });
      }
    }
    const tt = r.theory && r.theory.theta_tilde;
    if (S.opts.predicted && K === 2 && hat && tt) {
      const pred = s.filter(x => x.t >= 1).map(x => hat.map((h, q) => h * Math.log(x.t) + tt[q]));
      paths.push({ pts: pred.map(p => proj(p, 0)), color: cssVar("--pred"), width: 1.5, dash: [3, 3] });
      const pt = s[cur].t >= 1 ? hat.map((h, q) => h * Math.log(s[cur].t) + tt[q]) : null;
      if (pt) marks.push({ p: proj(pt, 0), color: cssVar("--pred"), r: 3.5 });
    }
    sub = `— θ${sub_(i)} vs θ${sub_(j)}` + (norm ? " (unit sphere)" : "") +
      (K === 2 ? " · blue: θ_t · dashed black: θ̂ direction" : " · colored: w_k per class · dashed black: ŵ_k direction") + (K === 2 && r.theory && r.theory.theta_tilde ? " · purple: θ̂ log t + θ̃" : "") +
      " · green: −∇L";
    buildDimSelector(D);
  } else if (s[cur].neurons) {
    const nW = s[cur].neurons.w.length;
    const amax = Math.max(...s.flatMap(x => x.neurons ? x.neurons.a : [])) || 1;
    for (let q = 0; q < nW; q++) {
      const pts = s.map(x => x.neurons ? x.neurons.w[q].slice() : null).filter(Boolean).map(p => {
        const n = norm ? Math.hypot(...p) || 1 : 1; return [p[0] / n, p[1] / n];
      });
      const color = RUN_COLORS[q % RUN_COLORS.length];
      paths.push({ pts, color, width: 1, alpha: 0.25 });
      paths.push({ pts: pts.slice(0, cur + 1), color, width: 1.6, alpha: 0.9 });
      marks.push({ p: pts[cur], color, r: 2 + 5 * s[cur].neurons.a[q] / amax });
    }
    sub = "— first-layer neurons w_j ∈ ℝ² (dot size: ‖outgoing weights‖)";
    $("dimsel").innerHTML = "";
  } else {
    ctx.fillStyle = muted; ctx.font = "12px sans-serif";
    ctx.fillText("Too many parameters to record — see layer norms below.", 20, 30);
    return;
  }
  $("weights-sub").textContent = sub;

  // view bounds over everything that will ever be drawn so the camera does not jump
  let x0 = 0, x1 = 0, y0 = 0, y1 = 0;
  for (const p of paths) for (const [x, y] of p.pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (norm) { x0 = Math.min(x0, -1); x1 = Math.max(x1, 1); y0 = Math.min(y0, -1); y1 = Math.max(y1, 1); }
  const span = Math.max(x1 - x0, y1 - y0, 1e-9) * 1.12, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const size = Math.min(w, h) - 30, ox = (w - size) / 2 + 10, oy = (h - size) / 2;
  const X = x => ox + (x - (cx - span / 2)) / span * size;
  const Y = y => oy + size - (y - (cy - span / 2)) / span * size;

  // grid + axes
  ctx.strokeStyle = cssVar("--grid"); ctx.fillStyle = muted; ctx.font = "10px sans-serif"; ctx.lineWidth = 1;
  for (const t of niceTicks(cx - span / 2, cx + span / 2, 6)) {
    ctx.beginPath(); ctx.moveTo(X(t), oy); ctx.lineTo(X(t), oy + size); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText(fmt(t), X(t), oy + size + 12);
  }
  for (const t of niceTicks(cy - span / 2, cy + span / 2, 6)) {
    ctx.beginPath(); ctx.moveTo(ox, Y(t)); ctx.lineTo(ox + size, Y(t)); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(fmt(t), ox - 4, Y(t) + 3);
  }
  ctx.strokeStyle = muted;
  ctx.beginPath(); ctx.moveTo(X(0), oy); ctx.lineTo(X(0), oy + size); ctx.moveTo(ox, Y(0)); ctx.lineTo(ox + size, Y(0)); ctx.stroke();
  if (norm) { ctx.strokeStyle = cssVar("--line"); ctx.beginPath(); ctx.arc(X(0), Y(0), size / span, 0, 7); ctx.stroke(); }

  for (const p of paths) {
    ctx.strokeStyle = p.color; ctx.lineWidth = p.width; ctx.setLineDash(p.dash || []); ctx.globalAlpha = p.alpha ?? 1;
    ctx.beginPath();
    p.pts.forEach(([x, y], q) => q ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)));
    ctx.stroke();
  }
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  for (const m of marks) { ctx.fillStyle = m.color; ctx.beginPath(); ctx.arc(X(m.p[0]), Y(m.p[1]), m.r, 0, 7); ctx.fill(); }
  for (const a of arrows) {
    const L = Math.hypot(a.d[0], a.d[1]);
    if (!(L > 0)) continue;
    const len = 0.13 * size;
    arrow(ctx, X(a.p[0]), Y(a.p[1]), X(a.p[0]) + len * a.d[0] / L, Y(a.p[1]) - len * a.d[1] / L, a.color, 2, 8);
  }
}

const sub_ = k => "₀₁₂₃₄₅₆₇₈₉"[k] || k;

function buildDimSelector(D) {
  const box = $("dimsel");
  if (D <= 2) { box.innerHTML = ""; return; }
  if (box.dataset.D == D) return;
  box.dataset.D = D;
  const opt = sel => [...Array(D)].map((_, q) => `<option value="${q}" ${q === sel ? "selected" : ""}>θ${sub_(q)}</option>`).join("");
  box.innerHTML = `axes <select id="dx">${opt(S.dims[0])}</select> × <select id="dy">${opt(S.dims[1])}</select>`;
  $("dx").onchange = e => { S.dims[0] = +e.target.value; render(); };
  $("dy").onchange = e => { S.dims[1] = +e.target.value; render(); };
}

// ------------------------------------------------------------------ charts
function chartDefs() {
  const r = run(), runs = S.data.runs;
  const all = (key, f) => runs.map((q, ri) => ({
    pts: q.snapshots.filter(x => x[key] !== undefined).map(x => [x.t, f ? f(x, q) : x[key]]),
    color: RUN_COLORS[ri % RUN_COLORS.length], width: ri === S.run ? 2 : 1.2, alpha: ri === S.run ? 1 : 0.55, marker: ri === S.run,
  })).filter(s => s.pts.length);
  const cur = snap() || {};
  const val = (k, d = 4) => cur[k] !== undefined ? `t=${cur.t}: ${(+cur[k]).toPrecision(d)}` : "";
  const th = r.theory || {};
  const defs = [];
  defs.push({ id: "loss", title: "Loss L(θ_t)", sub: val("loss"), xlog: true, ylog: true, series: all("loss") });
  if (runs.some(q => q.snapshots.some(x => x.angle_deg !== undefined)))
    defs.push({ id: "angle", title: "Angle to max-margin ∠(θ_t, θ̂)  [deg]", sub: val("angle_deg") + "   (theory: O(1/log t))",
      xlog: true, ylog: true, series: all("angle_deg") });
  if (r.linear) {
    const ser = all("param_norm");
    if (th.theta_hat) {
      const nh = Math.hypot(...th.theta_hat.flat());
      ser.push({ pts: snaps().filter(x => x.t >= 1).map(x => [x.t, nh * Math.log(x.t)]), color: cssVar("--pred"), dash: [4, 4], width: 1.2 });
    }
    defs.push({ id: "norm", title: "‖θ_t‖ vs log t", sub: val("param_norm") + (th.theta_hat ? "   dashed: ‖θ̂‖ log t" : ""), xlog: true, series: ser });
  }
  if (runs.some(q => q.snapshots.some(x => x.residual !== undefined)))
    defs.push({ id: "res", title: "Residual ‖r(t)‖ = ‖θ_t − θ̂ log t − θ̃‖", sub: val("residual") + "   (theorem: O(1))",
      xlog: true, series: all("residual"), ymin: 0 });
  if (runs.some(q => q.snapshots.some(x => x.t_loss !== undefined)))
    defs.push({ id: "tl", title: "t · L(θ_t)", sub: val("t_loss") + "   (L = Θ(1/t) ⇒ flat)", xlog: true, series: all("t_loss"), ymin: 0 });
  if (r.linear)
    defs.push({ id: "nm", title: "Normalized margin min_n u_n / ‖θ_t‖", sub: val("norm_margin") + (th.margin ? `   max-margin 1/‖θ̂‖ = ${th.margin.toPrecision(4)}` : ""),
      xlog: true, series: all("norm_margin"), hlines: th.margin ? [{ y: th.margin, color: cssVar("--hat") }] : [] });
  defs.push({ id: "V", title: "Expected reward V(θ_t)", sub: val("V"), xlog: true, series: all("V") });
  defs.push({ id: "acc", title: "Accuracy", sub: val("acc", 3), xlog: true, series: all("acc"), ymin: 0, ymax: 1 });
  defs.push({ id: "gn", title: "Gradient norm ‖∇L(θ_t)‖", sub: val("grad_norm"), xlog: true, ylog: true, series: all("grad_norm") });

  // current run only
  const s = snaps();
  if (r.linear && s.length && s[0].theta) {
    const D = s[0].theta.length, ser = [];
    const tt = th.theta_tilde, hat = th.theta_hat && th.theta_hat.flat();
    for (let q = 0; q < D; q++) {
      const color = RUN_COLORS[q % RUN_COLORS.length];
      ser.push({ pts: s.map(x => [x.t, x.theta[q]]), color, marker: true });
      if (hat && tt && r.K === 2) ser.push({ pts: s.filter(x => x.t >= 1).map(x => [x.t, hat[q] * Math.log(x.t) + tt[q]]), color, dash: [3, 3], width: 1 });
    }
    defs.push({ id: "w", title: `Weights θ_i(t) — ${r.name}`, sub: cur.theta ? "θ_t = (" + cur.theta.map(v => v.toPrecision(3)).join(", ") + ")" + (tt ? "   dashed: θ̂_i log t + θ̃_i" : "") : "",
      xlog: true, series: ser });
  } else if (s.length) {
    const names = Object.keys(s[0].layer_norms);
    defs.push({ id: "w", title: `Layer weight norms — ${r.name}`, sub: names.join(", "), xlog: true, ylog: true,
      series: names.map((nm, q) => ({ pts: s.map(x => [x.t, x.layer_norms[nm]]), color: RUN_COLORS[q % RUN_COLORS.length], marker: true })) });
  }
  if (s.length) {
    const support = new Set(th.support || []);
    const ser = r.X.map((_, n) => ({
      pts: s.map(x => [x.t, Math.abs(x.push[n])]),
      color: n === S.hover ? cssVar("--accent") : support.has(n) ? cssVar("--hat") : classColor(r.labels[n]),
      width: n === S.hover ? 2.5 : support.has(n) ? 1.8 : 1, alpha: n === S.hover || support.has(n) ? 1 : 0.35,
    }));
    defs.push({ id: "push", title: "Per-sample push |∂L/∂f(x_n)|", sub: support.size ? "black: support vectors — the others decay faster" : "hover a point to highlight it",
      xlog: true, ylog: true, series: ser });
    const mser = r.X.map((_, n) => ({
      pts: s.map(x => [x.t, x.margins[n]]),
      color: n === S.hover ? cssVar("--accent") : support.has(n) ? cssVar("--hat") : classColor(r.labels[n]),
      width: n === S.hover ? 2.5 : support.has(n) ? 1.8 : 1, alpha: n === S.hover || support.has(n) ? 1 : 0.35,
    }));
    defs.push({ id: "marg", title: "Margins u_n(t)", sub: "support vectors grow like log t together", xlog: true, series: mser });
  }
  return defs;
}

function drawCharts() {
  const box = $("charts"), defs = chartDefs();
  const key = defs.map(d => d.id).join(",");
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = defs.map(d => `<div class="chart"><div class="ct" id="ct-${d.id}"></div><div class="cs" id="cs-${d.id}"></div><canvas id="cc-${d.id}"></canvas></div>`).join("");
    for (const d of defs) {
      $("cc-" + d.id).addEventListener("click", e => {
        const inv = S.chartSeek[d.id];
        if (!inv) return;
        const rect = e.target.getBoundingClientRect();
        setFrame(nearestFrame(inv(e.clientX - rect.left)));
      });
    }
  }
  const t = snap() ? snap().t : 0;
  for (const d of defs) {
    $("ct-" + d.id).textContent = d.title;
    $("cs-" + d.id).textContent = d.sub || "";
    S.chartSeek[d.id] = lineChart($("cc-" + d.id), { ...d, cursor: Math.max(t, d.xlog ? 1 : t) });
  }
}

// ------------------------------------------------------------------ main render
let raf = 0;
function render() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    if (!S.data || !S.data.runs.length) return;
    if (!snaps().length) {
      for (const id of ["c-input", "c-weights"]) {
        const { ctx, w, h } = setupCanvas($(id));
        ctx.fillStyle = cssVar("--muted"); ctx.font = "13px sans-serif"; ctx.textAlign = "center";
        ctx.fillText("not trained yet — waiting for .train()", w / 2, h / 2);
      }
      $("tlabel").textContent = "t = –";
      $("charts").innerHTML = ""; $("charts").dataset.key = "";
      return;
    }
    $("frame").max = snaps().length - 1;
    $("frame").value = S.frame;
    const s = snap();
    $("tlabel").textContent = `t = ${s.t.toLocaleString()}  (${S.frame + 1}/${snaps().length})`;
    drawInput();
    drawWeights();
    drawCharts();
  });
}

function tick() {
  if (S.playing) {
    if (S.frame >= snaps().length - 1) { S.playing = false; $("play").textContent = "▶"; }
    else setFrame(S.frame + 1);
  }
  setTimeout(tick, 120 / +$("speed").value);
}

// ------------------------------------------------------------------ events
$("frame").addEventListener("input", e => setFrame(+e.target.value));
$("play").addEventListener("click", () => {
  if (!S.data) return;
  if (!S.playing && S.frame >= snaps().length - 1) S.frame = 0;
  S.playing = !S.playing;
  $("play").textContent = S.playing ? "❚❚" : "▶";
});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || !S.data) return;
  if (e.key === " ") { e.preventDefault(); $("play").click(); }
  if (e.key === "ArrowRight") setFrame(S.frame + 1);
  if (e.key === "ArrowLeft") setFrame(S.frame - 1);
});
document.querySelectorAll("[data-opt]").forEach(el => {
  el.addEventListener("change", () => { S.opts[el.dataset.opt] = el.checked; render(); });
});
$("c-input").addEventListener("mousemove", e => {
  const v = S.inputView, r = S.data && run();
  if (!v || !r) return;
  const rect = e.target.getBoundingClientRect(), px = e.clientX - rect.left, py = e.clientY - rect.top;
  let best = -1, bd = 12;
  r.X.forEach((x, n) => { const d = Math.hypot(v.X(x[0]) - px, v.Y(x[1]) - py); if (d < bd) { bd = d; best = n; } });
  const tip = $("tip");
  if (best !== S.hover) { S.hover = best; render(); }
  if (best < 0) { tip.style.display = "none"; return; }
  const s = snap(), n = best, support = new Set((r.theory || {}).support || []);
  tip.textContent = `#${n}  class ${r.labels[n]}${r.K === 2 ? ` (y=${r.labels[n] ? "+1" : "−1"})` : ""}${support.has(n) ? "  · support vector" : ""}\n` +
    `x = (${r.X[n].map(v => v.toFixed(3)).join(", ")})\n` +
    `rewards = [${r.R[n].map(v => +v.toPrecision(3)).join(", ")}]   Δ = ${r.gaps[n].toPrecision(3)}\n` +
    `margin u = ${s.margins[n].toPrecision(4)}\npush = ${s.push[n].toExponential(3)}`;
  tip.style.display = "block";
  tip.style.left = (px + e.target.offsetLeft + 14) + "px";
  tip.style.top = (py + e.target.offsetTop + 14) + "px";
});
$("c-input").addEventListener("mouseleave", () => { $("tip").style.display = "none"; if (S.hover >= 0) { S.hover = -1; render(); } });
window.addEventListener("resize", render);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

if (window.RLAB_DATA) accept(window.RLAB_DATA); else poll();
tick();
