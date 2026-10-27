// Minimal canvas line charts with log axes, a time cursor and click-to-seek.
"use strict";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  const w = Math.max(10, Math.round(r.width)), h = Math.max(10, Math.round(r.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) if (e >= lo - 1e-9 && e <= hi + 1e-9) out.push(e);
  if (out.length < 2) return niceTicks(lo, hi, 3);
  const every = Math.ceil(out.length / 6);
  return out.filter((_, i) => i % every === 0);
}

function fmt(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) return v.toExponential(0).replace("e+", "e");
  return +v.toPrecision(3) + "";
}

// cfg: {series:[{pts:[[x,y]], color, width, dash, alpha}], xlog, ylog, cursor, hlines:[{y,color,dash}], ymin, ymax}
function lineChart(canvas, cfg) {
  const { ctx, w, h } = setupCanvas(canvas);
  const pad = { l: 46, r: 10, t: 6, b: 22 };
  const tx = v => cfg.xlog ? Math.log10(v) : v, ty = v => cfg.ylog ? Math.log10(v) : v;
  const ok = (x, y) => isFinite(x) && isFinite(y) && (!cfg.xlog || x > 0) && (!cfg.ylog || y > 0);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of cfg.series) for (const [x, y] of s.pts) if (ok(x, y)) {
    x0 = Math.min(x0, tx(x)); x1 = Math.max(x1, tx(x)); y0 = Math.min(y0, ty(y)); y1 = Math.max(y1, ty(y));
  }
  for (const l of cfg.hlines || []) if (ok(1, l.y)) { y0 = Math.min(y0, ty(l.y)); y1 = Math.max(y1, ty(l.y)); }
  if (cfg.ymin !== undefined) y0 = Math.min(y0, ty(cfg.ymin));
  if (cfg.ymax !== undefined) y1 = Math.max(y1, ty(cfg.ymax));
  const muted = cssVar("--muted"), gridc = cssVar("--grid");
  ctx.font = "10px -apple-system, sans-serif";
  if (!isFinite(x0)) { ctx.fillStyle = muted; ctx.fillText("no data yet", w / 2 - 25, h / 2); return null; }
  if (x1 === x0) { x1 += 1; x0 -= cfg.xlog ? 0 : 1; }
  if (y1 === y0) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
  const ym = (y1 - y0) * 0.06; y0 -= ym; y1 += ym;
  const X = v => pad.l + (tx(v) - x0) / (x1 - x0) * (w - pad.l - pad.r);
  const Y = v => h - pad.b - (ty(v) - y0) / (y1 - y0) * (h - pad.t - pad.b);

  ctx.strokeStyle = gridc; ctx.fillStyle = muted; ctx.lineWidth = 1;
  const yt = cfg.ylog ? logTicks(y0, y1) : niceTicks(y0, y1, 4);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const v of yt) {
    const py = h - pad.b - (v - y0) / (y1 - y0) * (h - pad.t - pad.b);
    ctx.beginPath(); ctx.moveTo(pad.l, py); ctx.lineTo(w - pad.r, py); ctx.stroke();
    ctx.fillText(fmt(cfg.ylog ? Math.pow(10, v) : v), pad.l - 4, py);
  }
  const xt = cfg.xlog ? logTicks(x0, x1) : niceTicks(x0, x1, 5);
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of xt) {
    const px = pad.l + (v - x0) / (x1 - x0) * (w - pad.l - pad.r);
    ctx.beginPath(); ctx.moveTo(px, pad.t); ctx.lineTo(px, h - pad.b); ctx.stroke();
    ctx.fillText(fmt(cfg.xlog ? Math.pow(10, v) : v), px, h - pad.b + 4);
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b); ctx.clip();
  for (const l of cfg.hlines || []) {
    if (!ok(1, l.y)) continue;
    ctx.strokeStyle = l.color; ctx.setLineDash(l.dash || [4, 4]); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(l.y)); ctx.lineTo(w - pad.r, Y(l.y)); ctx.stroke();
  }
  for (const s of cfg.series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []);
    ctx.globalAlpha = s.alpha ?? 1;
    ctx.beginPath();
    let pen = false;
    for (const [x, y] of s.pts) {
      if (!ok(x, y)) { pen = false; continue; }
      pen ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y));
      pen = true;
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  ctx.setLineDash([]);
  if (cfg.cursor !== undefined && ok(cfg.cursor, 1)) {
    ctx.strokeStyle = cssVar("--accent"); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(cfg.cursor), pad.t); ctx.lineTo(X(cfg.cursor), h - pad.b); ctx.stroke();
    for (const s of cfg.series) {
      if (!s.marker) continue;
      const p = s.pts.find(([x]) => x >= cfg.cursor);
      if (p && ok(p[0], p[1])) {
        ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, 7); ctx.fill();
      }
    }
  }
  ctx.restore();
  // inverse map for click-to-seek
  return px => {
    const v = x0 + (px - pad.l) / (w - pad.l - pad.r) * (x1 - x0);
    return cfg.xlog ? Math.pow(10, v) : v;
  };
}
