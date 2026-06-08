import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "ESS/FluxSampler";
const WIDGET_HEIGHT = 170;
const PAD_LEFT = 26;
const PAD_RIGHT = 28;
const PAD_TOP = 16;
const PAD_BOTTOM = 22;
const TICK_STRIP_H = 10;

const COLOR_BG = "#1f2227";
const COLOR_GRID = "#2e3239";
const COLOR_AXIS = "#3a3f47";
const COLOR_LABEL = "#9aa3ad";
const COLOR_LABEL_SOFT = "#6c737c";
const COLOR_GUIDANCE = "#5fb3f7";
const COLOR_CFG_BAND = "rgba(255, 187, 92, 0.22)";
const COLOR_CFG_BAND_EDGE = "rgba(255, 187, 92, 0.7)";
const COLOR_SIGMA_PLAIN = "#6a7079";
const COLOR_SIGMA_SHIFTED = "#9aff7a";
const COLOR_PREVIEW = "#c084fc";
const COLOR_STEP_TICK = "#4a4f57";
const COLOR_DETAIL_POS_TINT = "rgba(125, 220, 140, 0.045)";
const COLOR_DETAIL_NEG_TINT = "rgba(220, 125, 125, 0.045)";
const COLOR_DETAIL_POS_TEXT = "#7ddc8c";
const COLOR_DETAIL_NEG_TEXT = "#dc7d7d";
const COLOR_WARN = "#dc7d7d";
const COLOR_PROGRESS_DONE = "#5ec567";
const COLOR_PROGRESS_CURRENT = "#9aff7a";
const COLOR_PROGRESS_BAR = "rgba(94, 197, 103, 0.9)";
const COLOR_PROGRESS_BAR_TRACK = "rgba(94, 197, 103, 0.15)";

function getWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}
function getValue(node, name, fallback) {
  const w = getWidget(node, name);
  if (!w) return fallback;
  const v = w.value;
  return v === undefined || v === null ? fallback : v;
}
function nodeHasNegative(node) {
  const inp = node.inputs?.find((i) => i.name === "negative");
  return !!(inp && inp.link != null);
}

let _progressListenerInstalled = false;
function installProgressListener() {
  if (_progressListenerInstalled) return;
  _progressListenerInstalled = true;

  const findNode = (rawId) => {
    if (rawId == null) return null;
    const asNum = typeof rawId === "number" ? rawId : parseInt(rawId, 10);
    let n = null;
    if (Number.isFinite(asNum)) n = app.graph?.getNodeById?.(asNum) || null;
    if (!n) n = app.graph?.getNodeById?.(rawId) || null;
    return n && n.type === NODE_TYPE ? n : null;
  };

  const setProgress = (rawId, value, max) => {
    const node = findNode(rawId);
    if (!node) return;
    node.__essProgress = {
      value: Number(value) || 0,
      max: Number(max) || 0,
      ts: Date.now(),
    };
    node.setDirtyCanvas?.(true, true);
  };

  const clearAll = () => {
    for (const n of app.graph?._nodes || []) {
      if (n.type === NODE_TYPE && n.__essProgress) {
        n.__essProgress = null;
        n.setDirtyCanvas?.(true, true);
      }
    }
  };

  api.addEventListener("progress", (event) => {
    const d = event?.detail || {};
    setProgress(d.node, d.value, d.max);
  });

  // Newer Comfy: per-node progress_state with running/finished states.
  api.addEventListener("progress_state", (event) => {
    const d = event?.detail || {};
    const nodes = d.nodes || {};
    for (const [rawId, st] of Object.entries(nodes)) {
      if (st == null) continue;
      if (st.state === "finished" || st.state === "error") continue;
      setProgress(rawId, st.value, st.max);
    }
  });

  // executing with null detail = whole prompt finished.
  api.addEventListener("executing", (event) => {
    if (event?.detail == null) clearAll();
  });
  api.addEventListener("execution_success", clearAll);
  api.addEventListener("execution_error", clearAll);
  api.addEventListener("execution_interrupted", clearAll);
}

// Approximate sigma curve per scheduler. Shape-correct, not numerically exact.
function baseSigma(scheduler, t) {
  switch (scheduler) {
    case "karras":
    case "exponential":
      return Math.pow(1 - t, 2.0);
    case "polyexponential":
      return Math.pow(1 - t, 1.5);
    case "beta":
      return Math.pow(1 - t, 0.7);
    case "sgm_uniform":
    case "ddim_uniform":
    case "normal":
    case "simple":
    default:
      return 1 - t;
  }
}
function applyShift(t, mu) {
  if (!mu || mu <= 1.0) return t;
  return (mu * t) / (1 + (mu - 1) * t);
}
function buildSigmaTrace(steps, scheduler, shift) {
  const n = Math.max(2, steps);
  const ys = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    ys[i] = baseSigma(scheduler, applyShift(t, shift));
  }
  return ys;
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}

function drawChart(ctx, node, widgetWidth, widgetY) {
  // Read live state
  const steps = Math.max(1, Math.round(Number(getValue(node, "steps", 20))));
  const scheduler = String(getValue(node, "scheduler", "simple"));
  const shift = Number(getValue(node, "sigma_shift", 0));
  const guidance = Number(getValue(node, "guidance", 3.5));
  const guidanceEnd = Number(getValue(node, "guidance_end", 3.5));
  const trueCfg = Number(getValue(node, "true_cfg", 1.0));
  const trueCfgUntil = Number(getValue(node, "true_cfg_until", 0.0));
  const detailBoost = Number(getValue(node, "detail_boost", 0));
  const previewEvery = Math.max(0, Math.round(Number(getValue(node, "preview_every", 0))));
  const hasNegative = nodeHasNegative(node);

  // Layout
  const x0 = 8;
  const y0 = widgetY + 2;
  const w0 = widgetWidth - 16;
  const h0 = WIDGET_HEIGHT - 4;

  // Outer frame
  ctx.fillStyle = COLOR_BG;
  ctx.fillRect(x0, y0, w0, h0);
  ctx.strokeStyle = COLOR_AXIS;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, w0 - 1, h0 - 1);

  // Chart inner area (between axes, above tick strip)
  const chartX = x0 + PAD_LEFT;
  const chartY = y0 + PAD_TOP;
  const chartW = w0 - PAD_LEFT - PAD_RIGHT;
  const chartH = h0 - PAD_TOP - PAD_BOTTOM;
  if (chartW < 30 || chartH < 30) return;

  // Detail-boost background tint over chart area
  if (Math.abs(detailBoost) > 1e-4) {
    ctx.fillStyle = detailBoost > 0 ? COLOR_DETAIL_POS_TINT : COLOR_DETAIL_NEG_TINT;
    ctx.fillRect(chartX, chartY, chartW, chartH);
  }

  const stepW = chartW / steps;

  // Guidance Y-axis range (right side)
  const gMin = Math.min(0, guidance, guidanceEnd);
  const gMax = Math.max(5, guidance, guidanceEnd) + 0.5;
  const gSpan = Math.max(0.001, gMax - gMin);
  const yFromG = (g) => chartY + chartH * (1 - (g - gMin) / gSpan);
  // Sigma Y-axis range (left side, always [0, 1])
  const yFromSigma = (s) => chartY + chartH * (1 - s);

  // True-CFG band
  const cfgActive = trueCfg > 1.0 && trueCfgUntil > 0.0 && hasNegative;
  const cfgConfiguredButNoNeg = trueCfg > 1.0 && trueCfgUntil > 0.0 && !hasNegative;
  if (cfgActive) {
    const cfgStepsIncl = Math.max(0, Math.min(steps - 1, Math.round(steps * trueCfgUntil) - 1));
    const bandX1 = chartX + stepW * (cfgStepsIncl + 1);
    ctx.fillStyle = COLOR_CFG_BAND;
    ctx.fillRect(chartX, chartY, bandX1 - chartX, chartH);
    ctx.strokeStyle = COLOR_CFG_BAND_EDGE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bandX1 + 0.5, chartY);
    ctx.lineTo(bandX1 + 0.5, chartY + chartH);
    ctx.stroke();
  }

  // Grid: zero baseline for guidance (only if 0 is in range)
  if (gMin < 0 && gMax > 0) {
    const y0g = yFromG(0);
    ctx.strokeStyle = COLOR_GRID;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(chartX, y0g);
    ctx.lineTo(chartX + chartW, y0g);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Y-axis labels (sigma left, guidance right)
  ctx.fillStyle = COLOR_LABEL_SOFT;
  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("σ 1", chartX - 4, yFromSigma(1));
  ctx.fillText("0", chartX - 4, yFromSigma(0));
  ctx.textAlign = "left";
  ctx.fillText(`${gMax.toFixed(1)} g`, chartX + chartW + 4, yFromG(gMax));
  ctx.fillText(`${gMin.toFixed(1)}`, chartX + chartW + 4, yFromG(gMin));

  // --- Sigma trace(s) ---
  const sigmaPlain = buildSigmaTrace(steps, scheduler, 1.0);
  const sigmaShifted = (shift > 1.0) ? buildSigmaTrace(steps, scheduler, shift) : null;
  function plotSigma(arr, color, lineW) {
    ctx.strokeStyle = color;
    ctx.lineWidth = lineW;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const cx = chartX + stepW * (i + 0.5) * (steps / arr.length);
      const cy = yFromSigma(arr[i]);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
  }
  if (sigmaShifted) plotSigma(sigmaPlain, COLOR_SIGMA_PLAIN, 1);
  plotSigma(sigmaShifted || sigmaPlain, sigmaShifted ? COLOR_SIGMA_SHIFTED : COLOR_SIGMA_PLAIN, 1.5);

  // --- Guidance trace ---
  ctx.strokeStyle = COLOR_GUIDANCE;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < steps; i++) {
    const t = steps <= 1 ? 0 : i / (steps - 1);
    const g = lerp(guidance, guidanceEnd, t);
    const cx = chartX + stepW * (i + 0.5);
    const cy = yFromG(g);
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();
  ctx.fillStyle = COLOR_GUIDANCE;
  for (let i = 0; i < steps; i++) {
    const t = steps <= 1 ? 0 : i / (steps - 1);
    const g = lerp(guidance, guidanceEnd, t);
    const cx = chartX + stepW * (i + 0.5);
    const cy = yFromG(g);
    ctx.beginPath();
    ctx.arc(cx, cy, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  // --- Progress (live, from sampler callbacks) ---
  const progress = node.__essProgress;
  // Stale guard: if no event for > 30s, treat as cleared.
  let progValue = 0;
  let progMax = 0;
  if (progress && Date.now() - progress.ts < 30000) {
    progValue = Math.max(0, progress.value);
    progMax = Math.max(0, progress.max);
  }
  const progFraction = progMax > 0 ? Math.min(1, progValue / progMax) : 0;

  // Thin green progress bar just above the tick strip.
  const barY = chartY + chartH + 1;
  const barH = 2;
  ctx.fillStyle = COLOR_PROGRESS_BAR_TRACK;
  ctx.fillRect(chartX, barY, chartW, barH);
  if (progFraction > 0) {
    ctx.fillStyle = COLOR_PROGRESS_BAR;
    ctx.fillRect(chartX, barY, chartW * progFraction, barH);
  }

  // --- Step ticks (bottom strip) ---
  const tickY = chartY + chartH + 4;
  let previewCount = 0;
  ctx.lineWidth = 1;
  for (let i = 0; i < steps; i++) {
    const cx = Math.round(chartX + stepW * (i + 0.5)) + 0.5;
    const stepIdx = i + 1;
    const isPreview = previewEvery > 0 &&
                      ((stepIdx % previewEvery === 0) || (stepIdx === steps));
    if (isPreview) previewCount++;
    let color;
    let tickH;
    if (progValue > 0 && stepIdx < progValue) {
      color = COLOR_PROGRESS_DONE;
      tickH = TICK_STRIP_H;
    } else if (progValue > 0 && stepIdx === progValue) {
      color = COLOR_PROGRESS_CURRENT;
      tickH = TICK_STRIP_H + 2;
    } else if (isPreview) {
      color = COLOR_PREVIEW;
      tickH = TICK_STRIP_H;
    } else {
      color = COLOR_STEP_TICK;
      tickH = TICK_STRIP_H - 3;
    }
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(cx, tickY);
    ctx.lineTo(cx, tickY + tickH);
    ctx.stroke();
  }

  // --- Top labels ---
  ctx.font = "10px sans-serif";
  ctx.textBaseline = "top";
  ctx.fillStyle = COLOR_LABEL;
  ctx.textAlign = "left";
  let topLeftLabel = scheduler;
  if (shift > 1.0) topLeftLabel += `  shift=${shift.toFixed(2)}`;
  ctx.fillText(topLeftLabel, chartX, y0 + 3);

  ctx.textAlign = "right";
  if (cfgActive) {
    ctx.fillStyle = COLOR_CFG_BAND_EDGE;
    ctx.fillText(`true CFG ${trueCfg.toFixed(1)}× → ${Math.round(trueCfgUntil * 100)}%`,
                 chartX + chartW, y0 + 3);
  } else if (cfgConfiguredButNoNeg) {
    ctx.fillStyle = COLOR_WARN;
    ctx.fillText("true CFG set but no 'negative' wired", chartX + chartW, y0 + 3);
  } else if (Math.abs(detailBoost) > 1e-4) {
    const factor = 1 + 0.25 * Math.max(-1, Math.min(1, detailBoost));
    ctx.fillStyle = detailBoost > 0 ? COLOR_DETAIL_POS_TEXT : COLOR_DETAIL_NEG_TEXT;
    ctx.fillText(`σ×${factor.toFixed(3)}  (boost ${detailBoost >= 0 ? "+" : ""}${detailBoost.toFixed(2)})`,
                 chartX + chartW, y0 + 3);
  }

  // --- Bottom step-axis labels ---
  ctx.fillStyle = COLOR_LABEL_SOFT;
  ctx.font = "9px sans-serif";
  ctx.textBaseline = "bottom";
  ctx.textAlign = "left";
  ctx.fillText("step 1", chartX, y0 + h0 - 3);
  ctx.textAlign = "center";
  if (previewEvery > 0) {
    ctx.fillStyle = COLOR_PREVIEW;
    ctx.fillText(`preview ×${previewCount}`, chartX + chartW / 2, y0 + h0 - 3);
  } else {
    ctx.fillStyle = COLOR_LABEL_SOFT;
    ctx.fillText("preview off", chartX + chartW / 2, y0 + h0 - 3);
  }
  ctx.fillStyle = COLOR_LABEL_SOFT;
  ctx.textAlign = "right";
  ctx.fillText(`${steps}`, chartX + chartW, y0 + h0 - 3);
}

app.registerExtension({
  name: "ess.flux_sampler_chart",
  async setup() {
    installProgressListener();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const widget = {
        type: "ESS_FLUX_TIMELINE",
        name: "__ess_flux_timeline",
        value: null,
        serializeValue: () => null,
        computeSize: (width) => [width, WIDGET_HEIGHT],
        draw: function (ctx, node, widgetWidth, y) {
          drawChart(ctx, node, widgetWidth, y);
        },
      };
      this.addCustomWidget(widget);
      if (this.computeSize && this.setSize) {
        this.setSize(this.computeSize());
      }
      return r;
    };
  },
});
