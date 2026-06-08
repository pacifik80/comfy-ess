import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "ess-recrop-style";
const NODE_NAME = "ESS/Recrop";
const DEFAULT_NODE_WIDTH = 960;
const DEFAULT_PANEL_HEIGHT = 560;
const MIN_PANEL_HEIGHT = 360;
const PANEL_VERTICAL_OFFSET = 260;
const PANEL_BOTTOM_MARGIN = 8;
const CONTROLS_COL_WIDTH = 320;

const ZONE_GROUPS = [
  {
    key: "composite", label: "Composites", defaultOpen: true,
    zones: [
      { key: "full",       label: "Full",       paired: false, defaultEnabled: true  },
      { key: "head_only",  label: "Head only",  paired: false, defaultEnabled: false },
      { key: "bust",       label: "Bust",       paired: false, defaultEnabled: false },
      { key: "upper_body", label: "Upper body", paired: false, defaultEnabled: false },
      { key: "half_body",  label: "Half body",  paired: false, defaultEnabled: false },
    ],
  },
  {
    key: "head", label: "Head", defaultOpen: false,
    zones: [
      { key: "face",         label: "Face",         paired: false, defaultEnabled: false },
      { key: "eyebrow",      label: "Eyebrow",      paired: true,  defaultEnabled: false },
      { key: "eye",          label: "Eye",          paired: true,  defaultEnabled: false },
      { key: "ear",          label: "Ear",          paired: true,  defaultEnabled: false },
      { key: "nose",         label: "Nose",         paired: false, defaultEnabled: false },
      { key: "cheek",        label: "Cheek",        paired: true,  defaultEnabled: false },
      { key: "mouth",        label: "Mouth",        paired: false, defaultEnabled: false },
      { key: "mouth_corner", label: "Mouth corner", paired: true,  defaultEnabled: false },
      { key: "chin",         label: "Chin",         paired: false, defaultEnabled: false },
      { key: "jaw",          label: "Jaw",          paired: true,  defaultEnabled: false },
    ],
  },
  {
    key: "torso", label: "Torso", defaultOpen: false,
    zones: [
      { key: "neck",     label: "Neck",     paired: false, defaultEnabled: false },
      { key: "shoulder", label: "Shoulder", paired: true,  defaultEnabled: false },
      { key: "chest",    label: "Chest",    paired: false, defaultEnabled: false },
      { key: "belly",    label: "Belly",    paired: false, defaultEnabled: false },
      { key: "hip",      label: "Hip",      paired: true,  defaultEnabled: false },
    ],
  },
  {
    key: "arms", label: "Arms", defaultOpen: false,
    zones: [
      { key: "upper_arm", label: "Upper arm", paired: true, defaultEnabled: false },
      { key: "elbow",     label: "Elbow",     paired: true, defaultEnabled: false },
      { key: "forearm",   label: "Forearm",   paired: true, defaultEnabled: false },
      { key: "wrist",     label: "Wrist",     paired: true, defaultEnabled: false },
    ],
  },
  {
    key: "legs", label: "Legs", defaultOpen: false,
    zones: [
      { key: "thigh", label: "Thigh", paired: true, defaultEnabled: false },
      { key: "knee",  label: "Knee",  paired: true, defaultEnabled: false },
      { key: "shin",  label: "Shin",  paired: true, defaultEnabled: false },
      { key: "ankle", label: "Ankle", paired: true, defaultEnabled: false },
    ],
  },
];

const ALL_ZONES = ZONE_GROUPS.flatMap((g) => g.zones.map((z) => ({ ...z, group: g.key })));

const GROUP_COLORS = {
  composite: "rgba(96,220,120,0.95)",
  head:      "rgba(96,196,255,0.95)",
  torso:     "rgba(255,196,64,0.95)",
  arms:      "rgba(255,128,200,0.95)",
  legs:      "rgba(180,128,255,0.95)",
};

const CLASS_A_WIDGETS = new Set(["confidence_threshold", "device"]);

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-recrop-root{position:relative;width:100%;height:100%;display:flex;flex-direction:column;color:#e5e7eb;font-family:"IBM Plex Sans","Segoe UI",sans-serif;pointer-events:auto;overflow:hidden;box-sizing:border-box;padding:0 6px 6px}
.ess-recrop-frame{flex:1 1 auto;width:100%;min-width:0;min-height:0;display:flex;flex-direction:row;border:1px solid #334155;border-radius:6px;background:#070f20;overflow:hidden;box-sizing:border-box}
.ess-recrop-controls{flex:0 0 ${CONTROLS_COL_WIDTH}px;width:${CONTROLS_COL_WIDTH}px;min-width:240px;border-right:1px solid #1f2937;background:#0a1424;display:flex;flex-direction:column;overflow:hidden}
.ess-recrop-controls-scroll{flex:1;min-height:0;overflow-y:auto;padding:8px}
.ess-recrop-header{padding:8px 10px;border-bottom:1px solid #1f2937;display:flex;align-items:center;justify-content:space-between;gap:6px;background:#0d1830;flex:0 0 auto}
.ess-recrop-status{display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8}
.ess-recrop-status-dot{width:8px;height:8px;border-radius:50%;background:#475569}
.ess-recrop-status-dot.fresh{background:#22c55e;box-shadow:0 0 6px #22c55e}
.ess-recrop-status-dot.stale{background:#f59e0b;box-shadow:0 0 6px #f59e0b}
.ess-recrop-status-dot.busy{background:#3b82f6;box-shadow:0 0 6px #3b82f6;animation:ess-recrop-pulse 1s infinite alternate}
@keyframes ess-recrop-pulse{from{opacity:0.4}to{opacity:1}}
.ess-recrop-btn{height:24px;padding:0 10px;border:1px solid #475569;border-radius:5px;background:#1e293b;color:#e5e7eb;font-size:11px;cursor:pointer}
.ess-recrop-btn:hover{background:#334155}
.ess-recrop-btn.primary{border-color:#2563eb;background:#1d4ed8;color:#fff}
.ess-recrop-btn.primary:hover{background:#2563eb}
.ess-recrop-btn.stale{border-color:#f59e0b;background:#78350f;color:#fde68a}
.ess-recrop-btn:disabled{opacity:0.5;cursor:not-allowed}
.ess-recrop-section{margin-bottom:6px;border:1px solid #1f2937;border-radius:6px;overflow:hidden;background:#0a1424}
.ess-recrop-section-head{display:flex;align-items:center;justify-content:space-between;padding:5px 8px;cursor:pointer;background:#0f1b30;user-select:none}
.ess-recrop-section-head:hover{background:#152340}
.ess-recrop-section-head-left{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:#93c5fd}
.ess-recrop-section-head-count{font-size:10px;color:#64748b;font-weight:400}
.ess-recrop-section-body{display:none;padding:4px 6px 6px;border-top:1px solid #1f2937}
.ess-recrop-section.open .ess-recrop-section-body{display:block}
.ess-recrop-section.open .ess-recrop-caret{transform:rotate(90deg)}
.ess-recrop-caret{display:inline-block;transition:transform .15s;font-size:9px;color:#64748b;width:9px}
.ess-recrop-row{display:grid;grid-template-columns:minmax(54px,1fr) 22px minmax(0,0.7fr) minmax(0,0.6fr) minmax(0,0.65fr) minmax(0,0.65fr);gap:4px;align-items:center;padding:2px 2px}
.ess-recrop-row label{font-size:11px;color:#e5e7eb;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ess-recrop-row.highlight{background:#1e3a8a44;border-radius:3px}
.ess-recrop-row input[type="checkbox"]{justify-self:center;width:14px;height:14px;margin:0;accent-color:#2563eb}
.ess-recrop-row .side-placeholder{font-size:10px;color:#475569;text-align:center}
.ess-recrop-row select,.ess-recrop-row input[type="number"]{width:100%;height:22px;box-sizing:border-box;border:1px solid #334155;border-radius:4px;background:#020617;color:#e5e7eb;padding:1px 4px;font-size:11px}
.ess-recrop-row select:disabled,.ess-recrop-row input:disabled{opacity:.45;background:#0a0f1c}
.ess-recrop-row-head{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.05em;padding:2px 2px}
.ess-recrop-options-row{display:grid;grid-template-columns:60px minmax(0,1fr) 56px minmax(0,1fr);gap:5px;align-items:center;padding:3px 2px}
.ess-recrop-options-row label{font-size:11px;color:#94a3b8}
.ess-recrop-options-row select,.ess-recrop-options-row input{height:22px;box-sizing:border-box;border:1px solid #334155;border-radius:4px;background:#020617;color:#e5e7eb;padding:1px 4px;font-size:11px;width:100%}
.ess-recrop-options-row input[type="color"]{padding:1px;height:22px}
.ess-recrop-preview{flex:1 1 0;min-width:0;min-height:0;width:auto;height:auto;display:flex;flex-direction:column;background:#000;position:relative}
.ess-recrop-preview-toolbar{flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:4px 8px;background:#0a1424;border-bottom:1px solid #1f2937;font-size:11px;color:#94a3b8}
.ess-recrop-preview-toolbar .spacer{flex:1}
.ess-recrop-preview-toolbar label{display:flex;align-items:center;gap:3px;cursor:pointer;user-select:none}
.ess-recrop-preview-toolbar input[type="checkbox"]{width:12px;height:12px;margin:0;accent-color:#2563eb}
.ess-recrop-canvas-wrap{flex:1;min-height:0;position:relative;overflow:hidden;background:#000;cursor:crosshair}
.ess-recrop-canvas-wrap canvas{position:absolute;left:0;top:0}
.ess-recrop-tooltip{position:absolute;pointer-events:none;background:rgba(15,23,42,0.96);color:#e5e7eb;border:1px solid #475569;border-radius:4px;padding:5px 7px;font-size:11px;line-height:1.4;max-width:240px;z-index:10;box-shadow:0 4px 12px rgba(0,0,0,0.5)}
.ess-recrop-empty{flex:1;display:flex;align-items:center;justify-content:center;color:#475569;font-size:13px;font-style:italic;padding:20px;text-align:center}
.ess-recrop-legend{position:absolute;left:6px;bottom:6px;background:rgba(15,23,42,0.85);border:1px solid #334155;border-radius:4px;padding:4px 6px;font-size:10px;color:#94a3b8;display:flex;gap:8px;flex-wrap:wrap;pointer-events:none;max-width:60%}
.ess-recrop-legend .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:3px;vertical-align:-1px}
.ess-recrop-zoom{position:absolute;right:6px;bottom:6px;display:flex;gap:4px;pointer-events:auto}
.ess-recrop-zoom .ess-recrop-btn{padding:0 6px;height:22px;font-size:10px}
`;
  document.head.appendChild(style);
}

function getWidget(node, name) {
  if (!Array.isArray(node?.widgets)) return null;
  return node.widgets.find((w) => String(w?.name || "") === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.serialize = true;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  const element = widget.element || widget.inputEl;
  if (element?.style) {
    element.style.display = "none";
    element.style.pointerEvents = "none";
  }
}

function setWidgetValue(node, widget, value) {
  if (!widget) return;
  if (widget.value === value) return;
  widget.value = value;
  if (typeof widget.callback === "function") {
    try { widget.callback(value); } catch {}
  }
}

function trapPointerEvents(element) {
  if (!element) return;
  ["pointerdown","pointermove","pointerup","mousedown","mouseup","click","dblclick","contextmenu","keydown","keyup","keypress"]
    .forEach((evt) => element.addEventListener(evt, (e) => e.stopPropagation()));
}

function buildPartRow(node, part, refs, onChange) {
  const partKey = part.key;
  const enabledWidget = refs[`enabled_${partKey}`];
  const modeWidget = refs[`mode_${partKey}`];
  const sideWidget = refs[`side_${partKey}`];
  const weightWidget = refs[`weight_${partKey}`];
  const marginWidget = refs[`margin_${partKey}`];

  const row = document.createElement("div");
  row.className = "ess-recrop-row";
  row.dataset.partKey = partKey;

  const label = document.createElement("label");
  label.textContent = part.label;

  const enabled = document.createElement("input");
  enabled.type = "checkbox";

  let sideControl;
  if (part.paired) {
    sideControl = document.createElement("select");
    ["both","left","right"].forEach((value) => {
      const opt = document.createElement("option");
      opt.value = value; opt.textContent = value;
      sideControl.appendChild(opt);
    });
  } else {
    sideControl = document.createElement("div");
    sideControl.className = "side-placeholder";
    sideControl.textContent = "—";
  }

  const mode = document.createElement("select");
  ["include","exclude"].forEach((value) => {
    const opt = document.createElement("option");
    opt.value = value; opt.textContent = value;
    mode.appendChild(opt);
  });

  const weight = document.createElement("input");
  weight.type = "number"; weight.step = "1"; weight.min = "0"; weight.max = "100";

  const margin = document.createElement("input");
  margin.type = "number"; margin.step = "0.1"; margin.min = "0"; margin.max = "10";

  const applyState = () => {
    const enabledNow = Boolean(enabledWidget?.value ?? part.defaultEnabled);
    enabled.checked = enabledNow;
    mode.value = String(modeWidget?.value ?? "include");
    if (part.paired && sideWidget && sideControl?.tagName === "SELECT") {
      sideControl.value = String(sideWidget.value ?? "both");
      sideControl.disabled = !enabledNow;
    }
    mode.disabled = !enabledNow;
    weight.disabled = !enabledNow;
    margin.disabled = !enabledNow;
    weight.value = String(weightWidget?.value ?? "100");
    margin.value = String(marginWidget?.value ?? "0");
  };

  enabled.addEventListener("change", () => {
    setWidgetValue(node, enabledWidget, !!enabled.checked);
    applyState();
    onChange?.("B");
  });
  mode.addEventListener("change", () => {
    setWidgetValue(node, modeWidget, mode.value || "include");
    onChange?.("B");
  });
  if (part.paired && sideWidget && sideControl?.tagName === "SELECT") {
    sideControl.addEventListener("change", () => {
      setWidgetValue(node, sideWidget, sideControl.value || "both");
      onChange?.("B");
    });
  }
  weight.addEventListener("input", () => {
    setWidgetValue(node, weightWidget, Number(weight.value || 0));
    onChange?.("B");
  });
  margin.addEventListener("input", () => {
    setWidgetValue(node, marginWidget, Number(margin.value || 0));
    onChange?.("B");
  });

  [enabled, sideControl, mode, weight, margin].forEach(trapPointerEvents);
  row.append(label, enabled, sideControl, mode, weight, margin);
  return { row, refresh: applyState, partKey };
}

function buildSection(node, group, refs, onChange) {
  const section = document.createElement("div");
  section.className = "ess-recrop-section" + (group.defaultOpen ? " open" : "");
  section.dataset.groupKey = group.key;

  const head = document.createElement("div");
  head.className = "ess-recrop-section-head";
  const left = document.createElement("div");
  left.className = "ess-recrop-section-head-left";
  const caret = document.createElement("span");
  caret.className = "ess-recrop-caret";
  caret.textContent = "▶";
  const title = document.createElement("span");
  title.textContent = group.label;
  left.append(caret, title);
  const count = document.createElement("span");
  count.className = "ess-recrop-section-head-count";
  head.append(left, count);

  const body = document.createElement("div");
  body.className = "ess-recrop-section-body";

  const headerRow = document.createElement("div");
  headerRow.className = "ess-recrop-row ess-recrop-row-head";
  ["Part","On","Side","Mode","Weight","Margin"].forEach((t) => {
    const c = document.createElement("div"); c.textContent = t; headerRow.appendChild(c);
  });
  body.appendChild(headerRow);

  const rows = group.zones.map((part) => buildPartRow(node, part, refs, onChange));
  rows.forEach((r) => body.appendChild(r.row));

  head.addEventListener("click", (e) => {
    e.stopPropagation();
    section.classList.toggle("open");
  });
  trapPointerEvents(head);

  section.append(head, body);

  const refreshCount = () => {
    let enabledCount = 0;
    group.zones.forEach((part) => {
      if (refs[`enabled_${part.key}`]?.value) enabledCount += 1;
    });
    count.textContent = enabledCount > 0 ? `${enabledCount} on` : "off";
  };

  const refresh = () => { rows.forEach((r) => r.refresh()); refreshCount(); };
  return { section, refresh, refreshCount, rows };
}

function buildOptionsPanel(node, refs, onChange) {
  const container = document.createElement("div");
  container.style.padding = "4px 2px";

  function makeRow(items) {
    const row = document.createElement("div");
    row.className = "ess-recrop-options-row";
    items.forEach((el) => row.appendChild(el));
    return row;
  }
  function makeLabel(text) {
    const el = document.createElement("label"); el.textContent = text; return el;
  }

  const framingSelect = document.createElement("select");
  ["crop","expand"].forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; framingSelect.appendChild(o); });
  framingSelect.addEventListener("change", () => { setWidgetValue(node, refs.framing_mode, framingSelect.value || "crop"); onChange?.("B"); refresh(); });

  const fillModeSelect = document.createElement("select");
  ["border_fill","fill_color"].forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; fillModeSelect.appendChild(o); });
  fillModeSelect.addEventListener("change", () => { setWidgetValue(node, refs.expand_fill_mode, fillModeSelect.value || "border_fill"); onChange?.("C"); refresh(); });

  const colorInput = document.createElement("input"); colorInput.type = "color";
  colorInput.addEventListener("input", () => { setWidgetValue(node, refs.expand_fill_color, colorInput.value || "#000000"); onChange?.("C"); });

  const genderSelect = document.createElement("select");
  ["any","female","male"].forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; genderSelect.appendChild(o); });
  genderSelect.addEventListener("change", () => { setWidgetValue(node, refs.preferred_gender, genderSelect.value || "any"); onChange?.("B"); });

  const ageMin = document.createElement("input"); ageMin.type = "number"; ageMin.min = "0"; ageMin.max = "120"; ageMin.step = "1";
  const ageMax = document.createElement("input"); ageMax.type = "number"; ageMax.min = "0"; ageMax.max = "120"; ageMax.step = "1";
  ageMin.addEventListener("input", () => { setWidgetValue(node, refs.target_age_min, Number(ageMin.value || 0)); onChange?.("B"); });
  ageMax.addEventListener("input", () => { setWidgetValue(node, refs.target_age_max, Number(ageMax.value || 0)); onChange?.("B"); });

  const confInput = document.createElement("input"); confInput.type = "number"; confInput.min = "0.01"; confInput.max = "0.99"; confInput.step = "0.01";
  confInput.addEventListener("input", () => { setWidgetValue(node, refs.confidence_threshold, Number(confInput.value || 0.25)); onChange?.("A"); });

  const deviceSelect = document.createElement("select");
  ["auto","cuda","cpu"].forEach((v) => { const o = document.createElement("option"); o.value = v; o.textContent = v; deviceSelect.appendChild(o); });
  deviceSelect.addEventListener("change", () => { setWidgetValue(node, refs.device, deviceSelect.value || "auto"); onChange?.("A"); });

  const rowFraming = makeRow([makeLabel("Framing"), framingSelect, makeLabel("Fill mode"), fillModeSelect]);
  const rowColor = makeRow([makeLabel("Fill color"), colorInput, document.createElement("div"), document.createElement("div")]);
  const rowSubject = makeRow([makeLabel("Subject"), genderSelect, makeLabel("Age min"), ageMin]);
  const rowAge = makeRow([makeLabel("Age max"), ageMax, makeLabel("Conf"), confInput]);
  const rowDevice = makeRow([makeLabel("Device"), deviceSelect, document.createElement("div"), document.createElement("div")]);
  container.append(rowFraming, rowColor, rowSubject, rowAge, rowDevice);

  [framingSelect, fillModeSelect, colorInput, genderSelect, ageMin, ageMax, confInput, deviceSelect].forEach(trapPointerEvents);

  function refresh() {
    framingSelect.value = String(refs.framing_mode?.value ?? "crop");
    fillModeSelect.value = String(refs.expand_fill_mode?.value ?? "border_fill");
    colorInput.value = String(refs.expand_fill_color?.value || "#000000");
    genderSelect.value = String(refs.preferred_gender?.value ?? "any");
    ageMin.value = String(refs.target_age_min?.value ?? 18);
    ageMax.value = String(refs.target_age_max?.value ?? 35);
    confInput.value = String(refs.confidence_threshold?.value ?? 0.25);
    deviceSelect.value = String(refs.device?.value ?? "auto");
    const expand = framingSelect.value === "expand";
    fillModeSelect.disabled = !expand;
    colorInput.disabled = !(expand && fillModeSelect.value === "fill_color");
  }
  return { container, refresh };
}

function makePreviewCanvas() {
  const wrap = document.createElement("div");
  wrap.className = "ess-recrop-canvas-wrap";
  const canvas = document.createElement("canvas");
  wrap.appendChild(canvas);
  const tooltip = document.createElement("div");
  tooltip.className = "ess-recrop-tooltip";
  tooltip.style.display = "none";
  wrap.appendChild(tooltip);

  const legend = document.createElement("div");
  legend.className = "ess-recrop-legend";
  Object.entries(GROUP_COLORS).forEach(([k, c]) => {
    const span = document.createElement("span");
    span.innerHTML = `<span class="dot" style="background:${c}"></span>${k}`;
    legend.appendChild(span);
  });
  wrap.appendChild(legend);

  const zoomBox = document.createElement("div");
  zoomBox.className = "ess-recrop-zoom";
  const zoomFit = document.createElement("button"); zoomFit.className = "ess-recrop-btn"; zoomFit.textContent = "Fit";
  const zoom100 = document.createElement("button"); zoom100.className = "ess-recrop-btn"; zoom100.textContent = "1:1";
  zoomBox.append(zoomFit, zoom100);
  wrap.appendChild(zoomBox);

  trapPointerEvents(wrap);
  return { wrap, canvas, tooltip, zoomFit, zoom100 };
}

function setupPreview(state, previewParts) {
  const { wrap, canvas, tooltip, zoomFit, zoom100 } = previewParts;
  const ctx = canvas.getContext("2d");
  state.preview = {
    image: null,
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    fitMode: true,
    panStart: null,
    width: 0,
    height: 0,
  };

  function getZonesForRender() {
    if (!state.zonesPayload) return [];
    const people = state.zonesPayload.people || [];
    const refs = state.refs || {};
    const flat = [];
    people.forEach((person, pIdx) => {
      (person.zones || []).forEach((z) => {
        const enabledWidget = refs[`enabled_${z.name}`];
        if (!enabledWidget?.value) return;
        const weightWidget = refs[`weight_${z.name}`];
        if (Number(weightWidget?.value ?? 0) <= 0) return;
        if (z.side === "left" || z.side === "right") {
          const sideWidget = refs[`side_${z.name}`];
          const sideFilter = String(sideWidget?.value ?? "both");
          if (sideFilter === "left" && z.side !== "left") return;
          if (sideFilter === "right" && z.side !== "right") return;
        }
        const modeWidget = refs[`mode_${z.name}`];
        const mode = String(modeWidget?.value ?? "include");
        flat.push({ ...z, personIdx: pIdx, selected: !!person.selected, mode });
      });
    });
    return flat;
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    state.preview.width = Math.max(1, wrap.offsetWidth | 0);
    state.preview.height = Math.max(1, wrap.offsetHeight | 0);
    canvas.style.width = state.preview.width + "px";
    canvas.style.height = state.preview.height + "px";
    canvas.width = Math.floor(state.preview.width * dpr);
    canvas.height = Math.floor(state.preview.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (state.preview.fitMode) fitToFrame();
    draw();
  }

  function fitToFrame() {
    const img = state.preview.image;
    if (!img || !img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) {
      state.preview.scale = 1;
      state.preview.offsetX = 0;
      state.preview.offsetY = 0;
      return;
    }
    const sx = state.preview.width / img.naturalWidth;
    const sy = state.preview.height / img.naturalHeight;
    state.preview.scale = Math.min(sx, sy);
    state.preview.offsetX = (state.preview.width - img.naturalWidth * state.preview.scale) / 2;
    state.preview.offsetY = (state.preview.height - img.naturalHeight * state.preview.scale) / 2;
    state.preview.fitMode = true;
  }

  function setZoom100() {
    const img = state.preview.image;
    if (!img) return;
    state.preview.scale = 1;
    state.preview.offsetX = (state.preview.width - img.naturalWidth) / 2;
    state.preview.offsetY = (state.preview.height - img.naturalHeight) / 2;
    state.preview.fitMode = false;
  }

  function imgToCanvas(x, y) {
    return [x * state.preview.scale + state.preview.offsetX, y * state.preview.scale + state.preview.offsetY];
  }
  function canvasToImg(cx, cy) {
    return [(cx - state.preview.offsetX) / state.preview.scale, (cy - state.preview.offsetY) / state.preview.scale];
  }

  function draw() {
    const img = state.preview.image;
    ctx.clearRect(0, 0, state.preview.width, state.preview.height);
    ctx.fillStyle = "#0a0f1c";
    ctx.fillRect(0, 0, state.preview.width, state.preview.height);
    if (!img || !img.complete || img.naturalWidth === 0 || img.naturalHeight === 0) return;
    ctx.save();
    ctx.translate(state.preview.offsetX, state.preview.offsetY);
    ctx.scale(state.preview.scale, state.preview.scale);
    ctx.drawImage(img, 0, 0);
    ctx.restore();

    const zones = getZonesForRender();
    const filters = state.filters || { anatomy: true, semantic: true, prompt: true, manual: true };

    ctx.lineWidth = 1.25;
    ctx.font = "11px sans-serif";

    zones.forEach((z) => {
      if (!filters.anatomy) return;
      const [x0, y0, x1, y1] = z.box;
      const [cx0, cy0] = imgToCanvas(x0, y0);
      const [cx1, cy1] = imgToCanvas(x1, y1);
      const color = GROUP_COLORS[z.group] || "rgba(200,200,200,0.8)";
      ctx.strokeStyle = color;
      ctx.fillStyle = color.replace("0.95", "0.12");
      ctx.lineWidth = z.confidence < 0.4 ? 1 : 1.25;
      if (z.confidence < 0.4) ctx.setLineDash([3, 3]); else ctx.setLineDash([]);
      ctx.fillRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
      ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
    });
    ctx.setLineDash([]);

    const cropBox = state.zonesPayload?.crop_box;
    if (cropBox) {
      const [x0, y0, x1, y1] = cropBox;
      const [cx0, cy0] = imgToCanvas(x0, y0);
      const [cx1, cy1] = imgToCanvas(x1, y1);
      ctx.strokeStyle = "rgba(255,0,0,0.95)";
      ctx.lineWidth = 2;
      ctx.strokeRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
    }

    if (state.highlightedZone) {
      const z = state.highlightedZone;
      const [x0, y0, x1, y1] = z.box;
      const [cx0, cy0] = imgToCanvas(x0, y0);
      const [cx1, cy1] = imgToCanvas(x1, y1);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(cx0 - 1, cy0 - 1, cx1 - cx0 + 2, cy1 - cy0 + 2);
    }
  }

  function findZoneUnder(cx, cy) {
    const [ix, iy] = canvasToImg(cx, cy);
    const filters = state.filters || { anatomy: true };
    if (!filters.anatomy) return null;
    const zones = getZonesForRender();
    let best = null;
    let bestArea = Infinity;
    zones.forEach((z) => {
      const [x0, y0, x1, y1] = z.box;
      if (ix < x0 || ix > x1 || iy < y0 || iy > y1) return;
      const area = (x1 - x0) * (y1 - y0);
      if (area < bestArea) { bestArea = area; best = z; }
    });
    return best;
  }

  const pointerToWrapCoords = (e) => {
    const rect = wrap.getBoundingClientRect();
    const sx = rect.width / Math.max(1, wrap.offsetWidth);
    const sy = rect.height / Math.max(1, wrap.offsetHeight);
    return [(e.clientX - rect.left) / (sx || 1), (e.clientY - rect.top) / (sy || 1)];
  };

  wrap.addEventListener("pointermove", (e) => {
    const [cx, cy] = pointerToWrapCoords(e);
    if (state.preview.panStart) {
      const dx = cx - state.preview.panStart.x;
      const dy = cy - state.preview.panStart.y;
      state.preview.offsetX = state.preview.panStart.ox + dx;
      state.preview.offsetY = state.preview.panStart.oy + dy;
      state.preview.fitMode = false;
      draw();
      return;
    }
    const zone = findZoneUnder(cx, cy);
    if (zone !== state.highlightedZone) {
      state.highlightedZone = zone;
      draw();
      highlightControlsRow(state, zone?.name);
    }
    if (zone) {
      tooltip.style.display = "block";
      const sideLabel = zone.side && zone.side !== "single" ? ` (${zone.side})` : "";
      const sourceLabel = zone.source ? zone.source : "?";
      tooltip.innerHTML = `<b>${zone.name}${sideLabel}</b><br>source: ${sourceLabel}<br>confidence: ${zone.confidence.toFixed(2)}<br>group: ${zone.group}`;
      const tx = Math.min(cx + 12, state.preview.width - 250);
      const ty = Math.min(cy + 12, state.preview.height - 80);
      tooltip.style.left = tx + "px";
      tooltip.style.top = ty + "px";
    } else {
      tooltip.style.display = "none";
    }
  });

  wrap.addEventListener("pointerleave", () => {
    tooltip.style.display = "none";
    state.highlightedZone = null;
    state.preview.panStart = null;
    highlightControlsRow(state, null);
    draw();
  });

  wrap.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 && e.button !== 1) return;
    if (e.button === 1 || e.shiftKey) {
      const [px, py] = pointerToWrapCoords(e);
      state.preview.panStart = { x: px, y: py, ox: state.preview.offsetX, oy: state.preview.offsetY };
      wrap.setPointerCapture(e.pointerId);
      e.preventDefault();
    }
  });

  wrap.addEventListener("pointerup", (e) => {
    state.preview.panStart = null;
    try { wrap.releasePointerCapture(e.pointerId); } catch {}
  });

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!state.preview.image) return;
    const [cx, cy] = pointerToWrapCoords(e);
    const [ix, iy] = canvasToImg(cx, cy);
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.max(0.05, Math.min(20, state.preview.scale * factor));
    state.preview.offsetX = cx - ix * newScale;
    state.preview.offsetY = cy - iy * newScale;
    state.preview.scale = newScale;
    state.preview.fitMode = false;
    draw();
  }, { passive: false });

  zoomFit.addEventListener("click", (e) => { e.stopPropagation(); fitToFrame(); draw(); });
  zoom100.addEventListener("click", (e) => { e.stopPropagation(); setZoom100(); draw(); });
  trapPointerEvents(zoomFit); trapPointerEvents(zoom100);

  state.preview.resize = resizeCanvas;
  state.preview.draw = draw;
  state.preview.fitToFrame = fitToFrame;
  return { resize: resizeCanvas, draw, fit: fitToFrame };
}

function highlightControlsRow(state, partKey) {
  const root = state.controlsRoot;
  if (!root) return;
  root.querySelectorAll(".ess-recrop-row.highlight").forEach((r) => r.classList.remove("highlight"));
  if (!partKey) return;
  const row = root.querySelector(`.ess-recrop-row[data-part-key="${partKey}"]`);
  if (!row) return;
  row.classList.add("highlight");
  const section = row.closest(".ess-recrop-section");
  if (section && !section.classList.contains("open")) section.classList.add("open");
  row.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function buildToolbar(state, refs, onDetect) {
  const bar = document.createElement("div");
  bar.className = "ess-recrop-preview-toolbar";

  const status = document.createElement("div"); status.className = "ess-recrop-status";
  const dot = document.createElement("span"); dot.className = "ess-recrop-status-dot";
  const label = document.createElement("span"); label.textContent = "no detection yet";
  status.append(dot, label);

  const spacer = document.createElement("div"); spacer.className = "spacer";

  const filterAnatomy = document.createElement("label");
  const filterAnatomyInput = document.createElement("input"); filterAnatomyInput.type = "checkbox"; filterAnatomyInput.checked = true;
  filterAnatomy.append(filterAnatomyInput, document.createTextNode("anatomy"));
  filterAnatomyInput.addEventListener("change", () => {
    state.filters = state.filters || {};
    state.filters.anatomy = filterAnatomyInput.checked;
    state.preview?.draw?.();
  });
  trapPointerEvents(filterAnatomy);

  const detectBtn = document.createElement("button");
  detectBtn.className = "ess-recrop-btn primary";
  detectBtn.textContent = "Detect";
  detectBtn.addEventListener("click", (e) => { e.stopPropagation(); onDetect?.(); });
  trapPointerEvents(detectBtn);

  bar.append(status, spacer, filterAnatomy, detectBtn);

  function setStatus(kind, text) {
    dot.classList.remove("fresh","stale","busy");
    if (kind) dot.classList.add(kind);
    label.textContent = text;
    if (kind === "stale") detectBtn.classList.add("stale"); else detectBtn.classList.remove("stale");
    detectBtn.disabled = kind === "busy";
  }

  return { bar, setStatus, detectBtn };
}

async function callDetectViaPartialPrompt(node) {
  const graph = await app.graphToPrompt();
  const output = graph?.output;
  const workflow = graph?.workflow;
  const nodeId = String(node.id);
  if (!output || !output[nodeId]) {
    throw new Error("Recrop node not found in current graph");
  }

  const keep = new Set([nodeId]);
  const queue = [nodeId];
  while (queue.length) {
    const id = queue.shift();
    const def = output[id];
    if (!def?.inputs) continue;
    for (const value of Object.values(def.inputs)) {
      if (Array.isArray(value) && value.length === 2 && (typeof value[0] === "string" || typeof value[0] === "number")) {
        const src = String(value[0]);
        if (!keep.has(src) && output[src]) {
          keep.add(src);
          queue.push(src);
        }
      }
    }
  }

  const trimmed = {};
  keep.forEach((id) => {
    if (output[id]) {
      const clone = JSON.parse(JSON.stringify(output[id]));
      trimmed[id] = clone;
    }
  });
  trimmed[nodeId].inputs = trimmed[nodeId].inputs || {};
  trimmed[nodeId].inputs.detect_only = true;

  const imageInput = trimmed[nodeId].inputs.image;
  if (!Array.isArray(imageInput)) {
    throw new Error("connect an image to the Recrop node first");
  }

  const promptPayload = { output: trimmed, workflow };
  await api.queuePrompt(-1, promptPayload);
}

function loadPreviewFromInfo(info) {
  if (!info?.filename) return Promise.resolve(null);
  const url = `/view?filename=${encodeURIComponent(info.filename)}&type=${encodeURIComponent(info.type || "temp")}&subfolder=${encodeURIComponent(info.subfolder || "")}&rand=${Math.random()}`;
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

app.registerExtension({
  name: "ess_recrop",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        if (this.__essRecropReady || typeof this.addDOMWidget !== "function") return result;
        ensureStyles();

        const refs = {};
        for (const spec of ALL_ZONES) {
          refs[`enabled_${spec.key}`] = getWidget(this, `enabled_${spec.key}`);
          refs[`mode_${spec.key}`] = getWidget(this, `mode_${spec.key}`);
          refs[`weight_${spec.key}`] = getWidget(this, `weight_${spec.key}`);
          refs[`margin_${spec.key}`] = getWidget(this, `margin_${spec.key}`);
          if (spec.paired) refs[`side_${spec.key}`] = getWidget(this, `side_${spec.key}`);
        }
        ["device","confidence_threshold","framing_mode","expand_fill_mode","expand_fill_color","preferred_gender","target_age_min","target_age_max","detect_only"].forEach((k) => {
          refs[k] = getWidget(this, k);
        });
        if (refs.detect_only) setWidgetValue(this, refs.detect_only, false);

        const allHide = Object.values(refs).filter(Boolean);
        if (!allHide.length) return result;
        allHide.forEach(hideWidget);

        const state = {
          zonesPayload: null,
          highlightedZone: null,
          filters: { anatomy: true, semantic: true, prompt: true, manual: true },
          controlsRoot: null,
          panelHeight: DEFAULT_PANEL_HEIGHT,
          refs,
        };
        this.__essRecropState = state;

        const root = document.createElement("div");
        root.className = "ess-recrop-root";

        const frame = document.createElement("div");
        frame.className = "ess-recrop-frame";

        const controls = document.createElement("div");
        controls.className = "ess-recrop-controls";

        const controlsHeader = document.createElement("div");
        controlsHeader.className = "ess-recrop-header";
        const headerTitle = document.createElement("div");
        headerTitle.style.fontWeight = "600";
        headerTitle.style.color = "#cbd5e1";
        headerTitle.textContent = "Recrop";
        controlsHeader.appendChild(headerTitle);
        controls.appendChild(controlsHeader);

        const controlsScroll = document.createElement("div");
        controlsScroll.className = "ess-recrop-controls-scroll";
        controls.appendChild(controlsScroll);
        state.controlsRoot = controlsScroll;

        const preview = document.createElement("div");
        preview.className = "ess-recrop-preview";

        const previewParts = makePreviewCanvas();
        const previewCtl = setupPreview(state, previewParts);

        let setStatus, toolbar, detectBtn;

        const sections = [];

        const markStale = () => { setStatus?.("stale", "knobs changed — press Detect or run"); };
        const markFresh = () => { setStatus?.("fresh", "preview up to date"); };
        const onChange = (cls) => {
          sections.forEach((s) => s.refreshCount());
          if (cls === "A") markStale();
          if (cls === "B") state.preview?.draw?.();
        };

        const optionsPanel = buildOptionsPanel(this, refs, onChange);
        controlsScroll.appendChild(optionsPanel.container);

        ZONE_GROUPS.forEach((group) => {
          const built = buildSection(this, group, refs, onChange);
          controlsScroll.appendChild(built.section);
          sections.push(built);
        });

        const toolbarBuild = buildToolbar(state, refs, async () => {
          if (!toolbar) return;
          setStatus("busy", "detecting…");
          try {
            await callDetectViaPartialPrompt(this);
          } catch (err) {
            setStatus("stale", String(err?.message || err));
          }
        });
        toolbar = toolbarBuild.bar;
        setStatus = toolbarBuild.setStatus;
        detectBtn = toolbarBuild.detectBtn;
        preview.appendChild(toolbar);
        preview.appendChild(previewParts.wrap);

        frame.append(controls, preview);
        root.appendChild(frame);

        sections.forEach((s) => s.refresh());
        optionsPanel.refresh();

        const node = this;
        let widget = null;
        const calcDesiredHeight = () => {
          const nodeH = Number(node.size?.[1] || 0);
          return Math.max(MIN_PANEL_HEIGHT, nodeH - PANEL_VERTICAL_OFFSET);
        };

        widget = this.addDOMWidget("recrop_panel", "ess_recrop_panel", root, {
          getValue: () => "",
          setValue: () => {},
          getMinHeight: () => calcDesiredHeight(),
          getMaxHeight: () => calcDesiredHeight(),
          hideOnZoom: false,
          margin: 0,
        });

        if (widget) {
          widget.serialize = false;
          widget.computeSize = function (width) {
            const w = Math.max(Number(width || node.size?.[0] || DEFAULT_NODE_WIDTH), 480);
            const h = calcDesiredHeight();
            return [w, h];
          };
        }

        this.size = Array.isArray(this.size) ? this.size : [DEFAULT_NODE_WIDTH, 0];
        this.size[0] = Math.max(Number(this.size[0] || 0), DEFAULT_NODE_WIDTH);
        this.size[1] = Math.max(Number(this.size[1] || 0), DEFAULT_PANEL_HEIGHT + PANEL_VERTICAL_OFFSET);

        const enforceHeight = () => {
          if (!widget?.element || node._removed) return;
          const desired = calcDesiredHeight();
          const cur = parseFloat(widget.element.style.height) || 0;
          if (Math.abs(cur - desired) > 0.5) {
            widget.element.style.height = desired + "px";
            if (Math.abs(state.panelHeight - desired) > 0.5) {
              state.panelHeight = desired;
            }
          }
        };

        let lastCanvasW = 0;
        let lastCanvasH = 0;
        const checkCanvasResize = () => {
          const wrapEl = previewParts.wrap;
          if (!wrapEl) return;
          const w = wrapEl.offsetWidth | 0;
          const h = wrapEl.offsetHeight | 0;
          if (w === lastCanvasW && h === lastCanvasH) return;
          lastCanvasW = w;
          lastCanvasH = h;
          previewCtl.resize();
          if (state.preview?.fitMode) previewCtl.fit();
          previewCtl.draw();
        };

        const originalOnResize = this.onResize;
        this.onResize = function (size) {
          if (originalOnResize) originalOnResize.apply(this, arguments);
          enforceHeight();
          this.setDirtyCanvas?.(true, true);
        };

        let heightTickRaf = 0;
        let heightTickStopped = false;
        let previewGeneration = 0;
        const heightTick = () => {
          if (heightTickStopped || node._removed) return;
          enforceHeight();
          checkCanvasResize();
          heightTickRaf = requestAnimationFrame(heightTick);
        };
        heightTickRaf = requestAnimationFrame(heightTick);

        const resizeObserver = new ResizeObserver(() => checkCanvasResize());
        resizeObserver.observe(previewParts.wrap);
        resizeObserver.observe(root);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function () {
          try {
            heightTickStopped = true;
            if (heightTickRaf) {
              cancelAnimationFrame(heightTickRaf);
              heightTickRaf = 0;
            }
            try { resizeObserver.disconnect(); } catch (e) {}
            state.preview.image = null;
            previewGeneration = (previewGeneration | 0) + 1;
          } catch (e) {
            console.error("[ess_recrop] onRemoved cleanup error", e);
          }
          if (originalOnRemoved) return originalOnRemoved.apply(this, arguments);
        };

        const refreshAll = () => {
          sections.forEach((s) => s.refresh());
          optionsPanel.refresh();
        };
        this.__essRecropRefresh = refreshAll;

        const onExecuted = nodeType.prototype.onExecuted;
        this.onExecuted = function (message) {
          if (onExecuted) onExecuted.apply(this, arguments);
          try {
            const raw = message?.zones?.[0];
            if (raw) {
              try { state.zonesPayload = typeof raw === "string" ? JSON.parse(raw) : raw; }
              catch (e) { state.zonesPayload = null; }
            }
            const imgInfo = state.zonesPayload?.preview;
            if (imgInfo) {
              const myGen = ++previewGeneration;
              loadPreviewFromInfo(imgInfo).then((img) => {
                if (myGen !== previewGeneration) return;
                if (img) {
                  state.preview.image = img;
                  previewCtl.fit();
                  previewCtl.draw();
                }
              });
            } else {
              previewCtl.draw();
            }
            markFresh();
          } catch (err) {
            console.error("[ess_recrop] onExecuted error", err);
          }
        };

        setStatus("stale", "queue prompt to detect");
        this.__essRecropReady = true;
        requestAnimationFrame(() => { previewCtl.resize(); refreshAll(); });
      } catch (err) {
        console.error("[ess_recrop] init failed", err);
      }
      return result;
    };
  },
});
