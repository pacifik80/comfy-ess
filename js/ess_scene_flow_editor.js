import { app } from "../../scripts/app.js";
import { renderHighlight } from "./ess_template_highlight.js";
import { installLoraTokenController } from "./ess_lora_token_editor.js";

const FLOW_STYLE_ID = "ess-scene-flow-style";
const TEMPLATE_STYLE_ID = "ess-scene-flow-template-style";
const DEFAULT_SECTIONS = [
  { id: "options", name: "options" },
  { id: "output", name: "output" },
];
const NODE_WIDTH = 230;
const NODE_HEIGHT_ESTIMATE = 92;
const COLUMN_GAP = 52;
const ROW_GAP = 84;
const COLUMN_PACK_GAP = 22;
const COLUMN_PACK_GAP_GROUP = 10;
const INPUT_PORT_GAP = 24;
const INPUT_PORT_TOP_PADDING = 56;
const INPUT_PORT_BOTTOM_PADDING = 16;
const RANDOM_INPUT_PORT_TOP_PADDING = 58;
const RANDOM_INPUT_PORT_BOTTOM_PADDING = 22;
const RANDOM_NODE_MIN_HEIGHT = 92;
const LANE_GAP = 20;
const BOARD_SIDE_PADDING = 20;
const LANE_TOP = 12;
const LANE_HEADER_HEIGHT = 28;
const LANE_INNER_LEFT = 18;
const LANE_INNER_RIGHT = 18;
const LANE_INNER_TOP = 20;
const LANE_INNER_BOTTOM = 20;

function ensureStyles() {
  if (!document.getElementById(FLOW_STYLE_ID)) {
    const style = document.createElement("style");
    style.id = FLOW_STYLE_ID;
    style.textContent = `
.ess-scene-flow-widget{display:flex;gap:10px;align-items:center;padding:8px 10px;border:1px solid #334155;border-radius:8px;background:#0b1222;color:#e2e8f0;font-family:"IBM Plex Sans","Segoe UI",sans-serif}
.ess-scene-flow-widget button{background:#1d4ed8;color:#eff6ff;border:1px solid #1e40af;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px;font-weight:600}
.ess-scene-flow-summary{font-size:12px;color:#bfdbfe}
.ess-flow-overlay{position:fixed;inset:0;z-index:10001;background:#030712;color:#e5e7eb;display:grid;grid-template-rows:auto 1fr;font-family:"IBM Plex Sans","Segoe UI",sans-serif}
.ess-flow-topbar{display:flex;gap:10px;align-items:center;padding:10px 14px;border-bottom:1px solid #334155;background:#020617}
.ess-flow-topbar h3{margin:0;font-size:15px;color:#dbeafe}.ess-flow-topbar .spacer{flex:1}
.ess-flow-topbar button,.ess-flow-topbar select{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:5px 8px;font-size:12px}
.ess-flow-topbar button.primary{background:#1d4ed8;border-color:#1e40af}
.ess-flow-main{display:grid;grid-template-columns:230px 1fr 360px;gap:10px;padding:10px;min-height:0}
.ess-flow-panel{border:1px solid #334155;border-radius:10px;background:#0b1222;display:flex;flex-direction:column;overflow:hidden;min-height:0}
.ess-flow-panel h4{margin:0;padding:9px 12px;font-size:12px;text-transform:uppercase;color:#93c5fd;border-bottom:1px solid #334155}
.ess-flow-body{padding:10px;overflow:auto;min-height:0}
.ess-flow-section{display:grid;grid-template-columns:1fr auto;gap:6px;align-items:center;background:#111827;border:1px solid #334155;border-radius:7px;padding:6px 7px;margin-bottom:6px}
.ess-flow-section.active{border-color:#60a5fa}.ess-flow-section button{border:none;background:transparent;color:#dbeafe;text-align:left;cursor:pointer;padding:0}
.ess-flow-chip{border-radius:999px;padding:1px 7px;font-size:11px;background:#1e3a8a;color:#dbeafe}
.ess-flow-grid-buttons{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.ess-flow-grid-buttons button,.ess-flow-actions button,.ess-flow-inspector button{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:6px 8px;cursor:pointer;font-size:12px}
.ess-flow-actions{display:grid;grid-template-columns:1fr;gap:6px}
.ess-flow-canvas{position:relative;overflow:auto;min-height:0;background:#050a15}
.ess-flow-board{position:relative}
.ess-flow-lanes{position:absolute;inset:0}
.ess-flow-lane{position:absolute;top:12px;border:1px solid #334155;border-radius:10px;background:rgba(15,23,42,.35);overflow:hidden}
.ess-flow-lane.active{border-color:#60a5fa;background:rgba(14,40,83,.3)}
.ess-flow-lane-header{height:28px;display:flex;align-items:center;justify-content:space-between;gap:8px;padding:0 8px;font-size:11px;color:#bfdbfe;border-bottom:1px solid #334155;background:rgba(2,6,23,.65);cursor:pointer}
.ess-flow-lane-header-title{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.ess-flow-lane-header-controls{display:flex;gap:4px}
.ess-flow-lane-header-controls button{width:20px;height:18px;padding:0;background:#0b1222;color:#dbeafe;border:1px solid #475569;border-radius:5px;cursor:pointer;font-size:11px;line-height:1}
.ess-flow-lane-header-controls button:disabled{opacity:.45;cursor:default}
.ess-flow-edges{position:absolute;inset:0;pointer-events:none;width:100%;height:100%;overflow:visible}
.ess-flow-edge{fill:none;stroke:#60a5fa;stroke-width:2.5;pointer-events:auto;cursor:pointer}.ess-flow-edge.random{stroke:#f59e0b}.ess-flow-edge.pending{stroke:#22c55e;stroke-dasharray:5 4}
.ess-flow-edge.muted{stroke:#64748b;stroke-dasharray:4 4;opacity:.9}
.ess-flow-edge-label{fill:#fbbf24;font-size:11px;font-weight:600;text-shadow:0 1px 0 #000}
.ess-flow-edge-label.muted{fill:#94a3b8}
.ess-flow-node{position:absolute;width:230px;min-height:78px;border:1px solid #4b5563;border-radius:9px;background:#0b1222;box-shadow:0 8px 16px rgba(0,0,0,.35);user-select:none}
.ess-flow-node.selected{border-color:#60a5fa}.ess-flow-node-header{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;cursor:pointer;padding:7px 9px;border-bottom:1px solid #374151;font-size:12px;font-weight:600;border-top-left-radius:8px;border-top-right-radius:8px}
.ess-flow-node.type-element .ess-flow-node-header{background:#14532d}.ess-flow-node.type-sequential .ess-flow-node-header{background:#1e3a8a}.ess-flow-node.type-random .ess-flow-node-header{background:#78350f}.ess-flow-node.type-output .ess-flow-node-header{background:#5b21b6}
.ess-flow-node-title{display:flex;align-items:center;gap:7px;min-width:0}
.ess-flow-node-title-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ess-flow-type-icon{width:14px;height:14px;display:inline-flex;align-items:center;justify-content:center;flex:0 0 auto;color:#f8fafc;opacity:.95}
.ess-flow-type-icon svg{width:14px;height:14px;display:block;stroke:currentColor;fill:none;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round}
.ess-flow-type-icon .fill{fill:currentColor;stroke:none}
.ess-flow-node-body{padding:8px 9px 10px;font-size:11px;line-height:1.4;color:#cbd5e1;white-space:pre-wrap}
.ess-flow-node-badge{border-radius:999px;padding:1px 7px;font-size:10px;background:#020617;border:1px solid #334155;color:#dbeafe}
.ess-flow-port{position:absolute;top:50%;width:13px;height:13px;border-radius:999px;border:2px solid #020617;transform:translateY(-50%);cursor:pointer}
.ess-flow-port.in{left:-7px;background:#22d3ee}.ess-flow-port.out{right:-7px;background:#34d399}
.ess-flow-inbound-control{position:absolute;display:grid;grid-template-columns:1fr 52px 30px 20px;align-items:center;gap:4px;padding:2px 4px;border:1px solid #475569;background:#0b1222;border-radius:6px;transform:translateY(-50%);z-index:2;box-sizing:border-box}
.ess-flow-inbound-control.simple{grid-template-columns:1fr 20px}
.ess-flow-inbound-label{font-size:11px;color:#cbd5e1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ess-flow-inbound-control input{width:52px;height:22px;background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:4px;padding:2px 4px;font-size:11px;box-sizing:border-box}
.ess-flow-inbound-control button{width:26px;height:22px;background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:4px;padding:0;font-size:11px;cursor:pointer}
.ess-flow-inbound-control button.is-muted{background:#3f1111;border-color:#7f1d1d;color:#fecaca}
.ess-flow-inbound-arrows{display:grid;grid-template-rows:1fr 1fr;gap:1px;height:22px}
.ess-flow-inbound-arrows button{width:20px;min-width:20px;height:10px;padding:0;line-height:1}
.ess-flow-inspector{display:grid;gap:9px}.ess-flow-field{display:grid;gap:4px}.ess-flow-field label{font-size:11px;color:#bfdbfe}
.ess-flow-field input,.ess-flow-field select,.ess-flow-field textarea{width:100%;box-sizing:border-box;background:#0b1222;color:#e5e7eb;border:1px solid #4b5563;border-radius:6px;padding:6px 7px;font-size:12px}
.ess-flow-weight-row{display:grid;grid-template-columns:1fr 90px auto auto;gap:6px;align-items:center;font-size:12px}
.ess-flow-muted{font-size:12px;color:#94a3b8}
.ess-flow-modal-backdrop{position:absolute;inset:0;background:rgba(2,6,23,.74);display:flex;align-items:center;justify-content:center;z-index:30}
.ess-flow-modal{width:min(420px,calc(100% - 28px));border:1px solid #334155;border-radius:10px;background:#0b1222;box-shadow:0 24px 48px rgba(0,0,0,.55);overflow:hidden}
.ess-flow-modal h5{margin:0;padding:10px 12px;font-size:13px;color:#dbeafe;border-bottom:1px solid #334155;background:#020617}
.ess-flow-modal p{margin:0;padding:12px;color:#cbd5e1;font-size:13px;line-height:1.45}
.ess-flow-modal-actions{display:flex;justify-content:flex-end;gap:8px;padding:10px 12px;border-top:1px solid #334155}
.ess-flow-modal-actions button{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px}
.ess-flow-modal-actions button.primary{background:#1d4ed8;border-color:#1e40af;color:#eff6ff}
.ess-flow-test-modal{width:min(1100px,calc(100% - 32px));max-height:calc(100% - 40px);display:flex;flex-direction:column}
.ess-flow-test-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px}
.ess-flow-test-toolbar{display:flex;gap:10px;align-items:flex-end;padding:12px 12px 0}
.ess-flow-test-field{display:grid;gap:4px}
.ess-flow-test-field label{font-size:11px;color:#bfdbfe}
.ess-flow-test-field input,.ess-flow-test-field select{background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:6px 8px;font-size:12px}
.ess-flow-test-toolbar button{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px}
.ess-flow-test-toolbar button.primary{background:#1d4ed8;border-color:#1e40af;color:#eff6ff}
.ess-flow-test-col{display:grid;gap:6px}
.ess-flow-test-col label{font-size:12px;color:#bfdbfe}
.ess-flow-test-col textarea{width:100%;min-height:320px;max-height:48vh;resize:vertical;box-sizing:border-box;background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.4;font-family:"JetBrains Mono","Fira Code","Consolas",monospace}
.ess-flow-test-meta{padding:0 12px 10px;color:#94a3b8;font-size:11px}
`;
    document.head.appendChild(style);
  }

  if (!document.getElementById(TEMPLATE_STYLE_ID)) {
    const style = document.createElement("style");
    style.id = TEMPLATE_STYLE_ID;
    style.textContent = `
.ess-template-editor{position:relative;width:100%;height:260px;box-sizing:border-box;border:1px solid #374151;background:#0b1222;border-radius:7px;overflow:hidden}
.ess-template-editor .ess-template-highlight-content,.ess-template-editor textarea{font-family:"JetBrains Mono","Fira Code","Consolas",monospace;font-size:12px;line-height:1.5;padding:8px 10px;box-sizing:border-box;width:100%;height:100%}
.ess-template-editor .ess-template-highlight{position:absolute;inset:0;color:#c9d1d9;overflow:hidden}
.ess-template-editor .ess-template-highlight-content{white-space:pre-wrap;pointer-events:none;will-change:transform}
.ess-template-editor textarea{position:relative;background:transparent;color:transparent;caret-color:#e6edf3;border:none;resize:none;outline:none;overflow:auto}
.ess-template-editor .ess-tpl-choice{color:#2f6f3e}.ess-template-editor .ess-tpl-header{color:#b48800}.ess-template-editor .ess-tpl-negative-marker,.ess-template-editor .ess-tpl-negative-text{color:#a43f3f}.ess-template-editor .ess-tpl-variant{color:#5b2c83}.ess-template-editor .ess-tpl-comment{color:#6e7681}
`;
    document.head.appendChild(style);
  }
}

function createTemplateEditor(initialValue, onChange) {
  const container = document.createElement("div");
  container.className = "ess-template-editor";
  const highlight = document.createElement("div");
  highlight.className = "ess-template-highlight";
  const highlightContent = document.createElement("div");
  highlightContent.className = "ess-template-highlight-content";
  highlight.appendChild(highlightContent);
  const textarea = document.createElement("textarea");
  textarea.value = initialValue || "";
  textarea.placeholder = "Write element template...";
  textarea.spellcheck = false;
  let loraController = null;
  const sync = () => {
    highlightContent.style.width = `${textarea.clientWidth || 0}px`;
    highlightContent.style.height = `${textarea.clientHeight || 0}px`;
    highlightContent.style.transform = `translate(${-textarea.scrollLeft}px, ${-textarea.scrollTop}px)`;
  };
  const render = () => {
    const knownLoras = loraController?.getKnownLoras?.() || null;
    highlightContent.innerHTML = renderHighlight(textarea.value || "", textarea.selectionStart || 0, { knownLoras });
    sync();
  };
  textarea.addEventListener("input", () => { onChange(textarea.value); render(); });
  textarea.addEventListener("scroll", sync);
  textarea.addEventListener("click", render);
  textarea.addEventListener("keyup", render);
  textarea.addEventListener("select", render);
  textarea.addEventListener("focus", render);
  container.appendChild(highlight);
  container.appendChild(textarea);
  loraController = installLoraTokenController(textarea, { container, requestRender: render });
  render();
  const ro = new ResizeObserver(sync);
  ro.observe(textarea);
  return {
    container,
    destroy() {
      ro.disconnect();
      loraController?.destroy?.();
    },
  };
}

function nextId(prefix, items) {
  let max = 0;
  for (const item of items) {
    const m = String(item?.id || "").match(new RegExp(`^${prefix}(\\d+)$`));
    if (!m) continue;
    const n = Number(m[1]);
    if (Number.isFinite(n)) max = Math.max(max, n);
  }
  return `${prefix}${max + 1}`;
}

function normalizeState(raw) {
  let payload = {};
  if (raw && raw.trim()) {
    try { const parsed = JSON.parse(raw); if (parsed && typeof parsed === "object") payload = parsed; } catch { payload = {}; }
  }
  const sections = [];
  const sectionIds = new Set();
  for (const section of Array.isArray(payload.sections) ? payload.sections : []) {
    if (!section || typeof section !== "object") continue;
    let id = String(section.id || "").trim();
    if (id === "scene") id = "output";
    if (!id || sectionIds.has(id)) continue;
    sectionIds.add(id);
    sections.push({ id, name: String(section.name || id).trim() || id });
  }
  if (!sections.length) {
    for (const section of buildDefaultSections()) {
      sectionIds.add(section.id);
      sections.push(section);
    }
  }
  const legacyAutoSections = ["person", "features", "clothes", "environment", "action", "pose", "emotion"];
  const shouldCollapseLegacyDefaults = !sectionIds.has("options") && legacyAutoSections.some((id) => sectionIds.has(id));
  if (shouldCollapseLegacyDefaults) {
    sections.length = 0;
    sectionIds.clear();
    for (const section of buildDefaultSections()) {
      sections.push(section);
      sectionIds.add(section.id);
    }
  }
  if (!sectionIds.has("output")) {
    sections.push({ id: "output", name: "output" });
    sectionIds.add("output");
  }
  const outputSection = sections.find((s) => s.id === "output");
  if (outputSection) outputSection.name = "output";
  const defaultWorkSectionId = (sections.find((s) => s.id !== "output") || sections[0]).id;

  const nodes = [];
  const nodeIds = new Set();
  for (const node of Array.isArray(payload.nodes) ? payload.nodes : []) {
    if (!node || typeof node !== "object") continue;
    const id = String(node.id || "").trim();
    if (!id || nodeIds.has(id)) continue;
    nodeIds.add(id);
    const type = ["element", "sequential", "random", "output"].includes(String(node.type || "").toLowerCase()) ? String(node.type).toLowerCase() : "element";
    const rawSection = String(node.section_id || "").trim();
    const normalizedSectionId = rawSection === "scene" ? "output" : rawSection;
    const sectionId = type === "output"
      ? "output"
      : (sectionIds.has(normalizedSectionId) ? normalizedSectionId : defaultWorkSectionId);
    nodes.push({
      id,
      type,
      title: String(node.title || type),
      section_id: sectionId,
      x: Number.isFinite(Number(node.x)) ? Number(node.x) : 0,
      y: Number.isFinite(Number(node.y)) ? Number(node.y) : 0,
      template: String(node.template || ""),
    });
  }
  const sceneSectionId = "output";
  if (!nodes.some((n) => n.type === "output")) {
    nodes.push({
      id: nodeIds.has("n1") ? nextId("n", nodes) : "n1",
      type: "output",
      title: "Scene Output",
      section_id: sceneSectionId,
      x: 0,
      y: 0,
      template: "",
    });
  }
  for (const n of nodes) {
    if (n.type === "output") n.title = "Scene Output";
  }

  const edges = [];
  for (const edge of Array.isArray(payload.edges) ? payload.edges : []) {
    if (!edge || typeof edge !== "object") continue;
    const from = String(edge.from || "").trim();
    const to = String(edge.to || "").trim();
    if (!from || !to) continue;
    edges.push({ id: String(edge.id || nextId("e", edges)), from, to, weight: Number(edge.weight) || 1, order: Number.isFinite(Number(edge.order)) ? Number(edge.order) : edges.length, enabled: edge.enabled !== false });
  }

  const outputs = nodes.filter((n) => n.type === "output");
  let sceneOutputId = String(payload.scene_output_id || "");
  if (!outputs.some((n) => n.id === sceneOutputId)) sceneOutputId = (outputs.find((n) => n.section_id === "output") || outputs[0] || nodes[0]).id;

  const preferredActive = (sections.find((s) => s.id !== "output") || sections[0]).id;
  const rawActive = String(payload.active_section_id || "").trim();
  const activeSectionId = rawActive === "scene" ? "output" : rawActive;
  return {
    version: 1,
    sections,
    nodes,
    edges,
    active_section_id: sectionIds.has(activeSectionId) ? activeSectionId : preferredActive,
    scene_section_id: "output",
    scene_output_id: sceneOutputId,
  };
}

function serializeState(state) {
  return JSON.stringify({ version: 1, sections: state.sections, nodes: state.nodes, edges: state.edges, active_section_id: state.active_section_id, scene_section_id: state.scene_section_id, scene_output_id: state.scene_output_id }, null, 2);
}

function hasInput(node) { return node.type !== "element"; }
function hasOutput(node) { return node.type !== "output"; }
function preview(node) {
  if (node.type === "element") { const line = String(node.template || "").split("\n").find((it) => it.trim()) || "(empty template)"; return line.slice(0, 70); }
  if (node.type === "random") return "";
  if (node.type === "sequential") return "Concatenate inbound branches";
  return "Final scene output";
}
function summaryText(state) { return `${state.sections.length} sections | ${state.nodes.length} nodes | ${state.edges.length} links`; }
function buildDefaultSections() { return DEFAULT_SECTIONS.map((s) => ({ id: s.id, name: s.name })); }
function createTypeIcon(type) {
  const span = document.createElement("span");
  span.className = "ess-flow-type-icon";
  const key = String(type || "").toLowerCase();
  const icons = {
    element: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 14V8"/><path d="M8 8c0-3 2-5 5-5 0 3-2 5-5 5z"/><path d="M8 8c-2.5 0-4.5-1.7-5-4 2.8-.5 5 1.2 5 4z"/></svg>',
    random: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2.2" y="2.2" width="11.6" height="11.6" rx="2.2"/><circle class="fill" cx="5" cy="5" r="1"/><circle class="fill" cx="8" cy="8" r="1"/><circle class="fill" cx="11" cy="11" r="1"/></svg>',
    sequential: '<svg viewBox="0 0 16 16" aria-hidden="true"><rect x="2.5" y="3" width="11" height="3.2" rx="1.1"/><rect x="2.5" y="6.4" width="11" height="3.2" rx="1.1"/><rect x="2.5" y="9.8" width="11" height="3.2" rx="1.1"/></svg>',
    output: '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 5h11v8h-11z"/><path d="M2.5 5l2-2h7l2 2"/><path d="M6 3l1.2 2"/><path d="M9 3l1.2 2"/></svg>',
  };
  span.innerHTML = icons[key] || icons.element;
  return span;
}
function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function findNearestFreeRow(desired, used) {
  let row = desired;
  while (used.has(row)) row += 1;
  return row;
}
function findNearestFreeRowBalanced(desired, used) {
  if (!used.has(desired)) return desired;
  for (let dist = 1; dist <= 2000; dist += 1) {
    const up = desired - dist;
    if (!used.has(up)) return up;
    const down = desired + dist;
    if (!used.has(down)) return down;
  }
  return findNearestFreeRow(desired, used);
}
function columnPairGap(a, b) {
  if (a?.type === "element" && b?.type === "element") return COLUMN_PACK_GAP;
  return COLUMN_PACK_GAP_GROUP;
}
function inputSlotCount(node, inboundCount) {
  if (!hasInput(node)) return 0;
  return Math.max(1, Number(inboundCount) || 0);
}
function inputPortTopPadding(nodeType) {
  return nodeType === "random" ? RANDOM_INPUT_PORT_TOP_PADDING : INPUT_PORT_TOP_PADDING;
}
function inputPortBottomPadding(nodeType) {
  return nodeType === "random" ? RANDOM_INPUT_PORT_BOTTOM_PADDING : INPUT_PORT_BOTTOM_PADDING;
}
function estimateNodeHeight(node, inboundCount) {
  const slots = inputSlotCount(node, inboundCount);
  if (!slots) return NODE_HEIGHT_ESTIMATE;
  const dynamic = inputPortTopPadding(node.type) + inputPortBottomPadding(node.type) + Math.max(0, slots - 1) * INPUT_PORT_GAP;
  if (node.type === "random") return Math.max(RANDOM_NODE_MIN_HEIGHT, dynamic);
  return Math.max(NODE_HEIGHT_ESTIMATE, dynamic + 16);
}
function portCenterY(slotIndex, slotCount, nodeHeight, nodeType) {
  const topPadding = inputPortTopPadding(nodeType);
  if (slotCount <= 1) {
    if (nodeType === "random") return Math.max(topPadding, nodeHeight / 2);
    return nodeHeight / 2;
  }
  const spread = (slotCount - 1) * INPUT_PORT_GAP;
  const top = Math.max(topPadding, (nodeHeight - spread) / 2);
  return top + slotIndex * INPUT_PORT_GAP;
}
function pickLaneByX(x, layout) {
  if (!layout.lanes.length) return null;
  for (const lane of layout.lanes) {
    if (x >= lane.left && x <= lane.right) return lane;
  }
  let nearest = layout.lanes[0];
  let best = Math.abs((nearest.left + nearest.right) / 2 - x);
  for (const lane of layout.lanes.slice(1)) {
    const score = Math.abs((lane.left + lane.right) / 2 - x);
    if (score < best) { best = score; nearest = lane; }
  }
  return nearest;
}
function computeAutoLayout(state) {
  const nodeMap = new Map(state.nodes.map((n) => [n.id, n]));
  const nodeOrder = new Map(state.nodes.map((n, idx) => [n.id, idx]));
  const layoutEdges = state.edges.filter((edge) => nodeMap.has(edge.from) && nodeMap.has(edge.to));
  const outgoing = new Map();
  const incoming = new Map();
  for (const node of state.nodes) {
    outgoing.set(node.id, []);
    incoming.set(node.id, []);
  }
  for (const edge of layoutEdges) {
    outgoing.get(edge.from)?.push(edge.to);
    incoming.get(edge.to)?.push(edge.from);
  }

  const sectionData = [];
  const sectionNodesMap = new Map();
  for (const section of state.sections) {
    const nodes = state.nodes.filter((n) => n.section_id === section.id);
    sectionNodesMap.set(section.id, nodes);
    const ids = new Set(nodes.map((n) => n.id));

    const columns = new Map();
    for (const node of nodes) {
      if (!hasInput(node)) columns.set(node.id, 0);
    }
    const cap = Math.max(2, nodes.length + 1);
    for (let iter = 0; iter < cap * 2; iter += 1) {
      let changed = false;
      for (const node of nodes) {
        if (!hasInput(node)) continue;
        const inSectionInputs = (incoming.get(node.id) || []).filter((src) => ids.has(src));
        let next = 1;
        if (inSectionInputs.length) {
          next = Math.max(1, ...inSectionInputs.map((src) => (columns.has(src) ? columns.get(src) + 1 : 1)));
        }
        const prev = columns.get(node.id);
        if (prev == null || prev !== next) {
          columns.set(node.id, Math.min(cap, next));
          changed = true;
        }
      }
      for (const edge of layoutEdges) {
        if (!ids.has(edge.from) || !ids.has(edge.to)) continue;
        const fromCol = columns.get(edge.from) ?? (hasInput(nodeMap.get(edge.from)) ? 1 : 0);
        const toCol = columns.get(edge.to) ?? 1;
        if (toCol <= fromCol) {
          columns.set(edge.to, Math.min(cap, fromCol + 1));
          changed = true;
        }
      }
      if (!changed) break;
    }
    for (const node of nodes) {
      if (!columns.has(node.id)) columns.set(node.id, hasInput(node) ? 1 : 0);
    }
    const minCol = nodes.length ? Math.min(...nodes.map((n) => columns.get(n.id) ?? 0)) : 0;
    if (minCol > 0) {
      for (const node of nodes) {
        columns.set(node.id, (columns.get(node.id) ?? 0) - minCol);
      }
    }
    const maxCol = nodes.length ? Math.max(...nodes.map((n) => columns.get(n.id) ?? 0)) : 0;

    const rows = new Map();
    let sinkCounter = 0;
    for (let col = maxCol; col >= 0; col -= 1) {
      const colNodes = nodes.filter((n) => (columns.get(n.id) ?? 0) === col);
      const preferred = [];
      for (const node of colNodes) {
        const successors = (outgoing.get(node.id) || []).filter((dst) => ids.has(dst) && (columns.get(dst) ?? 0) > col && rows.has(dst));
        const pref = successors.length ? average(successors.map((dst) => rows.get(dst) ?? 0)) : sinkCounter++;
        preferred.push({ node, pref });
      }
      preferred.sort((a, b) => (a.pref - b.pref) || ((nodeOrder.get(a.node.id) ?? 0) - (nodeOrder.get(b.node.id) ?? 0)));
      const used = new Set();
      for (const item of preferred) {
        const row = findNearestFreeRow(Math.round(item.pref), used);
        used.add(row);
        rows.set(item.node.id, row);
      }
    }

    // Refine rows left-to-right: reserve a virtual row span for multi-input nodes,
    // so unrelated nodes in the same column are pushed below that occupied span.
    const overlaps = (aStart, aEnd, bStart, bEnd) => !(aEnd < bStart || aStart > bEnd);
    for (let col = 1; col <= maxCol; col += 1) {
      const colNodes = nodes
        .filter((n) => (columns.get(n.id) ?? 0) === col)
        .sort((a, b) => ((rows.get(a.id) ?? 0) - (rows.get(b.id) ?? 0)) || ((nodeOrder.get(a.id) ?? 0) - (nodeOrder.get(b.id) ?? 0)));
      const occupied = [];
      for (const node of colNodes) {
        const inputIds = (incoming.get(node.id) || [])
          .filter((src) => ids.has(src) && (columns.get(src) ?? 0) < col && rows.has(src));
        let baseRow = rows.get(node.id) ?? 0;
        let spanStart = baseRow;
        let spanEnd = baseRow;
        if (inputIds.length) {
          const inputRows = inputIds
            .map((src) => rows.get(src))
            .filter((value) => Number.isFinite(value));
          if (inputRows.length) {
            const minInput = Math.min(...inputRows);
            const maxInput = Math.max(...inputRows);
            baseRow = Math.round((minInput + maxInput) / 2);
            spanStart = minInput;
            spanEnd = maxInput;
          }
        }
        let shift = 0;
        while (occupied.some((item) => overlaps(spanStart + shift, spanEnd + shift, item.start, item.end))) {
          shift += 1;
          if (shift > 2000) break;
        }
        rows.set(node.id, baseRow + shift);
        occupied.push({ start: spanStart + shift, end: spanEnd + shift });
      }
    }

    // Recenter upstream columns against the adjusted downstream rows.
    for (let col = maxCol - 1; col >= 0; col -= 1) {
      const colNodes = nodes.filter((n) => (columns.get(n.id) ?? 0) === col);
      const ranked = [];
      for (const node of colNodes) {
        const inRows = (incoming.get(node.id) || [])
          .filter((src) => ids.has(src) && (columns.get(src) ?? 0) < col && rows.has(src))
          .map((src) => rows.get(src) ?? 0);
        const successors = (outgoing.get(node.id) || [])
          .filter((dst) => ids.has(dst) && (columns.get(dst) ?? 0) > col && rows.has(dst));
        let pref = rows.get(node.id) ?? 0;
        const outRows = successors.map((dst) => rows.get(dst) ?? 0);
        if (inRows.length && outRows.length) {
          // Keep groups aligned to inbound clusters, with mild pull to the next column.
          pref = ((average(inRows) * 2) + average(outRows)) / 3;
        } else if (inRows.length) {
          pref = average(inRows);
        } else if (outRows.length) {
          pref = average(outRows);
        }
        ranked.push({ node, pref });
      }
      ranked.sort((a, b) => (a.pref - b.pref) || ((nodeOrder.get(a.node.id) ?? 0) - (nodeOrder.get(b.node.id) ?? 0)));
      const used = new Set();
      for (const item of ranked) {
        const row = findNearestFreeRow(Math.round(item.pref), used);
        used.add(row);
        rows.set(item.node.id, row);
      }
    }

    // Final inbound anchoring for group columns: place each node at the center
    // of its current input cluster after upstream rows have settled.
    for (let col = 1; col <= maxCol; col += 1) {
      const ranked = [];
      for (const node of nodes.filter((n) => (columns.get(n.id) ?? 0) === col)) {
        const inputRows = (incoming.get(node.id) || [])
          .filter((src) => ids.has(src) && (columns.get(src) ?? 0) < col && rows.has(src))
          .map((src) => rows.get(src) ?? 0);
        const pref = inputRows.length ? average(inputRows) : (rows.get(node.id) ?? 0);
        const key = inputRows.length
          ? [...inputRows].sort((a, b) => a - b).join(",")
          : `node:${node.id}`;
        ranked.push({ node, inputRows, pref, key });
      }
      ranked.sort((a, b) => (a.pref - b.pref) || ((nodeOrder.get(a.node.id) ?? 0) - (nodeOrder.get(b.node.id) ?? 0)));
      const usedRows = new Set();
      const rowsByKey = new Map();
      for (const item of ranked) {
        let baseRow = Math.round(item.pref);
        if (item.inputRows.length) {
          baseRow = Math.round(average(item.inputRows));
        }
        const keyRows = rowsByKey.get(item.key);
        let desiredRow = baseRow;
        if (keyRows && keyRows.length) {
          desiredRow = Math.round((baseRow + average(keyRows)) / 2);
        }
        const row = findNearestFreeRowBalanced(desiredRow, usedRows);
        rows.set(item.node.id, row);
        usedRows.add(row);
        const list = rowsByKey.get(item.key) || [];
        list.push(row);
        rowsByKey.set(item.key, list);
      }
    }

    // Compact unused virtual rows to avoid large visual gaps between independent clusters.
    const compactRows = Array.from(new Set(nodes.map((n) => rows.get(n.id) ?? 0))).sort((a, b) => a - b);
    const compactIndex = new Map(compactRows.map((row, index) => [row, index]));
    for (const node of nodes) {
      const row = rows.get(node.id) ?? 0;
      rows.set(node.id, compactIndex.get(row) ?? row);
    }

    const uniqueRows = Array.from(new Set(nodes.map((n) => rows.get(n.id) ?? 0))).sort((a, b) => a - b);
    const nodeHeights = new Map();
    for (const node of nodes) {
      // Count all inbound links (including cross-section links) so node height
      // matches rendered input rows.
      const inCount = (incoming.get(node.id) || []).length;
      nodeHeights.set(node.id, estimateNodeHeight(node, inCount));
    }
    const rowHeights = new Map();
    for (const row of uniqueRows) {
      const inRow = nodes.filter((n) => (rows.get(n.id) ?? 0) === row);
      const maxHeight = inRow.length ? Math.max(...inRow.map((n) => nodeHeights.get(n.id) ?? NODE_HEIGHT_ESTIMATE)) : NODE_HEIGHT_ESTIMATE;
      rowHeights.set(row, maxHeight);
    }
    const rowCenters = new Map();
    let cursor = 0;
    let prevHeight = 0;
    uniqueRows.forEach((row, index) => {
      const height = rowHeights.get(row) ?? NODE_HEIGHT_ESTIMATE;
      if (index === 0) {
        cursor = height / 2;
      } else {
        cursor += prevHeight / 2 + ROW_GAP + height / 2;
      }
      rowCenters.set(row, cursor);
      prevHeight = height;
    });
    const contentHeight = uniqueRows.length ? (cursor + prevHeight / 2) : NODE_HEIGHT_ESTIMATE;
    const colCount = Math.max(1, maxCol + 1);
    const contentWidth = colCount * NODE_WIDTH + (colCount - 1) * COLUMN_GAP;
    const width = Math.max(260, LANE_INNER_LEFT + contentWidth + LANE_INNER_RIGHT);

    sectionData.push({
      section,
      nodes,
      columns,
      rows,
      rowCenters,
      nodeHeights,
      contentHeight,
      colCount,
      width,
    });
  }

  const globalContentHeight = Math.max(...sectionData.map((s) => s.contentHeight), NODE_HEIGHT_ESTIMATE);
  let laneHeight = LANE_HEADER_HEIGHT + LANE_INNER_TOP + globalContentHeight + LANE_INNER_BOTTOM;
  let boardHeight = LANE_TOP * 2 + laneHeight;

  let left = BOARD_SIDE_PADDING;
  const lanes = [];
  const nodePositions = new Map();
  const nodeHeights = new Map();
  for (const data of sectionData) {
    const lane = {
      id: data.section.id,
      name: data.section.name,
      left,
      width: data.width,
      right: left + data.width,
    };
    lanes.push(lane);
    const sectionYOffset = LANE_INNER_TOP + Math.round((globalContentHeight - data.contentHeight) / 2);
    for (const node of data.nodes) {
      const col = data.columns.get(node.id) ?? 0;
      const row = data.rows.get(node.id) ?? 0;
      const centerY = data.rowCenters.get(row) ?? 0;
      const nodeHeight = data.nodeHeights.get(node.id) ?? NODE_HEIGHT_ESTIMATE;
      const x = lane.left + LANE_INNER_LEFT + col * (NODE_WIDTH + COLUMN_GAP);
      const y = LANE_TOP + LANE_HEADER_HEIGHT + sectionYOffset + centerY - nodeHeight / 2;
      nodePositions.set(node.id, { x, y, col, row });
      nodeHeights.set(node.id, nodeHeight);
    }
    left += data.width + LANE_GAP;
  }

  // Column compaction pass: tighten vertical spacing within each section/column
  // while preserving top-to-bottom order and preventing overlap.
  for (const data of sectionData) {
    const byCol = new Map();
    for (const node of data.nodes) {
      const col = data.columns.get(node.id) ?? 0;
      if (!byCol.has(col)) byCol.set(col, []);
      byCol.get(col).push(node);
    }
    for (const colNodes of byCol.values()) {
      if (colNodes.length < 2) continue;
      colNodes.sort((a, b) => {
        const pa = nodePositions.get(a.id);
        const pb = nodePositions.get(b.id);
        return (pa?.y ?? 0) - (pb?.y ?? 0);
      });
      const centers = colNodes.map((n) => {
        const pos = nodePositions.get(n.id);
        const h = nodeHeights.get(n.id) ?? NODE_HEIGHT_ESTIMATE;
        return (pos ? (pos.y + h / 2) : 0);
      });
      const targetCenter = average(centers);
      let packedHeight = 0;
      for (let i = 0; i < colNodes.length; i += 1) {
        const h = nodeHeights.get(colNodes[i].id) ?? NODE_HEIGHT_ESTIMATE;
        packedHeight += h;
        if (i > 0) packedHeight += columnPairGap(colNodes[i - 1], colNodes[i]);
      }
      let top = targetCenter - packedHeight / 2;
      for (let i = 0; i < colNodes.length; i += 1) {
        const node = colNodes[i];
        const pos = nodePositions.get(node.id);
        if (!pos) continue;
        const h = nodeHeights.get(node.id) ?? NODE_HEIGHT_ESTIMATE;
        pos.y = Math.round(top);
        top += h;
        if (i < colNodes.length - 1) top += columnPairGap(node, colNodes[i + 1]);
      }
    }
  }

  // Recompute section bounds after compaction and tighten board/lane height to
  // actual content instead of pre-compaction estimates.
  const sectionBounds = new Map();
  let adjustedGlobalContentHeight = NODE_HEIGHT_ESTIMATE;
  for (const data of sectionData) {
    if (!data.nodes.length) {
      const minTop = LANE_TOP + LANE_HEADER_HEIGHT + LANE_INNER_TOP;
      const maxBottom = minTop + NODE_HEIGHT_ESTIMATE;
      sectionBounds.set(data.section.id, { minTop, maxBottom, height: NODE_HEIGHT_ESTIMATE });
      continue;
    }
    let minTop = Infinity;
    let maxBottom = -Infinity;
    for (const node of data.nodes) {
      const pos = nodePositions.get(node.id);
      const h = nodeHeights.get(node.id) ?? NODE_HEIGHT_ESTIMATE;
      if (!pos) continue;
      minTop = Math.min(minTop, pos.y);
      maxBottom = Math.max(maxBottom, pos.y + h);
    }
    if (!Number.isFinite(minTop) || !Number.isFinite(maxBottom)) {
      minTop = LANE_TOP + LANE_HEADER_HEIGHT + LANE_INNER_TOP;
      maxBottom = minTop + NODE_HEIGHT_ESTIMATE;
    }
    const height = Math.max(NODE_HEIGHT_ESTIMATE, maxBottom - minTop);
    sectionBounds.set(data.section.id, { minTop, maxBottom, height });
    adjustedGlobalContentHeight = Math.max(adjustedGlobalContentHeight, height);
  }

  for (const data of sectionData) {
    const bounds = sectionBounds.get(data.section.id);
    if (!bounds) continue;
    const desiredTop = LANE_TOP + LANE_HEADER_HEIGHT + LANE_INNER_TOP + Math.round((adjustedGlobalContentHeight - bounds.height) / 2);
    const delta = desiredTop - bounds.minTop;
    if (!delta) continue;
    for (const node of data.nodes) {
      const pos = nodePositions.get(node.id);
      if (pos) pos.y += delta;
    }
  }

  laneHeight = LANE_HEADER_HEIGHT + LANE_INNER_TOP + adjustedGlobalContentHeight + LANE_INNER_BOTTOM;
  boardHeight = LANE_TOP * 2 + laneHeight;

  const boardWidth = Math.max(BOARD_SIDE_PADDING * 2 + 20, left - LANE_GAP + BOARD_SIDE_PADDING);
  return {
    lanes,
    byId: new Map(lanes.map((lane) => [lane.id, lane])),
    nodePositions,
    nodeHeights,
    boardWidth,
    boardHeight,
    laneHeight,
  };
}

function promptText(message, defaultValue) {
  try {
    if (typeof window.prompt === "function") {
      const value = window.prompt(message, defaultValue);
      if (value != null) return String(value);
    }
  } catch {}
  return String(defaultValue || "");
}

function alertText(message) {
  try {
    if (typeof window.alert === "function") {
      window.alert(message);
      return;
    }
  } catch {}
  console.warn(`[ess_scene_flow_editor] ${message}`);
}

function createOverlay(node, widget, stateRef, updateSummary) {
  const state = normalizeState(stateRef.value || "");
  let savedSignature = JSON.stringify({
    sections: state.sections,
    nodes: state.nodes,
    edges: state.edges,
    scene_output_id: state.scene_output_id,
  });
  let hasUnsavedChanges = false;
  let selectedNodeId = null;
  let pendingSourceId = null;
  let pendingPointer = null;
  let pendingStartPointer = null;
  let pendingDetachedEdge = null;
  let layoutCache = null;
  let inspectorCleanup = [];
  let nodeCardMap = new Map();
  let renamingSectionId = null;
  let renamingSectionValue = "";
  let renamingSectionSource = "sidebar";
  let renamingNodeId = null;
  let renamingNodeValue = "";
  let closePromptOpen = false;

  const overlay = document.createElement("div");
  overlay.className = "ess-flow-overlay";
  const topbar = document.createElement("div");
  topbar.className = "ess-flow-topbar";
  const title = document.createElement("h3");
  title.textContent = "ESS Scene Flow Editor";
  const testRunButton = document.createElement("button");
  testRunButton.textContent = "Test run";
  const saveButton = document.createElement("button");
  saveButton.className = "primary";
  saveButton.textContent = "Save";
  const closeButton = document.createElement("button");
  closeButton.textContent = "Close";
  const spacer = document.createElement("div");
  spacer.className = "spacer";
  topbar.appendChild(title);
  topbar.appendChild(spacer);
  topbar.appendChild(testRunButton);
  topbar.appendChild(saveButton);
  topbar.appendChild(closeButton);

  const main = document.createElement("div");
  main.className = "ess-flow-main";

  const left = document.createElement("div");
  left.className = "ess-flow-panel";
  left.innerHTML = '<h4>Sections & Nodes</h4>';
  const leftBody = document.createElement("div");
  leftBody.className = "ess-flow-body";
  left.appendChild(leftBody);

  const center = document.createElement("div");
  center.className = "ess-flow-panel";
  center.innerHTML = '<h4>Flow Graph</h4>';
  const canvasWrap = document.createElement("div");
  canvasWrap.className = "ess-flow-canvas";
  const board = document.createElement("div");
  board.className = "ess-flow-board";
  const laneLayer = document.createElement("div");
  laneLayer.className = "ess-flow-lanes";
  const edgeSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  edgeSvg.classList.add("ess-flow-edges");
  const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
  edgeSvg.appendChild(edgeLayer);
  const nodeLayer = document.createElement("div");
  nodeLayer.style.position = "absolute";
  nodeLayer.style.inset = "0";
  nodeLayer.style.pointerEvents = "none";
  board.appendChild(laneLayer);
  board.appendChild(edgeSvg);
  board.appendChild(nodeLayer);
  canvasWrap.appendChild(board);
  center.appendChild(canvasWrap);

  const right = document.createElement("div");
  right.className = "ess-flow-panel";
  right.innerHTML = '<h4>Inspector</h4>';
  const rightBody = document.createElement("div");
  rightBody.className = "ess-flow-body";
  right.appendChild(rightBody);

  main.appendChild(left);
  main.appendChild(center);
  main.appendChild(right);
  overlay.appendChild(topbar);
  overlay.appendChild(main);
  document.body.appendChild(overlay);

  const keyHandler = (event) => {
    if (!overlay.isConnected) return;
    if (event.key === "Escape") {
      event.preventDefault();
      void attemptCloseOverlay();
      return;
    }
    if ((event.key === "Delete" || event.key === "Backspace") && selectedNodeId) {
      const active = document.activeElement;
      const editing = active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT" || active.isContentEditable);
      if (editing) return;
      event.preventDefault();
      removeNode(selectedNodeId);
      renderAll();
      commitState();
    }
  };
  window.addEventListener("keydown", keyHandler);

  function closeOverlay() {
    for (const fn of inspectorCleanup) { try { fn(); } catch {} }
    inspectorCleanup = [];
    window.removeEventListener("keydown", keyHandler);
    overlay.remove();
  }

  function stateSignature() {
    return JSON.stringify({
      sections: state.sections,
      nodes: state.nodes,
      edges: state.edges,
      scene_output_id: state.scene_output_id,
    });
  }
  function refreshDirtyUi() {
    saveButton.textContent = hasUnsavedChanges ? "Save *" : "Save";
  }
  function showCloseConfirmModal() {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.className = "ess-flow-modal-backdrop";
      const modal = document.createElement("div");
      modal.className = "ess-flow-modal";
      const title = document.createElement("h5");
      title.textContent = "Unsaved changes";
      const message = document.createElement("p");
      message.textContent = "Save changes before closing?";
      const actions = document.createElement("div");
      actions.className = "ess-flow-modal-actions";
      const noBtn = document.createElement("button");
      noBtn.textContent = "No";
      noBtn.title = "Discard changes and close";
      const yesBtn = document.createElement("button");
      yesBtn.className = "primary";
      yesBtn.textContent = "Yes";
      yesBtn.title = "Save changes and close";
      actions.appendChild(noBtn);
      actions.appendChild(yesBtn);
      modal.appendChild(title);
      modal.appendChild(message);
      modal.appendChild(actions);
      backdrop.appendChild(modal);
      overlay.appendChild(backdrop);
      yesBtn.focus();

      const finish = (answer) => {
        window.removeEventListener("keydown", onKey, true);
        backdrop.remove();
        resolve(answer);
      };
      const onKey = (event) => {
        if (!overlay.isConnected) return;
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          finish(false);
        }
      };
      window.addEventListener("keydown", onKey, true);
      yesBtn.onclick = () => finish(true);
      noBtn.onclick = () => finish(false);
    });
  }
  function getNodeWidgetValue(name, fallback) {
    if (!Array.isArray(node.widgets)) return fallback;
    const widget = node.widgets.find((w) => String(w?.name || "").trim() === name);
    if (!widget) return fallback;
    return widget.value != null ? widget.value : fallback;
  }
  function getNodeWidgetValueAny(names, fallback) {
    if (!Array.isArray(node.widgets)) return fallback;
    const normalized = names.map((n) => String(n || "").trim().toLowerCase());
    const widget = node.widgets.find((w) => normalized.includes(String(w?.name || "").trim().toLowerCase()));
    if (!widget) return fallback;
    return widget.value != null ? widget.value : fallback;
  }
  function normalizeBoolean(value, fallback = true) {
    if (value == null) return fallback;
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    if (typeof value === "string") {
      const v = value.trim().toLowerCase();
      if (["1", "true", "yes", "on", "parse"].includes(v)) return true;
      if (["0", "false", "no", "off", "raw"].includes(v)) return false;
    }
    return fallback;
  }
  function normalizeSeedNumber(value, fallback = 0) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    const clamped = Math.max(0, Math.floor(n));
    return clamped;
  }
  function randomSeedNumber() {
    return Math.floor(Math.random() * 0x1fffffffffffff);
  }
  function normalizeControlMode(value, fallback = "fixed") {
    const raw = String(value == null ? "" : value).trim().toLowerCase();
    if (["randomize", "random"].includes(raw)) return "randomize";
    if (["fixed", "keep"].includes(raw)) return "fixed";
    if (["increase", "increment", "inc", "up"].includes(raw)) return "increase";
    if (["decrease", "decrement", "dec", "down"].includes(raw)) return "decrease";
    return fallback;
  }
  function splitLocalPrompt(text) {
    const raw = String(text || "");
    const marker = raw.indexOf("!>");
    if (marker < 0) return { positive: raw.trim(), negative: "" };
    const positive = raw.slice(0, marker).trim();
    const negative = raw.slice(marker + 2).trim();
    return { positive, negative };
  }
  function hashStringSeed(text, baseSeed) {
    let h = 2166136261 >>> 0;
    const src = `${baseSeed}|${text}`;
    for (let i = 0; i < src.length; i += 1) {
      h ^= src.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }
  function makeRng(seedValue) {
    let a = seedValue >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function pickWeightedIndex(weights, rand01) {
    const total = weights.reduce((sum, w) => sum + Math.max(0, Number(w) || 0), 0);
    if (total <= 0) return -1;
    let t = rand01() * total;
    for (let i = 0; i < weights.length; i += 1) {
      t -= Math.max(0, Number(weights[i]) || 0);
      if (t <= 0) return i;
    }
    return weights.length - 1;
  }
  function runLocalFlowPreview(flowScript, seed, parseTemplates) {
    const localState = normalizeState(flowScript || "");
    const nodes = Array.isArray(localState.nodes) ? localState.nodes : [];
    const edges = Array.isArray(localState.edges) ? localState.edges : [];
    const nodeById = new Map(nodes.map((n) => [String(n.id), n]));
    const inbound = new Map();
    for (const edge of edges) {
      const from = String(edge?.from || "");
      const to = String(edge?.to || "");
      if (!from || !to) continue;
      if (!nodeById.has(from) || !nodeById.has(to)) continue;
      if (edge.enabled === false) continue;
      if (!inbound.has(to)) inbound.set(to, []);
      inbound.get(to).push(edge);
    }
    for (const [to, list] of inbound.entries()) {
      list.sort((a, b) => ((Number(a.order ?? 0) - Number(b.order ?? 0)) || String(a.id || "").localeCompare(String(b.id || ""))));
      inbound.set(to, list);
    }
    let outputId = String(localState.scene_output_id || "");
    if (!outputId || !nodeById.has(outputId)) {
      const out = nodes.find((n) => String(n.type || "").toLowerCase() === "output");
      outputId = out ? String(out.id) : "";
    }
    const cache = new Map();
    const stack = new Set();
    const joinText = (parts) => parts.map((p) => String(p || "").trim()).filter(Boolean).join("\n");
    const evalNode = (nodeId) => {
      const id = String(nodeId || "");
      if (!id) return { positive: "", negative: "" };
      if (cache.has(id)) return cache.get(id);
      if (stack.has(id)) return { positive: "", negative: "" };
      const nodeObj = nodeById.get(id);
      if (!nodeObj) return { positive: "", negative: "" };
      stack.add(id);
      const type = String(nodeObj.type || "element").toLowerCase();
      let result = { positive: "", negative: "" };
      if (type === "element") {
        const template = String(nodeObj.template || "");
        if (parseTemplates) result = splitLocalPrompt(template);
        else result = { positive: template.trim(), negative: "" };
      } else if (type === "random") {
        const list = (inbound.get(id) || []).filter((e) => Number(e.weight ?? 1) > 0);
        if (!list.length) {
          result = { positive: "", negative: "" };
        } else {
          const rng = makeRng(hashStringSeed(`random-node:${id}`, seed));
          const idx = pickWeightedIndex(list.map((e) => Number(e.weight ?? 1) || 1), rng);
          const selected = idx >= 0 ? list[idx] : list[0];
          result = evalNode(String(selected.from || ""));
        }
      } else {
        const pos = [];
        const neg = [];
        for (const edge of inbound.get(id) || []) {
          const part = evalNode(String(edge.from || ""));
          if (part.positive?.trim()) pos.push(part.positive);
          if (part.negative?.trim()) neg.push(part.negative);
        }
        result = { positive: joinText(pos), negative: joinText(neg) };
      }
      stack.delete(id);
      cache.set(id, result);
      return result;
    };
    if (!outputId) return { positive: "", negative: "" };
    return evalNode(outputId);
  }
  async function executeTestFlow(flowScript, seed, parseTemplates) {
    const payload = {
      flow_script: flowScript,
      seed,
      parse_templates: parseTemplates,
    };

    const tryLocalPreview = (errorText, noteText = "Backend test endpoint unavailable; local preview used.") => {
      const local = runLocalFlowPreview(flowScript, seed, parseTemplates);
      return {
        positive: local.positive || "",
        negative: local.negative || "",
        error: String(errorText || ""),
        note: noteText,
        mode: "local-preview",
      };
    };

    try {
      const postResponse = await fetch("/ess/scene_flow/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const postData = await postResponse.json().catch(() => ({}));
      if (postResponse.ok && postData?.ok !== false) {
        return {
          positive: postData?.positive || "",
          negative: postData?.negative || "",
          mode: "server",
        };
      }

      const postError = String(postData?.error || `HTTP ${postResponse.status}`);

      // Some ComfyUI setups expose custom routes as GET-only.
      try {
        const params = new URLSearchParams({
          flow_script: String(flowScript || ""),
          seed: String(seed),
          parse_templates: String(!!parseTemplates),
        });
        const getResponse = await fetch(`/ess/scene_flow/test?${params.toString()}`, {
          method: "GET",
        });
        const getData = await getResponse.json().catch(() => ({}));
        if (getResponse.ok && getData?.ok !== false) {
          return {
            positive: getData?.positive || "",
            negative: getData?.negative || "",
            mode: "server",
          };
        }
        const getError = String(getData?.error || `HTTP ${getResponse.status}`);
        return tryLocalPreview(`${postError}; GET fallback failed: ${getError}`);
      } catch (getError) {
        return tryLocalPreview(`${postError}; GET fallback failed: ${String(getError?.message || getError || "unknown")}`);
      }
    } catch (error) {
      return tryLocalPreview(String(error?.message || error || "Test run failed"), "Server request failed; local preview used.");
    }
  }
  function showTestRunResultModal(options) {
    return new Promise((resolve) => {
      const flowScript = String(options?.flowScript || "");
      const parseTemplates = !!options?.parseTemplates;
      let currentSeed = normalizeSeedNumber(options?.seed, 0);
      let controlMode = normalizeControlMode(options?.controlMode, "fixed");

      const backdrop = document.createElement("div");
      backdrop.className = "ess-flow-modal-backdrop";
      const modal = document.createElement("div");
      modal.className = "ess-flow-modal ess-flow-test-modal";
      const header = document.createElement("h5");
      header.textContent = "Test Run Result";

      const toolbar = document.createElement("div");
      toolbar.className = "ess-flow-test-toolbar";
      const seedField = document.createElement("div");
      seedField.className = "ess-flow-test-field";
      const seedLabel = document.createElement("label");
      seedLabel.textContent = "Seed";
      const seedInput = document.createElement("input");
      seedInput.type = "number";
      seedInput.min = "0";
      seedInput.step = "1";
      seedInput.value = String(currentSeed);
      seedField.appendChild(seedLabel);
      seedField.appendChild(seedInput);

      const modeField = document.createElement("div");
      modeField.className = "ess-flow-test-field";
      const modeLabel = document.createElement("label");
      modeLabel.textContent = "Mode";
      const modeSelect = document.createElement("select");
      for (const mode of ["randomize", "fixed", "increase", "decrease"]) {
        const option = document.createElement("option");
        option.value = mode;
        option.textContent = mode;
        if (mode === controlMode) option.selected = true;
        modeSelect.appendChild(option);
      }
      modeField.appendChild(modeLabel);
      modeField.appendChild(modeSelect);

      const updateBtn = document.createElement("button");
      updateBtn.className = "primary";
      updateBtn.textContent = "Update";
      updateBtn.style.marginLeft = "auto";
      toolbar.appendChild(seedField);
      toolbar.appendChild(modeField);
      toolbar.appendChild(updateBtn);

      const grid = document.createElement("div");
      grid.className = "ess-flow-test-grid";
      const mkCol = (labelText) => {
        const col = document.createElement("div");
        col.className = "ess-flow-test-col";
        const label = document.createElement("label");
        label.textContent = labelText;
        const box = document.createElement("textarea");
        box.readOnly = true;
        col.appendChild(label);
        col.appendChild(box);
        return { col, box };
      };
      const positiveCol = mkCol("Positive");
      const negativeCol = mkCol("Negative");
      grid.appendChild(positiveCol.col);
      grid.appendChild(negativeCol.col);

      const meta = document.createElement("div");
      meta.className = "ess-flow-test-meta";

      const actions = document.createElement("div");
      actions.className = "ess-flow-modal-actions";
      const closeBtn = document.createElement("button");
      closeBtn.className = "primary";
      closeBtn.textContent = "Close";
      actions.appendChild(closeBtn);

      modal.appendChild(header);
      modal.appendChild(toolbar);
      modal.appendChild(grid);
      modal.appendChild(meta);
      modal.appendChild(actions);
      backdrop.appendChild(modal);
      overlay.appendChild(backdrop);

      const setRunning = (running) => {
        updateBtn.disabled = running;
        seedInput.disabled = running;
        modeSelect.disabled = running;
        updateBtn.textContent = running ? "Running..." : "Update";
      };
      const applyResult = (result, usedSeed) => {
        positiveCol.box.value = String(result?.positive || "");
        negativeCol.box.value = String(result?.negative || "");
        let text = `Seed: ${usedSeed} | Parse templates: ${parseTemplates ? "on" : "off"} | Mode: ${controlMode}`;
        if (result?.note) text += ` | ${result.note}`;
        if (result?.error && !(result.positive || result.negative)) text += ` | Error: ${result.error}`;
        else if (result?.error) text += ` | Warning: ${result.error}`;
        meta.textContent = text;
      };
      const updateSeedForNext = (usedSeed) => {
        if (controlMode === "increase") currentSeed = usedSeed + 1;
        else if (controlMode === "decrease") currentSeed = Math.max(0, usedSeed - 1);
        else if (controlMode === "randomize") currentSeed = randomSeedNumber();
        else currentSeed = usedSeed;
        seedInput.value = String(currentSeed);
      };
      const runOne = async () => {
        controlMode = normalizeControlMode(modeSelect.value, "fixed");
        currentSeed = normalizeSeedNumber(seedInput.value, currentSeed);
        const usedSeed = currentSeed;
        setRunning(true);
        const result = await executeTestFlow(flowScript, usedSeed, parseTemplates);
        setRunning(false);
        applyResult(result, usedSeed);
        updateSeedForNext(usedSeed);
      };

      const finish = () => {
        window.removeEventListener("keydown", onKey, true);
        backdrop.remove();
        resolve();
      };
      const onKey = (event) => {
        if (!overlay.isConnected) return;
        const active = document.activeElement;
        const editing = active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA" || active.tagName === "SELECT" || active.isContentEditable);
        if (event.key === "Escape" || event.key === "Enter") {
          if (event.key === "Enter" && editing) return;
          event.preventDefault();
          event.stopPropagation();
          if (updateBtn.disabled) return;
          finish();
        }
      };
      window.addEventListener("keydown", onKey, true);
      closeBtn.onclick = () => finish();
      updateBtn.onclick = () => { void runOne(); };
      closeBtn.focus();
      void runOne();
    });
  }
  async function runTestFlow() {
    const seedRaw = getNodeWidgetValueAny(["seed"], 0);
    const seed = normalizeSeedNumber(seedRaw, 0);
    const controlRaw = getNodeWidgetValueAny(["control_after_generate", "control after generate"], "fixed");
    const controlMode = normalizeControlMode(controlRaw, "fixed");
    const parseTemplates = normalizeBoolean(getNodeWidgetValue("parse_templates", true), true);
    const flowScript = serializeState(state);
    testRunButton.disabled = true;
    const oldLabel = testRunButton.textContent;
    testRunButton.textContent = "Opening...";
    try {
      await showTestRunResultModal({
        flowScript,
        parseTemplates,
        seed,
        controlMode,
      });
    } catch (error) {
      await showTestRunResultModal({
        flowScript,
        parseTemplates,
        seed,
        controlMode,
      });
    } finally {
      testRunButton.disabled = false;
      testRunButton.textContent = oldLabel || "Test run";
    }
  }
  function commitState(persist = false) {
    hasUnsavedChanges = stateSignature() !== savedSignature;
    refreshDirtyUi();
    if (!persist) return;
    ensureSectionOrder();
    pruneEdges();
    refreshSceneSelectors();
    const json = serializeState(state);
    stateRef.value = json;
    widget.value = json;
    if (typeof widget.callback === "function") widget.callback(json);
    if (Array.isArray(node.widgets_values)) {
      const idx = node.widgets?.indexOf(widget);
      if (idx != null && idx >= 0) node.widgets_values[idx] = json;
    }
    savedSignature = stateSignature();
    hasUnsavedChanges = false;
    refreshDirtyUi();
    updateSummary(state);
    node.setDirtyCanvas?.(true, true);
  }
  async function attemptCloseOverlay() {
    if (closePromptOpen) return;
    commitState();
    if (hasUnsavedChanges) {
      closePromptOpen = true;
      const saveBeforeClose = await showCloseConfirmModal();
      closePromptOpen = false;
      if (saveBeforeClose) commitState(true);
    }
    closeOverlay();
  }

  function byId(id) { return state.nodes.find((n) => n.id === id) || null; }
  function getFirstNonOutputSectionId() {
    return (state.sections.find((s) => s.id !== "output") || state.sections[0] || { id: "output" }).id;
  }
  function ensureWorkSection() {
    const active = state.sections.find((s) => s.id === state.active_section_id && s.id !== "output");
    if (active) return active.id;
    const existing = state.sections.find((s) => s.id !== "output");
    if (existing) return existing.id;
    let id = "options";
    let i = 2;
    while (state.sections.some((s) => s.id === id)) {
      id = `section_${i}`;
      i += 1;
    }
    state.sections.unshift({ id, name: id });
    state.active_section_id = id;
    return id;
  }
  function ensureSectionOrder() {
    const output = state.sections.find((s) => s.id === "output") || { id: "output", name: "output" };
    output.name = "output";
    const others = state.sections.filter((s) => s.id !== "output");
    state.sections = [...others, output];
    if (!state.sections.some((s) => s.id === state.active_section_id)) {
      state.active_section_id = getFirstNonOutputSectionId();
    }
  }
  function moveSectionById(sectionId, delta) {
    if (!sectionId || sectionId === "output") return false;
    const movable = state.sections.filter((s) => s.id !== "output");
    const idx = movable.findIndex((s) => s.id === sectionId);
    if (idx < 0) return false;
    const next = idx + delta;
    if (next < 0 || next >= movable.length) return false;
    const temp = movable[next];
    movable[next] = movable[idx];
    movable[idx] = temp;
    const output = state.sections.find((s) => s.id === "output") || { id: "output", name: "output" };
    state.sections = [...movable, output];
    return true;
  }
  function applySectionRename(sectionId, value) {
    if (sectionId === "output") {
      cancelSectionRename();
      return;
    }
    const target = state.sections.find((s) => s.id === sectionId);
    if (target) {
      const next = String(value || "").trim();
      target.name = next || target.name;
    }
    renamingSectionId = null;
    renamingSectionValue = "";
    renamingSectionSource = "sidebar";
    refreshSceneSelectors();
    renderAll();
    commitState();
  }
  function cancelSectionRename() {
    renamingSectionId = null;
    renamingSectionValue = "";
    renamingSectionSource = "sidebar";
    renderAll();
  }
  function startSectionRename(sectionId, source = "sidebar") {
    const target = state.sections.find((s) => s.id === sectionId);
    if (!target || target.id === "output") return;
    renamingSectionId = target.id;
    renamingSectionValue = target.name;
    renamingSectionSource = source === "lane" ? "lane" : "sidebar";
    renderSections();
    renderGraph();
    setTimeout(() => {
      const input = renamingSectionSource === "lane"
        ? laneLayer.querySelector(`input[data-section-rename="${sectionId}"]`)
        : leftBody.querySelector(".ess-flow-section input");
      if (input) { input.focus(); input.select(); }
    }, 0);
  }
  function applyNodeRename(nodeId, value) {
    const target = byId(nodeId);
    if (!target || target.type === "output") {
      renamingNodeId = null;
      renamingNodeValue = "";
      renderGraph();
      renderInspector();
      return;
    }
    const next = String(value || "").trim();
    target.title = next || target.title || target.type;
    renamingNodeId = null;
    renamingNodeValue = "";
    renderGraph();
    renderInspector();
    commitState();
  }
  function cancelNodeRename() {
    renamingNodeId = null;
    renamingNodeValue = "";
    renderGraph();
    renderInspector();
  }
  function startNodeRename(nodeId) {
    const target = byId(nodeId);
    if (!target || target.type === "output") return;
    renamingNodeId = target.id;
    renamingNodeValue = target.title || "";
    selectedNodeId = target.id;
    renderGraph();
    renderInspector();
    setTimeout(() => {
      const input = nodeLayer.querySelector(`input[data-node-rename="${nodeId}"]`);
      if (input) { input.focus(); input.select(); }
    }, 0);
  }
  function pruneEdges() {
    const map = new Map(state.nodes.map((n) => [n.id, n]));
    state.edges = state.edges.filter((e) => {
      const s = map.get(e.from); const t = map.get(e.to);
      return !!s && !!t && hasOutput(s) && hasInput(t);
    });
    // Output nodes accept only one inbound edge; keep the latest by order/id.
    const byTarget = new Map();
    for (const edge of state.edges) {
      if (!byTarget.has(edge.to)) byTarget.set(edge.to, []);
      byTarget.get(edge.to).push(edge);
    }
    const keep = new Set();
    for (const [targetId, edges] of byTarget.entries()) {
      const target = map.get(targetId);
      if (!target || target.type !== "output") {
        for (const edge of edges) keep.add(edge.id);
        continue;
      }
      edges.sort((a, b) => ((a.order ?? 0) - (b.order ?? 0)) || String(a.id).localeCompare(String(b.id)));
      keep.add(edges[edges.length - 1].id);
    }
    state.edges = state.edges.filter((e) => keep.has(e.id));
  }
  function removeNode(id) {
    const target = byId(id);
    if (target?.type === "output") {
      const outputCount = state.nodes.filter((n) => n.type === "output").length;
      if (outputCount <= 1) {
        alertText("At least one output node is required.");
        return;
      }
    }
    state.nodes = state.nodes.filter((n) => n.id !== id);
    state.edges = state.edges.filter((e) => e.from !== id && e.to !== id);
    if (selectedNodeId === id) selectedNodeId = null;
    if (renamingNodeId === id) {
      renamingNodeId = null;
      renamingNodeValue = "";
    }
    if (pendingSourceId === id) {
      pendingSourceId = null;
      pendingPointer = null;
    }
    const outs = state.nodes.filter((n) => n.type === "output");
    if (!outs.some((n) => n.id === state.scene_output_id) && outs.length) state.scene_output_id = outs[0].id;
  }
  function removeEdge(id) { state.edges = state.edges.filter((e) => e.id !== id); }
  function normalizeInboundOrder(targetId) {
    const inbound = state.edges
      .filter((e) => e.to === targetId)
      .sort((a, b) => ((a.order ?? 0) - (b.order ?? 0)) || String(a.id).localeCompare(String(b.id)));
    inbound.forEach((edge, index) => { edge.order = index; });
    return inbound;
  }
  function reorderInboundEdge(targetId, edgeId, delta) {
    const inbound = normalizeInboundOrder(targetId);
    const idx = inbound.findIndex((e) => e.id === edgeId);
    if (idx < 0) return;
    const next = idx + delta;
    if (next < 0 || next >= inbound.length) return;
    const temp = inbound[next];
    inbound[next] = inbound[idx];
    inbound[idx] = temp;
    inbound.forEach((edge, index) => { edge.order = index; });

    // Also reorder source nodes in state array to reflect inbound sequence.
    const target = byId(targetId);
    if (!target) return;
    const orderedSources = inbound
      .map((edge) => byId(edge.from))
      .filter((n) => n && n.section_id === target.section_id);
    if (orderedSources.length < 2) return;
    const sourceIds = new Set(orderedSources.map((n) => n.id));
    let firstIndex = -1;
    for (let i = 0; i < state.nodes.length; i += 1) {
      if (sourceIds.has(state.nodes[i].id)) {
        firstIndex = i;
        break;
      }
    }
    if (firstIndex < 0) return;
    const without = state.nodes.filter((n) => !sourceIds.has(n.id));
    const before = without.slice(0, firstIndex);
    const after = without.slice(firstIndex);
    state.nodes = [...before, ...orderedSources, ...after];
  }
  function addEdge(from, to) {
    if (!from || !to || from === to) return;
    const s = byId(from); const t = byId(to);
    if (!s || !t || !hasOutput(s) || !hasInput(t)) return;
    if (state.edges.some((e) => e.from === from && e.to === to)) return;
    if (t.type === "output") {
      state.edges = state.edges.filter((e) => e.to !== to);
    }
    state.edges.push({ id: nextId("e", state.edges), from, to, weight: 1, order: state.edges.length, enabled: true });
  }
  function addNode(type) {
    const id = nextId("n", state.nodes);
    const sectionId = type === "output" ? "output" : ensureWorkSection();
    state.nodes.push({
      id,
      type,
      title: type === "element" ? "Element" : (type === "sequential" ? "Sequential Group" : (type === "random" ? "Random Group" : "Scene Output")),
      section_id: sectionId,
      x: 0,
      y: 0,
      template: "",
    });
    if (type === "output") state.scene_output_id = id;
    selectedNodeId = id;
  }

  function refreshSceneSelectors() {
    state.scene_section_id = "output";
    const outputs = state.nodes.filter((n) => n.type === "output");
    for (const out of outputs) out.title = "Scene Output";
    if (outputs.length) {
      const fixedOutput = outputs.find((n) => n.section_id === "output") || outputs[0];
      state.scene_output_id = fixedOutput.id;
    }
  }

  function renderSections() {
    leftBody.innerHTML = "";
    for (const section of state.sections) {
      const row = document.createElement("div");
      row.className = `ess-flow-section${state.active_section_id === section.id ? " active" : ""}`;
      if (renamingSectionId === section.id && renamingSectionSource === "sidebar") {
        const input = document.createElement("input");
        input.type = "text";
        input.value = renamingSectionValue || section.name;
        input.style.width = "100%";
        input.style.background = "#0b1222";
        input.style.color = "#e5e7eb";
        input.style.border = "1px solid #475569";
        input.style.borderRadius = "5px";
        input.style.padding = "4px 6px";
        const save = () => applySectionRename(section.id, renamingSectionValue || input.value || "");
        input.oninput = () => { renamingSectionValue = input.value; };
        input.onkeydown = (event) => {
          if (event.key === "Enter") { event.preventDefault(); save(); }
          if (event.key === "Escape") {
            event.preventDefault();
            cancelSectionRename();
          }
        };
        const ok = document.createElement("button");
        ok.textContent = "OK";
        ok.onclick = (event) => { event.preventDefault(); save(); };
        row.appendChild(input);
        row.appendChild(ok);
      } else {
        const btn = document.createElement("button");
        btn.textContent = section.name;
        btn.addEventListener("click", () => {
          state.active_section_id = section.id;
          pendingSourceId = null;
          pendingPointer = null;
          renderAll();
          commitState();
        });
        btn.addEventListener("dblclick", (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (section.id === "output") return;
          startSectionRename(section.id);
        });
        const chip = document.createElement("span");
        chip.className = "ess-flow-chip";
        chip.textContent = String(state.nodes.filter((n) => n.section_id === section.id).length);
        row.appendChild(btn);
        row.appendChild(chip);
      }
      leftBody.appendChild(row);
    }

    const actions = document.createElement("div");
    actions.className = "ess-flow-actions";
    const addSection = document.createElement("button");
    addSection.textContent = "+ Add Section";
    addSection.addEventListener("click", () => {
      const fallbackName = `section_${state.sections.length + 1}`;
      const base = (promptText("Section name", fallbackName).trim() || fallbackName);
      let id = base.toLowerCase().replace(/[^a-z0-9_\-]+/g, "_") || "section";
      let i = 2;
      while (state.sections.some((s) => s.id === id)) { id = `${base}_${i}`.toLowerCase().replace(/[^a-z0-9_\-]+/g, "_"); i += 1; }
      state.sections.push({ id, name: base });
      state.active_section_id = id;
      refreshSceneSelectors();
      renderAll();
      commitState();
    });

    const remove = document.createElement("button");
    remove.textContent = "Remove Active";
    remove.addEventListener("click", () => {
      if (state.sections.length <= 1) return;
      const active = state.active_section_id;
      if (active === "output") {
        alertText("Section 'output' cannot be removed.");
        return;
      }
      if (state.nodes.some((n) => n.section_id === active)) { alertText("Move/delete nodes in this section first."); return; }
      state.sections = state.sections.filter((s) => s.id !== active);
      if (renamingSectionId === active) {
        renamingSectionId = null;
        renamingSectionValue = "";
      }
      state.active_section_id = getFirstNonOutputSectionId();
      if (state.scene_section_id === active) state.scene_section_id = "output";
      pendingSourceId = null;
      pendingPointer = null;
      refreshSceneSelectors();
      renderAll();
      commitState();
    });

    const moveUp = document.createElement("button");
    moveUp.textContent = "Move Active Up";
    moveUp.onclick = () => {
      const active = state.active_section_id;
      if (!moveSectionById(active, -1)) return;
      renderAll();
      commitState();
    };

    const moveDown = document.createElement("button");
    moveDown.textContent = "Move Active Down";
    moveDown.onclick = () => {
      const active = state.active_section_id;
      if (!moveSectionById(active, 1)) return;
      renderAll();
      commitState();
    };

    actions.appendChild(addSection); actions.appendChild(moveUp); actions.appendChild(moveDown); actions.appendChild(remove);
    leftBody.appendChild(actions);

    const nodeButtons = document.createElement("div");
    nodeButtons.className = "ess-flow-grid-buttons";
    const n1 = document.createElement("button"); n1.textContent = "+ Element"; n1.onclick = () => { addNode("element"); renderAll(); commitState(); };
    const n2 = document.createElement("button"); n2.textContent = "+ Sequential"; n2.onclick = () => { addNode("sequential"); renderAll(); commitState(); };
    const n3 = document.createElement("button"); n3.textContent = "+ Random"; n3.onclick = () => { addNode("random"); renderAll(); commitState(); };
    nodeButtons.appendChild(n1); nodeButtons.appendChild(n2); nodeButtons.appendChild(n3);
    leftBody.appendChild(nodeButtons);
  }

  function renderInspector() {
    for (const fn of inspectorCleanup) { try { fn(); } catch {} }
    inspectorCleanup = [];
    rightBody.innerHTML = "";
    const obj = byId(selectedNodeId);
    if (!obj) {
      const hint = document.createElement("div"); hint.className = "ess-flow-muted";
      hint.innerHTML = pendingSourceId ? `Connection mode: drop on input port or input node body. (source: ${pendingSourceId})<br><br>Ctrl/Cmd + click edge to remove.` : "Select a node to edit.<br><br>Drag from output to input (Comfy-style). Drag from an input point away to disconnect.";
      rightBody.appendChild(hint);
      return;
    }

    const inspector = document.createElement("div"); inspector.className = "ess-flow-inspector";
    const mkField = (title, control) => { const w = document.createElement("div"); w.className = "ess-flow-field"; const l = document.createElement("label"); l.textContent = title; w.appendChild(l); w.appendChild(control); return w; };

    const titleInput = document.createElement("input");
    titleInput.value = obj.type === "output" ? "Scene Output" : obj.title;
    titleInput.disabled = obj.type === "output";
    titleInput.oninput = () => {
      if (obj.type === "output") return;
      obj.title = titleInput.value;
      renderGraph();
      commitState();
    };
    inspector.appendChild(mkField("Title", titleInput));

    const sectionSelect = document.createElement("select");
    for (const section of state.sections) { const o = document.createElement("option"); o.value = section.id; o.textContent = section.name; if (section.id === obj.section_id) o.selected = true; sectionSelect.appendChild(o); }
    if (obj.type === "output") {
      sectionSelect.value = "output";
      sectionSelect.disabled = true;
    }
    sectionSelect.onchange = () => {
      obj.section_id = obj.type === "output" ? "output" : sectionSelect.value;
      refreshSceneSelectors();
      renderAll();
      commitState();
    };
    inspector.appendChild(mkField("Section", sectionSelect));

    const typeSelect = document.createElement("select");
    for (const type of ["element", "sequential", "random", "output"]) { const o = document.createElement("option"); o.value = type; o.textContent = type; if (type === obj.type) o.selected = true; typeSelect.appendChild(o); }
    typeSelect.onchange = () => {
      const previousType = obj.type;
      if (previousType === "output" && typeSelect.value !== "output") {
        const outputCount = state.nodes.filter((n) => n.type === "output").length;
        if (outputCount <= 1) {
          alertText("At least one output node is required.");
          typeSelect.value = "output";
          return;
        }
      }
      obj.type = typeSelect.value;
      if (obj.type !== "element") obj.template = "";
      if (obj.type === "output") {
        obj.section_id = "output";
        obj.title = "Scene Output";
        state.scene_output_id = obj.id;
      } else if (previousType === "output" && state.scene_output_id === obj.id) {
        const nextOutput = state.nodes.find((n) => n.id !== obj.id && n.type === "output");
        if (nextOutput) state.scene_output_id = nextOutput.id;
      }
      pruneEdges();
      renderAll();
      commitState();
    };
    inspector.appendChild(mkField("Node Type", typeSelect));

    if (obj.type === "element") {
      const editor = createTemplateEditor(obj.template || "", (value) => { obj.template = value; renderGraph(); commitState(); });
      inspectorCleanup.push(() => editor.destroy());
      inspector.appendChild(mkField("Template", editor.container));
    }

    const inbound = normalizeInboundOrder(obj.id);
    if (obj.type === "random") {
      const wrap = document.createElement("div"); wrap.className = "ess-flow-field"; const label = document.createElement("label"); label.textContent = "Inbound Weights"; wrap.appendChild(label);
      if (!inbound.length) { const empty = document.createElement("div"); empty.className = "ess-flow-muted"; empty.textContent = "No inbound branches connected."; wrap.appendChild(empty); }
      else {
        for (const edge of inbound) {
          const row = document.createElement("div"); row.className = "ess-flow-weight-row";
          const source = document.createElement("div"); source.textContent = byId(edge.from)?.title || edge.from;
          const weight = document.createElement("input"); weight.type = "number"; weight.min = "0"; weight.step = "0.1"; weight.value = String(edge.weight || 1);
          weight.oninput = () => { edge.weight = Number(weight.value) || 1; renderGraph(); commitState(); };
          const mute = document.createElement("button");
          mute.textContent = edge.enabled === false ? "Unmute" : "Mute";
          mute.onclick = () => {
            edge.enabled = edge.enabled === false ? true : false;
            renderAll();
            commitState();
          };
          const remove = document.createElement("button"); remove.textContent = "x"; remove.onclick = () => { removeEdge(edge.id); renderAll(); commitState(); };
          row.appendChild(source); row.appendChild(weight); row.appendChild(mute); row.appendChild(remove); wrap.appendChild(row);
        }
      }
      inspector.appendChild(wrap);
    }

    const del = document.createElement("button"); del.textContent = "Delete Node"; del.style.borderColor = "#7f1d1d"; del.style.color = "#fecaca"; del.style.background = "#3f1111";
    del.onclick = () => { removeNode(obj.id); renderAll(); commitState(); };
    inspector.appendChild(del);
    rightBody.appendChild(inspector);
  }

  function renderGraph() {
    layoutCache = computeAutoLayout(state);
    board.style.width = `${layoutCache.boardWidth}px`;
    board.style.height = `${layoutCache.boardHeight}px`;
    edgeSvg.setAttribute("width", String(layoutCache.boardWidth));
    edgeSvg.setAttribute("height", String(layoutCache.boardHeight));
    edgeSvg.setAttribute("viewBox", `0 0 ${layoutCache.boardWidth} ${layoutCache.boardHeight}`);

    laneLayer.innerHTML = "";
    nodeLayer.innerHTML = "";
    edgeLayer.innerHTML = "";
    const map = new Map();
    const edgeTargetPort = new Map();

    const inboundEdgesByTarget = new Map();
    for (const edge of state.edges) {
      if (!inboundEdgesByTarget.has(edge.to)) inboundEdgesByTarget.set(edge.to, []);
      inboundEdgesByTarget.get(edge.to).push(edge);
    }
    for (const [targetId, edges] of inboundEdgesByTarget.entries()) {
      edges.sort((a, b) => {
        const oa = Number(a.order ?? 0);
        const ob = Number(b.order ?? 0);
        if (oa !== ob) return oa - ob;
        const pa = layoutCache.nodePositions.get(a.from);
        const pb = layoutCache.nodePositions.get(b.from);
        const da = pa?.row ?? 0;
        const db = pb?.row ?? 0;
        if (da !== db) return da - db;
        return String(a.id).localeCompare(String(b.id));
      });
      inboundEdgesByTarget.set(targetId, edges);
    }

    for (const lane of layoutCache.lanes) {
      const laneEl = document.createElement("div");
      laneEl.className = `ess-flow-lane${state.active_section_id === lane.id ? " active" : ""}`;
      laneEl.style.left = `${lane.left}px`;
      laneEl.style.width = `${lane.width}px`;
      laneEl.style.height = `${layoutCache.laneHeight}px`;
      const header = document.createElement("div");
      header.className = "ess-flow-lane-header";
      if (renamingSectionId === lane.id && renamingSectionSource === "lane") {
        const input = document.createElement("input");
        input.type = "text";
        input.value = renamingSectionValue || lane.name;
        input.dataset.sectionRename = lane.id;
        input.style.width = "100%";
        input.style.background = "#0b1222";
        input.style.color = "#e5e7eb";
        input.style.border = "1px solid #475569";
        input.style.borderRadius = "5px";
        input.style.padding = "3px 6px";
        input.style.fontSize = "11px";
        input.style.textTransform = "none";
        input.onpointerdown = (event) => event.stopPropagation();
        input.onclick = (event) => event.stopPropagation();
        input.oninput = () => { renamingSectionValue = input.value; };
        input.onkeydown = (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            event.stopPropagation();
            applySectionRename(lane.id, renamingSectionValue || input.value || "");
          }
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            cancelSectionRename();
          }
        };
        header.appendChild(input);
      } else {
        const titleEl = document.createElement("div");
        titleEl.className = "ess-flow-lane-header-title";
        titleEl.textContent = lane.name;
        titleEl.title = lane.id === "output" ? "" : "Double-click to rename section";
        titleEl.ondblclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (lane.id === "output") return;
          startSectionRename(lane.id, "lane");
        };

        const controls = document.createElement("div");
        controls.className = "ess-flow-lane-header-controls";
        const moveLeftBtn = document.createElement("button");
        moveLeftBtn.textContent = "<";
        moveLeftBtn.title = "Move section left";
        const moveRightBtn = document.createElement("button");
        moveRightBtn.textContent = ">";
        moveRightBtn.title = "Move section right";
        const movable = state.sections.filter((s) => s.id !== "output");
        const idx = movable.findIndex((s) => s.id === lane.id);
        const isOutput = lane.id === "output";
        moveLeftBtn.disabled = isOutput || idx <= 0;
        moveRightBtn.disabled = isOutput || idx === -1 || idx >= movable.length - 1;
        moveLeftBtn.onclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (!moveSectionById(lane.id, -1)) return;
          state.active_section_id = lane.id;
          renderAll();
          commitState();
        };
        moveRightBtn.onclick = (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (!moveSectionById(lane.id, 1)) return;
          state.active_section_id = lane.id;
          renderAll();
          commitState();
        };
        controls.appendChild(moveLeftBtn);
        controls.appendChild(moveRightBtn);
        header.appendChild(titleEl);
        header.appendChild(controls);
      }
      laneEl.appendChild(header);
      laneEl.onclick = (event) => {
        event.stopPropagation();
        state.active_section_id = lane.id;
        renderSections();
        renderGraph();
        renderInspector();
        commitState();
      };
      laneLayer.appendChild(laneEl);
    }

    for (const obj of state.nodes) {
      if (!layoutCache.byId.has(obj.section_id)) obj.section_id = getFirstNonOutputSectionId();
      const pos = layoutCache.nodePositions.get(obj.id) || { x: 40, y: 100 };
      obj.x = Math.round(pos.x);
      obj.y = Math.round(pos.y);

      const card = document.createElement("div");
      card.className = `ess-flow-node type-${obj.type}${selectedNodeId === obj.id ? " selected" : ""}`;
      card.dataset.nodeId = obj.id;
      card.style.left = `${obj.x}px`;
      card.style.top = `${obj.y}px`;
      card.style.pointerEvents = "auto";
      const nodeHeight = layoutCache.nodeHeights.get(obj.id) ?? NODE_HEIGHT_ESTIMATE;
      card.style.height = `${Math.round(nodeHeight)}px`;

      const header = document.createElement("div"); header.className = "ess-flow-node-header";
      const t = document.createElement("div");
      t.className = "ess-flow-node-title";
      if (renamingNodeId === obj.id && obj.type !== "output") {
        const input = document.createElement("input");
        input.type = "text";
        input.dataset.nodeRename = obj.id;
        input.value = renamingNodeValue || obj.title || obj.type;
        input.style.width = "100%";
        input.style.background = "#0b1222";
        input.style.color = "#e5e7eb";
        input.style.border = "1px solid #475569";
        input.style.borderRadius = "5px";
        input.style.padding = "2px 6px";
        input.style.fontSize = "12px";
        input.style.fontWeight = "600";
        input.onpointerdown = (event) => event.stopPropagation();
        input.onclick = (event) => event.stopPropagation();
        input.oninput = () => { renamingNodeValue = input.value; };
        input.onkeydown = (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            event.stopPropagation();
            applyNodeRename(obj.id, renamingNodeValue || input.value || "");
          } else if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            cancelNodeRename();
          }
        };
        t.appendChild(input);
      } else {
        const icon = createTypeIcon(obj.type);
        const titleText = document.createElement("span");
        titleText.className = "ess-flow-node-title-text";
        titleText.textContent = obj.title || obj.type;
        t.appendChild(icon);
        t.appendChild(titleText);
      }
      const b = document.createElement("span"); b.className = "ess-flow-node-badge"; b.textContent = `${obj.type} | ${(state.sections.find((s) => s.id === obj.section_id)?.name || obj.section_id)}`;
      header.appendChild(t); header.appendChild(b);
      const body = document.createElement("div"); body.className = "ess-flow-node-body"; body.textContent = preview(obj);
      card.appendChild(header); card.appendChild(body);

      if (hasInput(obj)) {
        const inboundEdges = inboundEdgesByTarget.get(obj.id) || [];
        const slotCount = Math.max(1, inboundEdges.length);
        if ((obj.type === "random" || obj.type === "sequential") && inboundEdges.length > 0) {
          body.textContent = "";
        }
        for (let slot = 0; slot < slotCount; slot += 1) {
          const attachedEdge = inboundEdges[slot] || null;
          const inPort = document.createElement("div");
          inPort.className = "ess-flow-port in";
          inPort.style.top = `${portCenterY(slot, slotCount, nodeHeight, obj.type)}px`;
          if (attachedEdge && attachedEdge.enabled === false) {
            inPort.style.background = "#64748b";
          }
          if (attachedEdge) {
            edgeTargetPort.set(attachedEdge.id, inPort);
            inPort.title = "Drag away to disconnect/reconnect";
            inPort.onpointerdown = (event) => {
              if (!attachedEdge) return;
              event.preventDefault();
              event.stopPropagation();
              removeEdge(attachedEdge.id);
              pendingSourceId = attachedEdge.from;
              pendingPointer = { x: event.clientX, y: event.clientY };
              pendingStartPointer = { x: event.clientX, y: event.clientY };
              pendingDetachedEdge = { from: attachedEdge.from, to: attachedEdge.to };
              renderAll();
              commitState();
            };
          }
          if ((obj.type === "random" || obj.type === "sequential") && attachedEdge) {
            const control = document.createElement("div");
            control.className = "ess-flow-inbound-control";
            control.style.left = "12px";
            control.style.right = "12px";
            control.style.top = `${portCenterY(slot, slotCount, nodeHeight, obj.type)}px`;
            control.onpointerdown = (event) => event.stopPropagation();
            control.onclick = (event) => event.stopPropagation();

            const sourceLabel = document.createElement("div");
            sourceLabel.className = "ess-flow-inbound-label";
            sourceLabel.textContent = byId(attachedEdge.from)?.title || attachedEdge.from;

            const arrows = document.createElement("div");
            arrows.className = "ess-flow-inbound-arrows";
            const upButton = document.createElement("button");
            upButton.textContent = "^";
            upButton.title = "Move input up";
            upButton.disabled = slot === 0;
            upButton.onclick = (event) => {
              event.stopPropagation();
              reorderInboundEdge(obj.id, attachedEdge.id, -1);
              renderAll();
              commitState();
            };
            const downButton = document.createElement("button");
            downButton.textContent = "v";
            downButton.title = "Move input down";
            downButton.disabled = slot >= (slotCount - 1);
            downButton.onclick = (event) => {
              event.stopPropagation();
              reorderInboundEdge(obj.id, attachedEdge.id, 1);
              renderAll();
              commitState();
            };
            arrows.appendChild(upButton);
            arrows.appendChild(downButton);

            control.appendChild(sourceLabel);
            if (obj.type === "random") {
              const weightInput = document.createElement("input");
              weightInput.type = "number";
              weightInput.min = "0";
              weightInput.step = "0.1";
              weightInput.value = String(Number(attachedEdge.weight ?? 1) || 1);
              weightInput.title = "Weight";
              weightInput.oninput = () => {
                attachedEdge.weight = Number(weightInput.value) || 1;
                renderGraph();
                commitState();
              };

              const muteButton = document.createElement("button");
              const refreshMuteBtn = () => {
                const muted = attachedEdge.enabled === false;
                muteButton.textContent = muted ? "U" : "M";
                muteButton.title = muted ? "Unmute option" : "Mute option";
                muteButton.classList.toggle("is-muted", muted);
              };
              refreshMuteBtn();
              muteButton.onclick = (event) => {
                event.stopPropagation();
                attachedEdge.enabled = attachedEdge.enabled === false ? true : false;
                refreshMuteBtn();
                renderGraph();
                commitState();
              };

              control.appendChild(weightInput);
              control.appendChild(muteButton);
            } else {
              control.classList.add("simple");
            }
            control.appendChild(arrows);
            card.appendChild(control);
          }
          inPort.onclick = (event) => {
            event.stopPropagation();
            if (!pendingSourceId) return;
            if (!shouldConnectPendingTo(obj.id, event)) {
              pendingSourceId = null;
              pendingPointer = null;
              pendingStartPointer = null;
              pendingDetachedEdge = null;
              renderGraph();
              renderInspector();
              commitState();
              return;
            }
            addEdge(pendingSourceId, obj.id);
            pendingSourceId = null;
            pendingPointer = null;
            pendingStartPointer = null;
            pendingDetachedEdge = null;
            renderAll();
            commitState();
          };
          inPort.onpointerup = (event) => {
            event.stopPropagation();
            if (!pendingSourceId) return;
            if (!shouldConnectPendingTo(obj.id, event)) {
              pendingSourceId = null;
              pendingPointer = null;
              pendingStartPointer = null;
              pendingDetachedEdge = null;
              renderGraph();
              renderInspector();
              commitState();
              return;
            }
            addEdge(pendingSourceId, obj.id);
            pendingSourceId = null;
            pendingPointer = null;
            pendingStartPointer = null;
            pendingDetachedEdge = null;
            renderAll();
            commitState();
          };
          card.appendChild(inPort);
        }
        card.onpointerup = (event) => {
          if (!pendingSourceId) return;
          event.stopPropagation();
          if (!shouldConnectPendingTo(obj.id, event)) {
            pendingSourceId = null;
            pendingPointer = null;
            pendingStartPointer = null;
            pendingDetachedEdge = null;
            renderGraph();
            renderInspector();
            commitState();
            return;
          }
          addEdge(pendingSourceId, obj.id);
          pendingSourceId = null;
          pendingPointer = null;
          pendingStartPointer = null;
          pendingDetachedEdge = null;
          renderAll();
          commitState();
        };
      }
      if (hasOutput(obj)) {
        const outPort = document.createElement("div"); outPort.className = "ess-flow-port out";
        outPort.onpointerdown = (event) => {
          event.stopPropagation();
          pendingSourceId = obj.id;
          pendingPointer = { x: event.clientX, y: event.clientY };
          pendingStartPointer = { x: event.clientX, y: event.clientY };
          pendingDetachedEdge = null;
          renderGraph();
          renderInspector();
        };
        outPort.onclick = (event) => {
          event.stopPropagation();
          pendingSourceId = obj.id;
          pendingPointer = null;
          pendingStartPointer = null;
          pendingDetachedEdge = null;
          renderGraph();
          renderInspector();
        };
        card.appendChild(outPort);
      }

      card.onclick = (event) => {
        event.stopPropagation();
        if (pendingSourceId && hasInput(obj) && pendingSourceId !== obj.id) {
          if (!shouldConnectPendingTo(obj.id, event)) {
            pendingSourceId = null;
            pendingPointer = null;
            pendingStartPointer = null;
            pendingDetachedEdge = null;
            renderGraph();
            renderInspector();
            commitState();
            return;
          }
          addEdge(pendingSourceId, obj.id);
          pendingSourceId = null;
          pendingPointer = null;
          pendingStartPointer = null;
          pendingDetachedEdge = null;
          renderAll();
          commitState();
          return;
        }
        if (renamingNodeId && renamingNodeId !== obj.id) {
          renamingNodeId = null;
          renamingNodeValue = "";
        }
        selectedNodeId = obj.id;
        state.active_section_id = obj.section_id;
        renderSections();
        renderGraph();
        renderInspector();
      };
      header.onclick = (event) => {
        event.stopPropagation();
        if (renamingNodeId === obj.id) return;
        if (renamingNodeId && renamingNodeId !== obj.id) {
          renamingNodeId = null;
          renamingNodeValue = "";
        }
        selectedNodeId = obj.id;
        state.active_section_id = obj.section_id;
        renderSections();
        renderGraph();
        renderInspector();
      };
      header.ondblclick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        startNodeRename(obj.id);
      };

      nodeLayer.appendChild(card);
      map.set(obj.id, card);
    }
    nodeCardMap = map;

    for (const edge of state.edges) {
      const sourceCard = map.get(edge.from); const targetCard = map.get(edge.to);
      if (!sourceCard || !targetCard) continue;
      const sourcePort = sourceCard.querySelector(".ess-flow-port.out");
      const targetPort = edgeTargetPort.get(edge.id) || targetCard.querySelector(".ess-flow-port.in");
      if (!sourcePort || !targetPort) continue;
      const rect = board.getBoundingClientRect();
      const p0 = sourcePort.getBoundingClientRect();
      const p1 = targetPort.getBoundingClientRect();
      const x1 = p0.left + p0.width / 2 - rect.left;
      const y1 = p0.top + p0.height / 2 - rect.top;
      const x2 = p1.left + p1.width / 2 - rect.left;
      const y2 = p1.top + p1.height / 2 - rect.top;
      const c = Math.max(40, Math.abs(x2 - x1) * 0.35);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${x1} ${y1} C ${x1 + c} ${y1}, ${x2 - c} ${y2}, ${x2} ${y2}`);
      path.classList.add("ess-flow-edge");
      if (byId(edge.to)?.type === "random") path.classList.add("random");
      if (edge.enabled === false) path.classList.add("muted");
      path.title = "Ctrl/Cmd + click to remove";
      path.onclick = (event) => {
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault(); event.stopPropagation();
          removeEdge(edge.id);
          renderAll();
          commitState();
        }
      };
      edgeLayer.appendChild(path);
      if (byId(edge.to)?.type === "random") {
        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        label.classList.add("ess-flow-edge-label");
        if (edge.enabled === false) label.classList.add("muted");
        label.setAttribute("x", String((x1 + x2) / 2));
        label.setAttribute("y", String((y1 + y2) / 2 - 6));
        label.setAttribute("text-anchor", "middle");
        const w = Number(edge.weight || 1);
        label.textContent = edge.enabled === false ? `m w:${w}` : `w:${w}`;
        edgeLayer.appendChild(label);
      }
    }

    if (pendingSourceId) {
      const sourceNode = map.get(pendingSourceId);
      const sourcePort = sourceNode?.querySelector(".ess-flow-port.out");
      if (sourcePort) {
        const rect = board.getBoundingClientRect();
        const p = sourcePort.getBoundingClientRect();
        const x1 = p.left + p.width / 2 - rect.left;
        const y1 = p.top + p.height / 2 - rect.top;
        let x2 = x1 + 120;
        let y2 = y1 + 8;
        if (pendingPointer) {
          x2 = pendingPointer.x - rect.left;
          y2 = pendingPointer.y - rect.top;
        }
        const c = Math.max(40, Math.abs(x2 - x1) * 0.35);
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M ${x1} ${y1} C ${x1 + c} ${y1}, ${x2 - c} ${y2}, ${x2} ${y2}`);
        path.classList.add("ess-flow-edge", "pending");
        edgeLayer.appendChild(path);
      }
    }
  }

  function renderAll() {
    ensureSectionOrder();
    pruneEdges();
    refreshSceneSelectors();
    renderSections();
    renderGraph();
    renderInspector();
  }

  function shouldConnectPendingTo(targetId, event) {
    if (!pendingSourceId) return false;
    if (!pendingDetachedEdge || !pendingStartPointer) return true;
    if (pendingDetachedEdge.from !== pendingSourceId) return true;
    if (pendingDetachedEdge.to !== targetId) return true;
    const dx = event.clientX - pendingStartPointer.x;
    const dy = event.clientY - pendingStartPointer.y;
    return Math.hypot(dx, dy) > 10;
  }

  function pickInputTargetAt(clientX, clientY, sourceId) {
    const direct = document.elementFromPoint(clientX, clientY);
    const directCard = direct?.closest?.(".ess-flow-node");
    if (directCard?.dataset?.nodeId) {
      const directId = directCard.dataset.nodeId;
      const node = byId(directId);
      if (node && node.id !== sourceId && hasInput(node)) return node.id;
    }
    const rect = board.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let bestId = null;
    let bestDist = 32;
    for (const [nodeId, card] of nodeCardMap.entries()) {
      if (nodeId === sourceId) continue;
      const n = byId(nodeId);
      if (!n || !hasInput(n)) continue;
      const ports = card.querySelectorAll(".ess-flow-port.in");
      for (const port of ports) {
        const pr = port.getBoundingClientRect();
        const px = pr.left + pr.width / 2 - rect.left;
        const py = pr.top + pr.height / 2 - rect.top;
        const d = Math.hypot(px - x, py - y);
        if (d < bestDist) {
          bestDist = d;
          bestId = nodeId;
        }
      }
    }
    return bestId;
  }

  board.onpointermove = (event) => {
    if (pendingSourceId) {
      pendingPointer = { x: event.clientX, y: event.clientY };
      renderGraph();
    }
  };
  board.onpointerup = (event) => {
    if (pendingSourceId) {
      const targetId = pickInputTargetAt(event.clientX, event.clientY, pendingSourceId);
      if (targetId && shouldConnectPendingTo(targetId, event)) {
        addEdge(pendingSourceId, targetId);
        pendingSourceId = null;
        pendingPointer = null;
        pendingStartPointer = null;
        pendingDetachedEdge = null;
        renderAll();
        commitState();
        return;
      }
      pendingSourceId = null;
      pendingPointer = null;
      pendingStartPointer = null;
      pendingDetachedEdge = null;
      renderGraph();
      renderInspector();
      commitState();
    }
  };
  board.onpointercancel = () => {
    pendingSourceId = null;
    pendingPointer = null;
    pendingStartPointer = null;
    pendingDetachedEdge = null;
    renderGraph();
    renderInspector();
  };
  board.onclick = (event) => {
    const rect = board.getBoundingClientRect();
    const lane = layoutCache ? pickLaneByX(event.clientX - rect.left, layoutCache) : null;
    if (lane) state.active_section_id = lane.id;
    selectedNodeId = null;
    pendingSourceId = null;
    pendingPointer = null;
    pendingStartPointer = null;
    pendingDetachedEdge = null;
    renamingNodeId = null;
    renamingNodeValue = "";
    renderSections();
    renderGraph();
    renderInspector();
    commitState();
  };

  testRunButton.onclick = () => { void runTestFlow(); };
  saveButton.onclick = () => { commitState(true); };
  closeButton.onclick = () => { void attemptCloseOverlay(); };

  renderAll();
  refreshDirtyUi();
}

app.registerExtension({
  name: "ess_scene_flow_editor",
  async getCustomWidgets() {
    return {
      ESS_SCENE_FLOW_EDITOR(node, inputName, inputData) {
        ensureStyles();
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
          ? String(inputData.value || "")
          : String(config.value ?? config.default ?? "");
        const stateRef = { value: initialValue };

        const container = document.createElement("div");
        container.className = "ess-scene-flow-widget";
        const button = document.createElement("button");
        button.textContent = "Open Flow Editor";
        const summary = document.createElement("div");
        summary.className = "ess-scene-flow-summary";
        const updateSummary = (stateObj = null) => {
          if (!stateObj) stateObj = normalizeState(stateRef.value || "");
          summary.textContent = summaryText(stateObj);
        };
        updateSummary();
        container.appendChild(button);
        container.appendChild(summary);

        const widget = node.addDOMWidget(inputName, "ess_scene_flow_editor", container, {
          getValue: () => stateRef.value,
          setValue: (value) => { stateRef.value = String(value || ""); updateSummary(); },
          getMinHeight: () => Number(config.height || 58),
          hideOnZoom: false,
          margin: 12,
        });

        button.onclick = () => createOverlay(node, widget, stateRef, updateSummary);

        const originalRemove = widget.onRemove?.bind(widget);
        widget.onRemove = function () {
          originalRemove?.();
          if (container.isConnected) container.remove();
        };

        return { widget, minHeight: Number(config.height || 58) };
      },
    };
  },
});
