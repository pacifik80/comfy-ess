import { app } from "../../scripts/app.js";
import { renderHighlight } from "./ess_template_highlight.js";
import { installLoraTokenController } from "./ess_lora_token_editor.js";

const STYLE_ID = "ess-template-editor-style";
const MIN_EDITOR_HEIGHT = 160;
const DICT_INPUT_NAMES = new Set(["replace_dict", "replacements_dict", "input_dict"]);

function ensureStyles() {
  const existing = document.getElementById(STYLE_ID);
  const css = `
.ess-template-editor {
  position: relative;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  border: 1px solid #30363d;
  background: #0f1115;
  border-radius: 6px;
  overflow: hidden;
}
.ess-template-editor .ess-template-highlight-content,
.ess-template-editor textarea {
  font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
  font-size: 12px;
  line-height: 1.5;
  padding: 8px 10px;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}
.ess-template-editor .ess-template-highlight {
  position: absolute;
  inset: 0;
  color: #c9d1d9;
  overflow: hidden;
}
.ess-template-editor .ess-template-highlight-content {
  white-space: pre-wrap;
  pointer-events: none;
  will-change: transform;
}
.ess-template-editor textarea {
  position: relative;
  background: transparent;
  color: transparent;
  caret-color: #e6edf3;
  border: none;
  resize: none;
  outline: none;
  overflow: auto;
}
.ess-template-editor textarea::placeholder {
  color: #7d8590;
}
.ess-template-editor .ess-tpl-choice {
  color: #2f6f3e;
}
.ess-template-editor .ess-tpl-choice-active {
  color: #7ee787;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-header {
  color: #b48800;
}
.ess-template-editor .ess-tpl-header-active {
  color: #f9e076;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-header-name {
  color: #8b5a2b;
}
.ess-template-editor .ess-tpl-header-name-active {
  color: #d2a064;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-header-number {
  color: #d16ba5;
}
.ess-template-editor .ess-tpl-header-number-active {
  color: #ff9ad5;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-negative-marker {
  color: #a43f3f;
}
.ess-template-editor .ess-tpl-negative-marker-active {
  color: #ff7b72;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-negative-text {
  color: #a43f3f;
}
.ess-template-editor .ess-tpl-variant {
  color: #5b2c83;
}
.ess-template-editor .ess-tpl-variant-active {
  color: #d6a8ff;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-variant-label {
  color: #6b4fb3;
}
.ess-template-editor .ess-tpl-variant-label-active {
  color: #e0c1ff;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-comment {
  color: #6e7681;
}
.ess-template-editor .ess-tpl-placeholder {
  color: #38bdf8;
}
.ess-template-editor .ess-tpl-placeholder-active {
  color: #7dd3fc;
  font-weight: 700;
}
.ess-template-editor .ess-tpl-placeholder-unknown {
  color: #38bdf8;
  text-decoration-line: underline;
  text-decoration-style: wavy;
  text-decoration-color: #f87171;
  text-decoration-thickness: 1.2px;
  text-underline-offset: 2px;
}
.ess-template-editor .ess-tpl-placeholder-unknown-active {
  color: #7dd3fc;
  font-weight: 700;
  text-decoration-line: underline;
  text-decoration-style: wavy;
  text-decoration-color: #fca5a5;
  text-decoration-thickness: 1.4px;
  text-underline-offset: 2px;
}
`;
  if (existing) {
    existing.textContent = css;
    return;
  }
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = css;
  document.head.appendChild(style);
}

function applyContainerVars(container, minHeight, maxHeight) {
  const safeMin = Number.isFinite(minHeight) ? Math.max(minHeight, MIN_EDITOR_HEIGHT) : MIN_EDITOR_HEIGHT;
  const safeMax = Number.isFinite(maxHeight) ? Math.max(safeMin, maxHeight) : 100000;
  container.style.setProperty("--comfy-widget-min-height", `${safeMin}`);
  container.style.setProperty("--comfy-widget-max-height", `${safeMax}`);
}

function sameSet(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.size !== b.size) return false;
  for (const value of a) {
    if (!b.has(value)) return false;
  }
  return true;
}

function getGraphLinks(graph) {
  return graph?.links || graph?._links || null;
}

function getLinkOriginId(link) {
  if (!link) return null;
  if (Array.isArray(link)) return link[1] ?? null;
  return link.origin_id ?? null;
}

function getLinkOriginSlot(link) {
  if (!link) return null;
  if (Array.isArray(link)) return link[2] ?? null;
  return link.origin_slot ?? null;
}

function getNodeById(graph, nodeId) {
  if (!graph) return null;
  if (typeof graph.getNodeById === "function") return graph.getNodeById(nodeId);
  return graph?._nodes_by_id?.[nodeId] || null;
}

function isGroupRerouteNode(node) {
  const type = String(node?.type || "");
  const title = String(node?.title || "");
  return type.includes("GroupReroute") || title.includes("Group Reroute");
}

function isLiteRerouteNode(node) {
  const type = String(node?.type || "");
  const title = String(node?.title || "");
  return type === "Reroute" || title === "Reroute";
}

function resolveUpstreamNode(graph, link, depth = 0) {
  if (!graph || !link || depth > 24) return null;
  const links = getGraphLinks(graph);
  const originId = getLinkOriginId(link);
  const originNode = originId != null ? getNodeById(graph, originId) : null;
  if (!originNode) return null;
  const originSlot = originNode?.outputs?.[getLinkOriginSlot(link)];

  if (isGroupRerouteNode(originNode)) {
    const outputName = originSlot?.name || `output_${(getLinkOriginSlot(link) ?? 0) + 1}`;
    const match = outputName.match(/output_(\d+)/);
    const index = match ? Number(match[1]) : ((getLinkOriginSlot(link) ?? 0) + 1);
    const inputName = `input_${index}`;
    const inputSlot = originNode.inputs?.find((input) => input?.name === inputName);
    const inputLinkId = inputSlot?.link != null
      ? inputSlot.link
      : (Array.isArray(inputSlot?.links) && inputSlot.links.length ? inputSlot.links[0] : null);
    const upstreamLink = inputLinkId != null ? links?.[inputLinkId] : null;
    if (upstreamLink) return resolveUpstreamNode(graph, upstreamLink, depth + 1);
    return originNode;
  }

  if (isLiteRerouteNode(originNode)) {
    const inputSlot = originNode.inputs?.[0];
    const inputLinkId = inputSlot?.link != null
      ? inputSlot.link
      : (Array.isArray(inputSlot?.links) && inputSlot.links.length ? inputSlot.links[0] : null);
    const upstreamLink = inputLinkId != null ? links?.[inputLinkId] : null;
    if (upstreamLink) return resolveUpstreamNode(graph, upstreamLink, depth + 1);
    return originNode;
  }

  return originNode;
}

function getInputLinkIds(inputSlot) {
  const out = [];
  if (inputSlot?.link != null) out.push(inputSlot.link);
  if (Array.isArray(inputSlot?.links)) {
    for (const id of inputSlot.links) {
      if (id != null) out.push(id);
    }
  }
  return Array.from(new Set(out));
}

function normalizeReplaceKey(raw) {
  const key = String(raw || "").trim();
  if (!key) return "";
  if (key.includes("%")) return "";
  if (/\s/.test(key)) return "";
  return key;
}

function isReplaceDictNode(node) {
  const type = String(node?.type || "");
  const title = String(node?.title || "");
  return type.includes("ReplaceDict") || title.includes("Replace Dict");
}

function parseReplaceRows(raw) {
  try {
    const parsed = JSON.parse(String(raw || "[]"));
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

function getWidgetValueByName(node, targetName) {
  if (!Array.isArray(node?.widgets)) return "";
  const idx = node.widgets.findIndex((w) => String(w?.name || "").trim() === targetName);
  if (idx < 0) return "";
  const widget = node.widgets[idx];
  if (widget?.value != null) return widget.value;
  if (Array.isArray(node.widgets_values) && node.widgets_values[idx] != null) return node.widgets_values[idx];
  return "";
}

function collectKnownReplaceKeys(templateNode) {
  const graph = templateNode?.graph;
  if (!graph) return new Set();
  const links = getGraphLinks(graph);
  if (!links) return new Set();

  const known = new Set();
  const visitedNodes = new Set();

  const walkFromNode = (node) => {
    if (!node) return;
    const nodeId = node.id != null ? String(node.id) : `${node.type}|${node.title}`;
    if (visitedNodes.has(nodeId)) return;
    visitedNodes.add(nodeId);

    if (isReplaceDictNode(node)) {
      const raw = getWidgetValueByName(node, "replacements_editor");
      const rows = parseReplaceRows(raw);
      for (const row of rows) {
        const key = normalizeReplaceKey(row?.key);
        if (key) known.add(key);
      }
    }

    if (!Array.isArray(node.inputs)) return;
    for (const input of node.inputs) {
      const name = String(input?.name || "").trim().toLowerCase();
      if (!DICT_INPUT_NAMES.has(name)) continue;
      for (const linkId of getInputLinkIds(input)) {
        const link = links?.[linkId];
        const originId = getLinkOriginId(link);
        if (!link || originId == null) continue;
        const upstream = resolveUpstreamNode(graph, link) || getNodeById(graph, originId);
        if (upstream) walkFromNode(upstream);
      }
    }
  };

  if (!Array.isArray(templateNode?.inputs)) return known;
  for (const input of templateNode.inputs) {
    const name = String(input?.name || "").trim().toLowerCase();
    if (!DICT_INPUT_NAMES.has(name)) continue;
    for (const linkId of getInputLinkIds(input)) {
      const link = links?.[linkId];
      const originId = getLinkOriginId(link);
      if (!link || originId == null) continue;
      walkFromNode(resolveUpstreamNode(graph, link) || getNodeById(graph, originId));
    }
  }
  return known;
}

function createEditorElements(node, config, inputData) {
  const container = document.createElement("div");
  container.className = "ess-template-editor";

  const highlight = document.createElement("div");
  highlight.className = "ess-template-highlight";
  const highlightContent = document.createElement("div");
  highlightContent.className = "ess-template-highlight-content";
  highlight.appendChild(highlightContent);

  const textarea = document.createElement("textarea");
  textarea.spellcheck = false;
  textarea.setAttribute("data-capture-wheel", "true");
  const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
    ? inputData.value
    : (config.value ?? config.default ?? "");
  textarea.value = initialValue;
  if (config.placeholder) {
    textarea.placeholder = config.placeholder;
  }

  const syncSize = () => {
    const width = textarea.clientWidth || 0;
    const height = textarea.clientHeight || 0;
    highlightContent.style.width = `${width}px`;
    highlightContent.style.height = `${height}px`;
  };

  const syncScroll = () => {
    const offsetX = textarea.scrollLeft || 0;
    const offsetY = textarea.scrollTop || 0;
    highlightContent.style.transform = `translate(${-offsetX}px, ${-offsetY}px)`;
  };

  let raf = 0;
  let knownPlaceholders = collectKnownReplaceKeys(node);
  let loraController = null;
  const refreshKnownPlaceholders = () => {
    const next = collectKnownReplaceKeys(node);
    if (!sameSet(knownPlaceholders, next)) {
      knownPlaceholders = next;
      scheduleRender();
    }
  };

  const scheduleRender = () => {
    if (raf) {
      cancelAnimationFrame(raf);
    }
    raf = requestAnimationFrame(() => {
      raf = 0;
      syncSize();
      const caret = textarea.selectionStart || 0;
      const knownLoras = loraController?.getKnownLoras?.() || null;
      highlightContent.innerHTML = renderHighlight(textarea.value || "", caret, { knownPlaceholders, knownLoras });
      syncScroll();
    });
  };

  textarea.addEventListener("input", scheduleRender);
  textarea.addEventListener("click", scheduleRender);
  textarea.addEventListener("keyup", scheduleRender);
  textarea.addEventListener("select", scheduleRender);
  textarea.addEventListener("focus", scheduleRender);
  textarea.addEventListener("scroll", syncScroll);
  textarea.addEventListener("pointerdown", (event) => event.stopPropagation());
  textarea.addEventListener("pointermove", (event) => event.stopPropagation());
  textarea.addEventListener("pointerup", (event) => event.stopPropagation());
  textarea.addEventListener("contextmenu", (event) => event.stopPropagation());
  loraController = installLoraTokenController(textarea, {
    container,
    requestRender: scheduleRender,
  });

  container.appendChild(highlight);
  container.appendChild(textarea);
  scheduleRender();

  const resizeObserver = new ResizeObserver(() => {
    syncSize();
    syncScroll();
  });
  resizeObserver.observe(textarea);

  const onWheelCapture = (event) => {
    if (!container.isConnected) {
      return;
    }
    if (!container.contains(event.target)) {
      return;
    }
    const deltaX = event.deltaX || 0;
    const deltaY = event.deltaY || 0;
    if (event.shiftKey && Math.abs(deltaY) > Math.abs(deltaX)) {
      textarea.scrollLeft += deltaY;
    } else {
      textarea.scrollTop += deltaY;
      textarea.scrollLeft += deltaX;
    }
    syncScroll();
    event.preventDefault();
    event.stopPropagation();
  };

  window.addEventListener("wheel", onWheelCapture, { capture: true, passive: false });
  const onReplaceDictChanged = () => refreshKnownPlaceholders();
  window.addEventListener("ess:replace-dict-changed", onReplaceDictChanged);

  if (node && !node.__essTemplateOrigConnectionsChange) {
    node.__essTemplateOrigConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function () {
      const result = node.__essTemplateOrigConnectionsChange
        ? node.__essTemplateOrigConnectionsChange.apply(this, arguments)
        : undefined;
      this.__essTemplateRefreshKnown?.();
      return result;
    };
  }
  if (node) {
    node.__essTemplateRefreshKnown = refreshKnownPlaceholders;
  }

  const cleanup = () => {
    resizeObserver.disconnect();
    window.removeEventListener("ess:replace-dict-changed", onReplaceDictChanged);
    loraController?.destroy?.();
  };

  return { container, textarea, scheduleRender, onWheelCapture, cleanup };
}

app.registerExtension({
  name: "ess_template_editor",
  async getCustomWidgets() {
    return {
      ESS_TEMPLATE_EDITOR(node, inputName, inputData) {
        ensureStyles();

        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const minHeight = Number(config.minHeight) || Number(config.height) || MIN_EDITOR_HEIGHT;
        const rawMaxHeight = Number(config.maxHeight);
        const maxHeight = Number.isFinite(rawMaxHeight) ? Math.max(minHeight, rawMaxHeight) : 100000;

        const { container, textarea, scheduleRender, onWheelCapture, cleanup } = createEditorElements(node, config, inputData);
        applyContainerVars(container, minHeight, maxHeight);

        if (typeof node.addDOMWidget === "function") {
          const widget = node.addDOMWidget(inputName, "ess_template_editor", container, {
            getValue: () => textarea.value,
            setValue: (value) => {
              textarea.value = value ?? "";
              scheduleRender();
            },
            getMinHeight: () => minHeight,
            getMaxHeight: () => maxHeight,
            hideOnZoom: false,
            margin: 16,
          });

          const originalRemove = widget.onRemove?.bind(widget);
          widget.onRemove = function () {
            originalRemove?.();
            window.removeEventListener("wheel", onWheelCapture, { capture: true });
            cleanup?.();
            if (container.isConnected) {
              container.remove();
            }
          };

          return {
            widget,
            minHeight: minHeight,
          };
        }

        const fallback = node.addWidget("text", inputName, textarea.value ?? "", (value) => {
          textarea.value = value ?? "";
          scheduleRender();
        }, { multiline: true });

        return {
          widget: fallback,
          minHeight: minHeight,
        };
      },
    };
  },
});
