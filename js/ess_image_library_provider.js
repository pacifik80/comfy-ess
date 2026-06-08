import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "ess-image-library-style";
const MIN_WIDTH = 900;
const MIN_HEIGHT = 620;
const WIDGET_HEIGHT = 500;
const NODE_TOP_OFFSET = 156;
const WORKFLOW_EXTRA_KEY = "ess_workflow_path";
const attachedLibraries = new Set();

let activeWorkflowRef = "";
let saveHooksInstalled = false;
let loadHooksInstalled = false;
let executionHooksInstalled = false;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-image-lib{display:grid;grid-template-rows:auto minmax(0,1fr);gap:10px;width:100%;height:100%;min-height:${WIDGET_HEIGHT}px;box-sizing:border-box;padding:10px;border:1px solid #334155;border-radius:10px;background:#08111f;color:#e2e8f0;font-family:"IBM Plex Sans","Segoe UI",sans-serif;overflow:hidden;align-content:stretch}
.ess-image-lib *{box-sizing:border-box}
.ess-image-lib button,.ess-image-lib input,.ess-image-lib select,.ess-image-lib textarea{font:inherit}
.ess-image-lib button{background:#1d4ed8;color:#eff6ff;border:1px solid #1e40af;border-radius:7px;padding:7px 10px;cursor:pointer}
.ess-image-lib button.secondary{background:#132033;border-color:#334155;color:#dbeafe}
.ess-image-lib button:disabled{opacity:.5;cursor:default}
.ess-image-lib-top{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.ess-image-lib-top .spacer{flex:1}
.ess-image-lib-top label{font-size:12px;color:#bfdbfe}
.ess-image-lib-top select{min-width:130px;background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:7px;padding:6px 8px}
.ess-image-lib-shell{display:grid;grid-template-columns:260px minmax(0,1fr);gap:10px;min-height:0;height:100%;align-content:stretch}
.ess-image-lib-sidebar{display:grid;grid-template-rows:auto 1fr;gap:8px;min-height:0}
.ess-image-lib-sidebar-head{font-size:12px;color:#93c5fd;text-transform:uppercase;letter-spacing:.05em}
.ess-image-lib-list{overflow:auto;border:1px solid #334155;border-radius:9px;background:#020817;padding:6px;display:grid;gap:6px;min-height:0}
.ess-image-lib-item{display:grid;grid-template-columns:52px minmax(0,1fr);gap:8px;align-items:center;padding:6px;border:1px solid #233044;border-radius:8px;background:#0b1629;cursor:pointer}
.ess-image-lib-item.active{border-color:#60a5fa;background:#10213d}
.ess-image-lib-thumb{width:52px;height:52px;border-radius:6px;object-fit:cover;background:#020617;border:1px solid #243244}
.ess-image-lib-item-title{font-size:12px;font-weight:600;color:#f8fbff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ess-image-lib-item-meta{font-size:11px;color:#8fa6c1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ess-image-lib-main{display:grid;grid-template-rows:minmax(0,1fr) auto auto auto;gap:10px;min-height:0;height:100%;align-content:stretch}
.ess-image-lib-preview{position:relative;display:flex;align-items:center;justify-content:center;min-height:220px;height:100%;width:100%;border:1px solid #334155;border-radius:10px;background:linear-gradient(180deg,#08111f 0%,#050a14 100%);overflow:hidden}
.ess-image-lib-preview img{max-width:100%;max-height:100%;object-fit:contain;display:block}
.ess-image-lib-empty{padding:18px;color:#7f94ac;font-size:13px;text-align:center;max-width:320px}
.ess-image-lib-badge{position:absolute;top:10px;left:10px;padding:4px 8px;border-radius:999px;background:rgba(2,6,23,.82);border:1px solid #334155;color:#cfe6ff;font-size:11px}
.ess-image-lib-grid{display:grid;grid-template-columns:minmax(0,1fr) 120px;gap:10px}
.ess-image-lib-field{display:grid;gap:5px}
.ess-image-lib-field label{font-size:12px;color:#bfdbfe}
.ess-image-lib-field input,.ess-image-lib-field textarea{width:100%;background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:7px;padding:7px 9px}
.ess-image-lib-field textarea{min-height:120px;resize:vertical}
.ess-image-lib-actions{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:auto}
.ess-image-lib-status{margin-left:auto;font-size:12px;color:#8fa6c1}
`;
  document.head.appendChild(style);
}

function trapEvents(element) {
  if (!element) return;
  const stop = (event) => event.stopPropagation();
  [
    "pointerdown", "pointermove", "pointerup",
    "mousedown", "mouseup",
    "click", "dblclick", "contextmenu",
    "keydown", "keyup", "keypress",
    "wheel",
  ].forEach((name) => element.addEventListener(name, stop));
}

function collapseStorageWidget(widget) {
  if (!widget) return;
  widget.serialize = true;
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
  const element = widget.element || widget.inputEl;
  if (element?.style) {
    element.style.display = "none";
    element.style.height = "0px";
    element.style.minHeight = "0px";
    element.style.margin = "0";
    element.style.padding = "0";
    element.style.border = "0";
    element.style.overflow = "hidden";
  }
}

function syncStorageWidgetValue(node, widget, value) {
  const nextValue = String(value ?? "");
  widget.value = nextValue;
  const idx = Array.isArray(node?.widgets) ? node.widgets.indexOf(widget) : -1;
  if (idx >= 0 && Array.isArray(node.widgets_values)) {
    node.widgets_values[idx] = nextValue;
  }
  app.graph?.setDirtyCanvas?.(true, true);
  node.setDirtyCanvas?.(true, true);
}

function normalizeWorkflowRef(value) {
  let text = String(value ?? "").replace(/\\/g, "/").trim();
  while (text.startsWith("./")) {
    text = text.slice(2);
  }
  if (!/^[A-Za-z]:\//.test(text) && !text.startsWith("//")) {
    while (text.startsWith("/")) {
      text = text.slice(1);
    }
    if (text.toLowerCase() === "workflows") {
      text = "";
    } else if (text.toLowerCase().startsWith("workflows/")) {
      text = text.slice("workflows/".length);
    }
  }
  return text;
}

function extractWorkflowReference(raw) {
  if (!raw) return "";
  if (typeof raw === "string") return normalizeWorkflowRef(raw);
  if (typeof raw === "object") {
    for (const key of ["path", "file", "filename", "relative_path", "relativePath", "name", "id"]) {
      const value = raw[key];
      if (typeof value === "string" && value.trim()) {
        return normalizeWorkflowRef(value);
      }
    }
  }
  return "";
}

function ensureGraphExtra() {
  if (!app.graph) return null;
  if (!app.graph.extra || typeof app.graph.extra !== "object") {
    app.graph.extra = {};
  }
  return app.graph.extra;
}

function setWorkflowReference(raw) {
  const next = extractWorkflowReference(raw);
  if (!next) return "";
  activeWorkflowRef = next;
  const extra = ensureGraphExtra();
  if (extra) extra[WORKFLOW_EXTRA_KEY] = next;
  return next;
}

function getWorkflowReference() {
  const active = extractWorkflowReference(
    app.extensionManager?.workflow?.activeWorkflow
      ?? app.workflowManager?.activeWorkflow
      ?? app.graph?.extra?.[WORKFLOW_EXTRA_KEY]
      ?? activeWorkflowRef,
  );
  if (active) {
    setWorkflowReference(active);
    return active;
  }
  return normalizeWorkflowRef(activeWorkflowRef);
}

function createAssetUrl(item) {
  const assetId = String(item?.asset_id ?? "").trim();
  const workflow = extractWorkflowReference(item?.workflow_relative_path) || getWorkflowReference();
  if (!assetId || !workflow) return "";
  return `/ess/workflow_assets/file?workflow=${encodeURIComponent(workflow)}&asset_id=${encodeURIComponent(assetId)}`;
}

function parseState(raw) {
  let payload = {};
  if (raw && typeof raw === "object") {
    payload = raw;
  } else if (raw && String(raw).trim()) {
    try {
      const parsed = JSON.parse(String(raw));
      if (parsed && typeof parsed === "object") payload = parsed;
    } catch {
      payload = {};
    }
  }

  const items = [];
  for (const entry of Array.isArray(payload.items) ? payload.items : []) {
    if (!entry || typeof entry !== "object") continue;
    const imageData = String(entry.image_data ?? entry.imageData ?? entry.data ?? "");
    const assetId = String(entry.asset_id ?? entry.assetId ?? "");
    if (!imageData && !assetId) continue;
    items.push({
      name: String(entry.name ?? "").trim() || "Image",
      prompt: String(entry.prompt ?? ""),
      weight: Number.isFinite(Number(entry.weight)) ? Number(entry.weight) : 1,
      image_data: imageData,
      asset_id: assetId,
      workflow_relative_path: extractWorkflowReference(entry.workflow_relative_path ?? entry.workflowRelativePath),
      mime_type: String(entry.mime_type ?? entry.mimeType ?? ""),
      width: Number.isFinite(Number(entry.width)) ? Number(entry.width) : 0,
      height: Number.isFinite(Number(entry.height)) ? Number(entry.height) : 0,
    });
  }

  let selectedIndex = Number.isFinite(Number(payload.selected_index ?? payload.selectedIndex))
    ? Number(payload.selected_index ?? payload.selectedIndex)
    : 0;
  if (!items.length) selectedIndex = 0;
  selectedIndex = Math.max(0, Math.min(selectedIndex, Math.max(0, items.length - 1)));

  const mode = String(payload.mode ?? "manual").toLowerCase() === "random" ? "random" : "manual";
  return { mode, selected_index: selectedIndex, items };
}

function serializeState(state) {
  return JSON.stringify({
    mode: state.mode === "random" ? "random" : "manual",
    selected_index: Math.max(0, Number(state.selected_index) || 0),
    items: (Array.isArray(state.items) ? state.items : []).map((item, index) => ({
      name: String(item?.name ?? "").trim() || `Image ${index + 1}`,
      prompt: String(item?.prompt ?? ""),
      weight: Number.isFinite(Number(item?.weight)) ? Number(item.weight) : 1,
      image_data: String(item?.image_data ?? ""),
      asset_id: String(item?.asset_id ?? ""),
      workflow_relative_path: extractWorkflowReference(item?.workflow_relative_path),
      mime_type: String(item?.mime_type ?? ""),
      width: Number.isFinite(Number(item?.width)) ? Number(item.width) : 0,
      height: Number.isFinite(Number(item?.height)) ? Number(item.height) : 0,
    })),
  });
}

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read file."));
    reader.readAsDataURL(file);
  });
}

function readImageMeta(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({
      width: Number(image.naturalWidth || image.width || 0),
      height: Number(image.naturalHeight || image.height || 0),
    });
    image.onerror = () => reject(new Error("Failed to decode image preview."));
    image.src = dataUrl;
  });
}

function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function ensureNodeSize(node) {
  const width = Number(node?.size?.[0] || 0);
  const height = Number(node?.size?.[1] || 0);
  const nextWidth = Math.max(width || 0, MIN_WIDTH);
  const nextHeight = Math.max(height || 0, MIN_HEIGHT);
  if (!node?.size || width !== nextWidth || height !== nextHeight) {
    node.setSize?.([nextWidth, nextHeight]);
  }
}

function enforceNodeSize(node, requestedSize) {
  if (!node) return;
  const width = Math.max(Number(requestedSize?.[0] || node.size?.[0] || 0), MIN_WIDTH);
  const height = Math.max(Number(requestedSize?.[1] || node.size?.[1] || 0), MIN_HEIGHT);
  if (!node.size || node.size[0] !== width || node.size[1] !== height) {
    node.size = [width, height];
  }
}

function findAttachedLibraryStateByNodeId(nodeId) {
  const wanted = String(nodeId ?? "");
  if (!wanted) return null;
  for (const state of attachedLibraries) {
    if (String(state?.node?.id ?? "") === wanted) {
      return state;
    }
  }
  return null;
}

function extractExecutedSelectedIndex(detail) {
  const sources = [
    detail?.output,
    detail?.output?.ui,
    detail,
  ];
  for (const source of sources) {
    if (!source || typeof source !== "object") continue;
    const raw = source.selected_index ?? source.selectedIndex;
    const value = Array.isArray(raw) ? raw[0] : raw;
    if (Number.isFinite(Number(value))) {
      return Number(value);
    }
  }
  return null;
}

function currentItem(state) {
  const items = Array.isArray(state?.items) ? state.items : [];
  if (!items.length) return null;
  const index = Math.max(0, Math.min(Number(state?.selected_index) || 0, items.length - 1));
  return items[index] || null;
}

function syncInputValue(input, value) {
  if (!input) return;
  const nextValue = String(value ?? "");
  if (document.activeElement === input && input.value === nextValue) return;
  if (document.activeElement !== input || input.value !== nextValue) {
    input.value = nextValue;
  }
}

function getKnownLibraryNodeIds() {
  const nodes = Array.isArray(app.graph?._nodes) ? app.graph._nodes : [];
  return nodes
    .filter((node) => node && /ImageLibraryProvider$/.test(String(node.comfyClass ?? node.type ?? "")))
    .map((node) => String(node.id))
    .filter(Boolean);
}

async function fetchJson(url, options) {
  const response = api?.fetchApi
    ? await api.fetchApi(url, options)
    : await fetch(url, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok || payload?.ok === false) {
    throw new Error(String(payload?.error || response.statusText || "Request failed."));
  }
  return payload;
}

async function persistLibraryState(state, force = false) {
  if (!state || state.isRemoved) return;
  const workflow = getWorkflowReference();
  if (!workflow) return;
  if (state.persistPromise) {
    if (!force) return state.persistPromise;
    await state.persistPromise;
  }
  if (state.persistTimer) {
    clearTimeout(state.persistTimer);
    state.persistTimer = null;
  }

  const payload = {
    workflow,
    node_id: String(state.node?.id ?? ""),
    mode: state.data.mode,
    selected_index: state.data.selected_index,
    known_node_ids: getKnownLibraryNodeIds(),
    items: state.data.items.map((item, index) => ({
      name: String(item?.name ?? "").trim() || `Image ${index + 1}`,
      prompt: String(item?.prompt ?? ""),
      weight: Number.isFinite(Number(item?.weight)) ? Number(item.weight) : 1,
      image_data: String(item?.image_data ?? ""),
      asset_id: String(item?.asset_id ?? ""),
      workflow_relative_path: extractWorkflowReference(item?.workflow_relative_path),
      width: Number.isFinite(Number(item?.width)) ? Number(item.width) : 0,
      height: Number.isFinite(Number(item?.height)) ? Number(item.height) : 0,
      mime_type: String(item?.mime_type ?? ""),
    })),
  };

  state.status.textContent = state.data.items.length ? "Saving library assets..." : "Syncing library state...";
  state.persistPromise = (async () => {
    const result = await fetchJson("/ess/workflow_assets/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const workflowRelativePath = extractWorkflowReference(result.workflow_relative_path);
    if (workflowRelativePath) {
      setWorkflowReference(workflowRelativePath);
    }
    state.data = parseState(result.state);
    state.data.items = state.data.items.map((item) => ({
      ...item,
      workflow_relative_path: extractWorkflowReference(item.workflow_relative_path) || workflowRelativePath,
    }));
    syncStorageWidgetValue(state.node, state.storageWidget, serializeState(state.data));
    state.render();
  })()
    .catch((error) => {
      state.status.textContent = String(error?.message || error || "Failed to save library assets.");
      throw error;
    })
    .finally(() => {
      state.persistPromise = null;
    });

  return state.persistPromise;
}

function schedulePersist(state) {
  if (!state || state.isRemoved) return;
  if (!getWorkflowReference()) return;
  if (state.persistTimer) {
    clearTimeout(state.persistTimer);
  }
  state.persistTimer = setTimeout(() => {
    state.persistTimer = null;
    persistLibraryState(state).catch(() => {});
  }, 450);
}

async function flushAllLibraries() {
  for (const state of Array.from(attachedLibraries)) {
    await persistLibraryState(state, true);
  }
}

function refreshAllLibraries() {
  for (const state of attachedLibraries) {
    state.render?.();
  }
}

function wrapAsyncBefore(target, key, before) {
  if (!target || typeof target[key] !== "function" || target[key].__essWrapped) return;
  const original = target[key];
  const wrapped = async function (...args) {
    await before(...args);
    return await original.apply(this, args);
  };
  wrapped.__essWrapped = true;
  target[key] = wrapped;
}

function maybeCaptureWorkflowFromArgs(args) {
  for (const arg of args) {
    const ref = extractWorkflowReference(arg);
    if (ref) {
      setWorkflowReference(ref);
      return ref;
    }
    if (arg && typeof arg === "object") {
      const refFromExtra = extractWorkflowReference(arg.extra?.[WORKFLOW_EXTRA_KEY] ?? arg.workflow);
      if (refFromExtra) {
        setWorkflowReference(refFromExtra);
        return refFromExtra;
      }
    }
  }
  return "";
}

function installLifecycleHooks() {
  if (!loadHooksInstalled && typeof app.loadGraphData === "function") {
    const originalLoadGraphData = app.loadGraphData.bind(app);
    app.loadGraphData = async function (...args) {
      maybeCaptureWorkflowFromArgs(args);
      const graphData = args[0];
      const extraRef = extractWorkflowReference(graphData?.extra?.[WORKFLOW_EXTRA_KEY]);
      if (extraRef) setWorkflowReference(extraRef);
      const result = await originalLoadGraphData(...args);
      refreshAllLibraries();
      return result;
    };
    loadHooksInstalled = true;
  }

  if (saveHooksInstalled) return;
  const beforeSave = async (...args) => {
    maybeCaptureWorkflowFromArgs(args);
    const workflow = getWorkflowReference();
    if (!workflow) return;
    setWorkflowReference(workflow);
    await flushAllLibraries();
  };

  const workflowApi = app.extensionManager?.workflow || app.workflowManager;
  if (workflowApi && typeof workflowApi === "object") {
    for (const key of Object.keys(workflowApi)) {
      if (/save/i.test(key)) {
        wrapAsyncBefore(workflowApi, key, beforeSave);
      }
    }
  }

  const commands = app.extensionManager?.command?.commands;
  if (commands && typeof commands === "object") {
    for (const [key, value] of Object.entries(commands)) {
      if (!/save/i.test(key)) continue;
      if (typeof value === "function") {
        wrapAsyncBefore(commands, key, beforeSave);
        continue;
      }
      if (value && typeof value.execute === "function") {
        wrapAsyncBefore(value, "execute", beforeSave);
      }
      if (value && typeof value.action === "function") {
        wrapAsyncBefore(value, "action", beforeSave);
      }
    }
  }

  saveHooksInstalled = true;
}

function installExecutionHooks() {
  if (executionHooksInstalled || typeof api?.addEventListener !== "function") return;
  api.addEventListener("executed", ({ detail }) => {
    const nodeId = detail?.node;
    const selectedIndex = extractExecutedSelectedIndex(detail);
    if (nodeId == null || selectedIndex == null) return;
    const state = findAttachedLibraryStateByNodeId(nodeId);
    if (!state || !Array.isArray(state.data?.items) || !state.data.items.length) return;
    const boundedIndex = Math.max(0, Math.min(Number(selectedIndex) || 0, state.data.items.length - 1));
    if (boundedIndex === state.data.selected_index) {
      state.render?.();
      return;
    }
    state.data.selected_index = boundedIndex;
    syncStorageWidgetValue(state.node, state.storageWidget, serializeState(state.data));
    state.render?.();
  });
  executionHooksInstalled = true;
}

function attachImageLibraryWidget(node, storageWidget) {
  if (!node || !storageWidget || storageWidget.__essImageLibraryAttached || typeof node.addDOMWidget !== "function") return;

  ensureStyles();
  installLifecycleHooks();
  installExecutionHooks();
  collapseStorageWidget(storageWidget);
  ensureNodeSize(node);

  const container = createEl("div", "ess-image-lib");
  const top = createEl("div", "ess-image-lib-top");
  const addButton = createEl("button", "", "Add Images");
  const clearButton = createEl("button", "secondary", "Clear All");
  const spacer = createEl("div", "spacer");
  const modeLabel = createEl("label", "", "Output Mode");
  const modeSelect = document.createElement("select");
  [["manual", "Manual"], ["random", "Weighted Random"]].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    modeSelect.appendChild(option);
  });
  const hiddenInput = document.createElement("input");
  hiddenInput.type = "file";
  hiddenInput.accept = "image/*";
  hiddenInput.multiple = true;
  hiddenInput.style.display = "none";
  top.append(addButton, clearButton, spacer, modeLabel, modeSelect, hiddenInput);

  const shell = createEl("div", "ess-image-lib-shell");
  const sidebar = createEl("div", "ess-image-lib-sidebar");
  const sidebarHead = createEl("div", "ess-image-lib-sidebar-head", "Library");
  const list = createEl("div", "ess-image-lib-list");
  sidebar.append(sidebarHead, list);

  const main = createEl("div", "ess-image-lib-main");
  const preview = createEl("div", "ess-image-lib-preview");
  const badge = createEl("div", "ess-image-lib-badge");
  const previewImage = document.createElement("img");
  const empty = createEl("div", "ess-image-lib-empty", "Add images to the library. When the workflow path is known, ESS stores them in the workflow's adjacent .ess file.");
  preview.append(badge, previewImage, empty);

  const fieldGrid = createEl("div", "ess-image-lib-grid");
  const nameField = createEl("div", "ess-image-lib-field");
  const nameLabel = createEl("label", "", "Name");
  const nameInput = document.createElement("input");
  nameField.append(nameLabel, nameInput);
  const weightField = createEl("div", "ess-image-lib-field");
  const weightLabel = createEl("label", "", "Weight");
  const weightInput = document.createElement("input");
  weightInput.type = "number";
  weightInput.step = "0.1";
  weightInput.min = "0";
  weightField.append(weightLabel, weightInput);
  fieldGrid.append(nameField, weightField);

  const promptField = createEl("div", "ess-image-lib-field");
  const promptLabel = createEl("label", "", "Prompt");
  const promptInput = document.createElement("textarea");
  promptField.append(promptLabel, promptInput);

  const actions = createEl("div", "ess-image-lib-actions");
  const removeButton = createEl("button", "secondary", "Remove Selected");
  const status = createEl("div", "ess-image-lib-status");
  actions.append(removeButton, status);

  main.append(preview, fieldGrid, promptField, actions);
  shell.append(sidebar, main);
  container.append(top, shell);

  [container, addButton, clearButton, modeSelect, list, nameInput, weightInput, promptInput, removeButton].forEach(trapEvents);

  const state = {
    node,
    storageWidget,
    domWidget: null,
    list,
    badge,
    previewImage,
    empty,
    modeSelect,
    nameInput,
    weightInput,
    promptInput,
    status,
    data: parseState(storageWidget.value),
    persistTimer: null,
    persistPromise: null,
    isRemoved: false,
    render: null,
    syncLayout: null,
  };

  function syncLayout() {
    const nodeWidth = Math.max(Number(node?.size?.[0] || MIN_WIDTH), MIN_WIDTH);
    const nodeHeight = Math.max(Number(node?.size?.[1] || MIN_HEIGHT), MIN_HEIGHT);
    const innerWidth = Math.max(nodeWidth - 36, MIN_WIDTH - 36);
    const innerHeight = Math.max(nodeHeight - NODE_TOP_OFFSET, WIDGET_HEIGHT);
    container.style.width = `${innerWidth}px`;
    container.style.height = `${innerHeight}px`;
    container.style.minHeight = `${WIDGET_HEIGHT}px`;
  }

  function commit({ persist = true } = {}) {
    if (state.data.items.length) {
      state.data.selected_index = Math.max(0, Math.min(state.data.selected_index, state.data.items.length - 1));
    } else {
      state.data.selected_index = 0;
    }
    syncStorageWidgetValue(node, storageWidget, serializeState(state.data));
    render();
    if (persist) schedulePersist(state);
  }

  function renderList() {
    list.innerHTML = "";
    if (!state.data.items.length) {
      list.appendChild(createEl("div", "ess-image-lib-empty", "No images yet."));
      return;
    }
    state.data.items.forEach((item, index) => {
      const row = createEl("button", `ess-image-lib-item${index === state.data.selected_index ? " active" : ""}`);
      row.type = "button";
      const thumb = document.createElement("img");
      thumb.className = "ess-image-lib-thumb";
      const thumbSrc = item.image_data || createAssetUrl(item);
      if (thumbSrc) thumb.src = thumbSrc;
      const textWrap = document.createElement("div");
      const title = createEl("div", "ess-image-lib-item-title", item.name || `Image ${index + 1}`);
      const dims = item.width > 0 && item.height > 0 ? `${item.width}x${item.height}` : "unknown size";
      const meta = createEl("div", "ess-image-lib-item-meta", `${dims} • w=${Number(item.weight || 0)}`);
      textWrap.append(title, meta);
      row.append(thumb, textWrap);
      row.addEventListener("click", () => {
        state.data.selected_index = index;
        commit({ persist: false });
      });
      trapEvents(row);
      list.appendChild(row);
    });
  }

  function renderDetails() {
    modeSelect.value = state.data.mode;
    const item = currentItem(state.data);
    const hasItem = !!item;
    previewImage.style.display = hasItem ? "" : "none";
    badge.style.display = hasItem ? "" : "none";
    empty.style.display = hasItem ? "none" : "";
    nameInput.disabled = !hasItem;
    weightInput.disabled = !hasItem;
    promptInput.disabled = !hasItem;
    removeButton.disabled = !hasItem;

    if (!hasItem) {
      previewImage.removeAttribute("src");
      badge.textContent = "";
      nameInput.value = "";
      weightInput.value = "1";
      promptInput.value = "";
      status.textContent = getWorkflowReference()
        ? "Library is empty."
        : "Add images now; ESS will move them into the workflow's .ess file as soon as the workflow path is known.";
      return;
    }

    const previewSrc = item.image_data || createAssetUrl(item);
    if (previewSrc) {
      previewImage.src = previewSrc;
    } else {
      previewImage.removeAttribute("src");
    }
    const dims = item.width > 0 && item.height > 0 ? `${item.width} x ${item.height}` : "Stored image";
    const storageLabel = item.image_data ? "pending .ess sync" : ".ess asset";
    badge.textContent = `${dims} • ${storageLabel}${state.data.mode === "random" ? " • weighted random output" : " • manual output"}`;
    syncInputValue(nameInput, item.name || "");
    syncInputValue(weightInput, String(Number.isFinite(Number(item.weight)) ? Number(item.weight) : 1));
    syncInputValue(promptInput, item.prompt || "");
    if (state.persistPromise) {
      status.textContent = "Saving library assets...";
    } else if (!getWorkflowReference()) {
      status.textContent = "Workflow path not known yet. Assets stay temporary until the workflow is loaded from or saved to disk.";
    } else if (state.data.mode === "random") {
      status.textContent = "Execution chooses a weighted-random image. Weight 0 excludes an image from weighted picks.";
    } else {
      status.textContent = "Execution outputs the selected image.";
    }
  }

  function render() {
    syncLayout();
    renderList();
    renderDetails();
  }

  state.render = render;
  state.syncLayout = syncLayout;

  async function addFiles(files) {
    const incoming = Array.from(files || []).filter((file) => String(file?.type || "").startsWith("image/"));
    if (!incoming.length) return;
    addButton.disabled = true;
    clearButton.disabled = true;
    status.textContent = `Loading ${incoming.length} image${incoming.length === 1 ? "" : "s"}...`;
    try {
      for (const file of incoming) {
        const dataUrl = await readFileAsDataURL(file);
        const meta = await readImageMeta(dataUrl);
        state.data.items.push({
          name: String(file.name || "").trim() || `Image ${state.data.items.length + 1}`,
          prompt: "",
          weight: 1,
          image_data: dataUrl,
          asset_id: "",
          workflow_relative_path: getWorkflowReference(),
          mime_type: String(file.type || ""),
          width: meta.width,
          height: meta.height,
        });
      }
      state.data.selected_index = Math.max(0, state.data.items.length - incoming.length);
      commit();
    } catch (error) {
      status.textContent = String(error?.message || error || "Failed to add images.");
    } finally {
      addButton.disabled = false;
      clearButton.disabled = false;
      hiddenInput.value = "";
    }
  }

  addButton.addEventListener("click", () => hiddenInput.click());
  hiddenInput.addEventListener("change", () => addFiles(hiddenInput.files));
  clearButton.addEventListener("click", () => {
    state.data.items = [];
    state.data.selected_index = 0;
    commit();
  });
  modeSelect.addEventListener("change", () => {
    state.data.mode = modeSelect.value === "random" ? "random" : "manual";
    commit();
  });
  nameInput.addEventListener("input", () => {
    const item = currentItem(state.data);
    if (!item) return;
    item.name = String(nameInput.value || "").trim();
    commit();
  });
  weightInput.addEventListener("input", () => {
    const item = currentItem(state.data);
    if (!item) return;
    item.weight = Number.isFinite(Number(weightInput.value)) ? Number(weightInput.value) : 1;
    commit();
  });
  promptInput.addEventListener("input", () => {
    const item = currentItem(state.data);
    if (!item) return;
    item.prompt = String(promptInput.value || "");
    commit();
  });
  removeButton.addEventListener("click", () => {
    if (!state.data.items.length) return;
    state.data.items.splice(state.data.selected_index, 1);
    if (state.data.selected_index >= state.data.items.length) {
      state.data.selected_index = Math.max(0, state.data.items.length - 1);
    }
    commit();
  });

  const domWidget = node.addDOMWidget("image_library", "ess_image_library", container, {});
  domWidget.computeSize = (width) => [Math.max(Number(width || MIN_WIDTH), MIN_WIDTH), WIDGET_HEIGHT];
  state.domWidget = domWidget;

  storageWidget.__essImageLibraryAttached = true;
  storageWidget.__essImageLibrarySyncFromStorage = () => {
    state.data = parseState(storageWidget.value);
    render();
  };
  storageWidget.__essImageLibraryDomWidget = domWidget;

  attachedLibraries.add(state);

  const originalOnResize = node.onResize?.bind(node);
  node.onResize = function (size) {
    enforceNodeSize(node, size);
    const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
    state.syncLayout?.();
    node.setDirtyCanvas?.(true, true);
    return result;
  };

  const originalSetSize = node.setSize?.bind(node);
  if (originalSetSize && !node.__essImageLibrarySetSizeWrapped) {
    node.setSize = function (size) {
      const nextSize = [
        Math.max(Number(size?.[0] || 0), MIN_WIDTH),
        Math.max(Number(size?.[1] || 0), MIN_HEIGHT),
      ];
      return originalSetSize(nextSize);
    };
    node.__essImageLibrarySetSizeWrapped = true;
  }

  if (!node.__essImageLibraryLayoutTicker) {
    const tick = () => {
      if (state.isRemoved) {
        node.__essImageLibraryLayoutTicker = null;
        return;
      }
      state.syncLayout?.();
      requestAnimationFrame(tick);
    };
    node.__essImageLibraryLayoutTicker = requestAnimationFrame(tick);
  }

  const originalRemove = domWidget.onRemove?.bind(domWidget);
  domWidget.onRemove = function () {
    originalRemove?.();
    state.isRemoved = true;
    if (node.__essImageLibraryLayoutTicker) {
      cancelAnimationFrame(node.__essImageLibraryLayoutTicker);
      node.__essImageLibraryLayoutTicker = null;
    }
    if (state.persistTimer) clearTimeout(state.persistTimer);
    attachedLibraries.delete(state);
    storageWidget.__essImageLibraryAttached = false;
    storageWidget.__essImageLibrarySyncFromStorage = null;
    storageWidget.__essImageLibraryDomWidget = null;
    if (container.isConnected) container.remove();
  };

  render();
  schedulePersist(state);
}

function ensureImageLibraryWidgets(node) {
  if (!node?.widgets) return;
  const libraryWidgets = node.widgets.filter((widget) => widget?.__essImageLibraryConfig);
  if (!libraryWidgets.length) return;
  ensureNodeSize(node);
  for (const widget of libraryWidgets) {
    collapseStorageWidget(widget);
    if (widget.__essImageLibrarySyncFromStorage) {
      widget.__essImageLibrarySyncFromStorage();
      continue;
    }
    attachImageLibraryWidget(node, widget);
  }
}

app.registerExtension({
  name: "ess_image_library_provider",
  init() {
    ensureStyles();
    installLifecycleHooks();
    installExecutionHooks();
  },
  async getCustomWidgets() {
    return {
      ESS_IMAGE_LIBRARY(node, inputName, inputData) {
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
          ? String(inputData.value ?? "")
          : String(config.value ?? config.default ?? "");
        const storage = node.addWidget("text", inputName, initialValue, () => {}, {
          multiline: true,
          placeholder: config.placeholder,
          minHeight: config.minHeight,
          maxHeight: config.maxHeight,
          height: config.height,
          ess_image_library: true,
        });
        storage.value = initialValue;
        storage.__essImageLibraryConfig = { ...config };
        return { widget: storage, minHeight: 0 };
      },
    };
  },
  async beforeRegisterNodeDef(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        ensureImageLibraryWidgets(this);
      } catch (error) {
        console.error("[ess_image_library_provider] onNodeCreated failed:", error);
      }
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        ensureImageLibraryWidgets(this);
      } catch (error) {
        console.error("[ess_image_library_provider] onConfigure failed:", error);
      }
      return result;
    };
  },
});
