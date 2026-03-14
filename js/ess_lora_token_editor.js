import { app } from "../../scripts/app.js";
import {
  expandRangeToLoraTokenBoundaries,
  extractLoraTokens,
  findLoraTokenAt,
  normalizeLoraLookupKey,
  normalizeLoraWeight,
} from "./ess_lora_tokens_shared.js";

const STYLE_ID = "ess-lora-token-style";
const MENU_MAX_HEIGHT = 320;
const MENU_MAX_WIDTH = 460;

let loraNamesCache = null;
let loraLoadPromise = null;
let loraLoadStatus = "idle";

function stopEvent(event) {
  event.stopPropagation();
}

function uniqueSortedStrings(values) {
  const out = [];
  const seen = new Set();
  for (const value of values || []) {
    const text = String(value || "").replace(/\\/g, "/").replace(/^\/+/, "").trim();
    if (!text) continue;
    if (seen.has(text)) continue;
    seen.add(text);
    out.push(text);
  }
  out.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  return out;
}

function extractNamesFromObjectInfo(data) {
  const names = [];
  const readWidgetChoices = (nodeObj) => {
    const choices = nodeObj?.input?.required?.lora_name?.[0];
    if (Array.isArray(choices)) names.push(...choices);
  };

  readWidgetChoices(data);
  if (data && typeof data === "object") {
    for (const key of Object.keys(data)) {
      readWidgetChoices(data[key]);
    }
  }
  return names;
}

function extractNamesFromPayload(data) {
  if (Array.isArray(data)) {
    return data.filter((v) => typeof v === "string");
  }
  if (!data || typeof data !== "object") return [];
  if (Array.isArray(data.models)) return data.models.filter((v) => typeof v === "string");
  if (Array.isArray(data.loras)) return data.loras.filter((v) => typeof v === "string");
  if (Array.isArray(data.files)) return data.files.filter((v) => typeof v === "string");
  return extractNamesFromObjectInfo(data);
}

async function fetchJson(path) {
  const response = await app.api.fetchApi(path, { method: "GET" });
  if (!response || !response.ok) {
    throw new Error(`HTTP ${response?.status ?? "?"}`);
  }
  return response.json();
}

async function loadLoraNames() {
  if (Array.isArray(loraNamesCache)) return loraNamesCache;
  if (loraLoadPromise) return loraLoadPromise;

  loraLoadPromise = (async () => {
    const endpoints = [
      "/models/loras",
      "/api/models/loras",
      "/object_info/LoraLoader",
      "/api/object_info/LoraLoader",
      "/object_info",
      "/api/object_info",
    ];
    let hadSuccessfulResponse = false;
    let bestNames = [];
    for (const endpoint of endpoints) {
      try {
        const payload = await fetchJson(endpoint);
        hadSuccessfulResponse = true;
        const names = uniqueSortedStrings(extractNamesFromPayload(payload));
        if (names.length > bestNames.length) {
          bestNames = names;
        }
        if (names.length > 0) {
          loraLoadStatus = "ok";
          loraNamesCache = names;
          return names;
        }
      } catch {
        // try next endpoint
      }
    }
    if (hadSuccessfulResponse) {
      loraLoadStatus = "ok";
      loraNamesCache = bestNames;
      return loraNamesCache;
    }
    loraLoadStatus = "error";
    loraNamesCache = [];
    return loraNamesCache;
  })().finally(() => {
    loraLoadPromise = null;
  });

  return loraLoadPromise;
}

function getKnownLoraSet(names) {
  const out = new Set();
  for (const name of names || []) {
    const key = normalizeLoraLookupKey(name);
    if (!key) continue;
    out.add(key);
    if (key.endsWith(".safetensors")) out.add(key.slice(0, -12));
  }
  return out;
}

function getCaretClientRect(textarea, index) {
  const rect = textarea.getBoundingClientRect();
  const doc = textarea.ownerDocument || document;
  const mirror = doc.createElement("div");
  const span = doc.createElement("span");
  const style = window.getComputedStyle(textarea);
  const props = [
    "fontFamily", "fontSize", "fontWeight", "fontStyle", "fontVariant", "lineHeight",
    "letterSpacing", "textTransform", "textIndent", "textAlign",
    "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
    "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
    "boxSizing",
    "whiteSpace",
  ];
  mirror.style.position = "fixed";
  mirror.style.visibility = "hidden";
  mirror.style.pointerEvents = "none";
  mirror.style.left = `${Math.max(0, rect.left)}px`;
  mirror.style.top = `${Math.max(0, rect.top)}px`;
  mirror.style.width = `${textarea.clientWidth}px`;
  mirror.style.height = `${textarea.clientHeight}px`;
  mirror.style.overflow = "hidden";
  mirror.style.whiteSpace = "pre-wrap";
  mirror.style.wordBreak = "break-word";
  for (const prop of props) {
    mirror.style[prop] = style[prop];
  }
  mirror.scrollTop = textarea.scrollTop;
  mirror.scrollLeft = textarea.scrollLeft;

  const text = String(textarea.value || "");
  const safeIndex = Math.max(0, Math.min(index, text.length));
  const before = text.slice(0, safeIndex);
  mirror.textContent = before;
  span.textContent = "\u200b";
  mirror.appendChild(span);
  doc.body.appendChild(mirror);
  const spanRect = span.getBoundingClientRect();
  const result = {
    left: spanRect.left,
    top: spanRect.top,
    bottom: spanRect.bottom,
  };
  mirror.remove();
  return result;
}

function treeFromNames(names) {
  const root = { folders: new Map(), files: [] };
  for (const raw of names || []) {
    const normalized = String(raw || "").replace(/\\/g, "/").replace(/^\/+/, "");
    if (!normalized) continue;
    const parts = normalized.split("/").filter((p) => !!p);
    if (parts.length === 0) continue;
    let node = root;
    for (let i = 0; i < parts.length - 1; i += 1) {
      const folder = parts[i];
      if (!node.folders.has(folder)) {
        node.folders.set(folder, { folders: new Map(), files: [] });
      }
      node = node.folders.get(folder);
    }
    node.files.push(parts[parts.length - 1]);
  }
  return root;
}

function parseTrigger(text, caret) {
  if (caret < 0 || caret > text.length) return null;
  const before = text.slice(0, caret);
  const match = before.match(/<lora:([^>\n]*)$/i);
  if (!match) return null;
  return {
    start: caret - match[0].length,
    end: caret,
    query: String(match[1] || ""),
  };
}

function normalizeWeightInput(raw) {
  return normalizeLoraWeight(raw, "1");
}

export function ensureLoraTokenStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-tpl-lora-token{position:relative;display:inline-block;vertical-align:baseline;padding:0 1px;border-radius:4px;background:#1f2937;box-shadow:0 0 0 1px #4b5563 inset;color:#e5e7eb}
.ess-tpl-lora-token-active{box-shadow:0 0 0 1px #60a5fa inset}
.ess-tpl-lora-token-invalid{background:#2a1414;box-shadow:0 0 0 1px #b91c1c inset;color:#fecaca}
.ess-tpl-lora-token-ghost{opacity:0}
.ess-tpl-lora-token-label{position:absolute;inset:0;padding:0 1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none}
.ess-lora-picker{position:fixed;z-index:100500;background:#0b1222;border:1px solid #334155;border-radius:8px;box-shadow:0 12px 30px rgba(0,0,0,.45);width:min(${MENU_MAX_WIDTH}px,calc(100vw - 24px));max-height:min(${MENU_MAX_HEIGHT}px,calc(100vh - 24px));display:grid;grid-template-rows:auto auto minmax(0,1fr)}
.ess-lora-picker-head{display:grid;grid-template-columns:minmax(0,1fr) 92px 66px;gap:6px;padding:8px;border-bottom:1px solid #334155}
.ess-lora-picker-head input{width:100%;box-sizing:border-box;background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:5px 7px;font-size:12px}
.ess-lora-picker-btn{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:5px 7px;font-size:12px;cursor:pointer}
.ess-lora-picker-btn:hover{background:#172033}
.ess-lora-picker-meta{padding:0 8px 6px;color:#94a3b8;font-size:11px}
.ess-lora-picker-list{overflow:auto;padding:6px}
.ess-lora-picker-panels{display:flex;align-items:flex-start;gap:6px;min-height:100%}
.ess-lora-picker-panel{min-width:220px;max-width:220px;background:#0a1020;border:1px solid #334155;border-radius:6px;overflow:hidden}
.ess-lora-picker-panel-body{max-height:230px;overflow:auto}
.ess-lora-picker-row{display:flex;align-items:center;gap:6px;width:100%;box-sizing:border-box;background:transparent;border:0;color:#e5e7eb;text-align:left;padding:4px 8px;font-size:12px;cursor:pointer}
.ess-lora-picker-row:hover,.ess-lora-picker-row.active{background:#172033}
.ess-lora-picker-row-main{display:flex;align-items:center;gap:6px;min-width:0;flex:1}
.ess-lora-picker-name{min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ess-lora-picker-folder .ess-lora-picker-name{color:#f8fafc}
.ess-lora-picker-arrow{color:#93c5fd;font-size:11px;margin-left:4px}
.ess-lora-picker-empty{padding:10px;color:#94a3b8;font-size:12px}
.ess-lora-picker-path{color:#94a3b8;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px}
`;
  document.head.appendChild(style);
}

export function installLoraTokenController(textarea, options = {}) {
  if (!textarea) {
    return {
      getKnownLoras: () => null,
      refreshKnownLoras: async () => null,
      destroy: () => {},
    };
  }
  ensureLoraTokenStyles();

  const requestRender = typeof options.requestRender === "function" ? options.requestRender : () => {};
  let knownLoras = null;
  let menu = null;
  let menuState = null;
  let isApplying = false;

  const closeMenu = () => {
    if (!menu) return;
    const toRemove = menu;
    menu = null;
    menuState = null;
    if (toRemove.parentNode) toRemove.parentNode.removeChild(toRemove);
    document.removeEventListener("pointerdown", onOutsidePointerDown, true);
    document.removeEventListener("keydown", onGlobalEscape, true);
  };

  const onGlobalEscape = (event) => {
    if (event.key === "Escape") closeMenu();
  };

  const onOutsidePointerDown = (event) => {
    if (!menu) return;
    const target = event.target;
    if (menu.contains(target)) return;
    closeMenu();
  };

  const mutateText = (nextText, selStart, selEnd = selStart) => {
    isApplying = true;
    textarea.value = String(nextText ?? "");
    const start = Math.max(0, Math.min(Number(selStart || 0), textarea.value.length));
    const end = Math.max(start, Math.min(Number(selEnd ?? start), textarea.value.length));
    textarea.setSelectionRange(start, end);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
    isApplying = false;
    requestRender();
  };

  const applyToken = (name, weightText, start, end) => {
    const cleanName = String(name || "").trim();
    if (!cleanName) return;
    const weight = normalizeWeightInput(weightText);
    const token = `<lora:${cleanName}:${weight}>`;
    const source = String(textarea.value || "");
    const safeStart = Math.max(0, Math.min(start, source.length));
    const safeEnd = Math.max(safeStart, Math.min(end, source.length));
    const next = `${source.slice(0, safeStart)}${token}${source.slice(safeEnd)}`;
    const caret = safeStart + token.length;
    mutateText(next, caret);
  };

  const renderList = (
    listHost,
    allNames,
    treeRoot,
    query,
    selectName,
    activeName,
    openFolders,
    onOpenFoldersChange,
  ) => {
    listHost.textContent = "";
    const q = String(query || "").trim().toLowerCase();
    const filtered = q
      ? allNames.filter((name) => name.toLowerCase().includes(q))
      : allNames.slice();

    if (filtered.length === 0) {
      const empty = document.createElement("div");
      empty.className = "ess-lora-picker-empty";
      empty.textContent = allNames.length === 0
        ? "No LoRA models found. Check model paths."
        : "No matches.";
      listHost.appendChild(empty);
      return;
    }

    const addChoice = (fullName, label, pathHint = "") => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "ess-lora-picker-row";
      if (activeName && fullName === activeName) row.classList.add("active");
      const main = document.createElement("span");
      main.className = "ess-lora-picker-row-main";
      const nameEl = document.createElement("span");
      nameEl.className = "ess-lora-picker-name";
      nameEl.textContent = String(label || "");
      main.appendChild(nameEl);
      row.appendChild(main);
      if (pathHint) {
        const hint = document.createElement("span");
        hint.className = "ess-lora-picker-path";
        hint.textContent = String(pathHint || "");
        row.appendChild(hint);
      }
      row.addEventListener("pointerdown", stopEvent);
      row.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        selectName(fullName);
      });
      listHost.appendChild(row);
    };

    if (q) {
      for (const fullName of filtered) {
        const slash = fullName.lastIndexOf("/");
        if (slash >= 0) {
          addChoice(fullName, fullName.slice(slash + 1), fullName.slice(0, slash));
        } else {
          addChoice(fullName, fullName, "");
        }
      }
      return;
    }

    const panels = document.createElement("div");
    panels.className = "ess-lora-picker-panels";
    listHost.appendChild(panels);

    const buildPanel = (node, prefix, depth) => {
      const panel = document.createElement("div");
      panel.className = "ess-lora-picker-panel";
      const body = document.createElement("div");
      body.className = "ess-lora-picker-panel-body";
      panel.appendChild(body);

      const folders = Array.from(node.folders.keys())
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
      for (const folder of folders) {
        const row = document.createElement("button");
        row.type = "button";
        row.className = "ess-lora-picker-row ess-lora-picker-folder";
        if (openFolders[depth] === folder) row.classList.add("active");
        const main = document.createElement("span");
        main.className = "ess-lora-picker-row-main";
        const icon = document.createElement("span");
        icon.textContent = "\uD83D\uDCC1";
        const nameEl = document.createElement("span");
        nameEl.className = "ess-lora-picker-name";
        nameEl.textContent = String(folder || "");
        main.appendChild(icon);
        main.appendChild(nameEl);
        const arrow = document.createElement("span");
        arrow.className = "ess-lora-picker-arrow";
        arrow.textContent = ">";
        row.appendChild(main);
        row.appendChild(arrow);
        row.addEventListener("pointerdown", stopEvent);
        const openThis = (event) => {
          event.preventDefault();
          event.stopPropagation();
          const next = openFolders.slice(0, depth);
          next[depth] = folder;
          onOpenFoldersChange(next);
        };
        row.addEventListener("mouseenter", openThis);
        row.addEventListener("click", openThis);
        body.appendChild(row);
      }

      const files = (node.files || [])
        .slice()
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
      for (const file of files) {
        const fullName = prefix ? `${prefix}/${file}` : file;
        const row = document.createElement("button");
        row.type = "button";
        row.className = "ess-lora-picker-row";
        if (activeName && fullName === activeName) row.classList.add("active");
        const main = document.createElement("span");
        main.className = "ess-lora-picker-row-main";
        const nameEl = document.createElement("span");
        nameEl.className = "ess-lora-picker-name";
        nameEl.textContent = String(file || "");
        main.appendChild(nameEl);
        row.appendChild(main);
        row.addEventListener("pointerdown", stopEvent);
        row.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          selectName(fullName);
        });
        body.appendChild(row);
      }
      return panel;
    };

    let node = treeRoot;
    let prefix = "";
    let depth = 0;
    while (node) {
      panels.appendChild(buildPanel(node, prefix, depth));
      const folder = openFolders[depth];
      if (!folder || !node.folders.has(folder)) break;
      prefix = prefix ? `${prefix}/${folder}` : folder;
      node = node.folders.get(folder);
      depth += 1;
    }
  };

  const openMenu = async (state) => {
    closeMenu();
    menuState = state;
    const names = await loadLoraNames();
    if (!menuState || menuState !== state) return;

    const menuEl = document.createElement("div");
    menuEl.className = "ess-lora-picker";
    menuEl.addEventListener("pointerdown", stopEvent, true);
    menuEl.addEventListener("keydown", stopEvent, true);

    const head = document.createElement("div");
    head.className = "ess-lora-picker-head";
    const queryInput = document.createElement("input");
    queryInput.type = "text";
    queryInput.placeholder = "Filter list";
    queryInput.value = String(state.query || "");
    const weightInput = document.createElement("input");
    weightInput.type = "text";
    weightInput.placeholder = "weight";
    weightInput.value = normalizeWeightInput(state.weight || "1");
    const applyBtn = document.createElement("button");
    applyBtn.type = "button";
    applyBtn.className = "ess-lora-picker-btn";
    applyBtn.textContent = "Apply";
    head.appendChild(queryInput);
    head.appendChild(weightInput);
    head.appendChild(applyBtn);

    const meta = document.createElement("div");
    meta.className = "ess-lora-picker-meta";
    meta.textContent = `${names.length} LoRA models`;

    const listHost = document.createElement("div");
    listHost.className = "ess-lora-picker-list";
    const treeRoot = treeFromNames(names);

    let activeName = String(state.activeName || "");
    let openFolders = [];
    if (activeName && activeName.includes("/")) {
      const parts = activeName.split("/").filter((part) => !!part);
      openFolders = parts.slice(0, -1);
    }
    const selectAndApply = (name) => {
      activeName = name;
      applyToken(activeName, weightInput.value, state.start, state.end);
      closeMenu();
    };

    const rerenderList = () => {
      renderList(
        listHost,
        names,
        treeRoot,
        queryInput.value,
        selectAndApply,
        activeName,
        openFolders,
        (nextFolders) => {
          openFolders = Array.isArray(nextFolders) ? nextFolders : [];
          rerenderList();
        },
      );
    };
    rerenderList();

    queryInput.addEventListener("input", () => {
      rerenderList();
    });
    applyBtn.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!activeName) {
        const q = String(queryInput.value || "").trim();
        if (!q) return;
        const exact = names.find((name) => name.toLowerCase() === q.toLowerCase());
        if (exact) {
          activeName = exact;
        } else {
          return;
        }
      }
      applyToken(activeName, weightInput.value, state.start, state.end);
      closeMenu();
    });
    queryInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        applyBtn.click();
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeMenu();
      }
    });
    weightInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        applyBtn.click();
      } else if (event.key === "Escape") {
        event.preventDefault();
        closeMenu();
      }
    });

    menuEl.appendChild(head);
    menuEl.appendChild(meta);
    menuEl.appendChild(listHost);
    document.body.appendChild(menuEl);

    const viewportW = window.innerWidth || 0;
    const viewportH = window.innerHeight || 0;
    const rect = menuEl.getBoundingClientRect();
    const left = Math.max(8, Math.min(state.anchorX || 16, Math.max(8, viewportW - rect.width - 8)));
    const top = Math.max(8, Math.min(state.anchorY || 16, Math.max(8, viewportH - rect.height - 8)));
    menuEl.style.left = `${left}px`;
    menuEl.style.top = `${top}px`;

    menu = menuEl;
    document.addEventListener("pointerdown", onOutsidePointerDown, true);
    document.addEventListener("keydown", onGlobalEscape, true);

    queryInput.focus();
    queryInput.select();
  };

  const openInsertMenuFromTrigger = () => {
    const text = String(textarea.value || "");
    const caret = textarea.selectionStart || 0;
    const trigger = parseTrigger(text, caret);
    if (!trigger) return false;
    const caretRect = getCaretClientRect(textarea, caret);
    const state = {
      mode: "insert",
      start: trigger.start,
      end: trigger.end,
      query: trigger.query,
      weight: "1",
      anchorX: caretRect.left + 12,
      anchorY: caretRect.bottom + 8,
      activeName: "",
    };
    openMenu(state);
    return true;
  };

  const openEditMenuForToken = (token, anchorX, anchorY) => {
    if (!token) return false;
    const state = {
      mode: "replace",
      start: token.start,
      end: token.end,
      query: token.name || "",
      weight: token.displayWeight || "1",
      anchorX,
      anchorY,
      activeName: token.name || "",
    };
    openMenu(state);
    return true;
  };

  const onInput = () => {
    if (isApplying) return;
    const hasTrigger = openInsertMenuFromTrigger();
    if (!hasTrigger && menuState?.mode === "insert") {
      closeMenu();
    }
  };

  const onClick = (event) => {
    if (event.defaultPrevented) return;
    if (textarea.selectionStart !== textarea.selectionEnd) return;
    const text = String(textarea.value || "");
    const caret = textarea.selectionStart || 0;
    const token = findLoraTokenAt(text, caret);
    if (!token) return;
    event.preventDefault();
    event.stopPropagation();
    openEditMenuForToken(token, event.clientX + 8, event.clientY + 8);
  };

  const onKeydown = (event) => {
    if (event.defaultPrevented) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    const text = String(textarea.value || "");
    let selStart = textarea.selectionStart || 0;
    let selEnd = textarea.selectionEnd || 0;
    if (selEnd < selStart) {
      const tmp = selStart;
      selStart = selEnd;
      selEnd = tmp;
    }

    const key = event.key;
    const collapsed = selStart === selEnd;

    if (key === "Backspace" && collapsed) {
      const token = findLoraTokenAt(text, selStart, { inclusive: true })
        || findLoraTokenAt(text, Math.max(0, selStart - 1), { inclusive: true });
      if (token) {
        event.preventDefault();
        const next = `${text.slice(0, token.start)}${text.slice(token.end)}`;
        mutateText(next, token.start);
      }
      return;
    }
    if ((key === "Backspace" || key === "Delete") && !collapsed) {
      const expanded = expandRangeToLoraTokenBoundaries(text, selStart, selEnd);
      if (expanded.start !== selStart || expanded.end !== selEnd) {
        textarea.setSelectionRange(expanded.start, expanded.end);
      }
      return;
    }
    if (key === "Delete" && collapsed) {
      const token = findLoraTokenAt(text, selStart, { inclusive: true })
        || findLoraTokenAt(text, Math.min(text.length, selStart + 1), { inclusive: true });
      if (token) {
        event.preventDefault();
        const next = `${text.slice(0, token.start)}${text.slice(token.end)}`;
        mutateText(next, token.start);
      }
      return;
    }

    if (key === "ArrowLeft" || key === "ArrowRight") {
      const token = findLoraTokenAt(text, selStart);
      if (token && collapsed) {
        event.preventDefault();
        if (key === "ArrowLeft") {
          textarea.setSelectionRange(token.start, token.start);
        } else {
          textarea.setSelectionRange(token.end, token.end);
        }
      }
      return;
    }

    const writesText = key.length === 1 || key === "Enter" || key === "Tab";
    if (!writesText) return;

    if (collapsed) {
      const token = findLoraTokenAt(text, selStart);
      if (token) {
        event.preventDefault();
        textarea.setSelectionRange(token.end, token.end);
      }
      return;
    }

    const expanded = expandRangeToLoraTokenBoundaries(text, selStart, selEnd);
    if (expanded.start !== selStart || expanded.end !== selEnd) {
      textarea.setSelectionRange(expanded.start, expanded.end);
    }
  };

  const onPaste = (event) => {
    if (!event.clipboardData) return;
    const pasted = event.clipboardData.getData("text/plain");
    if (pasted == null) return;
    event.preventDefault();

    const text = String(textarea.value || "");
    let start = textarea.selectionStart || 0;
    let end = textarea.selectionEnd || 0;
    if (end < start) {
      const t = start;
      start = end;
      end = t;
    }
    if (start === end) {
      const token = findLoraTokenAt(text, start);
      if (token) {
        start = token.end;
        end = token.end;
      }
    } else {
      const expanded = expandRangeToLoraTokenBoundaries(text, start, end);
      start = expanded.start;
      end = expanded.end;
    }
    const next = `${text.slice(0, start)}${pasted}${text.slice(end)}`;
    const caret = start + pasted.length;
    mutateText(next, caret);
  };

  const refreshKnownLoras = async () => {
    const names = await loadLoraNames();
    knownLoras = loraLoadStatus === "ok" ? getKnownLoraSet(names) : null;
    requestRender();
    return knownLoras;
  };

  textarea.addEventListener("input", onInput);
  textarea.addEventListener("click", onClick);
  textarea.addEventListener("keydown", onKeydown);
  textarea.addEventListener("paste", onPaste);
  refreshKnownLoras();

  return {
    getKnownLoras: () => knownLoras,
    refreshKnownLoras,
    getTokens: () => extractLoraTokens(String(textarea.value || ""), { knownLoras }),
    destroy: () => {
      closeMenu();
      textarea.removeEventListener("input", onInput);
      textarea.removeEventListener("click", onClick);
      textarea.removeEventListener("keydown", onKeydown);
      textarea.removeEventListener("paste", onPaste);
    },
  };
}
