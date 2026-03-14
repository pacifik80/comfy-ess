import { app } from "../../scripts/app.js";
import { renderHighlight } from "./ess_template_highlight.js";
import { installLoraTokenController } from "./ess_lora_token_editor.js";

const STYLE_ID = "ess-replacements-editor-style";
const MAX_PAIRS = 24;
const VALUE_EDITOR_HEIGHT = 122;
const MIN_NODE_WIDTH = 460;
const STORAGE_PROP_KEY = "ess_replace_pairs";
const STORAGE_PROP_KEY_LEGACY = "__ess_replace_pairs";
const APPLIED_COUNT_PROP_KEY = "ess_replace_applied_count";
const APPLIED_COUNT_PROP_KEY_LEGACY = "__ess_replace_applied_count";

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-repl-slot{display:flex;align-items:flex-start;justify-content:center;width:100%;pointer-events:none}
.ess-repl-slot.expanded{height:100%;padding:0 0 12px;box-sizing:border-box}
.ess-repl-pair{display:grid;gap:6px;border:1px solid #334155;border-radius:8px;background:#070f20;padding:6px;box-sizing:border-box;width:calc(100% - 40px);max-width:calc(100% - 40px);margin:0 auto;overflow:hidden;pointer-events:auto}
.ess-repl-slot.expanded .ess-repl-pair{height:100%;grid-template-rows:auto minmax(0,1fr)}
.ess-repl-pair-header{display:grid;grid-template-columns:32px minmax(0,1fr) auto;gap:6px;align-items:center}
.ess-repl-pair-index{font-size:11px;color:#93c5fd;min-width:0}
.ess-repl-pair-key{background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:5px;padding:4px 6px;font-size:12px;height:30px;box-sizing:border-box;width:100%;min-width:0}
.ess-repl-pair-key.invalid{background:#1b1010;color:#fecaca;border-color:#b91c1c;box-shadow:0 0 0 1px rgba(185,28,28,.2) inset}
.ess-repl-pair-toggle{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:5px;padding:3px 8px;font-size:11px;cursor:pointer;height:30px;white-space:nowrap}
.ess-repl-pair-body{padding:2px 0 0}
.ess-repl-slot.expanded .ess-repl-pair-body{height:100%;min-height:0}
.ess-repl-editor{position:relative;width:100%;height:${VALUE_EDITOR_HEIGHT}px;border:1px solid #374151;background:#0b1222;border-radius:7px;overflow:hidden}
.ess-repl-slot.expanded .ess-repl-editor{height:100%;min-height:${VALUE_EDITOR_HEIGHT}px}
.ess-repl-editor .ess-repl-highlight-content,.ess-repl-editor textarea{font-family:"JetBrains Mono","Fira Code","Consolas",monospace;font-size:12px;line-height:1.45;padding:8px 10px;box-sizing:border-box;width:100%;height:100%}
.ess-repl-editor .ess-repl-highlight{position:absolute;inset:0;color:#c9d1d9;overflow:hidden}
.ess-repl-editor .ess-repl-highlight-content{white-space:pre-wrap;word-break:break-word;pointer-events:none;will-change:transform}
.ess-repl-editor textarea{position:relative;background:transparent;color:transparent;caret-color:#e6edf3;border:none;outline:none;resize:none;overflow:auto}
.ess-repl-editor .ess-tpl-choice{color:#2f6f3e}
.ess-repl-editor .ess-tpl-choice-active{color:#7ee787;font-weight:700}
.ess-repl-editor .ess-tpl-header{color:#b48800}
.ess-repl-editor .ess-tpl-header-active{color:#f9e076;font-weight:700}
.ess-repl-editor .ess-tpl-header-name{color:#8b5a2b}
.ess-repl-editor .ess-tpl-header-name-active{color:#d2a064;font-weight:700}
.ess-repl-editor .ess-tpl-header-number{color:#d16ba5}
.ess-repl-editor .ess-tpl-header-number-active{color:#ff9ad5;font-weight:700}
.ess-repl-editor .ess-tpl-negative-marker{color:#a43f3f}
.ess-repl-editor .ess-tpl-negative-marker-active{color:#ff7b72;font-weight:700}
.ess-repl-editor .ess-tpl-negative-text{color:#a43f3f}
.ess-repl-editor .ess-tpl-variant{color:#5b2c83}
.ess-repl-editor .ess-tpl-variant-active{color:#d6a8ff;font-weight:700}
.ess-repl-editor .ess-tpl-variant-label{color:#6b4fb3}
.ess-repl-editor .ess-tpl-variant-label-active{color:#e0c1ff;font-weight:700}
.ess-repl-editor .ess-tpl-comment{color:#6e7681}
.ess-repl-editor .ess-tpl-placeholder{color:#38bdf8}
.ess-repl-editor .ess-tpl-placeholder-active{color:#7dd3fc;font-weight:700}
.ess-repl-editor .ess-tpl-placeholder-unknown{color:#38bdf8;text-decoration-line:underline;text-decoration-style:wavy;text-decoration-color:#f87171;text-decoration-thickness:1.2px;text-underline-offset:2px}
.ess-repl-editor .ess-tpl-placeholder-unknown-active{color:#7dd3fc;font-weight:700;text-decoration-line:underline;text-decoration-style:wavy;text-decoration-color:#fca5a5;text-decoration-thickness:1.4px;text-underline-offset:2px}
.ess-repl-pair-menu{position:fixed;z-index:100000;background:#0b1222;border:1px solid #334155;border-radius:7px;min-width:170px;padding:4px;box-shadow:0 8px 24px rgba(0,0,0,.45)}
.ess-repl-pair-menu-btn{width:100%;display:block;text-align:left;background:transparent;color:#e5e7eb;border:0;border-radius:5px;padding:6px 10px;font-size:12px;cursor:pointer}
.ess-repl-pair-menu-btn:hover{background:#172033}
.ess-repl-pair-menu-btn:disabled{color:#64748b;cursor:default}
.ess-repl-pair-menu-sep{height:1px;background:#334155;margin:4px 0}
`;
  document.head.appendChild(style);
}

function trapEditorEvents(element) {
  if (!element) return;
  const stop = (event) => event.stopPropagation();
  [
    "keydown", "keyup", "keypress",
    "pointerdown", "pointermove", "pointerup",
    "mousedown", "mouseup",
    "click", "dblclick", "contextmenu",
  ].forEach((name) => element.addEventListener(name, stop));
}

function createTemplateEditor(initialValue, onInput, onBlur) {
  const wrap = document.createElement("div");
  wrap.className = "ess-repl-editor";
  const highlight = document.createElement("div");
  highlight.className = "ess-repl-highlight";
  const highlightContent = document.createElement("div");
  highlightContent.className = "ess-repl-highlight-content";
  highlight.appendChild(highlightContent);
  const input = document.createElement("textarea");
  input.value = initialValue || "";

  let raf = 0;
  let loraController = null;
  const syncSize = () => {
    highlightContent.style.width = `${input.clientWidth || 0}px`;
    highlightContent.style.height = `${input.clientHeight || 0}px`;
  };
  const syncScroll = () => {
    const x = input.scrollLeft || 0;
    const y = input.scrollTop || 0;
    highlightContent.style.transform = `translate(${-x}px, ${-y}px)`;
  };
  const render = () => {
    try {
      const knownLoras = loraController?.getKnownLoras?.() || null;
      const rendered = renderHighlight(input.value || "", input.selectionStart || 0, { knownLoras });
      if (typeof rendered === "string" && rendered.length > 0) {
        highlightContent.innerHTML = rendered;
      } else if ((input.value || "").length > 0) {
        highlightContent.textContent = input.value;
      } else {
        highlightContent.innerHTML = "&nbsp;";
      }
    } catch {
      // Fail-safe: never leave editor visually blank due to highlighting errors.
      highlightContent.textContent = input.value || " ";
    }
    syncScroll();
  };
  const scheduleSync = () => {
    if (raf) cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      raf = 0;
      syncSize();
      render();
    });
  };

  input.addEventListener("input", () => {
    scheduleSync();
    onInput?.(input.value);
  });
  input.addEventListener("blur", () => onBlur?.(input.value));
  input.addEventListener("scroll", syncScroll);
  input.addEventListener("click", scheduleSync);
  input.addEventListener("keyup", scheduleSync);
  input.addEventListener("select", scheduleSync);
  input.addEventListener("focus", scheduleSync);
  input.addEventListener("paste", () => {
    // Paste may update scroll/caret after input event.
    scheduleSync();
    requestAnimationFrame(scheduleSync);
  });
  input.addEventListener("cut", scheduleSync);
  input.addEventListener("drop", () => {
    scheduleSync();
    requestAnimationFrame(scheduleSync);
  });
  input.addEventListener("compositionend", scheduleSync);

  trapEditorEvents(input);
  loraController = installLoraTokenController(input, {
    container: wrap,
    requestRender: scheduleSync,
  });
  scheduleSync();

  wrap.appendChild(highlight);
  wrap.appendChild(input);
  const resizeObserver = new ResizeObserver(() => {
    syncSize();
    syncScroll();
  });
  resizeObserver.observe(input);

  wrap.__essReplDestroy = () => {
    resizeObserver.disconnect();
    loraController?.destroy?.();
    if (raf) cancelAnimationFrame(raf);
  };
  return { container: wrap, input, refresh: scheduleSync };
}

function parsePairs(raw) {
  try {
    const parsed = JSON.parse(String(raw || "[]"));
    if (!Array.isArray(parsed)) return [];
    return parsed.map((item) => ({
      key: String(item?.key || ""),
      value: String(item?.value || ""),
      expanded: !!item?.expanded,
    }));
  } catch {
    return [];
  }
}

function loadPersistedPairs(node, storageWidget) {
  const fromStorage = parsePairs(storageWidget?.value);
  if (fromStorage.length > 0) return fromStorage;
  const fromProps = parsePairs(
    node?.properties?.[STORAGE_PROP_KEY] ?? node?.properties?.[STORAGE_PROP_KEY_LEGACY],
  );
  if (fromProps.length > 0) return fromProps;
  // Recovery path: workflow may carry serialized pairs in widgets_values
  // (e.g., when widget order changed between versions).
  const values = Array.isArray(node?.widgets_values) ? node.widgets_values : [];
  let best = [];
  for (const raw of values) {
    const parsed = parsePairs(raw);
    if (parsed.length > best.length) best = parsed;
  }
  if (best.length > 0) return best;
  return fromStorage;
}

function serializePairs(pairs) {
  return JSON.stringify(
    (pairs || []).map((it) => ({
      key: String(it?.key || ""),
      value: String(it?.value || ""),
      expanded: !!it?.expanded,
    })),
    null,
    2,
  );
}

function detectLastUsedPairIndex(pairs) {
  if (!Array.isArray(pairs) || pairs.length === 0) return -1;
  for (let i = pairs.length - 1; i >= 0; i -= 1) {
    const row = pairs[i] || {};
    const key = String(row.key || "").trim();
    const value = String(row.value || "").trim();
    if (key || value || !!row.expanded) return i;
  }
  return -1;
}

function clampCount(value) {
  const n = Math.floor(Number(value) || 1);
  return Math.max(1, Math.min(n, MAX_PAIRS));
}

function isInvalidKey(raw) {
  const key = String(raw || "").trim();
  if (!key) return false;
  return key.includes("%") || /\s/.test(key);
}

function getCountWidget(node) {
  if (!Array.isArray(node?.widgets)) return null;
  return node.widgets.find((w) => String(w?.name || "").trim() === "count") || null;
}

function getStorageWidget(node) {
  if (!Array.isArray(node?.widgets)) return null;
  return node.widgets.find((w) => String(w?.name || "").trim() === "replacements_editor") || null;
}

function getSerializedWidgetValue(node, widget) {
  if (!node || !widget || !Array.isArray(node.widgets) || !Array.isArray(node.widgets_values)) return undefined;
  const idx = node.widgets.indexOf(widget);
  if (idx < 0) return undefined;
  return node.widgets_values[idx];
}

function hideStorageWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.serialize = true;
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
  const element = widget.element || widget.inputEl;
  if (element && element.style) {
    element.style.display = "none";
    element.style.height = "0px";
    element.style.minHeight = "0px";
    element.style.margin = "0";
    element.style.padding = "0";
    element.style.border = "0";
    element.style.overflow = "hidden";
  }
}

function resizeNodeKeepWidth(node) {
  if (!node) return;
  const previousWidth = Math.max(Number(node.size?.[0] || 0), MIN_NODE_WIDTH);
  const computed = node.computeSize ? node.computeSize() : node.size;
  if (!computed || !Array.isArray(computed)) return;
  const nextHeight = Math.max(Number(computed[1] || node.size?.[1] || 0), 80);
  node.setSize?.([previousWidth, nextHeight]);
}

function ensureNodeMinHeightKeepUserHeight(node) {
  if (!node) return;
  const previousWidth = Math.max(Number(node.size?.[0] || 0), MIN_NODE_WIDTH);
  const computed = node.computeSize ? node.computeSize() : node.size;
  if (!computed || !Array.isArray(computed)) return;
  const computedHeight = Math.max(Number(computed[1] || 0), 80);
  const currentHeight = Math.max(Number(node.size?.[1] || 0), 80);
  node.setSize?.([previousWidth, Math.max(currentHeight, computedHeight)]);
}

app.registerExtension({
  name: "ess_replacements_editor",

  async getCustomWidgets() {
    // Hidden data carrier. Visual widgets are injected in beforeRegisterNodeDef.
    return {
      ESS_REPLACEMENTS_EDITOR(node, inputName, inputData) {
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
          ? String(inputData.value || "")
          : String(config.value ?? config.default ?? "[]");

        // Use a plain widget here for maximum compatibility.
        // Complex UI is injected later in beforeRegisterNodeDef.
        const storage = node.addWidget("text", inputName, String(initialValue || "[]"), () => {}, { multiline: true });
        storage.value = String(initialValue || "[]");
        hideStorageWidget(storage);
        return { widget: storage, minHeight: 0 };
      },
    };
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/ReplaceDict") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      let result;
      try {
        result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      } catch (error) {
        console.error("[ess_replacements_editor] base onNodeCreated failed:", error);
        // Do not hard-fail node creation because of UI hook issues.
        return undefined;
      }
      try {
        ensureStyles();

        const countWidget = getCountWidget(this);
        const storageWidget = getStorageWidget(this);
        if (!countWidget || !storageWidget || this.__essReplUiReady) return result;
        if (typeof this.addDOMWidget !== "function") return result;
        this.__essReplUiReady = true;
        if (Array.isArray(this.size)) {
          this.size[0] = Math.max(Number(this.size[0] || 0), MIN_NODE_WIDTH);
        } else {
          this.size = [MIN_NODE_WIDTH, 240];
        }

        hideStorageWidget(storageWidget);

        let pairs = loadPersistedPairs(this, storageWidget);
        while (pairs.length < MAX_PAIRS) pairs.push({ key: "", value: "", expanded: false });
        if (pairs.length > MAX_PAIRS) pairs = pairs.slice(0, MAX_PAIRS);
        const savedAppliedInitial = Number(
          this.properties?.[APPLIED_COUNT_PROP_KEY] ?? this.properties?.[APPLIED_COUNT_PROP_KEY_LEGACY],
        );
        let appliedCount = Number.isFinite(savedAppliedInitial)
          ? clampCount(savedAppliedInitial)
          : clampCount(countWidget.value);

        const pairRows = [];
        let activePairMenu = null;

        const closePairMenu = () => {
          if (!activePairMenu) return;
          const menu = activePairMenu;
          activePairMenu = null;
          if (menu.parentNode) menu.parentNode.removeChild(menu);
          document.removeEventListener("pointerdown", closePairMenuOnOutside, true);
          document.removeEventListener("keydown", closePairMenuOnEscape, true);
        };
        const closePairMenuOnEscape = (event) => {
          if (event.key === "Escape") closePairMenu();
        };
        const closePairMenuOnOutside = (event) => {
          if (!activePairMenu) return;
          if (!activePairMenu.contains(event.target)) closePairMenu();
        };

        const stashLocal = () => {
          const safeApplied = Math.max(1, Math.min(MAX_PAIRS, Number(appliedCount) || 1));
          // Never truncate persisted rows to visible count; keep authored data safe.
          const lastUsedIndex = detectLastUsedPairIndex(pairs);
          const keepCount = Math.max(safeApplied, lastUsedIndex + 1, 1);
          const payload = serializePairs(pairs.slice(0, keepCount));
          storageWidget.value = payload;
          this.properties = this.properties || {};
          this.properties[STORAGE_PROP_KEY] = payload;
          this.properties[STORAGE_PROP_KEY_LEGACY] = payload;
          this.properties[APPLIED_COUNT_PROP_KEY] = safeApplied;
          this.properties[APPLIED_COUNT_PROP_KEY_LEGACY] = safeApplied;
          if (Array.isArray(this.widgets_values)) {
            const idx = this.widgets?.indexOf(storageWidget);
            if (idx != null && idx >= 0) this.widgets_values[idx] = payload;
          }
        };

        const commit = () => {
          stashLocal();
          if (typeof storageWidget.callback === "function") storageWidget.callback(storageWidget.value);
          try {
            window.dispatchEvent(new CustomEvent("ess:replace-dict-changed", {
              detail: { nodeId: this.id, graph: this.graph || null },
            }));
          } catch {}
          this.setDirtyCanvas?.(true, true);
        };

        const syncRowContentFromPairs = () => {
          for (let i = 0; i < pairRows.length; i += 1) {
            const row = pairRows[i];
            const pair = pairs[i] || { key: "", value: "", expanded: false };
            if (row.keyInput.value !== pair.key) {
              row.keyInput.value = pair.key;
            }
            row.keyInput.classList.toggle("invalid", isInvalidKey(pair.key || ""));
            if (row.editor.input.value !== pair.value) {
              row.editor.input.value = pair.value;
              row.editor.refresh?.();
            }
            const expanded = !!pair.expanded && i < appliedCount && row.visible;
            row.toggleButton.textContent = expanded ? "Collapse" : "Edit";
            row.body.style.display = expanded ? "" : "none";
            row.container.classList.toggle("expanded", expanded);
          }
        };

        const applyCount = (commitAfter = true) => {
          appliedCount = clampCount(countWidget.value);
          this.__essReplAppliedCount = appliedCount;
          while (pairs.length < appliedCount) pairs.push({ key: "", value: "", expanded: false });
          if (pairs.length > MAX_PAIRS) pairs = pairs.slice(0, MAX_PAIRS);

          for (let i = 0; i < pairRows.length; i += 1) {
            const row = pairRows[i];
            const visible = i < appliedCount;
            row.visible = visible;
            row.container.style.display = visible ? "" : "none";
            row.widget.hidden = !visible;
            row.indexLabel.textContent = `#${i + 1}`;
          }
          syncRowContentFromPairs();
          if (commitAfter) commit();
          resizeNodeKeepWidth(this);
          this.setDirtyCanvas?.(true, true);
        };

        const movePair = (fromIndex, toIndex) => {
          if (fromIndex < 0 || toIndex < 0 || fromIndex >= appliedCount || toIndex >= appliedCount) return;
          if (fromIndex === toIndex) return;
          const [moved] = pairs.splice(fromIndex, 1);
          pairs.splice(toIndex, 0, moved || { key: "", value: "", expanded: false });
          applyCount(true);
        };

        const deletePairAt = (index) => {
          if (index < 0 || index >= appliedCount) return;
          if (!window.confirm(`Delete pair #${index + 1}?`)) return;
          if (appliedCount <= 1) {
            pairs[0] = { key: "", value: "", expanded: false };
            applyCount(true);
            return;
          }
          pairs.splice(index, 1);
          while (pairs.length < MAX_PAIRS) pairs.push({ key: "", value: "", expanded: false });
          if (pairs.length > MAX_PAIRS) pairs = pairs.slice(0, MAX_PAIRS);
          countWidget.value = Math.max(1, appliedCount - 1);
          applyCount(true);
        };

        const openPairMenu = (event, index) => {
          event.preventDefault();
          event.stopPropagation();
          closePairMenu();
          if (index < 0 || index >= appliedCount) return;

          const menu = document.createElement("div");
          menu.className = "ess-repl-pair-menu";

          const addAction = (label, enabled, callback, danger = false) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "ess-repl-pair-menu-btn";
            btn.textContent = label;
            if (danger) btn.style.color = "#fca5a5";
            btn.disabled = !enabled;
            btn.addEventListener("click", (e) => {
              e.preventDefault();
              e.stopPropagation();
              closePairMenu();
              if (enabled) callback?.();
            });
            menu.appendChild(btn);
          };
          const addSeparator = () => {
            const sep = document.createElement("div");
            sep.className = "ess-repl-pair-menu-sep";
            menu.appendChild(sep);
          };

          addAction("Move Up", index > 0, () => movePair(index, index - 1));
          addAction("Move Down", index < appliedCount - 1, () => movePair(index, index + 1));
          addSeparator();
          addAction("Delete Pair", true, () => deletePairAt(index), true);

          document.body.appendChild(menu);
          const viewportW = window.innerWidth || 0;
          const viewportH = window.innerHeight || 0;
          const rect = menu.getBoundingClientRect();
          const left = Math.max(8, Math.min(event.clientX, Math.max(8, viewportW - rect.width - 8)));
          const top = Math.max(8, Math.min(event.clientY, Math.max(8, viewportH - rect.height - 8)));
          menu.style.left = `${left}px`;
          menu.style.top = `${top}px`;

          activePairMenu = menu;
          document.addEventListener("pointerdown", closePairMenuOnOutside, true);
          document.addEventListener("keydown", closePairMenuOnEscape, true);
        };

        const applyButton = this.addWidget("button", "apply", null, () => applyCount(true));
        applyButton.label = "Apply count";
        applyButton.serialize = false;

        const originalCountCallback = countWidget.callback;
        countWidget.callback = (value, canvas, node, pos, event) => {
          if (originalCountCallback) originalCountCallback(value, canvas, node, pos, event);
          this.setDirtyCanvas?.(true, true);
        };

        const originalConnectionsChange = this.onConnectionsChange;
        this.onConnectionsChange = function () {
          const out = originalConnectionsChange
            ? originalConnectionsChange.apply(this, arguments)
            : undefined;
          try {
            window.dispatchEvent(new CustomEvent("ess:replace-dict-changed", {
              detail: { nodeId: this.id, graph: this.graph || null },
            }));
          } catch {}
          return out;
        };

        for (let i = 0; i < MAX_PAIRS; i += 1) {
          const container = document.createElement("div");
          container.className = "ess-repl-slot";
          const pairCard = document.createElement("div");
          pairCard.className = "ess-repl-pair";
          container.appendChild(pairCard);

          const header = document.createElement("div");
          header.className = "ess-repl-pair-header";
          const indexLabel = document.createElement("div");
          indexLabel.className = "ess-repl-pair-index";
          indexLabel.textContent = `#${i + 1}`;

          const keyInput = document.createElement("input");
          keyInput.className = "ess-repl-pair-key";
          keyInput.type = "text";
          keyInput.placeholder = "key (use key only, without %)";
          keyInput.value = pairs[i].key || "";
          keyInput.classList.toggle("invalid", isInvalidKey(keyInput.value));
          trapEditorEvents(keyInput);
          keyInput.addEventListener("input", () => {
            pairs[i].key = keyInput.value;
            keyInput.classList.toggle("invalid", isInvalidKey(keyInput.value));
            stashLocal();
          });
          keyInput.addEventListener("blur", () => commit());

          const toggleButton = document.createElement("button");
          toggleButton.className = "ess-repl-pair-toggle";
          toggleButton.textContent = pairs[i].expanded ? "Collapse" : "Edit";
          trapEditorEvents(toggleButton);
          toggleButton.addEventListener("click", () => {
            const nextExpanded = !pairs[i].expanded;
            if (nextExpanded) {
              for (let j = 0; j < appliedCount; j += 1) {
                if (j === i || !pairs[j]?.expanded) continue;
                pairs[j].expanded = false;
                const other = pairRows[j];
                if (other) {
                  other.body.style.display = "none";
                  other.toggleButton.textContent = "Edit";
                  other.container.classList.remove("expanded");
                }
              }
            }
            pairs[i].expanded = nextExpanded;
            toggleButton.textContent = pairs[i].expanded ? "Collapse" : "Edit";
            body.style.display = pairs[i].expanded ? "" : "none";
            container.classList.toggle("expanded", pairs[i].expanded);
            commit();
            ensureNodeMinHeightKeepUserHeight(this);
          });

          header.appendChild(indexLabel);
          header.appendChild(keyInput);
          header.appendChild(toggleButton);
          pairCard.appendChild(header);

          const body = document.createElement("div");
          body.className = "ess-repl-pair-body";
          const editor = createTemplateEditor(
            pairs[i].value || "",
            (value) => {
              pairs[i].value = value;
              stashLocal();
            },
            (value) => {
              pairs[i].value = value;
              commit();
            },
          );
          body.appendChild(editor.container);
          body.style.display = pairs[i].expanded ? "" : "none";
          pairCard.appendChild(body);
          container.classList.toggle("expanded", pairs[i].expanded);
          // Capture phase ensures the pair menu opens even when nested inputs
          // stop propagation for their own editing behavior.
          pairCard.addEventListener("contextmenu", (event) => openPairMenu(event, i), true);

          const widget = this.addDOMWidget(`pair_${i + 1}`, "ess_replacements_pair", container, {
            getValue: () => "",
            setValue: () => {},
            getMinHeight: () => {
              const row = pairRows[i];
              const visible = row ? !!row.visible : (i < appliedCount);
              if (!visible) return 0;
              return pairs[i]?.expanded ? (VALUE_EDITOR_HEIGHT + 58) : 44;
            },
            getMaxHeight: () => {
              const row = pairRows[i];
              const visible = row ? !!row.visible : (i < appliedCount);
              if (!visible) return 0;
              return pairs[i]?.expanded ? 100000 : 44;
            },
            hideOnZoom: false,
            margin: 0,
          });

          if (!widget) continue;
          widget.serialize = false;
          const originalOnRemove = widget.onRemove?.bind(widget);
          widget.onRemove = () => {
            originalOnRemove?.();
            closePairMenu();
            container.__essReplDestroy?.();
          };
          pairRows.push({
            widget,
            container,
            indexLabel,
            keyInput,
            toggleButton,
            body,
            editor,
            visible: i < appliedCount,
          });
        }

        this.__essReplRefresh = () => {
          const parsed = loadPersistedPairs(this, storageWidget);
          if (parsed.length > 0 && parsePairs(storageWidget.value).length === 0) {
            // Promote recovered rows into canonical hidden widget for next saves.
            storageWidget.value = serializePairs(parsed);
          }
          let firstExpandedIndex = -1;
          for (let i = 0; i < pairRows.length; i += 1) {
            const src = parsed[i] || { key: "", value: "", expanded: false };
            let expanded = !!src.expanded;
            if (expanded) {
              if (firstExpandedIndex === -1) {
                firstExpandedIndex = i;
              } else {
                expanded = false;
              }
            }
            pairs[i] = {
              key: String(src.key || ""),
              value: String(src.value || ""),
              expanded,
            };
            const row = pairRows[i];
            row.keyInput.value = pairs[i].key;
            row.editor.input.value = pairs[i].value;
            row.editor.refresh?.();
          }
          const savedApplied = Number(
            this.properties?.[APPLIED_COUNT_PROP_KEY] ?? this.properties?.[APPLIED_COUNT_PROP_KEY_LEGACY],
          );
          const parsedLastUsed = detectLastUsedPairIndex(parsed);
          const parsedContentCount = clampCount(Math.max(1, parsedLastUsed + 1));
          const parsedLengthCount = clampCount(Math.max(1, parsed.length || 1));
          const serializedCount = getSerializedWidgetValue(this, countWidget);
          const widgetCount = clampCount(
            serializedCount != null ? serializedCount : countWidget.value,
          );
          const savedCount = Number.isFinite(savedApplied) ? clampCount(savedApplied) : 1;
          // Use the strongest signal to avoid collapsing to 1 when row data exists.
          appliedCount = clampCount(Math.max(parsedContentCount, parsedLengthCount, widgetCount, savedCount));
          this.__essReplAppliedCount = appliedCount;
          if (countWidget.value !== appliedCount) {
            countWidget.value = appliedCount;
          }
          applyCount(false);
        };

        setTimeout(() => {
          this.__essReplRefresh?.();
          resizeNodeKeepWidth(this);
        }, 0);
        // ComfyUI may assign widget values after configure hooks.
        // Run a few late refresh passes to pick real serialized state.
        let latePass = 0;
        const runLateRefresh = () => {
          this.__essReplRefresh?.();
          this.setDirtyCanvas?.(true, true);
          latePass += 1;
          if (latePass < 4) setTimeout(runLateRefresh, 40);
        };
        setTimeout(runLateRefresh, 40);
      } catch (error) {
        console.error("[ess_replacements_editor] onNodeCreated failed:", error);
      }

      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      this.__essReplRefresh?.();
      setTimeout(() => this.__essReplRefresh?.(), 30);
      setTimeout(() => this.__essReplRefresh?.(), 90);
      return result;
    };
  },
});
