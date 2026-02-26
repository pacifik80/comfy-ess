import { app } from "../../scripts/app.js";

const STYLE_ID = "ess-replacements-editor-style";
const MAX_PAIRS = 24;
const MIN_HEIGHT = 180;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-repl-widget{display:grid;gap:8px;border:1px solid #334155;border-radius:8px;background:#0b1222;padding:8px}
.ess-repl-top{display:flex;align-items:center;gap:8px}
.ess-repl-top .summary{font-size:11px;color:#bfdbfe;flex:1}
.ess-repl-top button{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:6px;padding:4px 8px;font-size:11px;cursor:pointer}
.ess-repl-rows{display:grid;gap:8px;max-height:420px;overflow:auto;padding-right:2px}
.ess-repl-row{border:1px solid #334155;border-radius:8px;background:#070f20}
.ess-repl-row-header{display:grid;grid-template-columns:auto 1fr auto;gap:6px;align-items:center;padding:6px 7px;border-bottom:1px solid #1f2937}
.ess-repl-row-index{font-size:11px;color:#93c5fd}
.ess-repl-row-header input{background:#020617;color:#e5e7eb;border:1px solid #475569;border-radius:5px;padding:4px 6px;font-size:12px}
.ess-repl-row-header input.invalid{background:#1b1010;color:#fecaca;border-color:#b91c1c;box-shadow:0 0 0 1px rgba(185,28,28,.2) inset}
.ess-repl-row-header button{background:#0b1222;color:#e5e7eb;border:1px solid #475569;border-radius:5px;padding:3px 7px;font-size:11px;cursor:pointer}
.ess-repl-row-body{padding:7px}
.ess-repl-editor{position:relative;width:100%;height:122px;border:1px solid #374151;background:#0b1222;border-radius:7px;overflow:hidden}
.ess-repl-editor .hl,.ess-repl-editor textarea{font-family:"JetBrains Mono","Fira Code","Consolas",monospace;font-size:12px;line-height:1.45;padding:8px 10px;box-sizing:border-box;width:100%;height:100%}
.ess-repl-editor .hl{position:absolute;inset:0;color:#c9d1d9;white-space:pre-wrap;overflow:hidden;pointer-events:none}
.ess-repl-editor textarea{position:relative;background:transparent;color:transparent;caret-color:#e6edf3;border:none;outline:none;resize:none;overflow:auto}
.ess-repl-editor .c{color:#2f6f3e}
.ess-repl-editor .h{color:#b48800}
.ess-repl-editor .n{color:#a43f3f}
.ess-repl-editor .v{color:#5b2c83}
.ess-repl-editor .m{color:#6e7681}
.ess-repl-editor .p{color:#38bdf8}
`;
  document.head.appendChild(style);
}

function escapeHtml(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function simpleHighlight(text) {
  const out = [];
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const pair = text.slice(i, i + 2);
    if (ch === "%") {
      let j = i + 1;
      while (j < text.length && text[j] !== "%" && !/\s/.test(text[j])) j += 1;
      if (j < text.length && text[j] === "%" && j > i + 1) {
        out.push(`<span class="p">${escapeHtml(text.slice(i, j + 1))}</span>`);
        i = j;
        continue;
      }
    }
    if (pair === "!>") { out.push('<span class="n">!&gt;</span>'); i += 1; continue; }
    if (pair === "<<" || pair === ">>") { out.push(`<span class="v">${escapeHtml(pair)}</span>`); i += 1; continue; }
    if (ch === "{" || ch === "}" || ch === "|") { out.push(`<span class="c">${escapeHtml(ch)}</span>`); continue; }
    if (ch === "[" || ch === "]" || ch === ":") { out.push(`<span class="h">${escapeHtml(ch)}</span>`); continue; }
    if (ch === "#") {
      let j = i;
      while (j < text.length && text[j] !== "\n") j += 1;
      out.push(`<span class="m">${escapeHtml(text.slice(i, j))}</span>`);
      i = j - 1;
      continue;
    }
    out.push(escapeHtml(ch));
  }
  return out.join("");
}

function trapEditorEvents(element) {
  if (!element) return;
  const stop = (event) => event.stopPropagation();
  ["keydown", "keyup", "keypress", "pointerdown", "pointermove", "pointerup", "mousedown", "mouseup", "click", "dblclick", "contextmenu"].forEach((name) => {
    element.addEventListener(name, stop);
  });
}

function createTemplateEditor(initialValue, onChange, onCommit) {
  const wrap = document.createElement("div");
  wrap.className = "ess-repl-editor";
  const highlight = document.createElement("div");
  highlight.className = "hl";
  const input = document.createElement("textarea");
  input.value = initialValue || "";

  const sync = () => {
    highlight.innerHTML = simpleHighlight(input.value || "") || "&nbsp;";
    highlight.scrollTop = input.scrollTop;
    highlight.scrollLeft = input.scrollLeft;
  };
  input.addEventListener("input", () => {
    sync();
    onChange(input.value);
  });
  input.addEventListener("blur", () => {
    onCommit?.(input.value);
  });
  input.addEventListener("scroll", () => {
    highlight.scrollTop = input.scrollTop;
    highlight.scrollLeft = input.scrollLeft;
  });
  sync();
  trapEditorEvents(input);

  wrap.appendChild(highlight);
  wrap.appendChild(input);
  return {
    container: wrap,
    destroy: () => {},
  };
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

function serializePairs(pairs) {
  const normalized = (pairs || []).map((it) => ({
    key: String(it?.key || ""),
    value: String(it?.value || ""),
    expanded: !!it?.expanded,
  }));
  return JSON.stringify(normalized, null, 2);
}

function getCountWidget(node) {
  if (!Array.isArray(node?.widgets)) return null;
  return node.widgets.find((w) => String(w?.name || "").trim() === "count") || null;
}

function clampCount(value) {
  const n = Math.floor(Number(value) || 1);
  return Math.max(1, Math.min(n, MAX_PAIRS));
}

function isInvalidKey(raw) {
  const text = String(raw || "");
  const key = text.trim();
  if (!key) return false;
  return key.includes("%") || /\s/.test(key);
}

app.registerExtension({
  name: "ess_replacements_editor",
  async getCustomWidgets() {
    ensureStyles();
    return {
      ESS_REPLACEMENTS_EDITOR(node, inputName, inputData) {
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
          ? String(inputData.value || "")
          : String(config.value ?? config.default ?? "[]");
        const stateRef = { value: initialValue };
        const container = document.createElement("div");
        container.className = "ess-repl-widget";
        const top = document.createElement("div");
        top.className = "ess-repl-top";
        const summary = document.createElement("div");
        summary.className = "summary";
        const applyBtn = document.createElement("button");
        applyBtn.textContent = "Apply count";
        top.appendChild(summary);
        top.appendChild(applyBtn);
        const rows = document.createElement("div");
        rows.className = "ess-repl-rows";
        container.appendChild(top);
        container.appendChild(rows);

        let pairs = parsePairs(stateRef.value);

        const widget = node.addDOMWidget(inputName, "ess_replacements_editor", container, {
          getValue: () => stateRef.value,
          setValue: (value) => {
            stateRef.value = String(value || "[]");
            pairs = parsePairs(stateRef.value);
            render();
          },
          getMinHeight: () => Number(config.height || MIN_HEIGHT),
          hideOnZoom: false,
          margin: 8,
        });

        const stashLocal = () => {
          stateRef.value = serializePairs(pairs);
        };

        const stashPublished = () => {
          stateRef.value = serializePairs(pairs);
          widget.value = stateRef.value;
          if (Array.isArray(node.widgets_values)) {
            const idx = node.widgets?.indexOf(widget);
            if (idx != null && idx >= 0) node.widgets_values[idx] = stateRef.value;
          }
        };

        const commit = (notify = true) => {
          stashPublished();
          if (notify && typeof widget.callback === "function") widget.callback(stateRef.value);
          if (notify) node.setDirtyCanvas?.(true, true);
        };

        const syncToCount = (shouldCommit = true) => {
          const countWidget = getCountWidget(node);
          const desired = clampCount(countWidget?.value ?? 1);
          while (pairs.length < desired) pairs.push({ key: "", value: "", expanded: false });
          if (pairs.length > desired) pairs = pairs.slice(0, desired);
          if (shouldCommit) commit();
          render();
        };

        const render = () => {
          const countWidget = getCountWidget(node);
          const desired = clampCount(countWidget?.value ?? 1);
          summary.textContent = `${pairs.length}/${desired} pairs`;
          rows.innerHTML = "";
          pairs.forEach((pair, index) => {
            const row = document.createElement("div");
            row.className = "ess-repl-row";
            const header = document.createElement("div");
            header.className = "ess-repl-row-header";
            const idx = document.createElement("div");
            idx.className = "ess-repl-row-index";
            idx.textContent = `#${index + 1}`;
            const keyInput = document.createElement("input");
            keyInput.type = "text";
            keyInput.placeholder = "key (for %key%)";
            keyInput.value = pair.key || "";
            keyInput.classList.toggle("invalid", isInvalidKey(keyInput.value));
            trapEditorEvents(keyInput);
            keyInput.oninput = () => {
              pair.key = keyInput.value;
              keyInput.classList.toggle("invalid", isInvalidKey(keyInput.value));
              stashLocal();
            };
            keyInput.onblur = () => commit(true);
            const toggle = document.createElement("button");
            toggle.textContent = pair.expanded ? "Collapse" : "Edit";
            toggle.onclick = () => {
              pair.expanded = !pair.expanded;
              commit();
              render();
            };
            header.appendChild(idx);
            header.appendChild(keyInput);
            header.appendChild(toggle);
            row.appendChild(header);

            if (pair.expanded) {
              const body = document.createElement("div");
              body.className = "ess-repl-row-body";
              const editor = createTemplateEditor(
                pair.value || "",
                (value) => {
                  pair.value = value;
                  stashLocal();
                },
                (value) => {
                  pair.value = value;
                  commit(true);
                },
              );
              body.appendChild(editor.container);
              row.appendChild(body);
            }
            rows.appendChild(row);
          });
        };

        const hookCountWidget = () => {
          const countWidget = getCountWidget(node);
          if (!countWidget || countWidget.__essReplWrapped) return;
          const original = countWidget.callback;
          countWidget.callback = (value, canvas, innerNode, pos, event) => {
            if (original) original(value, canvas, innerNode, pos, event);
            syncToCount(true);
          };
          countWidget.__essReplWrapped = true;
        };

        applyBtn.onclick = () => syncToCount(true);

        setTimeout(() => {
          hookCountWidget();
          syncToCount(false);
          render();
        }, 0);

        const originalRemove = widget.onRemove?.bind(widget);
        widget.onRemove = function () {
          originalRemove?.();
          if (container.isConnected) container.remove();
        };

        return { widget, minHeight: Number(config.height || MIN_HEIGHT) };
      },
    };
  },
});
