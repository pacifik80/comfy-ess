import { app } from "../../scripts/app.js";

const STYLE_ID = "ess-template-editor-style";
const MIN_EDITOR_HEIGHT = 160;

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
.ess-template-editor .ess-template-highlight,
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
  white-space: pre-wrap;
  word-break: break-word;
  pointer-events: none;
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
.ess-template-editor .ess-tpl-symbol {
  color: #79c0ff;
  font-weight: 600;
}
.ess-template-editor .ess-tpl-scope {
  background: rgba(121, 192, 255, 0.18);
  border-radius: 3px;
}
.ess-template-editor .ess-tpl-scope.ess-tpl-symbol {
  color: #d2a8ff;
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

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function buildPairs(text) {
  const pairs = new Map();
  const reverse = new Map();
  const stack = [];
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "{") {
      stack.push(i);
    } else if (ch === "}") {
      if (stack.length > 0) {
        const open = stack.pop();
        pairs.set(open, i);
        reverse.set(i, open);
      }
    }
  }
  return { pairs, reverse };
}

function findActiveScope(text, caret, pairs, reverse) {
  if (caret > 0) {
    const prev = text[caret - 1];
    if (prev === "{" && pairs.has(caret - 1)) {
      return { open: caret - 1, close: pairs.get(caret - 1) };
    }
    if (prev === "}" && reverse.has(caret - 1)) {
      const open = reverse.get(caret - 1);
      return { open, close: caret - 1 };
    }
  }
  if (caret < text.length) {
    const curr = text[caret];
    if (curr === "{" && pairs.has(caret)) {
      return { open: caret, close: pairs.get(caret) };
    }
    if (curr === "}" && reverse.has(caret)) {
      const open = reverse.get(caret);
      return { open, close: caret };
    }
  }

  const stack = [];
  for (let i = 0; i < caret; i += 1) {
    const ch = text[i];
    if (ch === "{") {
      stack.push(i);
    } else if (ch === "}" && stack.length > 0) {
      stack.pop();
    }
  }
  if (stack.length === 0) {
    return null;
  }
  const open = stack[stack.length - 1];
  const close = pairs.get(open);
  if (close == null) {
    return null;
  }
  return { open, close };
}

function collectScopeTokens(text, open, close) {
  const active = new Set();
  active.add(open);
  active.add(close);
  let depth = 0;
  for (let i = open + 1; i < close; i += 1) {
    const ch = text[i];
    if (ch === "{") {
      depth += 1;
    } else if (ch === "}") {
      if (depth > 0) {
        depth -= 1;
      }
    } else if (ch === "|" && depth === 0) {
      active.add(i);
    }
  }
  return active;
}

function renderHighlight(text, caret) {
  const { pairs, reverse } = buildPairs(text);
  const activeScope = findActiveScope(text, caret, pairs, reverse);
  const activeSet = activeScope ? collectScopeTokens(text, activeScope.open, activeScope.close) : new Set();

  let html = "";
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const escaped = escapeHtml(ch);
    const isSymbol = ch === "{" || ch === "}" || ch === "|";
    const isActive = activeSet.has(i);
    if (!isSymbol && !isActive) {
      html += escaped;
      continue;
    }
    const classes = [];
    if (isSymbol) {
      classes.push("ess-tpl-symbol");
    }
    if (isActive) {
      classes.push("ess-tpl-scope");
    }
    html += `<span class="${classes.join(" ")}">${escaped}</span>`;
  }

  return html || "";
}

function applyContainerVars(container, minHeight, maxHeight) {
  const safeMin = Number.isFinite(minHeight) ? Math.max(minHeight, MIN_EDITOR_HEIGHT) : MIN_EDITOR_HEIGHT;
  const safeMax = Number.isFinite(maxHeight) ? Math.max(safeMin, maxHeight) : 100000;
  container.style.setProperty("--comfy-widget-min-height", `${safeMin}`);
  container.style.setProperty("--comfy-widget-max-height", `${safeMax}`);
}

function createEditorElements(config, inputData) {
  const container = document.createElement("div");
  container.className = "ess-template-editor";

  const highlight = document.createElement("div");
  highlight.className = "ess-template-highlight";

  const textarea = document.createElement("textarea");
  textarea.spellcheck = false;
  const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
    ? inputData.value
    : (config.value ?? config.default ?? "");
  textarea.value = initialValue;
  if (config.placeholder) {
    textarea.placeholder = config.placeholder;
  }

  const syncScroll = () => {
    highlight.scrollTop = textarea.scrollTop;
    highlight.scrollLeft = textarea.scrollLeft;
  };

  let raf = 0;
  const scheduleRender = () => {
    if (raf) {
      cancelAnimationFrame(raf);
    }
    raf = requestAnimationFrame(() => {
      raf = 0;
      const caret = textarea.selectionStart || 0;
      highlight.innerHTML = renderHighlight(textarea.value || "", caret);
      syncScroll();
    });
  };

  textarea.addEventListener("input", scheduleRender);
  textarea.addEventListener("click", scheduleRender);
  textarea.addEventListener("keyup", scheduleRender);
  textarea.addEventListener("select", scheduleRender);
  textarea.addEventListener("focus", scheduleRender);
  textarea.addEventListener("scroll", syncScroll);

  container.appendChild(highlight);
  container.appendChild(textarea);
  scheduleRender();

  return { container, textarea, scheduleRender };
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

        const { container, textarea, scheduleRender } = createEditorElements(config, inputData);
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
