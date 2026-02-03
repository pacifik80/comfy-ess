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
.ess-template-editor .ess-tpl-comment {
  color: #6e7681;
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
  const activeRange = activeScope ? { start: activeScope.open, end: activeScope.close } : null;

  const length = text.length;
  const classSets = Array.from({ length }, () => new Set());
  const isComment = new Array(length).fill(false);
  const isSpecial = new Array(length).fill(false);
  const headerTypes = new Array(length).fill(null);

  const markClass = (index, className) => {
    if (index < 0 || index >= length) {
      return;
    }
    classSets[index].add(className);
    isSpecial[index] = true;
  };

  // Mark comment ranges: start with #, terminate on newline or special symbols or another #
  let inComment = false;
  for (let i = 0; i < length; i += 1) {
    const ch = text[i];
    if (inComment) {
      if (ch === "#") {
        isComment[i] = true;
        inComment = false;
        continue;
      }
      if (ch === "\n" || ch === "{" || ch === "}" || ch === "|" || ch === "[" || ch === "]" || ch === ":") {
        inComment = false;
        continue;
      }
      isComment[i] = true;
      continue;
    }
    if (ch === "#") {
      isComment[i] = true;
      inComment = true;
    }
  }

  // Identify header name/number spans for coloring
  for (let i = 0; i < length; i += 1) {
    const ch = text[i];
    if (ch !== "[") {
      continue;
    }
    const close = text.indexOf("]", i + 1);
    if (close === -1) {
      break;
    }
    const headerContent = text.slice(i + 1, close);
    const colonIndex = headerContent.indexOf(":");
    if (colonIndex >= 0) {
      const nameStart = i + 1;
      const nameEnd = i + 1 + colonIndex;
      const numStart = nameEnd + 1;
      const numEnd = close;
      for (let j = nameStart; j < nameEnd; j += 1) {
        headerTypes[j] = "name";
      }
      for (let j = numStart; j < numEnd; j += 1) {
        headerTypes[j] = "number";
      }
    } else {
      const isNumber = /^\s*\d+(?:\.\d+)?\s*$/.test(headerContent);
      const type = isNumber ? "number" : "name";
      for (let j = i + 1; j < close; j += 1) {
        headerTypes[j] = type;
      }
    }
    i = close;
  }

  const stack = [];
  let negativeDepthCount = 0;
  let bracketDepth = 0;

  for (let i = 0; i < length; i += 1) {
    if (isComment[i]) {
      continue;
    }

    const ch = text[i];
    const next = text[i + 1];
    const inActiveRange = activeRange ? i >= activeRange.start && i <= activeRange.end : false;

    if (ch === "!" && next === ">") {
      const markerClass = inActiveRange ? "ess-tpl-negative-marker-active" : "ess-tpl-negative-marker";
      markClass(i, markerClass);
      markClass(i + 1, markerClass);
      if (stack.length > 0) {
        const frame = stack[stack.length - 1];
        if (!frame.negative) {
          frame.negative = true;
          negativeDepthCount += 1;
        }
      }
      i += 1;
      continue;
    }

    if (ch === "{") {
      const choiceClass = activeSet.has(i) ? "ess-tpl-choice-active" : "ess-tpl-choice";
      markClass(i, choiceClass);
      stack.push({ negative: false });
      continue;
    }
    if (ch === "}") {
      const choiceClass = activeSet.has(i) ? "ess-tpl-choice-active" : "ess-tpl-choice";
      markClass(i, choiceClass);
      const frame = stack.pop();
      if (frame?.negative) {
        negativeDepthCount -= 1;
      }
      continue;
    }
    if (ch === "|") {
      const choiceClass = activeSet.has(i) ? "ess-tpl-choice-active" : "ess-tpl-choice";
      markClass(i, choiceClass);
      if (stack.length > 0) {
        const frame = stack[stack.length - 1];
        if (frame.negative) {
          frame.negative = false;
          negativeDepthCount -= 1;
        }
      }
      continue;
    }
    if (ch === "[") {
      const headerClass = inActiveRange ? "ess-tpl-header-active" : "ess-tpl-header";
      markClass(i, headerClass);
      bracketDepth += 1;
      continue;
    }
    if (ch === "]") {
      const headerClass = inActiveRange ? "ess-tpl-header-active" : "ess-tpl-header";
      markClass(i, headerClass);
      if (bracketDepth > 0) {
        bracketDepth -= 1;
      }
      continue;
    }
    if (ch === ":" && bracketDepth > 0) {
      const headerClass = inActiveRange ? "ess-tpl-header-active" : "ess-tpl-header";
      markClass(i, headerClass);
      continue;
    }

    const headerType = headerTypes[i];
    if (headerType) {
      const headerClass = inActiveRange
        ? (headerType === "number" ? "ess-tpl-header-number-active" : "ess-tpl-header-name-active")
        : (headerType === "number" ? "ess-tpl-header-number" : "ess-tpl-header-name");
      classSets[i].add(headerClass);
      isSpecial[i] = true;
    }

    if (negativeDepthCount > 0 && !isSpecial[i]) {
      classSets[i].add("ess-tpl-negative-text");
    }
  }

  let html = "";
  for (let i = 0; i < length; i += 1) {
    const ch = text[i];
    const escaped = escapeHtml(ch);
    let classes;
    if (isComment[i]) {
      classes = ["ess-tpl-comment"];
    } else {
      classes = Array.from(classSets[i]);
    }
    if (!classes.length) {
      html += escaped;
      continue;
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
  const scheduleRender = () => {
    if (raf) {
      cancelAnimationFrame(raf);
    }
    raf = requestAnimationFrame(() => {
      raf = 0;
      syncSize();
      const caret = textarea.selectionStart || 0;
      highlightContent.innerHTML = renderHighlight(textarea.value || "", caret);
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

  const cleanup = () => {
    resizeObserver.disconnect();
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

        const { container, textarea, scheduleRender, onWheelCapture, cleanup } = createEditorElements(config, inputData);
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
