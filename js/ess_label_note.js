import { app } from "../../scripts/app.js";

const FONT_SIZES = [8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 112, 128];
const MIN_FONT_SIZE = FONT_SIZES[0];
const MAX_FONT_SIZE = FONT_SIZES[FONT_SIZES.length - 1];
const PADDING = 10;
const LINE_HEIGHT = 1.2;
const NOTE_BG = "#fff5a6";
const NOTE_BORDER = "#e6cf6b";
const DEFAULT_SIZE = [240, 140];
const MIN_SIZE = [120, 60];
const NOTE_EDITOR_STYLE_ID = "ess-label-note-editor-style";
const NOTE_EDITOR_CLASS = "ess-label-note-editor";
const NOTE_HEADER_HEIGHT = 28;
const NOTE_EDITOR_MIN_HEIGHT = 32;
const RESIZE_MARGIN = 8;
const LIGHT_TEXT = "#f8f8f8";
const DARK_TEXT = "#222";

function getValidSize(node) {
  const width = Number(node?.size?.[0]);
  const height = Number(node?.size?.[1]);
  if (Number.isFinite(width) && Number.isFinite(height) && width > 1 && height > 1) {
    return [width, height];
  }
  return [DEFAULT_SIZE[0], DEFAULT_SIZE[1]];
}

function getMinSize(text) {
  const textValue = String(text ?? "").trim();
  const extraLines = textValue ? textValue.split("\n").length - 1 : 0;
  const minHeight = MIN_SIZE[1] + extraLines * 10;
  return [MIN_SIZE[0], minHeight];
}

function getEditorMinHeight(text) {
  const [, nodeMinHeight] = getMinSize(text);
  return Math.max(NOTE_EDITOR_MIN_HEIGHT, nodeMinHeight - NOTE_HEADER_HEIGHT);
}

function ensureNoteSize(node) {
  const [width, height] = getValidSize(node);
  if (!node) {
    return [width, height];
  }
  const [minWidth, minHeight] = getMinSize(getNoteText(node));
  const finalWidth = Math.max(width, minWidth);
  const finalHeight = Math.max(height, minHeight);
  if (!node.size || node.size[0] !== finalWidth || node.size[1] !== finalHeight) {
    if (node.setSize) {
      node.setSize([finalWidth, finalHeight]);
    } else {
      node.size = [finalWidth, finalHeight];
    }
  }
  return [finalWidth, finalHeight];
}

function ensureNoteEditorStyles() {
  const existing = document.getElementById(NOTE_EDITOR_STYLE_ID);
  const css = `
.${NOTE_EDITOR_CLASS} {
  position: relative;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
}
.${NOTE_EDITOR_CLASS} textarea {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  background: transparent;
  color: var(--ess-note-text, ${DARK_TEXT});
  border: none;
  outline: none;
  resize: none;
  padding: calc(${PADDING}px - 4px) ${PADDING}px ${PADDING}px ${PADDING}px;
  font-family: sans-serif;
  overflow: hidden;
  scrollbar-width: none;
}
.${NOTE_EDITOR_CLASS} textarea::placeholder {
  color: var(--ess-note-placeholder, rgba(34, 34, 34, 0.45));
}
.${NOTE_EDITOR_CLASS} textarea::-webkit-scrollbar {
  width: 0px;
  height: 0px;
}
`;
  if (existing) {
    existing.textContent = css;
    return;
  }
  const style = document.createElement("style");
  style.id = NOTE_EDITOR_STYLE_ID;
  style.textContent = css;
  document.head.appendChild(style);
}

function syncNoteEditor(node) {
  const editor = node?.__essNoteEditor;
  if (!editor?.textarea) {
    return;
  }
  if (document.activeElement === editor.textarea) {
    return;
  }
  const text = getNoteText(node);
  if (editor.textarea.value !== text) {
    editor.textarea.value = text ?? "";
  }
}

function ensureNoteEditor(node) {
  if (!node || node.__essNoteEditor || typeof node.addDOMWidget !== "function") {
    return;
  }
  if (typeof document === "undefined") {
    return;
  }

  ensureNoteEditorStyles();

  const container = document.createElement("div");
  container.className = NOTE_EDITOR_CLASS;
  const textarea = document.createElement("textarea");
  textarea.value = getNoteText(node);
  textarea.placeholder = "Type note...";
  textarea.spellcheck = false;
  textarea.addEventListener("input", () => {
    setNoteText(node, textarea.value);
    node.setDirtyCanvas?.(true, true);
  });
  textarea.addEventListener("pointerdown", (event) => event.stopPropagation());
  textarea.addEventListener("pointerup", (event) => event.stopPropagation());
  textarea.addEventListener("contextmenu", (event) => event.stopPropagation());

  container.appendChild(textarea);

  const widget = node.addDOMWidget("note", "ess_label_note", container, {
    getValue: () => getNoteText(node),
    setValue: (value) => {
      textarea.value = value ?? "";
      setNoteText(node, textarea.value);
    },
    getMinHeight: () => getEditorMinHeight(getNoteText(node)),
    getHeight: () => {
      const minHeight = getEditorMinHeight(getNoteText(node));
      const target = Math.max(
        minHeight,
        (node.size?.[1] || DEFAULT_SIZE[1]) - NOTE_HEADER_HEIGHT
      );
      return target;
    },
    hideOnZoom: false,
    margin: RESIZE_MARGIN,
  });

  const originalRemove = widget.onRemove?.bind(widget);
  widget.onRemove = function () {
    originalRemove?.();
    if (container.isConnected) {
      container.remove();
    }
  };

  node.__essNoteEditor = { widget, container, textarea };
}

function wrapLine(ctx, text, maxWidth) {
  if (!text) {
    return [""];
  }
  const words = text.split(/\s+/);
  const lines = [];
  let current = "";

  for (const word of words) {
    const test = current ? `${current} ${word}` : word;
    if (ctx.measureText(test).width <= maxWidth) {
      current = test;
      continue;
    }

    if (current) {
      lines.push(current);
      current = "";
    }

    if (ctx.measureText(word).width <= maxWidth) {
      current = word;
      continue;
    }

    let chunk = "";
    for (const ch of word) {
      const next = chunk + ch;
      if (ctx.measureText(next).width > maxWidth && chunk) {
        lines.push(chunk);
        chunk = ch;
      } else {
        chunk = next;
      }
    }
    if (chunk) {
      current = chunk;
    }
  }

  if (current) {
    lines.push(current);
  }

  return lines.length ? lines : [""];
}

function wrapText(ctx, text, maxWidth) {
  const paragraphs = String(text ?? "").split("\n");
  const lines = [];

  for (const paragraph of paragraphs) {
    const wrapped = wrapLine(ctx, paragraph, maxWidth);
    for (const line of wrapped) {
      lines.push(line);
    }
  }

  return lines;
}

function measureBlock(ctx, text, maxWidth, fontSize) {
  ctx.font = `${fontSize}px sans-serif`;
  const lines = wrapText(ctx, text, maxWidth);
  const lineHeight = fontSize * LINE_HEIGHT;
  const height = lines.length * lineHeight;
  let width = 0;
  for (const line of lines) {
    width = Math.max(width, ctx.measureText(line).width);
  }
  return { lines, width, height, lineHeight, fontSize };
}

function pickFontSize(ctx, text, maxWidth, maxHeight) {
  const safeWidth = Math.max(10, maxWidth);
  const safeHeight = Math.max(10, maxHeight);
  let low = MIN_FONT_SIZE;
  let high = MAX_FONT_SIZE;
  let best = measureBlock(ctx, text, safeWidth, low);

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const metrics = measureBlock(ctx, text, safeWidth, mid);
    if (metrics.width <= safeWidth && metrics.height <= safeHeight) {
      best = metrics;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  return best;
}

function getNoteText(node) {
  if (!node.properties) {
    node.properties = {};
  }
  return node.properties.text ?? "";
}

function setNoteText(node, value) {
  if (!node.properties) {
    node.properties = {};
  }
  node.properties.text = value ?? "";
  if (node) {
    const textValue = String(node.properties.text ?? "").trim();
    node.title = textValue ? textValue.split("\n")[0] : "Label Note";
  }
}

function clearNoteOutputs(node) {
  if (!node || !Array.isArray(node.outputs) || node.outputs.length === 0) {
    return;
  }
  for (let i = node.outputs.length - 1; i >= 0; i -= 1) {
    node.removeOutput?.(i);
  }
  node.outputs = [];
}

function getContentBounds(node) {
  const editor = node?.__essNoteEditor?.textarea;
  if (editor && editor.isConnected) {
    const style = window.getComputedStyle(editor);
    const padX = (parseFloat(style.paddingLeft) || 0) + (parseFloat(style.paddingRight) || 0);
    const padY = (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
    const width = Math.max(20, editor.clientWidth - padX);
    const height = Math.max(20, editor.clientHeight - padY);
    return { width, height };
  }

  const [nodeWidth, nodeHeight] = ensureNoteSize(node);
  const width = Math.max(20, nodeWidth - PADDING * 2 - RESIZE_MARGIN * 2);
  const height = Math.max(
    20,
    nodeHeight - NOTE_HEADER_HEIGHT - PADDING * 2 - RESIZE_MARGIN * 2
  );
  return { width, height };
}

function getNoteColors(node) {
  const bg = node?.bgcolor || NOTE_BG;
  const border = node?.color || NOTE_BORDER;
  return { bg, border };
}

function parseColor(color) {
  if (!color || typeof color !== "string") {
    return null;
  }
  const trimmed = color.trim();
  if (trimmed.startsWith("#")) {
    const hex = trimmed.slice(1);
    if (hex.length === 3) {
      const r = Number.parseInt(hex[0] + hex[0], 16);
      const g = Number.parseInt(hex[1] + hex[1], 16);
      const b = Number.parseInt(hex[2] + hex[2], 16);
      return { r, g, b };
    }
    if (hex.length === 6 || hex.length === 8) {
      const r = Number.parseInt(hex.slice(0, 2), 16);
      const g = Number.parseInt(hex.slice(2, 4), 16);
      const b = Number.parseInt(hex.slice(4, 6), 16);
      return { r, g, b };
    }
    return null;
  }
  const rgbMatch = trimmed.match(/^rgba?\(([^)]+)\)$/i);
  if (rgbMatch) {
    const parts = rgbMatch[1].split(",").map((value) => Number.parseFloat(value.trim()));
    if (parts.length >= 3) {
      return {
        r: Math.max(0, Math.min(255, parts[0])),
        g: Math.max(0, Math.min(255, parts[1])),
        b: Math.max(0, Math.min(255, parts[2])),
      };
    }
  }
  return null;
}

function getNoteTextColor(node) {
  const { bg } = getNoteColors(node);
  const parsed = parseColor(bg);
  if (!parsed) {
    return DARK_TEXT;
  }
  const luminance = (0.299 * parsed.r + 0.587 * parsed.g + 0.114 * parsed.b) / 255;
  return luminance > 0.6 ? DARK_TEXT : LIGHT_TEXT;
}

function promptForText(node) {
  const current = getNoteText(node);
  const canvasPrompt = app?.canvas?.prompt
    || (typeof LGraphCanvas !== "undefined" ? LGraphCanvas.active_canvas?.prompt : null);
  const promptFn = canvasPrompt
    || (typeof LiteGraph !== "undefined" && LiteGraph.prompt)
    || ((label, value, callback) => {
      const result = window.prompt(label, value ?? "");
      callback(result);
    });

  promptFn("Edit note", current, (value) => {
    if (value == null) {
      return;
    }
    setNoteText(node, value);
    syncNoteEditor(node);
    node.setDirtyCanvas?.(true, true);
  });
}

app.registerExtension({
  name: "ess_label_note",
  nodeCreated(node) {
    if (!node || (node.comfyClass !== "LabelNote" && node.type !== "ESS/LabelNote")) {
      return;
    }
    const applyDefaults = () => {
      ensureNoteSize(node);
      if (!node.__essNoteComputeSize) {
        node.computeSize = function () {
          const [minWidth, minHeight] = getMinSize(getNoteText(this));
          return [minWidth, minHeight];
        };
        node.__essNoteComputeSize = true;
      }
      node.resizable = true;
      node.collapsable = false;
      if (!node.bgcolor) {
        node.bgcolor = NOTE_BG;
      }
      if (!node.color) {
        node.color = NOTE_BORDER;
      }
      setNoteText(node, getNoteText(node));
      ensureNoteEditor(node);
      syncNoteEditor(node);
      clearNoteOutputs(node);
      node.flags = node.flags || {};
      node.flags.collapsed = false;
    };

    applyDefaults();

    const originalConfigure = node.onConfigure;
    node.onConfigure = function () {
      const result = originalConfigure ? originalConfigure.apply(this, arguments) : undefined;
      applyDefaults();
      return result;
    };

    const originalAdded = node.onAdded;
    node.onAdded = function () {
      const result = originalAdded ? originalAdded.apply(this, arguments) : undefined;
      applyDefaults();
      this.setDirtyCanvas?.(true, true);
      return result;
    };

    const originalResize = node.onResize;
    node.onResize = function () {
      const result = originalResize ? originalResize.apply(this, arguments) : undefined;
      this.setDirtyCanvas?.(true, true);
      return result;
    };

    const originalDblClick = node.onDblClick;
    node.onDblClick = function () {
      if (originalDblClick) {
        originalDblClick.apply(this, arguments);
      }
      promptForText(this);
      return true;
    };

    const originalGetMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
      const result = originalGetMenu ? originalGetMenu.call(this, canvas, options) : options;
      const items = result || options || [];
      items.push({
        content: "Edit note...",
        callback: () => promptForText(this),
      });
      return items;
    };

    const originalDrawBg = node.onDrawBackground;
    node.onDrawBackground = function (ctx) {
      if (originalDrawBg) {
        originalDrawBg.call(this, ctx);
      }

      const { bg, border } = getNoteColors(this);
      ctx.save();
      ctx.fillStyle = bg;
      ctx.strokeStyle = border;
      ctx.lineWidth = 2;

      const x = 0;
      const y = 0;
      const [w, h] = ensureNoteSize(this);
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 6);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
      }
      ctx.restore();
    };

    const originalDrawFg = node.onDrawForeground;
    node.onDrawForeground = function (ctx) {
      if (originalDrawFg) {
        originalDrawFg.call(this, ctx);
      }

      syncNoteEditor(this);
      const text = getNoteText(this);
      const [nodeWidth, nodeHeight] = ensureNoteSize(this);
      const { width, height } = getContentBounds(this);

      ctx.save();
      const textColor = getNoteTextColor(this);
      ctx.fillStyle = textColor;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";

      const metrics = pickFontSize(ctx, text, width, height);
      ctx.font = `${metrics.fontSize}px sans-serif`;
      if (this.__essNoteEditor?.textarea) {
        this.__essNoteEditor.textarea.style.fontSize = `${metrics.fontSize}px`;
        this.__essNoteEditor.textarea.style.lineHeight = `${metrics.lineHeight}px`;
        this.__essNoteEditor.textarea.style.setProperty("--ess-note-text", textColor);
        const placeholderAlpha = textColor === DARK_TEXT ? "rgba(34, 34, 34, 0.45)" : "rgba(248, 248, 248, 0.6)";
        this.__essNoteEditor.textarea.style.setProperty("--ess-note-placeholder", placeholderAlpha);
      }
      if (this.__essNoteEditor?.textarea) {
        ctx.restore();
        return;
      }

      let y = PADDING;
      const startX = PADDING;
      for (const line of metrics.lines) {
        ctx.fillText(line, startX, y);
        y += metrics.lineHeight;
        if (y > nodeHeight - NOTE_HEADER_HEIGHT - PADDING) {
          break;
        }
      }

      ctx.restore();
    };
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/LabelNote") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      ensureNoteSize(this);
      this.resizable = true;
      this.collapsable = false;
      if (!this.bgcolor) {
        this.bgcolor = NOTE_BG;
      }
      if (!this.color) {
        this.color = NOTE_BORDER;
      }
      setNoteText(this, getNoteText(this));
      ensureNoteEditor(this);
      syncNoteEditor(this);
      clearNoteOutputs(this);
      this.flags = this.flags || {};
      this.flags.collapsed = false;
      return result;
    };

    nodeType.prototype.computeSize = function () {
      const [minWidth, minHeight] = getMinSize(getNoteText(this));
      return [minWidth, minHeight];
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      if (onAdded) {
        onAdded.apply(this, arguments);
      }
      ensureNoteSize(this);
      this.setDirtyCanvas?.(true, true);
    };

    const onSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (data) {
      if (onSerialize) {
        onSerialize.call(this, data);
      }
      data.properties = data.properties || {};
      data.properties.text = getNoteText(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (data) {
      if (onConfigure) {
        onConfigure.call(this, data);
      }
      if (data?.properties?.text != null) {
        setNoteText(this, data.properties.text);
      }
      ensureNoteEditor(this);
      syncNoteEditor(this);
      ensureNoteSize(this);
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      if (onResize) {
        onResize.apply(this, arguments);
      }
      this.setDirtyCanvas?.(true, true);
    };

    nodeType.prototype.onDblClick = function () {
      promptForText(this);
      return true;
    };

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      const result = getExtraMenuOptions ? getExtraMenuOptions.call(this, canvas, options) : options;
      const items = result || options || [];
      items.push({
        content: "Edit note...",
        callback: () => promptForText(this),
      });
      return items;
    };

    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      if (onDrawBackground) {
        onDrawBackground.call(this, ctx);
      }

      const { bg, border } = getNoteColors(this);
      ctx.save();
      ctx.fillStyle = bg;
      ctx.strokeStyle = border;
      ctx.lineWidth = 2;

      const x = 0;
      const y = 0;
      const [w, h] = ensureNoteSize(this);
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, w, h, 6);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(x, y, w, h);
        ctx.strokeRect(x, y, w, h);
      }
      ctx.restore();
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) {
        onDrawForeground.call(this, ctx);
      }

      syncNoteEditor(this);
      const text = getNoteText(this);
      const [nodeWidth, nodeHeight] = ensureNoteSize(this);
      const { width, height } = getContentBounds(this);

      ctx.save();
      const textColor = getNoteTextColor(this);
      ctx.fillStyle = textColor;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";

      const metrics = pickFontSize(ctx, text, width, height);
      ctx.font = `${metrics.fontSize}px sans-serif`;
      if (this.__essNoteEditor?.textarea) {
        this.__essNoteEditor.textarea.style.fontSize = `${metrics.fontSize}px`;
        this.__essNoteEditor.textarea.style.lineHeight = `${metrics.lineHeight}px`;
        this.__essNoteEditor.textarea.style.setProperty("--ess-note-text", textColor);
        const placeholderAlpha = textColor === DARK_TEXT ? "rgba(34, 34, 34, 0.45)" : "rgba(248, 248, 248, 0.6)";
        this.__essNoteEditor.textarea.style.setProperty("--ess-note-placeholder", placeholderAlpha);
      }
      if (this.__essNoteEditor?.textarea) {
        ctx.restore();
        return;
      }

      let y = PADDING;
      const startX = PADDING;
      for (const line of metrics.lines) {
        ctx.fillText(line, startX, y);
        y += metrics.lineHeight;
        if (y > nodeHeight - NOTE_HEADER_HEIGHT - PADDING) {
          break;
        }
      }

      ctx.restore();
    };
  },
});
