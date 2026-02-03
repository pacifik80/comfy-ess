import { app } from "../../scripts/app.js";

const FONT_SIZES = [8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64];
const PADDING = 10;
const LINE_HEIGHT = 1.2;
const NOTE_BG = "#fff5a6";
const NOTE_BORDER = "#e6cf6b";

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
  for (let i = FONT_SIZES.length - 1; i >= 0; i -= 1) {
    const size = FONT_SIZES[i];
    const metrics = measureBlock(ctx, text, maxWidth, size);
    if (metrics.width <= maxWidth && metrics.height <= maxHeight) {
      return metrics;
    }
  }

  const size = FONT_SIZES[0];
  return measureBlock(ctx, text, maxWidth, size);
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
}

function promptForText(node) {
  const current = getNoteText(node);
  const promptFn = (typeof LiteGraph !== "undefined" && LiteGraph.prompt)
    ? LiteGraph.prompt
    : (label, value, callback) => {
      const result = window.prompt(label, value ?? "");
      callback(result);
    };

  promptFn("Edit note", current, (value) => {
    if (value == null) {
      return;
    }
    setNoteText(node, value);
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
      const initialSize = node.size && node.size[0] > 0 && node.size[1] > 0
        ? node.size
        : [240, 140];
      if (node.setSize) {
        node.setSize(initialSize);
      } else {
        node.size = initialSize;
      }
      node.resizable = true;
      node.collapsable = false;
      if (typeof LiteGraph !== "undefined" && LiteGraph.NO_TITLE != null) {
        node.title = "";
        node.title_mode = LiteGraph.NO_TITLE;
      }
      if (Array.isArray(node.outputs) && node.outputs[0]) {
        node.outputs[0].name = "";
        node.outputs[0].hidden = true;
        node.outputs[0].type = "*";
      }
      node.flags = node.flags || {};
      node.flags.collapsed = false;
      setNoteText(node, getNoteText(node));
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

    const originalDrawBg = node.onDrawBackground;
    node.onDrawBackground = function (ctx) {
      if (originalDrawBg) {
        originalDrawBg.call(this, ctx);
      }

      ctx.save();
      ctx.fillStyle = NOTE_BG;
      ctx.strokeStyle = NOTE_BORDER;
      ctx.lineWidth = 2;

      const x = 0;
      const y = 0;
      const w = this.size[0];
      const h = this.size[1];
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

      const text = getNoteText(this);
      const width = Math.max(20, this.size[0] - PADDING * 2);
      const height = Math.max(20, this.size[1] - PADDING * 2);

      ctx.save();
      ctx.fillStyle = "#222";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";

      const metrics = pickFontSize(ctx, text, width, height);
      ctx.font = `${metrics.fontSize}px sans-serif`;

      let y = PADDING;
      const startX = PADDING;
      for (const line of metrics.lines) {
        ctx.fillText(line, startX, y);
        y += metrics.lineHeight;
        if (y > this.size[1] - PADDING) {
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
      const initialSize = this.size && this.size[0] > 0 && this.size[1] > 0
        ? this.size
        : [240, 140];
      if (this.setSize) {
        this.setSize(initialSize);
      } else {
        this.size = initialSize;
      }
      this.resizable = true;
      this.collapsable = false;
      if (typeof LiteGraph !== "undefined" && LiteGraph.NO_TITLE != null) {
        this.title = "";
        this.title_mode = LiteGraph.NO_TITLE;
      }
      if (Array.isArray(this.outputs) && this.outputs[0]) {
        this.outputs[0].name = "";
        this.outputs[0].hidden = true;
        this.outputs[0].type = "*";
      }
      this.flags = this.flags || {};
      this.flags.collapsed = false;
      setNoteText(this, getNoteText(this));
      return result;
    };

    nodeType.prototype.computeSize = function () {
      if (this.size && this.size[0] > 0 && this.size[1] > 0) {
        return this.size;
      }
      return [240, 140];
    };

    const onAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      if (onAdded) {
        onAdded.apply(this, arguments);
      }
      if (!this.size || this.size[0] === 0 || this.size[1] === 0) {
        if (this.setSize) {
          this.setSize([240, 140]);
        } else {
          this.size = [240, 140];
        }
      }
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
      if (!this.size || this.size[0] === 0 || this.size[1] === 0) {
        if (this.setSize) {
          this.setSize([240, 140]);
        } else {
          this.size = [240, 140];
        }
      }
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

    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      if (onDrawBackground) {
        onDrawBackground.call(this, ctx);
      }

      ctx.save();
      ctx.fillStyle = NOTE_BG;
      ctx.strokeStyle = NOTE_BORDER;
      ctx.lineWidth = 2;

      const x = 0;
      const y = 0;
      const w = this.size[0];
      const h = this.size[1];
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

      const text = getNoteText(this);
      const width = Math.max(20, this.size[0] - PADDING * 2);
      const height = Math.max(20, this.size[1] - PADDING * 2);

      ctx.save();
      ctx.fillStyle = "#222";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";

      const metrics = pickFontSize(ctx, text, width, height);
      ctx.font = `${metrics.fontSize}px sans-serif`;

      let y = PADDING;
      const startX = PADDING;
      for (const line of metrics.lines) {
        ctx.fillText(line, startX, y);
        y += metrics.lineHeight;
        if (y > this.size[1] - PADDING) {
          break;
        }
      }

      ctx.restore();
    };
  },
});
