import { app } from "../../scripts/app.js";

const MAX_SLOTS = 16;
const LABEL_FONT = "12px sans-serif";
const LABEL_HEIGHT = 14;
const LABEL_PADDING = 4;

function normalizeColor(color) {
  if (!color) {
    return "#888";
  }
  if (Array.isArray(color) && color.length >= 3) {
    return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  }
  return color;
}

function parseHexColor(color) {
  if (typeof color !== "string") {
    return null;
  }
  const hex = color.startsWith("#") ? color.slice(1) : null;
  if (!hex || (hex.length !== 6 && hex.length !== 3)) {
    return null;
  }
  const value = hex.length === 3
    ? hex.split("").map((c) => c + c).join("")
    : hex;
  const num = Number.parseInt(value, 16);
  return {
    r: (num >> 16) & 255,
    g: (num >> 8) & 255,
    b: num & 255,
  };
}

function getTextColorForBackground(color) {
  const parsed = parseHexColor(color);
  if (!parsed) {
    return "#fff";
  }
  const luminance = (0.299 * parsed.r + 0.587 * parsed.g + 0.114 * parsed.b) / 255;
  return luminance > 0.6 ? "#000" : "#fff";
}

function getTypeColor(type) {
  const colors = (typeof LGraphCanvas !== "undefined" && LGraphCanvas.link_type_colors)
    || (typeof LiteGraph !== "undefined" && LiteGraph.link_type_colors)
    || {};
  return normalizeColor(colors[type] || colors[type?.toLowerCase?.()] || "#888");
}

function getInputLinkInfo(node, inputSlot) {
  if (!inputSlot) {
    return null;
  }
  const linkId = inputSlot.link != null
    ? inputSlot.link
    : (Array.isArray(inputSlot.links) && inputSlot.links.length ? inputSlot.links[0] : null);
  if (linkId == null) {
    return null;
  }
  const graph = node.graph;
  const links = graph?.links || graph?._links;
  const link = links?.[linkId];
  const linkType = link?.type && link.type !== "*" ? link.type : null;
  const getNodeById = graph?.getNodeById?.bind(graph)
    || (graph?._nodes_by_id ? (id) => graph._nodes_by_id[id] : null);
  const originNode = link?.origin_id != null ? getNodeById?.(link.origin_id) : null;
  const originSlot = originNode?.outputs?.[link?.origin_slot];
  const originType = originSlot?.type && originSlot.type !== "*" ? originSlot.type : null;
  const inputType = inputSlot.type && inputSlot.type !== "*" ? inputSlot.type : null;
  const resolved = resolveOriginLink(graph, link, 0);
  return {
    type: resolved?.type || linkType || originType || inputType || null,
    originNode: resolved?.node || originNode,
    originSlot: resolved?.slot || originSlot,
    hasLink: true,
  };
}

function formatTypeLabel(type) {
  if (Array.isArray(type)) {
    return type.map((item) => String(item)).join("|");
  }
  if (type && typeof type === "object" && type.name) {
    return String(type.name);
  }
  return type != null ? String(type) : null;
}

function formatSourceLabel(originNode, originSlot) {
  if (!originNode) {
    return null;
  }
  const nodeLabel = originNode.title || originNode.type || "Node";
  const slotLabel = originSlot?.name ? String(originSlot.name) : (originSlot?.label ? String(originSlot.label) : null);
  if (slotLabel) {
    return `${nodeLabel}:${slotLabel}`;
  }
  return String(nodeLabel);
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

function resolveOriginLink(graph, link, depth) {
  if (!graph || !link || depth > MAX_SLOTS) {
    return null;
  }
  const getNodeById = graph?.getNodeById?.bind(graph)
    || (graph?._nodes_by_id ? (id) => graph._nodes_by_id[id] : null);
  const originNode = link?.origin_id != null ? getNodeById?.(link.origin_id) : null;
  const originSlot = originNode?.outputs?.[link?.origin_slot];
  const originType = originSlot?.type && originSlot.type !== "*" ? originSlot.type : null;
  if (!originNode) {
    return null;
  }

  if (isGroupRerouteNode(originNode)) {
    const outputName = originSlot?.name || `output_${(link?.origin_slot ?? 0) + 1}`;
    const match = outputName.match(/output_(\d+)/);
    const index = match ? Number(match[1]) : (link?.origin_slot ?? 0) + 1;
    const inputName = `input_${index}`;
    const inputSlot = originNode.inputs?.find((input) => input?.name === inputName);
    const inputLinkId = inputSlot?.link != null
      ? inputSlot.link
      : (Array.isArray(inputSlot?.links) ? inputSlot.links[0] : null);
    const links = graph?.links || graph?._links;
    const upstreamLink = inputLinkId != null ? links?.[inputLinkId] : null;
    const resolved = upstreamLink ? resolveOriginLink(graph, upstreamLink, depth + 1) : null;
    return resolved || { node: originNode, slot: originSlot, type: originType };
  }

  if (isLiteRerouteNode(originNode)) {
    const inputSlot = originNode.inputs?.[0];
    const inputLinkId = inputSlot?.link != null
      ? inputSlot.link
      : (Array.isArray(inputSlot?.links) ? inputSlot.links[0] : null);
    const links = graph?.links || graph?._links;
    const upstreamLink = inputLinkId != null ? links?.[inputLinkId] : null;
    const resolved = upstreamLink ? resolveOriginLink(graph, upstreamLink, depth + 1) : null;
    return resolved || { node: originNode, slot: originSlot, type: originType };
  }

  return { node: originNode, slot: originSlot, type: originType };
}

function getTypeLabelAndColor(info) {
  if (!info || !info.hasLink) {
    return null;
  }
  const label = formatTypeLabel(info?.type) || "?";
  const primary = Array.isArray(info.type) ? info.type[0] : info.type;
  const source = formatSourceLabel(info?.originNode, info?.originSlot);
  const displayLabel = source ? `${source} (${label})` : `(${label})`;
  return {
    label: displayLabel,
    color: getTypeColor(primary ?? label),
  };
}

function drawTypeLabels(node, ctx) {
  if (!node.inputs || !node.outputs || node.flags?.collapsed) {
    return;
  }

  ctx.save();
  ctx.font = LABEL_FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  let maxLabelWidth = 0;

  for (let i = 1; i <= MAX_SLOTS; i++) {
    const inputIndex = node.inputs.findIndex((input) => input?.name === `input_${i}`);
    const outputIndex = node.outputs.findIndex((output) => output?.name === `output_${i}`);
    if (inputIndex === -1 || outputIndex === -1) {
      continue;
    }

    const inputSlot = node.inputs[inputIndex];
    const linkInfo = getInputLinkInfo(node, inputSlot);
    const labelInfo = getTypeLabelAndColor(linkInfo);
    if (!labelInfo) {
      continue;
    }

    const inputPos = node.getConnectionPos(true, inputIndex);
    const outputPos = node.getConnectionPos(false, outputIndex);
    const x = (inputPos[0] + outputPos[0]) / 2 - node.pos[0];
    const y = (inputPos[1] + outputPos[1]) / 2 - node.pos[1];
    const label = labelInfo.label;
    const textWidth = ctx.measureText(label).width;
    if (textWidth > maxLabelWidth) {
      maxLabelWidth = textWidth;
    }
    const width = textWidth + LABEL_PADDING * 2;
    const height = LABEL_HEIGHT;
    const rectX = x - width / 2;
    const rectY = y - height / 2;

    ctx.fillStyle = labelInfo.color;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(rectX, rectY, width, height, 3);
    } else {
      ctx.rect(rectX, rectY, width, height);
    }
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = getTextColorForBackground(labelInfo.color);
    ctx.fillText(label, x, y + 0.5);
  }

  if (maxLabelWidth > 0) {
    const desiredWidth = Math.max(node.size[0], maxLabelWidth + LABEL_PADDING * 2 + 80);
    if (desiredWidth !== node.size[0]) {
      node.size[0] = desiredWidth;
    }
  }

  ctx.restore();
}

function syncSlots(node, targetCount) {
  const desired = Math.max(1, Math.min(Number(targetCount) || 1, MAX_SLOTS));

  if (!node.inputs) {
    node.inputs = [];
  }
  if (!node.outputs) {
    node.outputs = [];
  }

  for (let i = node.inputs.length - 1; i >= 0; i--) {
    const name = node.inputs[i]?.name || "";
    if (name.startsWith("input_")) {
      const index = Number(name.slice(6));
      if (index > desired) {
        node.removeInput(i);
      }
    }
  }

  for (let i = node.outputs.length - 1; i >= 0; i--) {
    const name = node.outputs[i]?.name || "";
    if (name.startsWith("output_")) {
      const index = Number(name.slice(7));
      if (index > desired) {
        node.removeOutput(i);
      }
    }
  }

  for (let i = 1; i <= desired; i++) {
    if (!node.inputs.some((input) => input?.name === `input_${i}`)) {
      node.addInput(`input_${i}`, "*");
    }
  }

  for (let i = 1; i <= desired; i++) {
    if (!node.outputs.some((output) => output?.name === `output_${i}`)) {
      node.addOutput(`output_${i}`, "*");
    }
  }

  if (node.computeSize) {
    node.setSize(node.computeSize());
  }
}

app.registerExtension({
  name: "ess_group_reroute",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/GroupReroute") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const countWidget = this.widgets?.find((w) => w.name === "count");

      const applyButton = this.addWidget("button", "apply", null, () => {
        syncSlots(this, countWidget?.value);
      });
      applyButton.label = "Apply slots";

      if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = (value, canvas, node, pos, event) => {
          if (originalCallback) {
            originalCallback(value, canvas, node, pos, event);
          }
          syncSlots(this, value);
        };

        const countIndex = this.widgets.findIndex((w) => w.name === "count");
        if (countIndex !== -1) {
          this.widgets.pop();
          this.widgets.splice(countIndex + 1, 0, applyButton);
        }
      }

      setTimeout(() => syncSlots(this, countWidget?.value), 0);
      return result;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) {
        onDrawForeground.call(this, ctx);
      }
      drawTypeLabels(this, ctx);
    };

    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      if (onDrawBackground) {
        onDrawBackground.call(this, ctx);
      }
      drawTypeLabels(this, ctx);
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      if (onConnectionsChange) {
        onConnectionsChange.apply(this, arguments);
      }
      this.setDirtyCanvas?.(true, true);
    };
  },
});
