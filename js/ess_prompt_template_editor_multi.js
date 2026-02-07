import { app } from "../../scripts/app.js";

const MAX_VARIANTS = 10;
const VARIANTS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"];

function clampCount(value) {
  const num = Number(value) || 1;
  return Math.max(1, Math.min(MAX_VARIANTS, Math.floor(num)));
}

function buildOutputNames(count) {
  const names = [];
  for (let i = 0; i < count; i += 1) {
    const letter = VARIANTS[i];
    names.push(`positive_${letter}`);
    names.push(`negative_${letter}`);
  }
  return names;
}

function syncOutputs(node, count) {
  if (!node) {
    return;
  }
  const desired = clampCount(count);
  const keep = new Set(buildOutputNames(desired));

  if (!Array.isArray(node.outputs)) {
    node.outputs = [];
  }

  for (let i = node.outputs.length - 1; i >= 0; i -= 1) {
    const name = node.outputs[i]?.name || "";
    if (name.startsWith("positive_") || name.startsWith("negative_")) {
      if (!keep.has(name)) {
        node.removeOutput?.(i);
      }
    }
  }

  const existing = new Set(node.outputs.map((out) => out?.name));
  for (const name of buildOutputNames(desired)) {
    if (!existing.has(name)) {
      node.addOutput?.(name, "STRING");
    }
  }

  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "ess_prompt_template_editor_multi",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/PromptTemplateEditorMulti") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      const countWidget = this.widgets?.find((w) => w.name === "count");
      const applyButton = this.addWidget("button", "apply", null, () => {
        syncOutputs(this, countWidget?.value);
      });
      applyButton.label = "Apply outputs";

      if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = (value, canvas, node, pos, event) => {
          if (originalCallback) {
            originalCallback(value, canvas, node, pos, event);
          }
          syncOutputs(this, value);
        };

        const countIndex = this.widgets.findIndex((w) => w.name === "count");
        if (countIndex !== -1) {
          this.widgets.pop();
          this.widgets.splice(countIndex + 1, 0, applyButton);
        }
      }

      setTimeout(() => syncOutputs(this, countWidget?.value), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      const countWidget = this.widgets?.find((w) => w.name === "count");
      syncOutputs(this, countWidget?.value);
      return result;
    };
  },
});
