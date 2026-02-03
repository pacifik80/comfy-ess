import { app } from "../../scripts/app.js";

const MAX_SLOTS = 16;

function syncSlots(node, targetCount) {
  const desired = Math.max(1, Math.min(Number(targetCount) || 1, MAX_SLOTS));

  if (!node.inputs) {
    node.inputs = [];
  }

  for (let i = node.inputs.length - 1; i >= 0; i -= 1) {
    const name = node.inputs[i]?.name || "";
    if (name.startsWith("input_")) {
      const index = Number(name.slice(6));
      if (index > desired) {
        node.removeInput(i);
      }
    }
  }

  for (let i = 1; i <= desired; i += 1) {
    if (!node.inputs.some((input) => input?.name === `input_${i}`)) {
      node.addInput(`input_${i}`, "STRING");
    }
  }

  if (node.computeSize) {
    node.setSize(node.computeSize());
  }
}

app.registerExtension({
  name: "ess_string_concat",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/StringConcatenate") {
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
  },
});
