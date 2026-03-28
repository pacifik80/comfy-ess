import { app } from "../../scripts/app.js";

const MAX_SLOTS = 16;

function syncImageInputs(node, targetCount) {
  const desired = Math.max(1, Math.min(Number(targetCount) || 1, MAX_SLOTS));

  if (!node.inputs) {
    node.inputs = [];
  }

  for (let i = node.inputs.length - 1; i >= 0; i--) {
    const name = node.inputs[i]?.name || "";
    if (name.startsWith("image_")) {
      const index = Number(name.slice(6));
      if (index > desired) {
        node.removeInput(i);
      }
    }
  }

  for (let i = 1; i <= desired; i++) {
    if (!node.inputs.some((input) => input?.name === `image_${i}`)) {
      node.addInput(`image_${i}`, "IMAGE");
    }
  }

  if (typeof node.computeSize === "function") {
    node.setSize(node.computeSize());
  }
}

app.registerExtension({
  name: "ess_image_batch_merge",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/ImageBatchMerge") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      const countWidget = this.widgets?.find((w) => w.name === "count");

      const applyButton = this.addWidget("button", "apply", null, () => {
        syncImageInputs(this, countWidget?.value);
      });
      applyButton.label = "Apply inputs";

      if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = (value, canvas, node, pos, event) => {
          if (originalCallback) {
            originalCallback(value, canvas, node, pos, event);
          }
          syncImageInputs(this, value);
        };

        const countIndex = this.widgets.findIndex((w) => w.name === "count");
        if (countIndex !== -1) {
          this.widgets.pop();
          this.widgets.splice(countIndex + 1, 0, applyButton);
        }
      }

      setTimeout(() => syncImageInputs(this, countWidget?.value), 0);
      return result;
    };
  },
});
