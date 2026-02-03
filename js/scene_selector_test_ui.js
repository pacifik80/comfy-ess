import { app } from "../../scripts/app.js";

const MAX_SCENES = 10;

const findWidget = (node, name) =>
  node.widgets?.find((w) => w.name === name);

const toggleWidgetVisibility = (widget, show) => {
  if (!widget) return;

  if (!widget._essOriginal) {
    widget._essOriginal = {
      computeSize: widget.computeSize,
      hidden: widget.hidden ?? false,
    };
  }

  widget.hidden = !show;
  widget.computeSize = show
    ? widget._essOriginal.computeSize
    : () => [0, -4];
};

const toggleInputVisibility = (node, inputName, show) => {
  if (!node.inputs) return;
  const index = node.inputs.findIndex((input) => input.name === inputName);
  if (index === -1) return;

  const input = node.inputs[index];
  const wasHidden = !!input.hidden;
  input.hidden = !show;

  if (!show && input.link != null) {
    node.disconnectInput(index);
  }
  if (wasHidden !== !show) {
    node.updateInputLinks?.();
  }
};

const refreshSceneBlocks = (node) => {
  const numWidget = findWidget(node, "num_scenes");
  const rawValue = typeof numWidget?.value === "number" ? numWidget.value : 1;
  const target = Math.max(1, Math.min(MAX_SCENES, Math.round(rawValue)));

  for (let i = 1; i <= MAX_SCENES; i += 1) {
    const show = i <= target;
    toggleInputVisibility(node, `scene_${i}`, show);
    toggleWidgetVisibility(findWidget(node, `name_${i}`), show);
    toggleWidgetVisibility(findWidget(node, `weight_${i}`), show);
  }

  if (node.computeSize) {
    const [width, height] = node.computeSize(node.size);
    node.size = [width, height];
  }

  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas(true, true);
};

const scheduleRefresh = (node) => {
  requestAnimationFrame(() => refreshSceneBlocks(node));
};

app.registerExtension({
  name: "ess.scene_selector_test_ui",
  nodeCreated(node) {
    if (node?.comfyClass !== "SceneSelectorTest") {
      return;
    }

    const numWidget = findWidget(node, "num_scenes");
    if (numWidget) {
      const originalCallback = numWidget.callback;
      numWidget.callback = function (...args) {
        const result = originalCallback?.apply(this, args);
        scheduleRefresh(node);
        return result;
      };
    }

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      scheduleRefresh(node);
      return result;
    };

    const originalOnAdded = node.onAdded;
    node.onAdded = function (...args) {
      const result = originalOnAdded?.apply(this, args);
      scheduleRefresh(node);
      return result;
    };

    scheduleRefresh(node);
  },
});
