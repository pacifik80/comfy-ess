import { app } from "../../scripts/app.js";

const STYLE_ID = "ess-composition-crop-style";
const NODE_NAME = "ESS/CompositionCrop";
const PARTS = [
  { key: "nose", label: "Nose", paired: false },
  { key: "eye", label: "Eye", paired: true },
  { key: "eyebrow", label: "Eyebrow", paired: true },
  { key: "ear", label: "Ear", paired: true },
  { key: "shoulder", label: "Shoulder", paired: true },
  { key: "elbow", label: "Elbow", paired: true },
  { key: "wrist", label: "Wrist", paired: true },
  { key: "hip", label: "Hip", paired: true },
  { key: "knee", label: "Knee", paired: true },
  { key: "ankle", label: "Ankle", paired: true },
  { key: "face", label: "Face", paired: false },
  { key: "mouth", label: "Mouth", paired: false },
  { key: "mouth_corner", label: "Mouth Corner", paired: true },
  { key: "chin", label: "Chin", paired: false },
  { key: "full_person", label: "Full", paired: false },
];
const DEFAULT_NODE_WIDTH = 470;

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-comp-crop-slot{display:flex;justify-content:center;align-items:flex-start;width:100%;padding:0 12px 14px;box-sizing:border-box;pointer-events:none}
.ess-comp-crop{display:grid;gap:6px;width:100%;max-width:100%;margin:0 auto;padding:8px 10px 6px;border:1px solid #334155;border-radius:8px;background:#070f20;box-sizing:border-box;color:#e5e7eb;font-family:"IBM Plex Sans","Segoe UI",sans-serif;pointer-events:auto;overflow:hidden}
.ess-comp-crop-options{display:grid;gap:6px}
.ess-comp-crop-option-row{display:grid;grid-template-columns:66px minmax(0,1fr) 52px minmax(0,1fr);gap:6px;align-items:center}
.ess-comp-crop-option-row label{font-size:12px;color:#e5e7eb}
.ess-comp-crop-head,.ess-comp-crop-row{display:grid;grid-template-columns:minmax(44px,1fr) 34px minmax(0,0.7fr) minmax(0,0.9fr) minmax(0,0.72fr) minmax(0,0.72fr);gap:6px;align-items:center}
.ess-comp-crop-head{font-size:11px;color:#93c5fd;text-transform:uppercase;letter-spacing:.04em}
.ess-comp-crop-row label{font-size:12px;color:#e5e7eb;min-width:0}
.ess-comp-crop-head > div,.ess-comp-crop-row > *{min-width:0}
.ess-comp-crop-row input[type="checkbox"]{justify-self:center;width:15px;height:15px;margin:0;accent-color:#2563eb}
.ess-comp-crop-side-placeholder{font-size:12px;color:#64748b;text-align:center}
.ess-comp-crop-row select,.ess-comp-crop-row input[type="number"],.ess-comp-crop-option-row select,.ess-comp-crop-option-row input[type="color"]{width:100%;height:28px;box-sizing:border-box;border:1px solid #475569;border-radius:6px;background:#020617;color:#e5e7eb;padding:3px 6px;font-size:12px}
.ess-comp-crop-option-row input[type="color"]{padding:2px}
.ess-comp-crop-row select:disabled,.ess-comp-crop-row input:disabled,.ess-comp-crop-option-row select:disabled,.ess-comp-crop-option-row input:disabled{opacity:.55}
`;
  document.head.appendChild(style);
}

function trapEvents(element) {
  if (!element) return;
  const stop = (event) => event.stopPropagation();
  [
    "pointerdown", "pointermove", "pointerup",
    "mousedown", "mouseup",
    "click", "dblclick", "contextmenu",
    "keydown", "keyup", "keypress",
    "wheel",
  ].forEach((name) => element.addEventListener(name, stop));
}

function getWidget(node, name) {
  if (!Array.isArray(node?.widgets)) return null;
  return node.widgets.find((widget) => String(widget?.name || "") === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.serialize = true;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  const element = widget.element || widget.inputEl;
  if (element?.style) {
    element.style.display = "none";
    element.style.pointerEvents = "none";
  }
}

function setWidgetValue(node, widget, value) {
  if (!widget) return;
  widget.value = value;
  if (typeof widget.callback === "function") {
    try {
      widget.callback(value);
    } catch {}
  }
}

function collectWidgetState(node) {
  const state = {};
  if (!Array.isArray(node?.widgets)) return state;
  for (const widget of node.widgets) {
    if (!widget?.name || widget.serialize === false) continue;
    state[widget.name] = widget.value;
  }
  return state;
}

function applyWidgetState(node, state) {
  if (!node || !state || typeof state !== "object") return;
  for (const [name, value] of Object.entries(state)) {
    const widget = getWidget(node, name);
    if (!widget || widget.serialize === false) continue;
    setWidgetValue(node, widget, value);
  }
}

function buildRow(node, part, refs) {
  const partName = part.key;
  const enabledWidget = refs[`enabled_${partName}`];
  const modeWidget = refs[`mode_${partName}`];
  const sideWidget = refs[`side_${partName}`];
  const weightWidget = refs[`weight_${partName}`];
  const marginWidget = refs[`margin_${partName}`];

  const row = document.createElement("div");
  row.className = "ess-comp-crop-row";

  const label = document.createElement("label");
  label.textContent = part.label;

  const enabled = document.createElement("input");
  enabled.type = "checkbox";

  let sideControl = null;
  if (part.paired) {
    sideControl = document.createElement("select");
    ["both", "left", "right"].forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      sideControl.appendChild(option);
    });
  } else {
    sideControl = document.createElement("div");
    sideControl.className = "ess-comp-crop-side-placeholder";
    sideControl.textContent = "—";
  }

  const mode = document.createElement("select");
  ["include", "exclude"].forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    mode.appendChild(option);
  });

  const weight = document.createElement("input");
  weight.type = "number";
  weight.step = "1";
  weight.min = "0";
  weight.max = "100";

  const margin = document.createElement("input");
  margin.type = "number";
  margin.step = "0.1";
  margin.min = "0";
  margin.max = "10";

  const applyState = () => {
    const enabledNow = Boolean(enabledWidget?.value ?? true);
    enabled.checked = enabledNow;
    mode.value = String(modeWidget?.value ?? "include");
    if (part.paired && sideWidget && sideControl?.tagName === "SELECT") {
      sideControl.value = String(sideWidget.value ?? "both");
    }
    if (part.paired && sideControl?.tagName === "SELECT") {
      sideControl.disabled = !enabledNow;
    }
    mode.disabled = !enabledNow;
    weight.disabled = !enabledNow;
    margin.disabled = !enabledNow;
    weight.value = String(weightWidget?.value ?? "0");
    margin.value = String(marginWidget?.value ?? "0");
  };

  enabled.addEventListener("change", () => {
    setWidgetValue(node, enabledWidget, !!enabled.checked);
    applyState();
    node.setDirtyCanvas?.(true, true);
  });
  mode.addEventListener("change", () => {
    setWidgetValue(node, modeWidget, mode.value || "include");
    node.setDirtyCanvas?.(true, true);
  });
  if (part.paired && sideWidget && sideControl?.tagName === "SELECT") {
    sideControl.addEventListener("change", () => {
      setWidgetValue(node, sideWidget, sideControl.value || "both");
      node.setDirtyCanvas?.(true, true);
    });
  }
  weight.addEventListener("input", () => {
    setWidgetValue(node, weightWidget, Number(weight.value || 0));
    node.setDirtyCanvas?.(true, true);
  });
  margin.addEventListener("input", () => {
    setWidgetValue(node, marginWidget, Number(margin.value || 0));
    node.setDirtyCanvas?.(true, true);
  });

  [enabled, sideControl, mode, weight, margin].forEach((item) => item && trapEvents(item));
  row.append(label, enabled, sideControl, mode, weight, margin);

  return { row, refresh: applyState };
}

function buildOptions(node, refs) {
  const container = document.createElement("div");
  container.className = "ess-comp-crop-options";

  const rowA = document.createElement("div");
  rowA.className = "ess-comp-crop-option-row";

  const modeLabel = document.createElement("label");
  modeLabel.textContent = "Framing";
  const modeSelect = document.createElement("select");
  ["crop", "expand"].forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    modeSelect.appendChild(option);
  });

  const fillLabel = document.createElement("label");
  fillLabel.textContent = "Fill";
  const fillSelect = document.createElement("select");
  ["border_fill", "fill_color"].forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    fillSelect.appendChild(option);
  });

  rowA.append(modeLabel, modeSelect, fillLabel, fillSelect);
  container.appendChild(rowA);

  const colorRow = document.createElement("div");
  colorRow.className = "ess-comp-crop-option-row";
  const colorLabel = document.createElement("label");
  colorLabel.textContent = "Color";
  const colorInput = document.createElement("input");
  colorInput.type = "color";
  const spacerA = document.createElement("div");
  const spacerB = document.createElement("div");
  colorRow.append(colorLabel, colorInput, spacerA, spacerB);
  container.appendChild(colorRow);

  const refresh = () => {
    const framingMode = String(refs.framing_mode?.value ?? "crop");
    const fillMode = String(refs.expand_fill_mode?.value ?? "border_fill");
    modeSelect.value = framingMode;
    fillSelect.value = fillMode;
    colorInput.value = String(refs.expand_fill_color?.value || "#000000");

    const expand = framingMode === "expand";
    rowA.style.gridTemplateColumns = expand ? "66px minmax(0,1fr) 52px minmax(0,1fr)" : "66px minmax(0,1fr)";
    fillLabel.style.display = expand ? "" : "none";
    fillSelect.style.display = expand ? "" : "none";
    colorRow.style.display = expand && fillMode === "fill_color" ? "" : "none";
  };

  modeSelect.addEventListener("change", () => {
    setWidgetValue(node, refs.framing_mode, modeSelect.value || "crop");
    refresh();
    node.setDirtyCanvas?.(true, true);
  });
  fillSelect.addEventListener("change", () => {
    setWidgetValue(node, refs.expand_fill_mode, fillSelect.value || "border_fill");
    refresh();
    node.setDirtyCanvas?.(true, true);
  });
  colorInput.addEventListener("input", () => {
    setWidgetValue(node, refs.expand_fill_color, colorInput.value || "#000000");
    node.setDirtyCanvas?.(true, true);
  });

  [modeSelect, fillSelect, colorInput].forEach(trapEvents);
  return { container, refresh };
}

function moveWidgetAfter(node, widget, anchorWidget) {
  if (!widget || !anchorWidget || !Array.isArray(node?.widgets)) return;
  const widgetIndex = node.widgets.indexOf(widget);
  const anchorIndex = node.widgets.indexOf(anchorWidget);
  if (widgetIndex < 0 || anchorIndex < 0) return;
  node.widgets.splice(widgetIndex, 1);
  const nextAnchorIndex = node.widgets.indexOf(anchorWidget);
  node.widgets.splice(nextAnchorIndex + 1, 0, widget);
}

function computePanelHeight(framingMode, fillMode) {
  const paddingTop = 8;
  const paddingBottom = 6;
  const gridGap = 6;
  const optionsRowHeight = 28;
  const headerHeight = 16;
  const partRowHeight = 28;
  const partRows = PARTS.length;
  const visibleOptionRows = framingMode === "expand" && fillMode === "fill_color" ? 2 : 1;
  const optionsHeight = (visibleOptionRows * optionsRowHeight) + (Math.max(0, visibleOptionRows - 1) * gridGap);
  const rowsHeight = (partRows * partRowHeight) + (partRows * gridGap);
  return Math.max(120, paddingTop + paddingBottom + optionsHeight + headerHeight + rowsHeight + 4);
}

app.registerExtension({
  name: "ess_composition_crop",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      const state = info?.properties?.__ess_comp_crop_state || this.properties?.__ess_comp_crop_state || null;
      if (state && typeof state === "object") {
        this.__essCompositionCropPendingState = state;
        if (typeof this.__essCompositionCropApplySavedState === "function") {
          this.__essCompositionCropApplySavedState(state);
        }
      }
      return result;
    };

    const onSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (info) {
      const result = onSerialize ? onSerialize.apply(this, arguments) : undefined;
      const payload = info || {};
      payload.properties = payload.properties || {};
      payload.properties.__ess_comp_crop_state = collectWidgetState(this);
      return result;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        if (this.__essCompositionCropReady || typeof this.addDOMWidget !== "function") return result;
        ensureStyles();

        const refs = {};
        const nativeWidgets = {
          preferred_gender: getWidget(this, "preferred_gender"),
          target_age_min: getWidget(this, "target_age_min"),
          target_age_max: getWidget(this, "target_age_max"),
        };
        for (const part of PARTS) {
          const partName = part.key;
          refs[`enabled_${partName}`] = getWidget(this, `enabled_${partName}`);
          refs[`mode_${partName}`] = getWidget(this, `mode_${partName}`);
          refs[`weight_${partName}`] = getWidget(this, `weight_${partName}`);
          refs[`margin_${partName}`] = getWidget(this, `margin_${partName}`);
          if (part.paired) {
            refs[`side_${partName}`] = getWidget(this, `side_${partName}`);
          }
        }
        refs.framing_mode = getWidget(this, "framing_mode");
        refs.expand_fill_mode = getWidget(this, "expand_fill_mode");
        refs.expand_fill_color = getWidget(this, "expand_fill_color");
        const allWidgets = Object.values(refs).filter(Boolean);
        if (!allWidgets.length) return result;
        allWidgets.forEach(hideWidget);

        const confidenceWidget = getWidget(this, "confidence_threshold");
        moveWidgetAfter(this, nativeWidgets.preferred_gender, confidenceWidget);
        moveWidgetAfter(this, nativeWidgets.target_age_min, nativeWidgets.preferred_gender || confidenceWidget);
        moveWidgetAfter(this, nativeWidgets.target_age_max, nativeWidgets.target_age_min || nativeWidgets.preferred_gender || confidenceWidget);
        const slot = document.createElement("div");
        slot.className = "ess-comp-crop-slot";
        const container = document.createElement("div");
        container.className = "ess-comp-crop";
        slot.appendChild(container);

        const options = buildOptions(this, refs);
        container.appendChild(options.container);

        const header = document.createElement("div");
        header.className = "ess-comp-crop-head";
        ["Part", "Use", "Side", "Mode", "Weight", "Margin"].forEach((text) => {
          const item = document.createElement("div");
          item.textContent = text;
          header.appendChild(item);
        });
        container.appendChild(header);

        const rows = PARTS.map((part) => buildRow(this, part, refs));
        rows.forEach((entry) => container.appendChild(entry.row));

        let widgetHeight = 120;
        this.__essCompositionCropWidgetHeight = widgetHeight;
        const widget = this.addDOMWidget("parts_compact", "ess_composition_crop_parts", slot, {
          getValue: () => "",
          setValue: () => {},
          getMinHeight: () => widgetHeight,
          getMaxHeight: () => widgetHeight,
          hideOnZoom: false,
          margin: 0,
        });
        if (widget) widget.serialize = false;
        if (widget) {
          widget.computeSize = (width) => [Math.max(Number(width || this.size?.[0] || DEFAULT_NODE_WIDTH), 120), widgetHeight];
        }
        if (widget && Array.isArray(this.widgets)) {
          const widgetIndex = this.widgets.indexOf(widget);
          const anchorWidget = nativeWidgets.target_age_max || nativeWidgets.preferred_gender;
          const anchorIndex = anchorWidget ? this.widgets.indexOf(anchorWidget) : -1;
          if (widgetIndex >= 0 && anchorIndex >= 0 && widgetIndex > anchorIndex + 1) {
            this.widgets.splice(widgetIndex, 1);
            this.widgets.splice(anchorIndex + 1, 0, widget);
          }
        }

        moveWidgetAfter(this, refs.framing_mode, widget);
        moveWidgetAfter(this, refs.expand_fill_mode, refs.framing_mode || widget);
        moveWidgetAfter(this, refs.expand_fill_color, refs.expand_fill_mode || refs.framing_mode || widget);

        const refresh = () => {
          options.refresh();
          rows.forEach((entry) => entry.refresh());
          const framingMode = String(refs.framing_mode?.value ?? "crop");
          const fillMode = String(refs.expand_fill_mode?.value ?? "border_fill");
          const nextHeight = computePanelHeight(framingMode, fillMode);
          if (Math.abs(nextHeight - widgetHeight) < 1) return;
          widgetHeight = nextHeight;
          this.__essCompositionCropWidgetHeight = widgetHeight;
          if (widget) {
            widget.computeSize = (width) => [Math.max(Number(width || this.size?.[0] || DEFAULT_NODE_WIDTH), 120), widgetHeight];
          }
          this.setDirtyCanvas?.(true, true);
        };
        this.__essCompositionCropRefresh = refresh;
        this.__essCompositionCropApplySavedState = (state) => {
          applyWidgetState(this, state);
          refresh();
          this.setDirtyCanvas?.(true, true);
          this.__essCompositionCropPendingState = null;
        };
        this.__essCompositionCropReady = true;
        this.size = Array.isArray(this.size) ? this.size : [420, 0];
        this.size[0] = Math.max(Number(this.size[0] || 0), DEFAULT_NODE_WIDTH);

        requestAnimationFrame(() => {
          refresh();
          const pendingState = this.__essCompositionCropPendingState || this.properties?.__ess_comp_crop_state || null;
          if (pendingState && typeof this.__essCompositionCropApplySavedState === "function") {
            this.__essCompositionCropApplySavedState(pendingState);
          }
        });
      } catch (error) {
        console.error("[ess_composition_crop] failed to build compact UI:", error);
      }
      return result;
    };
  },
});
