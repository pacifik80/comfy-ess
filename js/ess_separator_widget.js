import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ess_separator_widget",
  async getCustomWidgets() {
    return {
      ESS_SEPARATOR(node, inputName, inputData) {
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const label = config.label || inputName;
        const height = config.height || 26;

        const widget = {
          type: "ESS_SEPARATOR",
          name: inputName,
          value: label,
          options: config,
          computeSize: (width) => [width, height],
          draw: function (ctx, node, widgetWidth, y, H) {
            const h = H || height;
            const margin = 10;
            const radius = h * 0.35;
            const text = (this.value || "").toString().toUpperCase();

            ctx.save();
            ctx.beginPath();
            ctx.fillStyle = "#2b2f34";
            ctx.strokeStyle = "#3a3f45";
            ctx.roundRect(margin, y + 2, widgetWidth - margin * 2, h - 4, radius);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = "#c9d1d9";
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(text, margin + 10, y + h / 2);

            ctx.strokeStyle = "#4b525a";
            ctx.beginPath();
            ctx.moveTo(margin + 8, y + h - 6);
            ctx.lineTo(widgetWidth - margin - 8, y + h - 6);
            ctx.stroke();
            ctx.restore();
          },
          serializeValue: () => null,
        };

        node.addCustomWidget(widget);
        return widget;
      },
    };
  },
});
