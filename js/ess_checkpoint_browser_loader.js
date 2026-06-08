import { app } from "../../scripts/app.js";

const STYLE_ID = "ess-checkpoint-browser-style";
const SETTINGS = {
  directory: "ESS.ModelLoader.CheckpointDirectory",
  legacyDirectory: "ESS.Checkpoints.Directory",
  apiKey: "ESS.ModelLoader.CivitAIKey",
  legacyApiKey: "ESS.Checkpoints.CivitAIKey",
};
const KNOWN_BASE_MODELS = [
  "SD 1.4",
  "SD 1.5",
  "SD 1.5 LCM",
  "SD 2.0",
  "SD 2.1",
  "SDXL 0.9",
  "SDXL 1.0",
  "SDXL Distilled",
  "Pony",
  "Illustrious",
  "FLUX.1 D",
  "FLUX.1 S",
  "FLUX.1 C",
  "Wan 2.1",
  "Wan 2.2",
  "Qwen",
  "LTXV",
  "Hunyuan Video",
  "NoobAI",
  "PixArt a",
  "Playground v2.5",
  "Lumina",
  "Sana",
];
const EXTENSION_SETTINGS = [
  {
    id: SETTINGS.directory,
    category: ["ESS", "Model Loader"],
    name: "Model loader checkpoint directory",
    type: "text",
    defaultValue: "ess/checkpoints",
    tooltip: "Directory under ComfyUI models used by the ESS model loader. Downloads are organized as Family/Author/Name/Version inside this directory.",
    attrs: {
      placeholder: "ess/checkpoints",
    },
    onChange(value) {
      settingsState.directory = String(value ?? "ess/checkpoints");
      refreshAttachedWidgets();
    },
  },
  {
    id: SETTINGS.apiKey,
    category: ["ESS", "Model Loader"],
    name: "Model loader CivitAI API key",
    type: "text",
    defaultValue: "",
    tooltip: "Optional API key used for CivitAI search, downloads, and local metadata enrichment.",
    attrs: {
      placeholder: "Bearer ... or raw API key",
    },
    onChange(value) {
      settingsState.apiKey = String(value ?? "");
    },
  },
];

let settingsRefs = null;
let settingsState = {
  directory: "ess/checkpoints",
  apiKey: "",
};
const attachedWidgetStates = new Set();
const previewMetadataCache = new Map();

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-ckpt-widget {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  color: #d9e2ef;
  font-size: 12px;
}
.ess-ckpt-widget-card {
  border: 1px solid #2b3340;
  background: #0d1218;
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ess-ckpt-widget-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ess-ckpt-widget-title {
  color: #9ecbff;
  font-weight: 600;
}
.ess-ckpt-widget-actions {
  display: flex;
  gap: 6px;
}
.ess-ckpt-widget button,
.ess-ckpt-browser button,
.ess-ckpt-browser select,
.ess-ckpt-browser input {
  font: inherit;
}
.ess-ckpt-widget button,
.ess-ckpt-browser button {
  background: #273141;
  color: #eef5ff;
  border: 1px solid #3c4b61;
  border-radius: 6px;
  padding: 6px 10px;
  cursor: pointer;
}
.ess-ckpt-widget button:hover,
.ess-ckpt-browser button:hover {
  background: #31415b;
}
.ess-ckpt-widget button:disabled,
.ess-ckpt-browser button:disabled {
  opacity: 0.55;
  cursor: default;
}
.ess-ckpt-widget-name {
  color: #f7fbff;
  font-size: 13px;
  font-weight: 600;
}
.ess-ckpt-widget-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ess-ckpt-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 7px;
  border-radius: 999px;
  border: 1px solid #35506f;
  background: #12202f;
  color: #cfe6ff;
  font-size: 11px;
}
.ess-ckpt-widget-path {
  color: #95a6bc;
  word-break: break-word;
}
.ess-ckpt-widget-config {
  color: #7f94ac;
  word-break: break-word;
  font-size: 11px;
}
.ess-ckpt-browser {
  position: fixed;
  inset: 0;
  z-index: 10000;
  background: rgba(5, 8, 12, 0.86);
  display: flex;
  align-items: stretch;
  justify-content: center;
  padding: 18px;
  box-sizing: border-box;
  font-family: "Segoe UI", sans-serif;
}
.ess-ckpt-browser-shell {
  width: min(1320px, 100%);
  height: min(880px, 100%);
  background: #0d1218;
  border: 1px solid #243244;
  border-radius: 16px;
  display: grid;
  grid-template-rows: auto auto minmax(0, 1fr);
  overflow: hidden;
  box-shadow: 0 18px 80px rgba(0, 0, 0, 0.55);
}
.ess-ckpt-browser-head {
  padding: 14px 16px;
  border-bottom: 1px solid #202b39;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.ess-ckpt-browser-head-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.ess-ckpt-browser-head h3 {
  margin: 0;
  font-size: 18px;
  color: #f4f9ff;
}
.ess-ckpt-browser-head small {
  color: #93a6be;
}
.ess-ckpt-browser-toolbar {
  padding: 12px 16px;
  border-bottom: 1px solid #202b39;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}
.ess-ckpt-toolbar-search {
  flex: 1 1 320px;
  min-width: 240px;
}
.ess-ckpt-toolbar-select {
  flex: 0 1 220px;
  min-width: 170px;
}
.ess-ckpt-toolbar-compact {
  flex: 0 1 170px;
  min-width: 140px;
}
.ess-ckpt-toolbar-button {
  flex: 0 0 auto;
}
.ess-ckpt-toolbar-status {
  margin-left: auto;
}
.ess-ckpt-browser-tabs {
  display: flex;
  gap: 8px;
}
.ess-ckpt-browser-tabs button.active {
  background: #355781;
  border-color: #5b8dc5;
}
.ess-ckpt-browser input,
.ess-ckpt-browser select {
  width: 100%;
  min-width: 0;
  background: #111923;
  color: #eef5ff;
  border: 1px solid #324459;
  border-radius: 6px;
  padding: 7px 9px;
}
.ess-ckpt-inline-group {
  display: flex;
  gap: 8px;
  min-width: 0;
}
.ess-ckpt-inline-group > input {
  flex: 1 1 auto;
}
.ess-ckpt-settings-panel,
.ess-ckpt-settings-dialog {
  border: 1px solid #243244;
  border-radius: 12px;
  background: #101722;
  color: #eef5ff;
}
.ess-ckpt-settings-panel {
  margin-top: 16px;
  padding: 16px;
  max-width: 920px;
}
.ess-ckpt-settings-dialog {
  width: min(760px, calc(100vw - 48px));
  padding: 18px;
  box-shadow: 0 18px 80px rgba(0, 0, 0, 0.55);
}
.ess-ckpt-settings-title {
  font-size: 18px;
  font-weight: 600;
  color: #f5f9ff;
}
.ess-ckpt-settings-subtitle {
  color: #9bb1c8;
  font-size: 12px;
}
.ess-ckpt-settings-form {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 14px;
}
.ess-ckpt-settings-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ess-ckpt-settings-field label {
  font-size: 13px;
  font-weight: 600;
  color: #edf4ff;
}
.ess-ckpt-settings-field small {
  color: #93a6be;
  font-size: 11px;
}
.ess-ckpt-settings-field input {
  width: 100%;
  box-sizing: border-box;
  background: #111923;
  color: #eef5ff;
  border: 1px solid #324459;
  border-radius: 8px;
  padding: 10px 12px;
  font: inherit;
}
.ess-ckpt-settings-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 4px;
}
.ess-ckpt-settings-actions .ess-ckpt-status {
  flex: 1 1 auto;
}
.ess-ckpt-browser-body {
  min-height: 0;
  overflow: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.ess-ckpt-browser-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ess-ckpt-browser-section-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 0 4px;
}
.ess-ckpt-browser-section-title {
  color: #f1f7ff;
  font-size: 13px;
  font-weight: 600;
}
.ess-ckpt-browser-section-meta {
  color: #90a5bc;
  font-size: 11px;
}
.ess-ckpt-card-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  align-content: start;
}
.ess-ckpt-browser-empty {
  color: #95a6bc;
  border: 1px dashed #304355;
  border-radius: 10px;
  padding: 18px;
  background: #101823;
}
.ess-ckpt-card {
  border: 1px solid #243244;
  border-radius: 12px;
  background: linear-gradient(180deg, #101722 0%, #0d1218 100%);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
}
.ess-ckpt-card:hover {
  border-color: #456284;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.28);
  transform: translateY(-1px);
}
.ess-ckpt-card:focus-visible {
  outline: 2px solid #68a5f2;
  outline-offset: 2px;
}
.ess-ckpt-card-preview {
  height: 180px;
  background: #091019;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
}
.ess-ckpt-card-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  cursor: pointer;
}
.ess-ckpt-card-preview video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  cursor: pointer;
}
.ess-ckpt-card-preview-nav {
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ess-ckpt-card-preview-nav button {
  background: rgba(10, 16, 24, 0.82);
  border-color: rgba(125, 160, 201, 0.8);
  padding: 4px 8px;
}
.ess-ckpt-card-preview-nav span {
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(10, 16, 24, 0.82);
  border: 1px solid rgba(125, 160, 201, 0.5);
  color: #eef5ff;
  font-size: 11px;
}
.ess-ckpt-card-body {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ess-ckpt-card-title {
  color: #f5f9ff;
  font-size: 15px;
  font-weight: 600;
}
.ess-ckpt-card-subtitle {
  color: #89a1bc;
  font-size: 11px;
}
.ess-ckpt-card-path {
  color: #7e95ae;
  font-size: 11px;
  word-break: break-word;
}
.ess-ckpt-card-desc {
  color: #c6d4e4;
  font-size: 12px;
  line-height: 1.45;
  min-height: 52px;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.ess-ckpt-card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.ess-ckpt-card-row {
  display: grid;
  gap: 8px;
}
.ess-ckpt-card-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ess-ckpt-card-action-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.ess-ckpt-card-extra {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ess-ckpt-card-textblock {
  border: 1px solid #213041;
  border-radius: 8px;
  background: #0f1822;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ess-ckpt-card-texthead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.ess-ckpt-card-texttitle {
  color: #cfe5ff;
  font-size: 11px;
  font-weight: 600;
}
.ess-ckpt-card-textcontent {
  color: #bdd0e4;
  font-size: 11px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 112px;
  overflow: auto;
}
.ess-ckpt-status {
  color: #99adbf;
  font-size: 11px;
}
.ess-ckpt-progress {
  height: 6px;
  border-radius: 999px;
  overflow: hidden;
  background: #172130;
}
.ess-ckpt-progress > span {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #4ba2ff, #88d1ff);
}
.ess-ckpt-preview-dialog {
  width: min(1180px, calc(100vw - 48px));
  max-height: calc(100vh - 48px);
  padding: 18px;
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
  gap: 16px;
  border: 1px solid #243244;
  border-radius: 16px;
  background: #0d1218;
  color: #eef5ff;
  overflow: hidden;
}
.ess-ckpt-preview-stage {
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #091019;
  border-radius: 12px;
  overflow: hidden;
}
.ess-ckpt-preview-stage img,
.ess-ckpt-preview-stage video {
  max-width: 100%;
  max-height: calc(100vh - 120px);
  display: block;
}
.ess-ckpt-preview-side {
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.ess-ckpt-preview-meta {
  min-height: 0;
  overflow: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ess-ckpt-preview-grid {
  display: grid;
  grid-template-columns: minmax(110px, 160px) minmax(0, 1fr);
  gap: 8px 10px;
  font-size: 12px;
}
.ess-ckpt-preview-grid strong {
  color: #c8dcf3;
}
.ess-ckpt-preview-grid span {
  color: #eef5ff;
  word-break: break-word;
}
.ess-ckpt-preview-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.ess-ckpt-preview-text {
  border: 1px solid #243244;
  border-radius: 10px;
  background: #101722;
  padding: 10px;
}
.ess-ckpt-preview-texthead {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}
.ess-ckpt-preview-textcontent {
  color: #d7e5f6;
  font-size: 12px;
  line-height: 1.45;
  max-height: 180px;
  overflow: auto;
  white-space: pre-wrap;
}
@media (max-width: 960px) {
  .ess-ckpt-preview-dialog {
    grid-template-columns: 1fr;
  }
}
`;
  document.head.appendChild(style);
}

function ensureSettings() {
  if (settingsRefs) return settingsRefs;
  const settingsApi = app.ui?.settings;

  const settingDefinitions = [
    {
      id: SETTINGS.directory,
      category: ["ESS", "Model Loader"],
      name: "Model loader checkpoint directory",
      type: "text",
      defaultValue: "ess/checkpoints",
      tooltip: "Directory under ComfyUI models used by the ESS model loader. Downloads are organized as Family/Author/Name/Version inside this directory.",
      attrs: { placeholder: "ess/checkpoints" },
      onChange(value) {
        settingsState.directory = String(value ?? "ess/checkpoints");
        refreshAttachedWidgets();
      },
    },
    {
      id: SETTINGS.apiKey,
      category: ["ESS", "Model Loader"],
      name: "Model loader CivitAI API key",
      type: "text",
      defaultValue: "",
      tooltip: "Optional API key used for CivitAI search, downloads, and local metadata enrichment.",
      attrs: { placeholder: "Bearer ... or raw API key" },
      onChange(value) {
        settingsState.apiKey = String(value ?? "");
      },
    },
  ];

  if (settingsApi?.addSetting) {
    for (const definition of settingDefinitions) {
      try {
        settingsApi.addSetting(definition);
      } catch (error) {
        console.warn("[ess_checkpoint_browser_loader] addSetting failed for", definition.id, error);
      }
    }
  }

  const migrateLegacyValue = (targetId, legacyId) => {
    const settingsValues = settingsApi?.settingsValues;
    const currentValue = settingsValues?.[targetId];
    const legacyValue = settingsValues?.[legacyId] ?? localStorage.getItem(legacyId);
    if ((currentValue == null || currentValue === "") && legacyValue != null && legacyValue !== "") {
      try {
        if (typeof settingsApi?.setSettingValue === "function") {
          settingsApi.setSettingValue(targetId, legacyValue);
        }
      } catch (error) {
        console.warn("[ess_checkpoint_browser_loader] failed to migrate setting", legacyId, "->", targetId, error);
      }
      localStorage.setItem(targetId, String(legacyValue));
    }
  };

  migrateLegacyValue(SETTINGS.directory, SETTINGS.legacyDirectory);
  migrateLegacyValue(SETTINGS.apiKey, SETTINGS.legacyApiKey);

  const fallback = (id, defaultValue) => ({
    get value() {
      return localStorage.getItem(id) ?? defaultValue;
    },
    set value(next) {
      localStorage.setItem(id, next ?? "");
    },
  });

  const proxy = (id, defaultValue) => ({
    get value() {
      const settingsValues = app.ui?.settings?.settingsValues;
      if (settingsValues && Object.prototype.hasOwnProperty.call(settingsValues, id)) {
        const value = settingsValues[id];
        if (value != null && value !== "") {
          return value;
        }
      }
      if (id === SETTINGS.directory && settingsValues && Object.prototype.hasOwnProperty.call(settingsValues, SETTINGS.legacyDirectory)) {
        const value = settingsValues[SETTINGS.legacyDirectory];
        if (value != null && value !== "") {
          return value;
        }
      }
      if (id === SETTINGS.apiKey && settingsValues && Object.prototype.hasOwnProperty.call(settingsValues, SETTINGS.legacyApiKey)) {
        const value = settingsValues[SETTINGS.legacyApiKey];
        if (value != null && value !== "") {
          return value;
        }
      }
      if (id === SETTINGS.directory) {
        const legacy = localStorage.getItem(SETTINGS.legacyDirectory);
        if (legacy != null && legacy !== "") {
          return legacy;
        }
      }
      if (id === SETTINGS.apiKey) {
        const legacy = localStorage.getItem(SETTINGS.legacyApiKey);
        if (legacy != null && legacy !== "") {
          return legacy;
        }
      }
      return localStorage.getItem(id) ?? defaultValue;
    },
    set value(next) {
      const text = String(next ?? "");
      const settingsApi = app.ui?.settings;
      if (typeof settingsApi?.setSettingValue === "function") {
        settingsApi.setSettingValue(id, text);
      } else {
        localStorage.setItem(id, text);
      }
      localStorage.setItem(id, text);
      if (id === SETTINGS.directory) {
        localStorage.setItem(SETTINGS.legacyDirectory, text);
      }
      if (id === SETTINGS.apiKey) {
        localStorage.setItem(SETTINGS.legacyApiKey, text);
      }
    },
  });

  settingsRefs = {
    directory: proxy(SETTINGS.directory, "ess/checkpoints"),
    apiKey: proxy(SETTINGS.apiKey, ""),
  };
  settingsState.directory = String(settingsRefs.directory.value ?? "ess/checkpoints");
  settingsState.apiKey = String(settingsRefs.apiKey.value ?? "");
  return settingsRefs;
}

function refreshAttachedWidgets() {
  for (const state of attachedWidgetStates) {
    try {
      updateSummary(state);
    } catch (error) {
      console.warn("[ess_checkpoint_browser_loader] failed to refresh widget summary:", error);
    }
  }
}

async function fetchJson(path, options = {}) {
  const response = await app.api.fetchApi(path, options);
  if (!response?.ok) {
    const text = await response?.text?.().catch(() => "");
    throw new Error(text || `HTTP ${response?.status ?? "?"}`);
  }
  return response.json();
}

function collapseStorageWidget(widget) {
  if (!widget) return;
  widget.serialize = true;
  widget.computeSize = () => [0, 0];
  widget.draw = () => {};
  const element = widget.element || widget.inputEl;
  if (element?.style) {
    element.style.display = "none";
    element.style.height = "0";
    element.style.minHeight = "0";
    element.style.margin = "0";
    element.style.padding = "0";
    element.style.border = "0";
    element.style.overflow = "hidden";
  }
}

function readSettingsValues() {
  const refs = ensureSettings();
  return {
    checkpointDirectory: String(refs.directory?.value ?? settingsState.directory ?? "ess/checkpoints").trim() || "ess/checkpoints",
    apiKey: String(refs.apiKey?.value ?? settingsState.apiKey ?? "").trim(),
  };
}

function parseSelection(raw) {
  const text = String(raw ?? "").trim();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function stringifySelection(selection) {
  return selection ? JSON.stringify(selection) : "";
}

function createEl(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function buildRemotePreviewProxyUrl(rawUrl) {
  const params = new URLSearchParams({ url: rawUrl });
  const path = `/ess/checkpoints/remote_preview?${params.toString()}`;
  return app.api?.apiURL ? app.api.apiURL(path) : path;
}

function resolvePreviewUrl(url, previewType = "") {
  const raw = String(url || "").trim();
  if (!raw) return "";
  if (/^https?:/i.test(raw)) {
    return String(previewType || "").toLowerCase() === "video" ? raw : buildRemotePreviewProxyUrl(raw);
  }
  if (/^(data:|blob:)/i.test(raw)) return raw;
  if (app.api?.apiURL) return app.api.apiURL(raw);
  if (raw.startsWith("/api/")) return raw;
  if (raw.startsWith("/")) return `/api${raw}`;
  return raw;
}

function previewKind(preview) {
  const type = String(preview?.type || "").trim().toLowerCase();
  const resolvedUrl = resolvePreviewUrl(preview?.url || "", type);
  if (type === "video" || /\.(mp4|webm|mov)(\?|$)/i.test(resolvedUrl)) {
    return "video";
  }
  return "image";
}

function formatBytes(value) {
  const num = Number(value || 0);
  if (!Number.isFinite(num) || num <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = num;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size >= 100 ? size.toFixed(0) : size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatStateLabel(value) {
  if (value === true) return "yes";
  if (value === false) return "no";
  return "?";
}

function effectiveDownloadFolder() {
  const settings = readSettingsValues();
  return String(settings.checkpointDirectory || "ess/checkpoints").replace(/\/+/g, "/");
}

function writeCheckpointDirectory(value) {
  const text = String(value ?? "").trim() || "ess/checkpoints";
  const refs = ensureSettings();
  refs.directory.value = text;
  settingsState.directory = text;
  refreshAttachedWidgets();
  return text;
}

function writeApiKey(value) {
  const text = String(value ?? "").trim();
  const refs = ensureSettings();
  refs.apiKey.value = text;
  settingsState.apiKey = text;
  return text;
}

async function copyToClipboard(text) {
  const content = String(text || "").trim();
  if (!content) return false;
  if (navigator?.clipboard?.writeText) {
    await navigator.clipboard.writeText(content);
    return true;
  }
  const textarea = document.createElement("textarea");
  textarea.value = content;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  const succeeded = document.execCommand("copy");
  textarea.remove();
  return !!succeeded;
}

function examplePromptForVersion(version) {
  for (const image of version?.images || []) {
    const prompt = String(image?.meta?.prompt || "").trim();
    if (prompt) {
      return {
        prompt,
        negativePrompt: String(image?.meta?.negativePrompt || image?.meta?.negative_prompt || "").trim(),
      };
    }
  }
  return null;
}

function createCopyableTextBlock(title, value, buttonLabel, onCopy) {
  const text = String(value || "").trim();
  if (!text) return null;
  const block = createEl("div", "ess-ckpt-card-textblock");
  const head = createEl("div", "ess-ckpt-card-texthead");
  head.appendChild(createEl("div", "ess-ckpt-card-texttitle", title));
  const copyButton = createEl("button", "", buttonLabel);
  copyButton.addEventListener("click", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    await onCopy(text);
  });
  head.appendChild(copyButton);
  block.appendChild(head);
  block.appendChild(createEl("div", "ess-ckpt-card-textcontent", text));
  return block;
}

function parseTagList(value) {
  return String(value || "")
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function updateTagSuggestions(datalist, items) {
  datalist.innerHTML = "";
  for (const item of items || []) {
    const option = document.createElement("option");
    option.value = item.name;
    option.label = item.modelCount ? `${item.name} (${item.modelCount})` : item.name;
    datalist.appendChild(option);
  }
}

function metadataEntriesFromPreview(preview) {
  const meta = (preview && typeof preview.meta === "object" && preview.meta) ? preview.meta : {};
  const entries = [];
  const seen = new Set();
  const push = (label, value) => {
    const text = Array.isArray(value) ? value.join(", ") : String(value ?? "").trim();
    if (!text) return;
    const key = String(label || "").trim().toLowerCase();
    if (key) seen.add(key);
    entries.push([label, text]);
  };

  push("Prompt", meta.prompt);
  push("Negative", meta.negativePrompt || meta.negative_prompt);
  push("Seed", meta.seed);
  push("Steps", meta.steps);
  push("CFG", meta.cfgScale || meta.cfg || meta["cfg scale"]);
  push("Sampler", meta.sampler);
  push("Scheduler", meta.scheduler);
  push("Clip skip", meta.clipSkip || meta.clip_skip);
  push("Model", meta.Model || meta.model);
  push("Model hash", meta.ModelHash || meta.modelHash || meta.model_hash);
  push("VAE", meta.VAE || meta.vae);
  push("Size", meta.Size || (meta.width && meta.height ? `${meta.width}x${meta.height}` : ""));
  push("Denoise", meta.denoise);
  push("Hires upscaler", meta.hiresUpscaler || meta.hires_upscaler);
  push("Hires steps", meta.hiresSteps || meta.hires_steps);
  push("Hires upscale", meta.hiresUpscale || meta.hires_upscale);
  push("Generation process", meta.generationProcess || meta.generation_process);

  for (const [key, value] of Object.entries(meta)) {
    const normalizedKey = String(key || "").trim().toLowerCase();
    if (seen.has(normalizedKey)) continue;
    if (["prompt", "negativeprompt", "negative_prompt", "resources"].includes(normalizedKey)) continue;
    if (typeof value === "object") continue;
    push(key, value);
  }
  return entries;
}

function formatTimestamp(value) {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric) || numeric <= 0) return "";
  const timestamp = numeric > 9999999999 ? numeric : numeric * 1000;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function previewPromptInfo(preview) {
  const meta = (preview && typeof preview.meta === "object" && preview.meta) ? preview.meta : {};
  return {
    prompt: String(meta.prompt || "").trim(),
    negativePrompt: String(meta.negativePrompt || meta.negative_prompt || "").trim(),
    resources: meta.resources ?? null,
  };
}

function stringifyPreviewResource(value) {
  if (value == null) return "";
  if (typeof value === "string") return value.trim();
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function imageKeyParts(preview) {
  const url = String(preview?.url || "").trim();
  return {
    id: String(preview?.id || "").trim(),
    hash: String(preview?.hash || "").trim(),
    postId: String(preview?.postId || "").trim(),
    urlPath: url ? url.split("?")[0].toLowerCase() : "",
    dimensions: preview?.width && preview?.height ? `${preview.width}x${preview.height}` : "",
  };
}

function mergePreviewPayload(target, source) {
  if (!target || !source) return target;
  const merged = { ...target, ...source };
  if ((target?.meta && typeof target.meta === "object") || (source?.meta && typeof source.meta === "object")) {
    merged.meta = {
      ...(target?.meta && typeof target.meta === "object" ? target.meta : {}),
      ...(source?.meta && typeof source.meta === "object" ? source.meta : {}),
    };
  }
  return merged;
}

function matchAndMergePreviewEntries(entries, fetchedItems) {
  let changed = false;
  for (let index = 0; index < entries.length; index += 1) {
    const current = entries[index];
    const currentKeys = imageKeyParts(current);
    let matched = null;
    for (const candidate of fetchedItems || []) {
      const candidateKeys = imageKeyParts(candidate);
      if (currentKeys.id && candidateKeys.id && currentKeys.id === candidateKeys.id) {
        matched = candidate;
      } else if (currentKeys.hash && candidateKeys.hash && currentKeys.hash === candidateKeys.hash) {
        matched = candidate;
      } else if (currentKeys.postId && candidateKeys.postId && currentKeys.postId === candidateKeys.postId) {
        matched = candidate;
      } else if (currentKeys.urlPath && candidateKeys.urlPath && currentKeys.urlPath === candidateKeys.urlPath) {
        matched = candidate;
      } else if (currentKeys.dimensions && candidateKeys.dimensions && currentKeys.dimensions === candidateKeys.dimensions) {
        matched = candidate;
      }
      if (matched) break;
    }
    if (!matched) continue;
    const merged = mergePreviewPayload(current, matched);
    if (JSON.stringify(merged) !== JSON.stringify(current)) {
      entries[index] = merged;
      changed = true;
    }
  }
  return changed;
}

function openPreviewDialog(previews, startIndex = 0, context = {}) {
  const entries = Array.isArray(previews) ? previews.filter(Boolean) : [];
  if (!entries.length) return;

  let currentIndex = Math.max(0, Math.min(entries.length - 1, Number(startIndex || 0)));
  const overlay = createEl("div", "ess-ckpt-browser");
  const dialog = createEl("div", "ess-ckpt-preview-dialog");
  overlay.appendChild(dialog);

  const stage = createEl("div", "ess-ckpt-preview-stage");
  const side = createEl("div", "ess-ckpt-preview-side");
  const metaWrap = createEl("div", "ess-ckpt-preview-meta");
  side.appendChild(metaWrap);
  dialog.appendChild(stage);
  dialog.appendChild(side);

  const close = () => overlay.remove();
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  const render = () => {
    const preview = entries[currentIndex];
    const kind = previewKind(preview);
    const url = resolvePreviewUrl(preview?.url || "", preview?.type || "");
    stage.innerHTML = "";
    metaWrap.innerHTML = "";

    if (kind === "video") {
      const video = document.createElement("video");
      video.src = url;
      video.controls = true;
      video.autoplay = true;
      video.loop = true;
      video.playsInline = true;
      stage.appendChild(video);
    } else {
      const img = document.createElement("img");
      img.src = url;
      img.alt = "preview";
      stage.appendChild(img);
    }

    const head = createEl("div", "ess-ckpt-preview-actions");
    const counter = createEl("span", "ess-ckpt-pill", `${currentIndex + 1}/${entries.length}`);
    const prevButton = createEl("button", "", "Prev");
    const nextButton = createEl("button", "", "Next");
    const openButton = createEl("button", "", "Open");
    const closeButton = createEl("button", "", "Close");
    prevButton.addEventListener("click", () => {
      currentIndex = (currentIndex - 1 + entries.length) % entries.length;
      render();
    });
    nextButton.addEventListener("click", () => {
      currentIndex = (currentIndex + 1) % entries.length;
      render();
    });
    openButton.addEventListener("click", () => {
      window.open(url, "_blank", "noopener,noreferrer");
    });
    closeButton.addEventListener("click", close);
    head.appendChild(counter);
    head.appendChild(prevButton);
    head.appendChild(nextButton);
    head.appendChild(openButton);
    head.appendChild(closeButton);
    metaWrap.appendChild(head);

    const grid = createEl("div", "ess-ckpt-preview-grid");
    const pushGrid = (label, value) => {
      const text = Array.isArray(value) ? value.join(", ") : String(value ?? "").trim();
      if (!text) return;
      grid.appendChild(createEl("strong", "", label));
      grid.appendChild(createEl("span", "", text));
    };
    pushGrid("Model", context.item?.name);
    pushGrid("Author", context.item?.creator);
    pushGrid("Version", context.version?.name);
    pushGrid("Base Model", context.version?.baseModel);
    pushGrid("File", context.file?.name);
    pushGrid("Format", context.file?.format);
    pushGrid("Image NSFW", preview?.nsfw == null ? "" : String(preview.nsfw));
    pushGrid("NSFW Level", preview?.nsfwLevel);
    pushGrid("Image Size", preview?.width && preview?.height ? `${preview.width}x${preview.height}` : "");
    pushGrid("Posted by", preview?.username);
    pushGrid("Created", formatTimestamp(preview?.createdAt));
    pushGrid("Post ID", preview?.postId);
    pushGrid("Image Hash", preview?.hash);
    pushGrid("MIME Type", preview?.mimeType);
    pushGrid("Tags", context.item?.tags);
    pushGrid("Trigger Words", context.version?.trainedWords);
    pushGrid("Prompt available", preview?.meta?.prompt ? "yes" : "");
    pushGrid("Additional models", Array.isArray(preview?.meta?.resources) ? String(preview.meta.resources.length) : "");
    for (const [label, value] of metadataEntriesFromPreview(preview)) {
      pushGrid(label, value);
    }
    if (grid.childNodes.length) {
      metaWrap.appendChild(grid);
    }

    const trainedWords = (context.version?.trainedWords || []).map((entry) => String(entry || "").trim()).filter(Boolean);
    if (trainedWords.length) {
      const block = createCopyableTextBlock("Trigger Words", trainedWords.join(", "), "Copy", async (text) => {
        await copyToClipboard(text);
      });
      if (block) {
        block.classList.add("ess-ckpt-preview-text");
        metaWrap.appendChild(block);
      }
    }

    const promptInfo = previewPromptInfo(preview);
    const prompt = promptInfo.prompt;
    if (prompt) {
      const block = createCopyableTextBlock("Prompt", prompt, "Copy prompt", async (text) => {
        await copyToClipboard(text);
      });
      if (block) {
        block.classList.add("ess-ckpt-preview-text");
        metaWrap.appendChild(block);
      }
    }
    const negative = promptInfo.negativePrompt;
    if (negative) {
      const block = createCopyableTextBlock("Negative Prompt", negative, "Copy negative", async (text) => {
        await copyToClipboard(text);
      });
      if (block) {
        block.classList.add("ess-ckpt-preview-text");
        metaWrap.appendChild(block);
      }
    }
    const resources = promptInfo.resources;
    if (resources) {
      const text = stringifyPreviewResource(resources);
      const block = createCopyableTextBlock("Additional Models", text, "Copy", async (content) => {
        await copyToClipboard(content);
      });
      if (block) {
        block.classList.add("ess-ckpt-preview-text");
        metaWrap.appendChild(block);
      }
    }

    if (!prompt && !negative && !resources && context.version?.id) {
      const note = createEl("div", "ess-ckpt-status", "Prompt metadata is not in the current preview payload. The browser will try to load richer image metadata for this version.");
      metaWrap.appendChild(note);
    }

    const rawMeta = preview && typeof preview.meta === "object" && preview.meta ? preview.meta : null;
    if (rawMeta) {
      const text = stringifyPreviewResource(rawMeta);
      const block = createCopyableTextBlock("Raw Image Metadata", text, "Copy JSON", async (content) => {
        await copyToClipboard(content);
      });
      if (block) {
        block.classList.add("ess-ckpt-preview-text");
        metaWrap.appendChild(block);
      }
    }
  };

  render();
  document.body.appendChild(overlay);

  const versionId = String(context.version?.id || "").trim();
  if (versionId) {
    loadCivitaiVersionImages(versionId)
      .then((items) => {
        if (matchAndMergePreviewEntries(entries, items)) {
          render();
        }
      })
      .catch(() => {});
  }
}

function itemSearchHaystack(item) {
  return [
    item?.name,
    item?.creator,
    item?.description,
    item?.browserRelativePath,
    item?.browserFolder,
    ...(item?.tags || []),
    ...(item?.baseModels || []),
    ...((item?.versions || []).flatMap((version) => [
      version?.name,
      version?.baseModel,
      ...(version?.trainedWords || []),
      ...((version?.files || []).map((file) => file?.name)),
    ])),
  ]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function itemFormats(item) {
  const labels = new Set();
  for (const version of item?.versions || []) {
    for (const file of version?.files || []) {
      const text = String(file?.format || "").trim();
      if (text) labels.add(text);
    }
  }
  return [...labels].sort(compareText);
}

function collectFormats(items) {
  const labels = new Set();
  for (const item of items || []) {
    for (const format of itemFormats(item)) labels.add(format);
  }
  return [...labels].sort(compareText);
}

function componentStateForItem(item, key) {
  const values = [];
  for (const version of item?.versions || []) {
    for (const file of version?.files || []) {
      values.push(file?.[key]);
    }
  }
  if (values.some((value) => value === true)) return true;
  if (values.some((value) => value === false)) return false;
  return null;
}

function filterItems(items, filters, tab) {
  const queryTerms = String(filters?.query || "")
    .toLowerCase()
    .split(/\s+/)
    .map((term) => term.trim())
    .filter(Boolean);
  const authorTerm = String(filters?.author || "").trim().toLowerCase();
  const tagTerms = parseTagList(filters?.tag).map((entry) => entry.toLowerCase());
  const wantedBase = String(filters?.baseModel || "").trim();
  const wantedFormat = String(filters?.format || "").trim();
  const wantedClip = String(filters?.clip || "").trim();
  const wantedVae = String(filters?.vae || "").trim();
  const wantedNsfw = String(filters?.nsfw || "").trim();

  return (items || []).filter((item) => {
    const haystack = itemSearchHaystack(item);
    if (queryTerms.length && !queryTerms.every((term) => haystack.includes(term))) return false;
    if (authorTerm && !String(item?.creator || "").toLowerCase().includes(authorTerm)) return false;
    if (tagTerms.length) {
      const tags = (item?.tags || []).map((entry) => String(entry || "").toLowerCase());
      if (!tagTerms.every((term) => tags.some((entry) => entry.includes(term)))) return false;
    }
    if (wantedBase && !baseModelsForItem(item).includes(wantedBase)) return false;
    if (wantedFormat && !itemFormats(item).includes(wantedFormat)) return false;

    const hasClip = componentStateForItem(item, "hasClip");
    if (wantedClip === "yes" && hasClip !== true) return false;
    if (wantedClip === "no" && hasClip !== false) return false;

    const hasVae = componentStateForItem(item, "hasVae");
    if (wantedVae === "yes" && hasVae !== true) return false;
    if (wantedVae === "no" && hasVae !== false) return false;

    if (tab === "civitai") {
      if (wantedNsfw === "safe" && item?.nsfw === true) return false;
      if (wantedNsfw === "nsfw" && item?.nsfw !== true) return false;
    }
    return true;
  });
}

function sortItems(items, sortKey, tab) {
  const sorted = [...(items || [])];
  const byName = (left, right) => compareText(left?.name, right?.name);
  const newestValue = (item) => Number(item?.updatedAt || item?.createdAt || item?.modifiedAt || 0);
  const sizeValue = (item) => Math.max(0, ...((item?.versions || []).flatMap((version) => (version?.files || []).map((file) => Number(file?.sizeBytes || 0)))));
  const downloadsValue = (item) => Number(item?.downloadCount || 0);
  const ratingValue = (item) => Number(item?.rating || 0);

  switch (sortKey) {
    case "name_desc":
      sorted.sort((left, right) => byName(right, left));
      break;
    case "updated_desc":
      sorted.sort((left, right) => newestValue(right) - newestValue(left) || byName(left, right));
      break;
    case "updated_asc":
      sorted.sort((left, right) => newestValue(left) - newestValue(right) || byName(left, right));
      break;
    case "size_desc":
      sorted.sort((left, right) => sizeValue(right) - sizeValue(left) || byName(left, right));
      break;
    case "size_asc":
      sorted.sort((left, right) => sizeValue(left) - sizeValue(right) || byName(left, right));
      break;
    case "downloads_desc":
      sorted.sort((left, right) => downloadsValue(right) - downloadsValue(left) || byName(left, right));
      break;
    case "rating_desc":
      sorted.sort((left, right) => ratingValue(right) - ratingValue(left) || byName(left, right));
      break;
    default:
      if (tab === "local") {
        sorted.sort(byName);
      } else {
        sorted.sort((left, right) => newestValue(right) - newestValue(left) || byName(left, right));
      }
      break;
  }
  return sorted;
}

function syncSelectOptions(select, options, currentValue, placeholder) {
  const normalized = [{ value: "", label: placeholder }, ...(options || []).map((entry) => {
    if (typeof entry === "string") return { value: entry, label: entry };
    return entry;
  })];
  const existing = Array.from(select.options).map((option) => `${option.value}::${option.textContent}`);
  const target = normalized.map((option) => `${option.value}::${option.label}`);
  const sameShape = existing.length === target.length && existing.every((value, index) => value === target[index]);
  if (!sameShape) {
    select.innerHTML = "";
    for (const optionData of normalized) {
      const option = document.createElement("option");
      option.value = optionData.value;
      option.textContent = optionData.label;
      select.appendChild(option);
    }
  }
  select.value = normalized.some((entry) => entry.value === currentValue) ? currentValue : "";
  return select.value;
}

function buildSettingsForm() {
  const wrap = createEl("div", "ess-ckpt-settings-form");

  const directoryField = createEl("div", "ess-ckpt-settings-field");
  const directoryLabel = document.createElement("label");
  directoryLabel.textContent = "Checkpoint directory";
  const directoryHelp = createEl("small", "", "Directory under ComfyUI models used for ESS model downloads. Files are stored as Family / Author / Name / Version.");
  const directoryInput = document.createElement("input");
  directoryInput.type = "text";
  directoryInput.placeholder = "ess/checkpoints";
  directoryInput.value = effectiveDownloadFolder();
  directoryField.appendChild(directoryLabel);
  directoryField.appendChild(directoryHelp);
  directoryField.appendChild(directoryInput);
  wrap.appendChild(directoryField);

  const apiKeyField = createEl("div", "ess-ckpt-settings-field");
  const apiKeyLabel = document.createElement("label");
  apiKeyLabel.textContent = "CivitAI API key";
  const apiKeyHelp = createEl("small", "", "Optional. Used for CivitAI search, downloads, and metadata enrichment.");
  const apiKeyInput = document.createElement("input");
  apiKeyInput.type = "password";
  apiKeyInput.placeholder = "Bearer ... or raw API key";
  apiKeyInput.value = readSettingsValues().apiKey;
  apiKeyField.appendChild(apiKeyLabel);
  apiKeyField.appendChild(apiKeyHelp);
  apiKeyField.appendChild(apiKeyInput);
  wrap.appendChild(apiKeyField);

  const actions = createEl("div", "ess-ckpt-settings-actions");
  const saveButton = createEl("button", "", "Save");
  const status = createEl("div", "ess-ckpt-status", "");
  actions.appendChild(saveButton);
  actions.appendChild(status);
  wrap.appendChild(actions);

  async function save() {
    const directory = writeCheckpointDirectory(directoryInput.value);
    const apiKey = writeApiKey(apiKeyInput.value);
    directoryInput.value = directory;
    apiKeyInput.value = apiKey;
    status.textContent = "Saved.";
  }

  saveButton.addEventListener("click", () => {
    save().catch((error) => {
      status.textContent = String(error?.message || error || "Failed to save settings.");
    });
  });
  directoryInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    saveButton.click();
  });
  apiKeyInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    saveButton.click();
  });

  return {
    element: wrap,
    directoryInput,
    apiKeyInput,
    status,
  };
}

function openSettingsDialog(onSaved) {
  ensureStyles();
  const overlay = createEl("div", "ess-ckpt-browser");
  const dialog = createEl("div", "ess-ckpt-settings-dialog");
  overlay.appendChild(dialog);

  dialog.appendChild(createEl("div", "ess-ckpt-settings-title", "ESS Model Loader Settings"));
  dialog.appendChild(createEl("div", "ess-ckpt-settings-subtitle", "Configure the checkpoint folder and CivitAI API key used by the ESS checkpoint browser."));
  const form = buildSettingsForm();
  dialog.appendChild(form.element);

  const footer = createEl("div", "ess-ckpt-settings-actions");
  const closeButton = createEl("button", "", "Close");
  footer.appendChild(closeButton);
  dialog.appendChild(footer);

  const close = () => overlay.remove();
  closeButton.addEventListener("click", close);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  const saveButton = form.element.querySelector("button");
  if (saveButton) {
    saveButton.addEventListener("click", () => {
      if (typeof onSaved === "function") onSaved();
    });
  }

  document.body.appendChild(overlay);
}

let settingsPageObserverStarted = false;

function ensureSettingsPagePanel() {
  if (settingsPageObserverStarted) return;
  settingsPageObserverStarted = true;

  const inject = () => {
    const headings = Array.from(document.querySelectorAll("h1,h2,h3,h4,div,span"));
    const modelLoaderHeading = headings.find((el) => String(el.textContent || "").trim() === "Model Loader");
    if (!modelLoaderHeading) return;

    const pageRoot = modelLoaderHeading.parentElement?.parentElement || modelLoaderHeading.parentElement;
    if (!pageRoot || pageRoot.querySelector(".ess-ckpt-settings-panel")) return;

    const panel = createEl("div", "ess-ckpt-settings-panel");
    panel.appendChild(createEl("div", "ess-ckpt-settings-title", "ESS Model Loader Settings"));
    panel.appendChild(createEl("div", "ess-ckpt-settings-subtitle", "Use this panel if the built-in settings controls do not render correctly in your Desktop build."));
    panel.appendChild(buildSettingsForm().element);
    pageRoot.appendChild(panel);
  };

  const observer = new MutationObserver(() => {
    inject();
  });
  observer.observe(document.body, { childList: true, subtree: true });
  inject();
}

function updateSummary(state) {
  if (!state?.summaryName) return;
  const selection = parseSelection(state.storageWidget?.value);
  const hasSelection = !!selection;
  state.summaryName.textContent = hasSelection ? selection.model_name || selection.file_name || "Selected checkpoint" : "No checkpoint selected";
  state.summaryPath.textContent = hasSelection ? (selection.local_path || selection.relative_path || "") : "Open the browser to choose a local model or download one from CivitAI.";
  if (state.summaryConfig) {
    state.summaryConfig.textContent = `Download folder: ${effectiveDownloadFolder()}`;
  }
  state.summaryMeta.innerHTML = "";
  if (!hasSelection) return;
  const pills = [
    `Format: ${selection.format || "Unknown"}`,
    `CLIP: ${formatStateLabel(selection.has_clip)}`,
    `VAE: ${formatStateLabel(selection.has_vae)}`,
  ];
  for (const label of pills) {
    state.summaryMeta.appendChild(createEl("span", "ess-ckpt-pill", label));
  }
}

function setSelection(state, selection) {
  state.storageWidget.value = stringifySelection(selection);
  if (typeof state.storageWidget.callback === "function") {
    state.storageWidget.callback(state.storageWidget.value, app.canvas, state.node, app.canvas?.graph_mouse || null, null);
  }
  updateSummary(state);
}

async function loadLocalCatalog(query = "") {
  const settings = readSettingsValues();
  const params = new URLSearchParams({
    checkpoint_directory: settings.checkpointDirectory,
    query,
  });
  return fetchJson(`/ess/checkpoints/local?${params.toString()}`, { method: "GET" });
}

async function loadCivitaiCatalog(query = "", cursor = "", filters = {}) {
  const settings = readSettingsValues();
  const params = new URLSearchParams({
    query,
    cursor,
    limit: "20",
    api_key: settings.apiKey,
  });
  if (filters.author) params.set("username", filters.author);
  if (filters.tag) params.set("tag", filters.tag);
  if (filters.baseModel) params.set("base_model", filters.baseModel);
  if (filters.sort) params.set("sort", filters.sort);
  if (filters.period) params.set("period", filters.period);
  if (filters.nsfw) params.set("nsfw", filters.nsfw);
  return fetchJson(`/ess/checkpoints/civitai/search?${params.toString()}`, { method: "GET" });
}

async function startDownload(model, version, file) {
  const settings = readSettingsValues();
  return fetchJson("/ess/checkpoints/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      checkpoint_directory: settings.checkpointDirectory,
      api_key: settings.apiKey,
      model,
      version,
      file,
    }),
  });
}

async function getDownloadStatus(jobId) {
  const params = new URLSearchParams({ job_id: jobId });
  return fetchJson(`/ess/checkpoints/download_status?${params.toString()}`, { method: "GET" });
}

async function loadCivitaiTags(query = "") {
  const settings = readSettingsValues();
  const params = new URLSearchParams({
    query,
    limit: "20",
    api_key: settings.apiKey,
  });
  return fetchJson(`/ess/checkpoints/civitai/tags?${params.toString()}`, { method: "GET" });
}

async function loadCivitaiVersionImages(modelVersionId) {
  const versionId = String(modelVersionId || "").trim();
  if (!versionId) return [];
  if (previewMetadataCache.has(versionId)) {
    return previewMetadataCache.get(versionId);
  }
  const settings = readSettingsValues();
  const params = new URLSearchParams({
    model_version_id: versionId,
    limit: "100",
    api_key: settings.apiKey,
  });
  const promise = fetchJson(`/ess/checkpoints/civitai/images?${params.toString()}`, { method: "GET" })
    .then((payload) => payload.items || [])
    .catch((error) => {
      previewMetadataCache.delete(versionId);
      throw error;
    });
  previewMetadataCache.set(versionId, promise);
  return promise;
}

async function enrichLocalModel(localPath) {
  const settings = readSettingsValues();
  const payload = {
    checkpoint_directory: settings.checkpointDirectory,
    api_key: settings.apiKey,
    local_path: localPath,
  };
  try {
    return await fetchJson("/ess/checkpoints/enrich_local", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    const message = String(error?.message || error || "");
    if (!message.includes("405")) {
      throw error;
    }
    const params = new URLSearchParams(payload);
    return fetchJson(`/ess/checkpoints/enrich_local?${params.toString()}`, { method: "GET" });
  }
}

function renderTags(container, tags) {
  container.innerHTML = "";
  for (const tag of (tags || []).slice(0, 8)) {
    container.appendChild(createEl("span", "ess-ckpt-pill", String(tag)));
  }
}

function compareText(a, b) {
  return String(a || "").localeCompare(String(b || ""), undefined, { sensitivity: "base" });
}

function baseModelsForItem(item) {
  const labels = new Set();
  for (const label of item?.baseModels || []) {
    const text = String(label || "").trim();
    if (text) labels.add(text);
  }
  for (const version of item?.versions || []) {
    const text = String(version?.baseModel || "").trim();
    if (text) labels.add(text);
  }
  return [...labels].sort(compareText);
}

function collectBaseModels(items) {
  const labels = new Set();
  for (const item of items || []) {
    for (const label of baseModelsForItem(item)) {
      labels.add(label);
    }
  }
  return [...labels].sort(compareText);
}

function filterItemsByBaseModel(items, baseModel) {
  const wanted = String(baseModel || "").trim();
  if (!wanted) return items || [];
  return (items || []).filter((item) => baseModelsForItem(item).includes(wanted));
}

function syncBaseModelOptions(select, labels, currentValue) {
  const normalized = ["", ...(labels || []).map((entry) => String(entry || "").trim()).filter(Boolean)];
  const existing = Array.from(select.options).map((option) => option.value);
  const sameShape = existing.length === normalized.length && existing.every((value, index) => value === normalized[index]);
  if (!sameShape) {
    select.innerHTML = "";
    for (const value of normalized) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value || "All base models";
      select.appendChild(option);
    }
  }
  select.value = normalized.includes(currentValue) ? currentValue : "";
  return select.value;
}

function formatLocalSectionTitle(item) {
  const sourceLabel = String(item?.sourceLabel || "Local");
  const sourceRootName = String(item?.sourceRootName || "").trim();
  const browserFolder = String(item?.browserFolder || "").trim();
  const parts = [sourceLabel];
  if (sourceLabel !== "ESS" && sourceRootName && sourceRootName.toLowerCase() !== sourceLabel.toLowerCase()) {
    parts.push(sourceRootName);
  }
  parts.push(browserFolder || "(root)");
  return parts.join(" / ");
}

function groupLocalItems(items) {
  const groups = new Map();
  for (const item of items || []) {
    const key = [
      String(item?.sourceKind || ""),
      String(item?.sourceRootPath || item?.sourceRootName || ""),
      String(item?.browserFolder || ""),
    ].join("|");
    if (!groups.has(key)) {
      groups.set(key, {
        title: formatLocalSectionTitle(item),
        items: [],
      });
    }
    groups.get(key).items.push(item);
  }
  return [...groups.values()]
    .sort((left, right) => compareText(left.title, right.title))
    .map((group) => ({
      ...group,
      items: [...group.items].sort((left, right) => compareText(left?.name, right?.name)),
    }));
}

function createPreview(previews, currentIndex, onChangeIndex, onOpenPreview) {
  const wrap = createEl("div", "ess-ckpt-card-preview");
  const entries = Array.isArray(previews) ? previews.filter(Boolean) : [];
  const safeIndex = entries.length ? Math.max(0, Math.min(entries.length - 1, Number(currentIndex || 0))) : 0;
  const preview = entries[safeIndex] || null;
  const previewType = String(preview?.type || "").trim().toLowerCase();
  const resolvedUrl = resolvePreviewUrl(preview?.url || "", previewType);
  if (resolvedUrl) {
    if (previewKind(preview) === "video") {
      const video = document.createElement("video");
      video.src = resolvedUrl;
      video.muted = true;
      video.autoplay = true;
      video.loop = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (typeof onOpenPreview === "function") onOpenPreview(safeIndex);
      });
      wrap.appendChild(video);
    } else {
      const img = document.createElement("img");
      img.src = resolvedUrl;
      img.loading = "lazy";
      img.alt = "preview";
      img.addEventListener("error", () => {
        wrap.innerHTML = "";
        wrap.appendChild(createEl("div", "ess-ckpt-status", "Preview unavailable"));
      });
      img.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (typeof onOpenPreview === "function") onOpenPreview(safeIndex);
      });
      wrap.appendChild(img);
    }
  } else {
    wrap.appendChild(createEl("div", "ess-ckpt-status", "No preview"));
  }
  if (entries.length > 1) {
    const nav = createEl("div", "ess-ckpt-card-preview-nav");
    const prevButton = createEl("button", "", "Prev");
    const counter = createEl("span", "", `${safeIndex + 1}/${entries.length}`);
    const nextButton = createEl("button", "", "Next");
    prevButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      onChangeIndex((safeIndex - 1 + entries.length) % entries.length);
    });
    nextButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      onChangeIndex((safeIndex + 1) % entries.length);
    });
    nav.appendChild(prevButton);
    nav.appendChild(counter);
    nav.appendChild(nextButton);
    wrap.appendChild(nav);
  }
  return wrap;
}

function selectedVersionForItem(item, versionId) {
  const versions = item?.versions || [];
  return versions.find((entry) => String(entry.id) === String(versionId)) || versions[0] || null;
}

function selectedFileForVersion(version, fileId) {
  const files = version?.files || [];
  return files.find((entry) => String(entry.id) === String(fileId)) || files[0] || null;
}

function createCard(item, { tab, onUse, onDownload, onEnrichLocal, downloads }) {
  const card = createEl("article", "ess-ckpt-card");
  const initialVersion = selectedVersionForItem(item, item?.versions?.[0]?.id);
  let currentVersion = initialVersion;
  let currentFile = selectedFileForVersion(initialVersion, initialVersion?.files?.[0]?.id);
  let previewIndex = 0;
  const openCurrentPreview = (index = previewIndex) => {
    openPreviewDialog(currentVersion?.images || [], index, {
      item,
      version: currentVersion,
      file: currentFile,
    });
  };
  let previewNode = createPreview(currentVersion?.images || [], previewIndex, (nextIndex) => {
    previewIndex = nextIndex;
    refreshCard();
  }, openCurrentPreview);
  card.appendChild(previewNode);

  const body = createEl("div", "ess-ckpt-card-body");
  card.appendChild(body);

  body.appendChild(createEl("div", "ess-ckpt-card-title", item.name || "Untitled checkpoint"));
  body.appendChild(createEl("div", "ess-ckpt-card-subtitle", item.creator ? `by ${item.creator}` : item.source === "local" ? "Local checkpoint" : "CivitAI checkpoint"));
  if (item.source === "local" && (item.browserRelativePath || currentFile?.browserRelativePath)) {
    body.appendChild(createEl("div", "ess-ckpt-card-path", item.browserRelativePath || currentFile?.browserRelativePath || ""));
  }
  body.appendChild(createEl("div", "ess-ckpt-card-desc", item.description || (item.source === "local" ? "Local checkpoint" : "No description available.")));

  const tagsWrap = createEl("div", "ess-ckpt-card-tags");
  renderTags(tagsWrap, item.tags);
  body.appendChild(tagsWrap);

  const versionRow = createEl("div", "ess-ckpt-card-row");
  const versionSelect = document.createElement("select");
  for (const version of item.versions || []) {
    const option = document.createElement("option");
    option.value = String(version.id);
    option.textContent = version.name || "Version";
    versionSelect.appendChild(option);
  }
  versionRow.appendChild(versionSelect);
  body.appendChild(versionRow);

  const fileRow = createEl("div", "ess-ckpt-card-row");
  const fileSelect = document.createElement("select");
  fileRow.appendChild(fileSelect);
  body.appendChild(fileRow);

  const infoRow = createEl("div", "ess-ckpt-widget-meta");
  body.appendChild(infoRow);

  const statusLine = createEl("div", "ess-ckpt-status", "");
  body.appendChild(statusLine);

  const progressWrap = createEl("div", "ess-ckpt-progress");
  const progressBar = document.createElement("span");
  progressBar.style.width = "0%";
  progressWrap.appendChild(progressBar);
  body.appendChild(progressWrap);

  const actions = createEl("div", "ess-ckpt-card-actions");
  const actionButtons = createEl("div", "ess-ckpt-card-action-buttons");
  const primaryButton = createEl("button", "", tab === "local" ? "Use Model" : "Download");
  actionButtons.appendChild(primaryButton);
  let enrichButton = null;
  if (tab === "local") {
    enrichButton = createEl("button", "", "Fill from CivitAI");
    actionButtons.appendChild(enrichButton);
  }
  const secondary = createEl("div", "ess-ckpt-status", "");
  actions.appendChild(actionButtons);
  actions.appendChild(secondary);
  body.appendChild(actions);

  const extraWrap = createEl("div", "ess-ckpt-card-extra");
  body.appendChild(extraWrap);

  function refreshFileSelect() {
    fileSelect.innerHTML = "";
    currentVersion = selectedVersionForItem(item, versionSelect.value);
    for (const file of currentVersion?.files || []) {
      const option = document.createElement("option");
      option.value = String(file.id);
      option.textContent = `${file.name} (${file.format || "Unknown"})`;
      fileSelect.appendChild(option);
    }
    currentFile = selectedFileForVersion(currentVersion, fileSelect.value);
  }

  function refreshCard() {
    currentVersion = selectedVersionForItem(item, versionSelect.value);
    currentFile = selectedFileForVersion(currentVersion, fileSelect.value);
    const images = currentVersion?.images || [];
    if (previewIndex >= images.length) previewIndex = 0;
    const nextPreview = createPreview(images, previewIndex, (nextIndex) => {
      previewIndex = nextIndex;
      refreshCard();
    }, openCurrentPreview);
    card.replaceChild(nextPreview, previewNode);
    previewNode = nextPreview;
    infoRow.innerHTML = "";
    const pills = [
      currentVersion?.baseModel ? `Base: ${currentVersion.baseModel}` : null,
      currentFile?.format ? `Format: ${currentFile.format}` : null,
      `CLIP: ${formatStateLabel(currentFile?.hasClip)}`,
      `VAE: ${formatStateLabel(currentFile?.hasVae)}`,
      currentFile?.sizeBytes ? formatBytes(currentFile.sizeBytes) : null,
    ].filter(Boolean);
    for (const label of pills) {
      infoRow.appendChild(createEl("span", "ess-ckpt-pill", label));
    }
    secondary.textContent = (currentVersion?.trainedWords || []).slice(0, 4).join(", ");
    extraWrap.innerHTML = "";
    const trainedWords = (currentVersion?.trainedWords || []).map((entry) => String(entry || "").trim()).filter(Boolean);
    if (trainedWords.length) {
      const keywordBlock = createCopyableTextBlock("Trigger Words", trainedWords.join(", "), "Copy", async (text) => {
        await copyToClipboard(text);
        statusLine.textContent = "Trigger words copied.";
      });
      if (keywordBlock) extraWrap.appendChild(keywordBlock);
    }
    const promptInfo = examplePromptForVersion(currentVersion);
    if (promptInfo?.prompt) {
      const promptBlock = createCopyableTextBlock("Example prompt", promptInfo.prompt, "Copy prompt", async (text) => {
        await copyToClipboard(text);
        statusLine.textContent = "Prompt copied.";
      });
      if (promptBlock) extraWrap.appendChild(promptBlock);
    }
    if (promptInfo?.negativePrompt) {
      const negativeBlock = createCopyableTextBlock("Negative prompt", promptInfo.negativePrompt, "Copy negative", async (text) => {
        await copyToClipboard(text);
        statusLine.textContent = "Negative prompt copied.";
      });
      if (negativeBlock) extraWrap.appendChild(negativeBlock);
    }
    const downloadState = downloads.get(String(currentFile?.id || ""));
    if (downloadState) {
      statusLine.textContent = downloadState.message || downloadState.stage || "";
      progressBar.style.width = `${Math.max(0, Math.min(100, (downloadState.progress || 0) * 100))}%`;
      primaryButton.disabled = downloadState.status === "running" || downloadState.status === "queued";
      if (enrichButton) {
        enrichButton.disabled = downloadState.status === "running" || downloadState.status === "queued";
      }
    } else {
      statusLine.textContent = "";
      progressBar.style.width = "0%";
      primaryButton.disabled = false;
      if (enrichButton) {
        enrichButton.disabled = false;
      }
    }
  }

  versionSelect.addEventListener("change", () => {
    refreshFileSelect();
    refreshCard();
  });
  fileSelect.addEventListener("change", () => {
    currentFile = selectedFileForVersion(currentVersion, fileSelect.value);
    refreshCard();
  });

  primaryButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (tab === "local") {
      if (!currentVersion || !currentFile) return;
      onUse(item, currentVersion, currentFile);
    } else {
      onDownload(item, currentVersion, currentFile, statusLine, progressBar, primaryButton);
    }
  });
  if (enrichButton) {
    enrichButton.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!currentFile?.localPath) return;
      await onEnrichLocal(item, currentVersion, currentFile, statusLine, progressBar, enrichButton, primaryButton);
    });
  }
  refreshFileSelect();
  refreshCard();
  return card;
}

async function openBrowser(state) {
  ensureStyles();
  if (state.overlay) {
    state.overlay.style.display = "flex";
    return;
  }

  const overlay = createEl("div", "ess-ckpt-browser");
  const shell = createEl("div", "ess-ckpt-browser-shell");
  overlay.appendChild(shell);
  document.body.appendChild(overlay);
  state.overlay = overlay;

  const head = createEl("div", "ess-ckpt-browser-head");
  const titleWrap = document.createElement("div");
  titleWrap.appendChild(createEl("h3", "", "ESS Checkpoint Browser"));
  titleWrap.appendChild(createEl("small", "", "Browse local checkpoints or search CivitAI and download directly into the ESS library."));
  const headActions = createEl("div", "ess-ckpt-browser-head-actions");
  const settingsButton = createEl("button", "", "Model Loader Settings");
  const closeButton = createEl("button", "", "Close");
  headActions.appendChild(settingsButton);
  headActions.appendChild(closeButton);
  head.appendChild(titleWrap);
  head.appendChild(headActions);
  shell.appendChild(head);

  const toolbar = createEl("div", "ess-ckpt-browser-toolbar");
  const tabs = createEl("div", "ess-ckpt-browser-tabs");
  const localTab = createEl("button", "active", "Local");
  const civitaiTab = createEl("button", "", "CivitAI");
  tabs.appendChild(localTab);
  tabs.appendChild(civitaiTab);
  toolbar.appendChild(tabs);

  const baseModelSelect = document.createElement("select");
  baseModelSelect.className = "ess-ckpt-toolbar-select";
  toolbar.appendChild(baseModelSelect);

  const formatSelect = document.createElement("select");
  formatSelect.className = "ess-ckpt-toolbar-select";
  toolbar.appendChild(formatSelect);

  const clipSelect = document.createElement("select");
  clipSelect.className = "ess-ckpt-toolbar-compact";
  toolbar.appendChild(clipSelect);

  const vaeSelect = document.createElement("select");
  vaeSelect.className = "ess-ckpt-toolbar-compact";
  toolbar.appendChild(vaeSelect);

  const authorInput = document.createElement("input");
  authorInput.type = "text";
  authorInput.placeholder = "Author";
  authorInput.className = "ess-ckpt-toolbar-compact";
  toolbar.appendChild(authorInput);

  const tagInput = document.createElement("input");
  tagInput.type = "text";
  tagInput.placeholder = "Tags (comma separated)";
  tagInput.className = "ess-ckpt-toolbar-compact";
  const tagDatalistId = `ess-ckpt-tags-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  tagInput.setAttribute("list", tagDatalistId);
  toolbar.appendChild(tagInput);
  const tagDatalist = document.createElement("datalist");
  tagDatalist.id = tagDatalistId;
  shell.appendChild(tagDatalist);

  const nsfwSelect = document.createElement("select");
  nsfwSelect.className = "ess-ckpt-toolbar-compact";
  toolbar.appendChild(nsfwSelect);

  const sortSelect = document.createElement("select");
  sortSelect.className = "ess-ckpt-toolbar-select";
  toolbar.appendChild(sortSelect);

  const periodSelect = document.createElement("select");
  periodSelect.className = "ess-ckpt-toolbar-compact";
  toolbar.appendChild(periodSelect);

  const searchInput = document.createElement("input");
  searchInput.type = "text";
  searchInput.placeholder = "Search name, description, keywords, tags...";
  searchInput.className = "ess-ckpt-toolbar-search";
  toolbar.appendChild(searchInput);

  const refreshButton = createEl("button", "ess-ckpt-toolbar-button", "Search");
  toolbar.appendChild(refreshButton);

  const loadMoreButton = createEl("button", "ess-ckpt-toolbar-button", "Load More");
  loadMoreButton.style.display = "none";
  toolbar.appendChild(loadMoreButton);

  const rootInfo = createEl("div", "ess-ckpt-status ess-ckpt-toolbar-status", "");
  toolbar.appendChild(rootInfo);
  shell.appendChild(toolbar);

  const body = createEl("div", "ess-ckpt-browser-body");
  shell.appendChild(body);

  state.browser = {
    tab: "local",
    localItems: [],
    civitaiItems: [],
    nextCursor: "",
    downloads: new Map(),
    filters: {
      query: "",
      baseModel: "",
      format: "",
      clip: "",
      vae: "",
      author: "",
      tag: "",
      nsfw: "",
      sort: "",
      period: "",
    },
    baseModelSelect,
    formatSelect,
    clipSelect,
    vaeSelect,
    authorInput,
    tagInput,
    tagDatalist,
    nsfwSelect,
    sortSelect,
    periodSelect,
    localAvailableBaseModels: [],
    civitaiAvailableBaseModels: [],
    availableFormats: [],
    tagSuggestionsTimer: null,
    searchInput,
    body,
    loadMoreButton,
    rootInfo,
  };

  const close = () => {
    overlay.style.display = "none";
  };
  closeButton.addEventListener("click", close);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && overlay.style.display !== "none") close();
  });

  async function pollDownload(jobId, cardKey) {
    while (true) {
      const status = await getDownloadStatus(jobId);
      state.browser.downloads.set(cardKey, status);
      renderCurrentTab();
      if (status.status === "completed") {
        setSelection(state, status.result);
        await fetchLocal();
        return;
      }
      if (status.status === "failed") {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 1200));
    }
  }

  function cardKeyFor(file) {
    return String(file?.id || "");
  }

  function onUse(item, version, file) {
    setSelection(state, {
      model_name: item.name,
      version_name: version?.name,
      file_name: file.name,
      format: file.format,
      local_path: file.localPath,
      relative_path: file.relativePath,
      managed: file.managed,
      has_clip: file.hasClip,
      has_vae: file.hasVae,
      description: item.description,
      tags: item.tags || [],
      trained_words: version?.trainedWords || [],
    });
    close();
  }

  async function onDownload(item, version, file) {
    const key = cardKeyFor(file);
    state.browser.downloads.set(key, {
      status: "queued",
      progress: 0,
      message: "Queued for download.",
    });
    renderCurrentTab();
    try {
      const started = await startDownload(item, version, file);
      await pollDownload(started.job_id, key);
    } catch (error) {
      state.browser.downloads.set(key, {
        status: "failed",
        progress: 0,
        message: String(error?.message || error || "Download failed."),
      });
      renderCurrentTab();
    }
  }

  async function onEnrichLocal(item, version, file, statusLine, progressBar, enrichButton, primaryButton) {
    if (!file?.localPath) return;
    const key = `enrich:${file.localPath}`;
    state.browser.downloads.set(key, {
      status: "running",
      progress: 0,
      message: "Fetching metadata from CivitAI...",
    });
    enrichButton.disabled = true;
    primaryButton.disabled = true;
    statusLine.textContent = "Fetching metadata from CivitAI...";
    progressBar.style.width = "15%";
    try {
      const result = await enrichLocalModel(file.localPath);
      statusLine.textContent = "Local metadata updated.";
      progressBar.style.width = "100%";
      if (result?.selection) {
        setSelection(state, result.selection);
      }
      await fetchLocal();
    } catch (error) {
      statusLine.textContent = String(error?.message || error || "Failed to enrich metadata.");
      progressBar.style.width = "0%";
    } finally {
      state.browser.downloads.delete(key);
      enrichButton.disabled = false;
      primaryButton.disabled = false;
    }
  }

  function createCardGrid(items) {
    const grid = createEl("div", "ess-ckpt-card-grid");
    for (const item of items) {
      grid.appendChild(createCard(item, {
        tab: state.browser.tab,
        onUse,
        onDownload,
        onEnrichLocal,
        downloads: state.browser.downloads,
      }));
    }
    return grid;
  }

  function renderItems(items) {
    body.innerHTML = "";
    if (!items?.length) {
      body.appendChild(createEl("div", "ess-ckpt-browser-empty", state.browser.tab === "local" ? "No local checkpoints matched the current search." : "No CivitAI checkpoints matched the current search."));
      return;
    }
    body.appendChild(createCardGrid(items));
  }

  function syncFilterControls() {
    const isLocal = state.browser.tab === "local";
    const baseLabels = isLocal
      ? state.browser.localAvailableBaseModels
      : [...new Set([...KNOWN_BASE_MODELS, ...state.browser.civitaiAvailableBaseModels])].sort(compareText);
    const sourceItems = isLocal ? state.browser.localItems : state.browser.civitaiItems;
    const formats = collectFormats(sourceItems);
    state.browser.availableFormats = formats;

    state.browser.filters.baseModel = syncSelectOptions(baseModelSelect, baseLabels, state.browser.filters.baseModel, "All base models");
    state.browser.filters.format = syncSelectOptions(formatSelect, formats, state.browser.filters.format, "All formats");
    state.browser.filters.clip = syncSelectOptions(clipSelect, [
      { value: "yes", label: "CLIP: yes" },
      { value: "no", label: "CLIP: no" },
    ], state.browser.filters.clip, "Any CLIP");
    state.browser.filters.vae = syncSelectOptions(vaeSelect, [
      { value: "yes", label: "VAE: yes" },
      { value: "no", label: "VAE: no" },
    ], state.browser.filters.vae, "Any VAE");
    state.browser.filters.nsfw = syncSelectOptions(nsfwSelect, [
      { value: "safe", label: "Safe models only" },
      { value: "true", label: "Allow NSFW models" },
      { value: "nsfw", label: "NSFW-flagged models only" },
    ], state.browser.filters.nsfw, "Any model NSFW");
    nsfwSelect.style.display = isLocal ? "none" : "";

    const sortOptions = isLocal
      ? [
          { value: "name_desc", label: "Name Z-A" },
          { value: "updated_desc", label: "Newest files" },
          { value: "updated_asc", label: "Oldest files" },
          { value: "size_desc", label: "Largest files" },
          { value: "size_asc", label: "Smallest files" },
        ]
      : [
          { value: "downloads_desc", label: "Most downloaded" },
          { value: "rating_desc", label: "Highest rated" },
          { value: "name_desc", label: "Name Z-A" },
        ];
    state.browser.filters.sort = syncSelectOptions(sortSelect, sortOptions, state.browser.filters.sort, isLocal ? "Name A-Z" : "Newest releases");

    const periodOptions = [
      { value: "AllTime", label: "All time" },
      { value: "Year", label: "Year" },
      { value: "Month", label: "Month" },
      { value: "Week", label: "Week" },
      { value: "Day", label: "Day" },
    ];
    state.browser.filters.period = syncSelectOptions(periodSelect, periodOptions, state.browser.filters.period, "All time");
    periodSelect.style.display = isLocal || !["downloads_desc", "rating_desc", ""].includes(state.browser.filters.sort) ? "none" : "";
  }

  function renderCurrentTab() {
    rootInfo.textContent = `Download folder: ${effectiveDownloadFolder()}`;
    const isLocal = state.browser.tab === "local";
    localTab.classList.toggle("active", isLocal);
    civitaiTab.classList.toggle("active", !isLocal);
    loadMoreButton.style.display = !isLocal && state.browser.nextCursor ? "" : "none";
    syncFilterControls();
    const sourceItems = isLocal ? state.browser.localItems : state.browser.civitaiItems;
    const filtered = filterItems(sourceItems, state.browser.filters, state.browser.tab);
    renderItems(sortItems(filtered, state.browser.filters.sort, state.browser.tab));
  }

  async function fetchLocal() {
    rootInfo.textContent = `Download folder: ${effectiveDownloadFolder()}`;
    const payload = await loadLocalCatalog(searchInput.value);
    state.browser.localItems = payload.items || [];
    state.browser.localAvailableBaseModels = payload.availableBaseModels || collectBaseModels(state.browser.localItems);
    renderCurrentTab();
  }

  async function fetchCivitai(append = false) {
    rootInfo.textContent = `Download folder: ${effectiveDownloadFolder()}`;
    const payload = await loadCivitaiCatalog(searchInput.value, append ? state.browser.nextCursor : "", {
      author: state.browser.filters.author,
      tag: parseTagList(state.browser.filters.tag)[0] || "",
      baseModel: state.browser.filters.baseModel,
      sort: state.browser.filters.sort === "downloads_desc" ? "Most Downloaded" : state.browser.filters.sort === "rating_desc" ? "Highest Rated" : "Newest",
      period: state.browser.filters.period || "AllTime",
      nsfw: state.browser.filters.nsfw === "safe" ? "false" : state.browser.filters.nsfw === "true" ? "true" : "",
    });
    if (append) {
      state.browser.civitaiItems = [...state.browser.civitaiItems, ...(payload.items || [])];
      state.browser.civitaiAvailableBaseModels = collectBaseModels(state.browser.civitaiItems);
    } else {
      state.browser.civitaiItems = payload.items || [];
      state.browser.civitaiAvailableBaseModels = payload.availableBaseModels || collectBaseModels(state.browser.civitaiItems);
    }
    state.browser.nextCursor = payload.nextCursor || "";
    renderCurrentTab();
  }

  refreshButton.addEventListener("click", () => {
    state.browser.filters.query = searchInput.value.trim();
    state.browser.filters.author = authorInput.value.trim();
    state.browser.filters.tag = tagInput.value.trim();
    if (state.browser.tab === "local") {
      fetchLocal().catch((error) => {
        body.innerHTML = "";
        body.appendChild(createEl("div", "ess-ckpt-browser-empty", String(error?.message || error || "Failed to load local checkpoints.")));
      });
    } else {
      fetchCivitai(false).catch((error) => {
        body.innerHTML = "";
        body.appendChild(createEl("div", "ess-ckpt-browser-empty", String(error?.message || error || "Failed to search CivitAI.")));
      });
    }
  });
  loadMoreButton.addEventListener("click", () => {
    fetchCivitai(true).catch((error) => {
      body.innerHTML = "";
      body.appendChild(createEl("div", "ess-ckpt-browser-empty", String(error?.message || error || "Failed to load more models.")));
    });
  });
  searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") refreshButton.click();
  });
  authorInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") refreshButton.click();
  });
  tagInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") refreshButton.click();
  });
  tagInput.addEventListener("input", () => {
    state.browser.filters.tag = tagInput.value.trim();
    if (state.browser.tagSuggestionsTimer) {
      clearTimeout(state.browser.tagSuggestionsTimer);
    }
    state.browser.tagSuggestionsTimer = window.setTimeout(async () => {
      if (state.browser.tab !== "civitai") return;
      const lastToken = parseTagList(tagInput.value).at(-1) || String(tagInput.value || "").trim();
      if (!lastToken) {
        updateTagSuggestions(tagDatalist, []);
        return;
      }
      try {
        const payload = await loadCivitaiTags(lastToken);
        updateTagSuggestions(tagDatalist, payload.items || []);
      } catch {
        updateTagSuggestions(tagDatalist, []);
      }
    }, 180);
  });
  settingsButton.addEventListener("click", () => {
    openSettingsDialog(() => {
      rootInfo.textContent = `Download folder: ${effectiveDownloadFolder()}`;
      if (state.browser.tab === "local") {
        fetchLocal().catch((error) => {
          body.innerHTML = "";
          body.appendChild(createEl("div", "ess-ckpt-browser-empty", String(error?.message || error || "Failed to load local checkpoints.")));
        });
      }
    });
  });
  baseModelSelect.addEventListener("change", () => {
    state.browser.filters.baseModel = baseModelSelect.value;
    renderCurrentTab();
  });
  formatSelect.addEventListener("change", () => {
    state.browser.filters.format = formatSelect.value;
    renderCurrentTab();
  });
  clipSelect.addEventListener("change", () => {
    state.browser.filters.clip = clipSelect.value;
    renderCurrentTab();
  });
  vaeSelect.addEventListener("change", () => {
    state.browser.filters.vae = vaeSelect.value;
    renderCurrentTab();
  });
  nsfwSelect.addEventListener("change", () => {
    state.browser.filters.nsfw = nsfwSelect.value;
    if (state.browser.tab === "civitai") {
      fetchCivitai(false).catch(() => renderCurrentTab());
    } else {
      renderCurrentTab();
    }
  });
  sortSelect.addEventListener("change", () => {
    state.browser.filters.sort = sortSelect.value;
    if (state.browser.tab === "civitai") {
      fetchCivitai(false).catch(() => renderCurrentTab());
    } else {
      renderCurrentTab();
    }
  });
  periodSelect.addEventListener("change", () => {
    state.browser.filters.period = periodSelect.value;
    if (state.browser.tab === "civitai") {
      fetchCivitai(false).catch(() => renderCurrentTab());
    } else {
      renderCurrentTab();
    }
  });
  localTab.addEventListener("click", () => {
    state.browser.tab = "local";
    fetchLocal().catch(() => renderCurrentTab());
  });
  civitaiTab.addEventListener("click", () => {
    state.browser.tab = "civitai";
    fetchCivitai(false).catch(() => renderCurrentTab());
  });

  await fetchLocal().catch((error) => {
    body.innerHTML = "";
    body.appendChild(createEl("div", "ess-ckpt-browser-empty", String(error?.message || error || "Failed to load checkpoints.")));
  });
}

function attachCheckpointWidget(node, storageWidget) {
  if (!node || !storageWidget || storageWidget.__essCheckpointBrowserAttached || typeof node.addDOMWidget !== "function") return;

  ensureStyles();
  ensureSettings();
  collapseStorageWidget(storageWidget);

  const container = createEl("div", "ess-ckpt-widget");
  const card = createEl("div", "ess-ckpt-widget-card");
  container.appendChild(card);

  const top = createEl("div", "ess-ckpt-widget-top");
  top.appendChild(createEl("div", "ess-ckpt-widget-title", "Checkpoint Loader"));
  const actions = createEl("div", "ess-ckpt-widget-actions");
  const openButton = createEl("button", "", "Open Browser");
  const clearButton = createEl("button", "", "Clear");
  actions.appendChild(openButton);
  actions.appendChild(clearButton);
  top.appendChild(actions);
  card.appendChild(top);

  const summaryName = createEl("div", "ess-ckpt-widget-name", "No checkpoint selected");
  const summaryMeta = createEl("div", "ess-ckpt-widget-meta");
  const summaryPath = createEl("div", "ess-ckpt-widget-path", "Open the browser to choose a local model or download one from CivitAI.");
  const summaryConfig = createEl("div", "ess-ckpt-widget-config", `Download folder: ${effectiveDownloadFolder()}`);
  card.appendChild(summaryName);
  card.appendChild(summaryMeta);
  card.appendChild(summaryPath);
  card.appendChild(summaryConfig);

  const domWidget = node.addDOMWidget("checkpoint_browser", "ess_checkpoint_browser", container, {});
  domWidget.computeSize = (width) => [Math.max(width || 320, 320), 168];

  const state = {
    node,
    storageWidget,
    summaryName,
    summaryMeta,
    summaryPath,
    summaryConfig,
    overlay: null,
    browser: null,
  };

  openButton.addEventListener("click", () => {
    openBrowser(state).catch((error) => {
      summaryPath.textContent = String(error?.message || error || "Failed to open checkpoint browser.");
    });
  });
  clearButton.addEventListener("click", () => setSelection(state, null));

  storageWidget.__essCheckpointBrowserAttached = true;
  node.__essCheckpointBrowserState = state;
  attachedWidgetStates.add(state);
  updateSummary(state);
}

function ensureCheckpointWidgets(node) {
  if (!node?.widgets) return;
  for (const widget of node.widgets) {
    if (!widget?.__essCheckpointBrowserConfig) continue;
    attachCheckpointWidget(node, widget);
  }
}

app.registerExtension({
  name: "ess_checkpoint_browser_loader",
  settings: EXTENSION_SETTINGS,
  init() {
    ensureStyles();
    ensureSettings();
    ensureSettingsPagePanel();
  },
  async getCustomWidgets() {
    return {
      ESS_CHECKPOINT_BROWSER(node, inputName, inputData) {
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
          ? String(inputData.value ?? "")
          : String(config.value ?? config.default ?? "");
        const storage = node.addWidget("text", inputName, initialValue, () => {}, {
          multiline: true,
          placeholder: config.placeholder,
          minHeight: config.minHeight,
          maxHeight: config.maxHeight,
          height: config.height,
          ess_checkpoint_browser: true,
        });
        storage.value = initialValue;
        storage.__essCheckpointBrowserConfig = { ...config };
        return { widget: storage, minHeight: 0 };
      },
    };
  },
  async beforeRegisterNodeDef(nodeType) {
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      try {
        ensureCheckpointWidgets(this);
      } catch (error) {
        console.error("[ess_checkpoint_browser_loader] onConfigure failed:", error);
      }
      return result;
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      try {
        ensureCheckpointWidgets(this);
      } catch (error) {
        console.error("[ess_checkpoint_browser_loader] onNodeCreated failed:", error);
      }
      return result;
    };
  },
});
