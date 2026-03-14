import { app } from "../../scripts/app.js";

const STYLE_ID = "ess-pose-mesh-editor-style";

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.ess-pose-widget {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  color: #d6d6d6;
  font-size: 12px;
}
.ess-pose-widget button {
  background: #2c313a;
  color: #f2f2f2;
  border: 1px solid #3f4754;
  border-radius: 4px;
  padding: 6px 10px;
  cursor: pointer;
}
.ess-pose-widget button:hover {
  background: #3a414c;
}
.ess-pose-overlay {
  position: fixed;
  inset: 0;
  background: rgba(5, 6, 8, 0.9);
  color: #e6e6e6;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}
.ess-pose-topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-bottom: 1px solid #2f3540;
  background: #0f1115;
}
.ess-pose-topbar input[type=file] {
  color: #e6e6e6;
}
.ess-pose-main {
  display: grid;
  grid-template-columns: 300px 1fr 360px;
  gap: 12px;
  padding: 12px;
  height: calc(100% - 54px);
  box-sizing: border-box;
}
.ess-pose-panel {
  background: #11161d;
  border: 1px solid #2a303b;
  border-radius: 8px;
  padding: 10px;
  box-sizing: border-box;
  overflow: hidden;
}
.ess-pose-panel h4 {
  margin: 0 0 8px;
  font-size: 13px;
  letter-spacing: 0.3px;
  color: #9fc5ff;
}
.ess-pose-bone-list {
  height: 460px;
  overflow: auto;
  background: #0b0e14;
  border: 1px solid #1f2530;
  border-radius: 6px;
}
.ess-pose-bone-row {
  display: flex;
  align-items: center;
  gap: 6px;
  border-bottom: 1px solid #1a1f28;
}
.ess-pose-bone-toggle {
  width: 22px;
  border: none;
  background: transparent;
  color: #d6d6d6;
  cursor: pointer;
  padding: 0;
  font-size: 11px;
}
.ess-pose-bone-btn {
  flex: 1;
  border: none;
  background: transparent;
  color: #d6d6d6;
  text-align: left;
  padding: 8px 6px;
  font-size: 12px;
  cursor: pointer;
  border-radius: 4px;
}
.ess-pose-bone-btn:hover {
  background: #1a2030;
}
.ess-pose-bone-btn.active {
  background: #32507a;
  color: #f5f9ff;
  box-shadow: inset 0 0 0 1px #5e87ba;
}
.ess-pose-controls label {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  font-size: 12px;
  margin-bottom: 6px;
  min-width: 0;
  overflow: hidden;
}
.ess-pose-controls input[type=range] {
  flex: 1;
}
.ess-pose-controls input[type=number] {
  width: 76px;
  background: #0f1115;
  color: #e6e6e6;
  border: 1px solid #2a303b;
  border-radius: 4px;
  padding: 3px 6px;
}
.ess-pose-viewport {
  width: 100%;
  height: 100%;
  background: #080b10;
  border-radius: 8px;
  position: relative;
  overflow: hidden;
}
.ess-pose-hint {
  font-size: 11px;
  color: #9ba3b0;
  margin-top: 6px;
}
.ess-pose-badge {
  background: #253043;
  color: #e6e6e6;
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 11px;
}
.ess-pose-right {
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
  overflow-x: hidden;
  min-width: 0;
}
.ess-pose-console {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 4px;
  border-top: 1px solid #2a303b;
  padding-top: 8px;
  background: #11161d;
}
.ess-pose-console textarea {
  width: 100%;
  min-height: 100px;
  max-height: 180px;
  resize: vertical;
  background: #0f1115;
  color: #e6e6e6;
  border: 1px solid #2a303b;
  box-sizing: border-box;
}
.ess-pose-footer {
  display: flex;
  gap: 10px;
  align-items: center;
}
.ess-pose-footer button.save {
  background: #2563eb;
  border-color: #1d4ed8;
}
.ess-pose-footer button.save:hover {
  background: #1e40af;
}
.ess-pose-footer button.secondary {
  background: #2f3540;
}
`;
  document.head.appendChild(style);
}

let threeBundlePromise = null;
const sessionStore = new WeakMap();

function createDefaultCameraSlots() {
  const base = {
    fov: 45,
    position: [0.0, 1.45, 2.75],
    target: [0, 1.0, 0],
    near: 0.001,
    far: 3000,
    resolution: [1024, 768],
  };
  return [0].map((idx) => ({
    ...base,
    name: `Camera`,
  }));
}

function normalizeCameraSlot(slot, fallbackName = "Cam") {
  const src = slot && typeof slot === "object" ? slot : {};
  const safeVec3 = (arr, dflt) => (
    Array.isArray(arr) && arr.length === 3 && arr.every((v) => Number.isFinite(Number(v)))
      ? [Number(arr[0]), Number(arr[1]), Number(arr[2])]
      : dflt.slice()
  );
  const safeRes = (arr) => {
    if (!Array.isArray(arr) || arr.length !== 2) return [1024, 768];
    const w = Math.max(64, Math.min(8192, Math.round(Number(arr[0]) || 1024)));
    const h = Math.max(64, Math.min(8192, Math.round(Number(arr[1]) || 768)));
    return [w, h];
  };
  return {
    name: String(src.name || fallbackName),
    fov: Number.isFinite(Number(src.fov)) ? Number(src.fov) : 45,
    position: safeVec3(src.position, [0.0, 1.45, 2.75]),
    target: safeVec3(src.target, [0, 1.0, 0]),
    near: Number.isFinite(Number(src.near)) && Number(src.near) > 0 ? Number(src.near) : 0.001,
    far: Number.isFinite(Number(src.far)) && Number(src.far) > 0 ? Number(src.far) : 3000,
    resolution: safeRes(src.resolution),
  };
}

function normalizeCameraSlots(slots) {
  const defaults = createDefaultCameraSlots();
  const out = [];
  for (let i = 0; i < 1; i += 1) {
    const src = Array.isArray(slots) && slots[i] ? slots[i] : defaults[i];
    out.push(normalizeCameraSlot(src, "Camera"));
  }
  return out;
}

function createEmptySession() {
  return {
    renderer: null,
    scene: null,
    camera: null,
    orbit: null,
    transform: null,
    renderLoopHandle: null,
    characters: [],
    activeCharacterId: null,
    groundPlane: null,
    groundGrid: null,
    originAxes: null,
    originDot: null,
    cameraSlots: createDefaultCameraSlots(),
    activeCameraIndex: 0,
    logLines: ["[log] editor initialized"],
  };
}

function getSessionForNode(node) {
  if (!node || typeof node !== "object") {
    return createEmptySession();
  }
  if (!sessionStore.has(node)) {
    sessionStore.set(node, createEmptySession());
  }
  return sessionStore.get(node);
}

const OPENPOSE_NAMES = [
  "Nose",
  "Neck",
  "RShoulder",
  "RElbow",
  "RWrist",
  "LShoulder",
  "LElbow",
  "LWrist",
  "RHip",
  "RKnee",
  "RAnkle",
  "LHip",
  "LKnee",
  "LAnkle",
  "REye",
  "LEye",
  "REar",
  "LEar",
];

const OPENPOSE_EDGES = [
  [1, 2], [1, 5],
  [2, 3], [3, 4],
  [5, 6], [6, 7],
  [1, 8], [8, 9], [9, 10],
  [1, 11], [11, 12], [12, 13],
  [1, 0], [0, 14], [14, 16],
  [0, 15], [15, 17],
];

// OpenPose COCO-18 palette (from OpenPose source poseParameters.cpp)
const OPENPOSE_COLORS = [
  "#ff0055", "#ff0000", "#ff5500", "#ffaa00", "#ffff00", "#aaff00",
  "#55ff00", "#00ff00", "#00ff55", "#00ffaa", "#00ffff", "#00aaff",
  "#0055ff", "#0000ff", "#ff00aa", "#aa00ff", "#ff00ff", "#5500ff",
];

const OPENPOSE_TO_BONE = {
  // Keep Nose strict; if absent we synthesize from neck/head.
  Nose: ["mixamorigNose", "Nose"],
  Neck: ["mixamorigNeck", "Neck"],
  // Prefer the limb chain itself over clavicle joints for Mixamo-style rigs.
  RShoulder: ["mixamorigRightArm", "RightArm", "mixamorigRightShoulder", "RightShoulder"],
  RElbow: ["mixamorigRightForeArm", "RightForeArm", "mixamorigRightArm", "RightArm"],
  RWrist: ["mixamorigRightHand", "RightHand", "mixamorigRightForeArm", "RightForeArm"],
  LShoulder: ["mixamorigLeftArm", "LeftArm", "mixamorigLeftShoulder", "LeftShoulder"],
  LElbow: ["mixamorigLeftForeArm", "LeftForeArm", "mixamorigLeftArm", "LeftArm"],
  LWrist: ["mixamorigLeftHand", "LeftHand", "mixamorigLeftForeArm", "LeftForeArm"],
  RHip: ["mixamorigRightUpLeg", "RightUpLeg"],
  RKnee: ["mixamorigRightLeg", "RightLeg"],
  RAnkle: ["mixamorigRightFoot", "RightFoot"],
  LHip: ["mixamorigLeftUpLeg", "LeftUpLeg"],
  LKnee: ["mixamorigLeftLeg", "LeftLeg"],
  LAnkle: ["mixamorigLeftFoot", "LeftFoot"],
  // If these are missing in the rig, we synthesize them from head/neck.
  REye: ["mixamorigRightEye", "RightEye"],
  LEye: ["mixamorigLeftEye", "LeftEye"],
  REar: ["mixamorigRightEar", "RightEar"],
  LEar: ["mixamorigLeftEar", "LeftEar"],
};
async function loadThreeBundle() {
  if (threeBundlePromise) return threeBundlePromise;
  const localBase = "./vendor/three";
  const cdnBase = "https://esm.sh/three@0.160.0";

  const loadBundle = async (base, isCdn = false) => {
    const suffix = isCdn ? "?deps=three@0.160.0" : "";
    return Promise.all([
      import(isCdn ? base : `${base}/build/three.module.js`),
      import(isCdn ? `${base}/examples/jsm/controls/OrbitControls.js${suffix}` : `${base}/examples/jsm/controls/OrbitControls.js`),
      import(isCdn ? `${base}/examples/jsm/controls/TransformControls.js${suffix}` : `${base}/examples/jsm/controls/TransformControls.js`),
      import(isCdn ? `${base}/examples/jsm/loaders/GLTFLoader.js${suffix}` : `${base}/examples/jsm/loaders/GLTFLoader.js`),
      import(isCdn ? `${base}/examples/jsm/loaders/FBXLoader.js${suffix}` : `${base}/examples/jsm/loaders/FBXLoader.js`),
      import(isCdn ? `${base}/examples/jsm/loaders/OBJLoader.js${suffix}` : `${base}/examples/jsm/loaders/OBJLoader.js`),
      import(isCdn ? `${base}/examples/jsm/libs/fflate.module.js${suffix}` : `${base}/examples/jsm/libs/fflate.module.js`),
    ]).then(([THREE, Orbit, Transform, GLTF, FBX, OBJ, FFLATE]) => ({
      THREE,
      OrbitControls: Orbit.OrbitControls,
      TransformControls: Transform.TransformControls,
      GLTFLoader: GLTF.GLTFLoader,
      FBXLoader: FBX.FBXLoader,
      OBJLoader: OBJ.OBJLoader,
      fflate: FFLATE,
    }));
  };

  threeBundlePromise = loadBundle(localBase).catch((err) => {
    console.error("Failed to load local three bundle, falling back to CDN", err);
    return loadBundle(cdnBase, true);
  });

  return threeBundlePromise;
}

function createNumberControl(labelText, min, max, step, onChange) {
  const wrap = document.createElement("label");
  wrap.textContent = labelText;
  const input = document.createElement("input");
  input.type = "number";
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = "0";
  input.addEventListener("input", () => onChange(Number(input.value) || 0));
  wrap.appendChild(input);
  return { wrap, input };
}

function createRangeNumberControl(labelText, min, max, step, onChange) {
  const wrap = document.createElement("label");
  wrap.style.flexDirection = "column";
  wrap.style.alignItems = "stretch";
  const row = document.createElement("div");
  row.style.display = "flex";
  row.style.alignItems = "center";
  row.style.gap = "6px";

  const label = document.createElement("span");
  label.textContent = labelText;
  label.style.minWidth = "56px";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  const initial = labelText.toLowerCase().includes("scale") ? 1 : 0;
  slider.value = String(initial);
  slider.style.flex = "1";

  const number = document.createElement("input");
  number.type = "number";
  number.min = String(min);
  number.max = String(max);
  number.step = String(step);
  number.value = String(initial);
  number.style.width = "64px";
  number.style.background = "#0f1115";
  number.style.color = "#e6e6e6";
  number.style.border = "1px solid #2a303b";
  number.style.borderRadius = "4px";
  number.style.padding = "3px 6px";

  const updateBoth = (val, emit = true) => {
    const clamped = Math.min(max, Math.max(min, val));
    slider.value = String(clamped);
    number.value = String(clamped);
    if (emit) {
      onChange(clamped);
    }
  };

  slider.addEventListener("input", () => updateBoth(Number(slider.value) || 0, true));
  number.addEventListener("input", () => updateBoth(Number(number.value) || 0, true));

  row.append(label, slider, number);
  wrap.appendChild(row);

  return {
    wrap,
    slider,
    number,
    setValue: (val) => updateBoth(val, false),
  };
}

function bufferToBase64(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function base64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

function toArrayBuffer(raw) {
  if (raw instanceof ArrayBuffer) {
    return raw;
  }
  if (ArrayBuffer.isView(raw)) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  }
  if (raw && raw.buffer instanceof ArrayBuffer) {
    const byteOffset = Number(raw.byteOffset || 0);
    const byteLength = Number(raw.byteLength || raw.buffer.byteLength || 0);
    return raw.buffer.slice(byteOffset, byteOffset + byteLength);
  }
  throw new Error("Unsupported binary payload type.");
}

function cloneArrayBuffer(buffer) {
  return buffer.slice(0);
}

function findBoneByAliases(map, aliases) {
  for (const name of aliases) {
    if (map.has(name)) return map.get(name);
    const aliasLower = String(name || "").toLowerCase();
    if (!aliasLower) continue;

    // case-insensitive exact match
    for (const [k, v] of map.entries()) {
      if (k.toLowerCase() === aliasLower) {
        return v;
      }
    }

    // fallback substring match only for sufficiently long aliases
    // to avoid false positives like "rear" -> "forearm".
    if (aliasLower.length < 6) {
      continue;
    }
    for (const [k, v] of map.entries()) {
      if (k.toLowerCase().includes(aliasLower)) {
        return v;
      }
    }
  }
  return null;
}

function radiansToDegrees(rad) {
  return rad * (180 / Math.PI);
}

function degreesToRadians(deg) {
  return deg * (Math.PI / 180);
}

function buildMirroredNameCandidates(name) {
  const source = String(name || "");
  if (!source) return [];
  const out = new Set();
  const addSwap = (a, b) => {
    if (source.includes(a)) out.add(source.replace(a, b));
    if (source.includes(b)) out.add(source.replace(b, a));
  };

  addSwap("Left", "Right");
  addSwap("left", "right");
  addSwap("LEFT", "RIGHT");
  addSwap("_L", "_R");
  addSwap("_l", "_r");
  addSwap(".L", ".R");
  addSwap(".l", ".r");
  addSwap("-L", "-R");
  addSwap("-l", "-r");
  addSwap("L_", "R_");
  addSwap("l_", "r_");

  out.delete(source);
  return Array.from(out);
}

function findBoneByNameCI(boneList, name) {
  const n = String(name || "");
  if (!n || !Array.isArray(boneList)) return null;
  const exact = boneList.find((b) => b?.name === n);
  if (exact) return exact;
  const lower = n.toLowerCase();
  return boneList.find((b) => String(b?.name || "").toLowerCase() === lower) || null;
}

function findMirroredBoneForName(boneList, boneName) {
  const candidates = buildMirroredNameCandidates(boneName);
  for (const candidate of candidates) {
    const found = findBoneByNameCI(boneList, candidate);
    if (found) return found;
  }
  return null;
}

function getBonePath(bone) {
  if (!bone) return "";
  const parts = [];
  let cur = bone;
  while (cur) {
    if (cur.isBone) {
      parts.push(cur.name || "(unnamed)");
    }
    cur = cur.parent;
  }
  return parts.reverse().join("/");
}

function raycastFromOrigin(threeRef, modelRoot, origin, direction, maxDistance, fallback = null) {
  const dir = direction?.clone?.();
  if (!threeRef || !modelRoot || !origin || !dir || dir.lengthSq() <= 1e-10) {
    return fallback ? fallback.clone() : origin.clone();
  }
  dir.normalize();
  const ray = new threeRef.Raycaster(origin.clone(), dir, 0, Math.max(0.05, maxDistance || 1.0));
  const hits = ray.intersectObject(modelRoot, true);
  for (const hit of hits) {
    if (hit?.object?.isMesh && hit?.point) {
      return hit.point.clone();
    }
  }
  return fallback ? fallback.clone() : origin.clone();
}

function synthesizeHeadLandmarks(threeRef, boneMap, camera, width, height, modelRoot = null, asWorld = false) {
  if (!threeRef || !boneMap || !camera) return null;
  const neck = findBoneByAliases(boneMap, OPENPOSE_TO_BONE.Neck || []);
  const head = findBoneByAliases(boneMap, ["mixamorigHead", "Head"]);
  if (!neck || !head) return null;
  const headTop = findBoneByAliases(boneMap, ["mixamorigHeadTop_End", "HeadTop_End", "HeadTop"]);
  const noseBone = findBoneByAliases(boneMap, ["mixamorigNose", "Nose"]);
  const rShoulderBone = findBoneByAliases(boneMap, OPENPOSE_TO_BONE.RShoulder || []);
  const lShoulderBone = findBoneByAliases(boneMap, OPENPOSE_TO_BONE.LShoulder || []);

  const neckPos = new threeRef.Vector3();
  const headPos = new threeRef.Vector3();
  neck.getWorldPosition(neckPos);
  head.getWorldPosition(headPos);
  const headTopPos = headTop ? new threeRef.Vector3() : null;
  if (headTopPos) {
    headTop.getWorldPosition(headTopPos);
  }

  const up = (headTopPos ? headTopPos.clone().sub(neckPos) : headPos.clone().sub(neckPos));
  const upLen = up.length();
  if (!Number.isFinite(upLen) || upLen <= 1e-6) return null;
  up.normalize();

  const worldQ = new threeRef.Quaternion();
  head.getWorldQuaternion(worldQ);
  const quatRight = new threeRef.Vector3(1, 0, 0).applyQuaternion(worldQ);
  const quatForward = new threeRef.Vector3(0, 0, 1).applyQuaternion(worldQ);

  let shoulderWidth = 0;
  let right = null;
  if (rShoulderBone && lShoulderBone) {
    const rPos = new threeRef.Vector3();
    const lPos = new threeRef.Vector3();
    rShoulderBone.getWorldPosition(rPos);
    lShoulderBone.getWorldPosition(lPos);
    shoulderWidth = rPos.distanceTo(lPos);
    right = rPos.clone().sub(lPos);
    // Remove vertical component so shoulder axis remains horizontal.
    right.addScaledVector(up, -right.dot(up));
    if (right.lengthSq() > 1e-10) {
      right.normalize();
    } else {
      right = null;
    }
  }

  if (!right) {
    right = quatRight.clone();
    right.addScaledVector(up, -right.dot(up));
    if (right.lengthSq() <= 1e-10) {
      right.set(1, 0, 0);
    }
    right.normalize();
  }

  let forward = new threeRef.Vector3().crossVectors(right, up);
  if (forward.lengthSq() <= 1e-10) {
    forward = quatForward.clone();
  }
  forward.addScaledVector(up, -forward.dot(up));
  if (forward.lengthSq() <= 1e-10) {
    forward.set(0, 0, 1);
  }
  forward.normalize();
  // Keep orientation consistent with head local forward when available.
  if (quatForward.lengthSq() > 1e-10 && forward.dot(quatForward) < 0) {
    forward.multiplyScalar(-1);
  }

  let headHeight = headTopPos ? neckPos.distanceTo(headTopPos) : neckPos.distanceTo(headPos) * 1.9;
  if (!Number.isFinite(headHeight) || headHeight <= 1e-6) {
    return null;
  }
  if (Number.isFinite(shoulderWidth) && shoulderWidth > 1e-6) {
    const minH = shoulderWidth * 0.24;
    const maxH = shoulderWidth * 0.50;
    headHeight = Math.min(maxH, Math.max(minH, headHeight));
  } else {
    headHeight = Math.max(headHeight, neckPos.distanceTo(headPos) * 1.8);
  }
  headHeight = Math.max(headHeight, upLen * 1.3);

  const headCenter = headTopPos
    ? neckPos.clone().lerp(headTopPos, 0.52)
    : neckPos.clone().add(up.clone().multiplyScalar(headHeight * 0.52));
  const faceAnchor = headCenter.clone()
    .add(forward.clone().multiplyScalar(0.10 * headHeight))
    .add(up.clone().multiplyScalar(0.02 * headHeight));

  let noseWorld = null;
  if (noseBone) {
    noseWorld = new threeRef.Vector3();
    noseBone.getWorldPosition(noseWorld);
    // Align synthetic forward sign with explicit nose when present.
    if (noseWorld.clone().sub(faceAnchor).dot(forward) < 0) {
      forward.multiplyScalar(-1);
    }
  }

  const noseFallback = faceAnchor.clone().add(forward.clone().multiplyScalar(0.20 * headHeight));
  const noseOrigin = headCenter.clone()
    .add(forward.clone().multiplyScalar(0.74 * headHeight))
    .add(up.clone().multiplyScalar(0.02 * headHeight));
  const nosePoint = noseWorld || raycastFromOrigin(
    threeRef,
    modelRoot,
    noseOrigin,
    forward.clone().multiplyScalar(-1),
    headHeight * 2.0,
    noseFallback,
  );

  const eyeFront = 0.66 * headHeight;
  const eyeUp = 0.11 * headHeight;
  const eyeSide = 0.16 * headHeight;
  const reyeFallback = headCenter.clone()
    .add(right.clone().multiplyScalar(eyeSide))
    .add(up.clone().multiplyScalar(0.08 * headHeight))
    .add(forward.clone().multiplyScalar(0.14 * headHeight));
  const leyeFallback = headCenter.clone()
    .add(right.clone().multiplyScalar(-eyeSide))
    .add(up.clone().multiplyScalar(0.08 * headHeight))
    .add(forward.clone().multiplyScalar(0.14 * headHeight));
  const reyeOrigin = headCenter.clone()
    .add(right.clone().multiplyScalar(eyeSide))
    .add(up.clone().multiplyScalar(eyeUp))
    .add(forward.clone().multiplyScalar(eyeFront));
  const leyeOrigin = headCenter.clone()
    .add(right.clone().multiplyScalar(-eyeSide))
    .add(up.clone().multiplyScalar(eyeUp))
    .add(forward.clone().multiplyScalar(eyeFront));
  const reyePoint = raycastFromOrigin(
    threeRef,
    modelRoot,
    reyeOrigin,
    forward.clone().multiplyScalar(-1),
    headHeight * 2.0,
    reyeFallback,
  );
  const leyePoint = raycastFromOrigin(
    threeRef,
    modelRoot,
    leyeOrigin,
    forward.clone().multiplyScalar(-1),
    headHeight * 2.0,
    leyeFallback,
  );

  const earSide = 0.70 * headHeight;
  const earUp = 0.04 * headHeight;
  const earForward = 0.02 * headHeight;
  const rearFallback = headCenter.clone()
    .add(right.clone().multiplyScalar(0.36 * headHeight))
    .add(up.clone().multiplyScalar(0.02 * headHeight))
    .add(forward.clone().multiplyScalar(-0.02 * headHeight));
  const learFallback = headCenter.clone()
    .add(right.clone().multiplyScalar(-0.36 * headHeight))
    .add(up.clone().multiplyScalar(0.02 * headHeight))
    .add(forward.clone().multiplyScalar(-0.02 * headHeight));
  const rearOrigin = headCenter.clone()
    .add(right.clone().multiplyScalar(earSide))
    .add(up.clone().multiplyScalar(earUp))
    .add(forward.clone().multiplyScalar(earForward));
  const learOrigin = headCenter.clone()
    .add(right.clone().multiplyScalar(-earSide))
    .add(up.clone().multiplyScalar(earUp))
    .add(forward.clone().multiplyScalar(earForward));
  let rearPoint = raycastFromOrigin(
    threeRef,
    modelRoot,
    rearOrigin,
    right.clone().multiplyScalar(-1),
    headHeight * 2.0,
    rearFallback,
  );
  let learPoint = raycastFromOrigin(
    threeRef,
    modelRoot,
    learOrigin,
    right.clone(),
    headHeight * 2.0,
    learFallback,
  );
  // If ear collapses too close to eye, try a slightly rear-biased side cast.
  if (rearPoint.distanceTo(reyePoint) < headHeight * 0.11) {
    const rearOriginBack = rearOrigin.clone().add(forward.clone().multiplyScalar(-0.10 * headHeight));
    rearPoint = raycastFromOrigin(threeRef, modelRoot, rearOriginBack, right.clone().multiplyScalar(-1), headHeight * 2.0, rearFallback);
  }
  if (learPoint.distanceTo(leyePoint) < headHeight * 0.11) {
    const learOriginBack = learOrigin.clone().add(forward.clone().multiplyScalar(-0.10 * headHeight));
    learPoint = raycastFromOrigin(threeRef, modelRoot, learOriginBack, right.clone(), headHeight * 2.0, learFallback);
  }

  const worldPoints = {
    nose: nosePoint,
    reye: reyePoint,
    leye: leyePoint,
    rear: rearPoint,
    lear: learPoint,
  };

  if (asWorld) {
    return {
      headBone: head,
      world: worldPoints,
    };
  }

  const to2d = (worldPos) => {
    const projected = worldPos.clone().project(camera);
    return {
      x: (projected.x * 0.5 + 0.5) * width,
      y: (-projected.y * 0.5 + 0.5) * height,
    };
  };

  return {
    nose: to2d(worldPoints.nose),
    reye: to2d(worldPoints.reye),
    leye: to2d(worldPoints.leye),
    rear: to2d(worldPoints.rear),
    lear: to2d(worldPoints.lear),
  };
}

function buildOverlay(node, widget, stateRef) {
  ensureStyles();
  const sharedSession = getSessionForNode(node);
  let overlayCanvas = null;
  if (widget && typeof widget.value === "string" && widget.value !== stateRef.value) {
    stateRef.value = widget.value;
  }
  const hasExplicitNodeState = () => {
    const values = node?.widgets_values;
    if (!Array.isArray(values)) return false;
    return values.some((v) => typeof v === "string" && v.trim().length > 0);
  };

  const overlay = document.createElement("div");
  overlay.className = "ess-pose-overlay";
  let consoleArea = null;

  const topbar = document.createElement("div");
  topbar.className = "ess-pose-topbar";

  const modelBadge = document.createElement("span");
  modelBadge.className = "ess-pose-badge";
  modelBadge.textContent = "No model loaded";

  const topActions = document.createElement("div");
  topActions.style.display = "flex";
  topActions.style.alignItems = "center";
  topActions.style.gap = "6px";
  topActions.style.flexWrap = "wrap";

  const updateNodeBtn = document.createElement("button");
  updateNodeBtn.className = "save";
  updateNodeBtn.textContent = "Update Node";

  const undoBtn = document.createElement("button");
  undoBtn.className = "secondary";
  undoBtn.textContent = "Undo";

  const redoBtn = document.createElement("button");
  redoBtn.className = "secondary";
  redoBtn.textContent = "Redo";

  const savePoseBtn = document.createElement("button");
  savePoseBtn.className = "secondary";
  savePoseBtn.textContent = "Save Pose";

  const loadPoseBtn = document.createElement("button");
  loadPoseBtn.className = "secondary";
  loadPoseBtn.textContent = "Load Pose";

  const saveSceneBtn = document.createElement("button");
  saveSceneBtn.className = "secondary";
  saveSceneBtn.textContent = "Save Scene";

  const loadSceneBtn = document.createElement("button");
  loadSceneBtn.className = "secondary";
  loadSceneBtn.textContent = "Load Scene";

  const initFromImageBtn = document.createElement("button");
  initFromImageBtn.className = "secondary";
  initFromImageBtn.textContent = "Init From Image";

  topActions.append(
    updateNodeBtn,
    undoBtn,
    redoBtn,
    savePoseBtn,
    loadPoseBtn,
    saveSceneBtn,
    loadSceneBtn,
    initFromImageBtn,
  );

  const topbarSpacer = document.createElement("span");
  topbarSpacer.style.flex = "1";

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "Close";
  closeBtn.className = "secondary";

  topbar.append(modelBadge, topActions, topbarSpacer, closeBtn);

  async function pickRiggedMeshName(items) {
    return new Promise((resolve) => {
      const shade = document.createElement("div");
      shade.style.position = "absolute";
      shade.style.inset = "0";
      shade.style.background = "rgba(0, 0, 0, 0.45)";
      shade.style.display = "flex";
      shade.style.alignItems = "center";
      shade.style.justifyContent = "center";
      shade.style.zIndex = "10002";

      const card = document.createElement("div");
      card.style.width = "420px";
      card.style.maxWidth = "calc(100vw - 48px)";
      card.style.background = "#11161d";
      card.style.border = "1px solid #2a303b";
      card.style.borderRadius = "8px";
      card.style.padding = "12px";
      card.style.display = "flex";
      card.style.flexDirection = "column";
      card.style.gap = "10px";

      const title = document.createElement("div");
      title.textContent = "Choose character from meshes/human_rig";
      title.style.fontSize = "13px";
      title.style.color = "#d6e8ff";

      const select = document.createElement("select");
      select.style.width = "100%";
      select.style.background = "#0f1115";
      select.style.color = "#e6e6e6";
      select.style.border = "1px solid #2a303b";
      select.style.borderRadius = "4px";
      select.style.padding = "6px 8px";
      items.forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      });

      const actions = document.createElement("div");
      actions.style.display = "flex";
      actions.style.justifyContent = "flex-end";
      actions.style.gap = "8px";

      const cancel = document.createElement("button");
      cancel.textContent = "Cancel";
      cancel.className = "secondary";
      cancel.style.background = "#2f3540";
      cancel.style.color = "#e6e6e6";
      cancel.style.border = "1px solid #3f4754";
      cancel.style.borderRadius = "4px";
      cancel.style.padding = "6px 10px";
      cancel.style.cursor = "pointer";

      const ok = document.createElement("button");
      ok.textContent = "Load";
      ok.className = "save";
      ok.style.background = "#2563eb";
      ok.style.color = "#ffffff";
      ok.style.border = "1px solid #1d4ed8";
      ok.style.borderRadius = "4px";
      ok.style.padding = "6px 10px";
      ok.style.cursor = "pointer";

      const close = (value) => {
        shade.remove();
        resolve(value);
      };

      cancel.addEventListener("click", () => close(null));
      ok.addEventListener("click", () => close(select.value || null));
      shade.addEventListener("click", (ev) => {
        if (ev.target === shade) close(null);
      });

      actions.append(cancel, ok);
      card.append(title, select, actions);
      shade.appendChild(card);
      overlay.appendChild(shade);
      select.focus();
    });
  }

  const main = document.createElement("div");
  main.className = "ess-pose-main";

  const leftPanel = document.createElement("div");
  leftPanel.className = "ess-pose-panel";
  const leftTitle = document.createElement("h4");
  leftTitle.textContent = "Bones";
  const boneList = document.createElement("div");
  boneList.className = "ess-pose-bone-list";
  leftPanel.style.position = "relative";
  boneList.style.marginRight = "46px";
  const characterTabs = document.createElement("div");
  characterTabs.style.position = "absolute";
  characterTabs.style.top = "40px";
  characterTabs.style.right = "6px";
  characterTabs.style.bottom = "10px";
  characterTabs.style.width = "36px";
  characterTabs.style.display = "flex";
  characterTabs.style.flexDirection = "column";
  characterTabs.style.gap = "6px";
  characterTabs.style.overflowY = "auto";
  leftPanel.append(leftTitle, boneList, characterTabs);

  const centerPanel = document.createElement("div");
  centerPanel.className = "ess-pose-panel";
  centerPanel.style.display = "flex";
  centerPanel.style.flexDirection = "column";
  centerPanel.style.padding = "0";

  const cameraBar = document.createElement("div");
  cameraBar.style.display = "flex";
  cameraBar.style.alignItems = "center";
  cameraBar.style.gap = "8px";
  cameraBar.style.padding = "8px 10px";
  cameraBar.style.borderBottom = "1px solid #2a303b";
  cameraBar.style.background = "#0f131b";

  const cameraTabsWrap = document.createElement("div");
  cameraTabsWrap.style.display = "none";
  cameraTabsWrap.style.gap = "6px";

  const camResolutionWrap = document.createElement("div");
  camResolutionWrap.style.display = "flex";
  camResolutionWrap.style.alignItems = "center";
  camResolutionWrap.style.gap = "6px";
  camResolutionWrap.style.marginLeft = "8px";
  const camResolutionLabel = document.createElement("span");
  camResolutionLabel.textContent = "Resolution";
  camResolutionLabel.style.fontSize = "12px";
  camResolutionLabel.style.color = "#c5d4eb";
  const camWidthInput = document.createElement("input");
  camWidthInput.type = "number";
  camWidthInput.min = "64";
  camWidthInput.max = "8192";
  camWidthInput.step = "1";
  camWidthInput.value = "1024";
  camWidthInput.style.width = "86px";
  camWidthInput.style.background = "#0f1115";
  camWidthInput.style.color = "#e6e6e6";
  camWidthInput.style.border = "1px solid #2a303b";
  camWidthInput.style.borderRadius = "4px";
  camWidthInput.style.padding = "3px 6px";
  const camMul = document.createElement("span");
  camMul.textContent = "x";
  camMul.style.color = "#8ca0bb";
  const camHeightInput = document.createElement("input");
  camHeightInput.type = "number";
  camHeightInput.min = "64";
  camHeightInput.max = "8192";
  camHeightInput.step = "1";
  camHeightInput.value = "768";
  camHeightInput.style.width = "86px";
  camHeightInput.style.background = "#0f1115";
  camHeightInput.style.color = "#e6e6e6";
  camHeightInput.style.border = "1px solid #2a303b";
  camHeightInput.style.borderRadius = "4px";
  camHeightInput.style.padding = "3px 6px";
  camResolutionWrap.append(camResolutionLabel, camWidthInput, camMul, camHeightInput);

  cameraBar.append(camResolutionWrap);
  const viewport = document.createElement("div");
  viewport.className = "ess-pose-viewport";
  viewport.style.flex = "1";
  viewport.style.minHeight = "480px";
  viewport.style.position = "relative";
  centerPanel.append(cameraBar, viewport);

  const rightPanel = document.createElement("div");
  rightPanel.className = "ess-pose-panel ess-pose-right";
  rightPanel.style.display = "grid";
  rightPanel.style.gridTemplateRows = "auto minmax(0, 1fr) auto auto auto";
  rightPanel.style.gap = "10px";
  rightPanel.style.overflow = "hidden";
  const controlsTitle = document.createElement("h4");
  controlsTitle.textContent = "Transforms";
  const controlsWrap = document.createElement("div");
  controlsWrap.className = "ess-pose-controls";
  controlsWrap.style.overflowY = "auto";
  controlsWrap.style.overflowX = "hidden";
  controlsWrap.style.paddingRight = "4px";
  controlsWrap.style.minHeight = "0";

  const selectedBoneInfo = document.createElement("div");
  selectedBoneInfo.className = "ess-pose-hint";
  selectedBoneInfo.style.marginTop = "-2px";
  selectedBoneInfo.style.marginBottom = "6px";
  selectedBoneInfo.textContent = "Selected: (none)";

  const mirrorWrap = document.createElement("label");
  mirrorWrap.style.display = "none";
  mirrorWrap.style.alignItems = "center";
  mirrorWrap.style.justifyContent = "flex-start";
  mirrorWrap.style.gap = "8px";
  mirrorWrap.style.marginBottom = "8px";
  mirrorWrap.style.fontSize = "12px";
  const mirrorCheck = document.createElement("input");
  mirrorCheck.type = "checkbox";
  const mirrorText = document.createElement("span");
  mirrorText.textContent = "Mirror paired bone";
  mirrorWrap.append(mirrorCheck, mirrorText);

  const rotX = createRangeNumberControl("Rot X", -180, 180, 0.5, (v) => applyEuler("x", v));
  const rotY = createRangeNumberControl("Rot Y", -180, 180, 0.5, (v) => applyEuler("y", v));
  const rotZ = createRangeNumberControl("Rot Z", -180, 180, 0.5, (v) => applyEuler("z", v));
  const posX = createRangeNumberControl("Pos X", -5, 5, 0.01, (v) => applyPosition("x", v));
  const posY = createRangeNumberControl("Pos Y", -5, 5, 0.01, (v) => applyPosition("y", v));
  const posZ = createRangeNumberControl("Pos Z", -5, 5, 0.01, (v) => applyPosition("z", v));
  const scaleU = createRangeNumberControl("Scale All", 0.1, 3, 0.01, (v) => applyScale("u", v));
  const scaleX = createRangeNumberControl("Scale X", 0.1, 3, 0.01, (v) => applyScale("x", v));
  const scaleY = createRangeNumberControl("Scale Y", 0.1, 3, 0.01, (v) => applyScale("y", v));
  const scaleZ = createRangeNumberControl("Scale Z", 0.1, 3, 0.01, (v) => applyScale("z", v));
  const fovRange = createRangeNumberControl("Camera FOV", 15, 90, 1, () => {});

  controlsWrap.append(
    selectedBoneInfo, mirrorWrap,
    rotX.wrap, rotY.wrap, rotZ.wrap,
    posX.wrap, posY.wrap, posZ.wrap,
    scaleU.wrap, scaleX.wrap, scaleY.wrap, scaleZ.wrap,
    fovRange.wrap,
  );

  const footer = document.createElement("div");
  footer.className = "ess-pose-footer";
  const saveBtn = document.createElement("button");
  saveBtn.className = "save";
  saveBtn.textContent = "Save to Node";
  const centerBtn = document.createElement("button");
  centerBtn.className = "secondary";
  centerBtn.textContent = "Frame Model";
  footer.append(saveBtn, centerBtn);

  const hint = document.createElement("div");
  hint.className = "ess-pose-hint";
  hint.textContent = "W/E/R: translate/rotate/scale - Hold right mouse to orbit - Mouse wheel to zoom";

  const consolePanel = document.createElement("div");
  consolePanel.className = "ess-pose-console";
  consolePanel.style.minHeight = "120px";
  const consoleTitle = document.createElement("h4");
  consoleTitle.textContent = "Log";
  consoleArea = document.createElement("textarea");
  consoleArea.readOnly = true;
  consoleArea.style.whiteSpace = "pre-wrap";
  consoleArea.style.wordBreak = "break-word";
  consoleArea.style.overflowX = "hidden";
  consoleArea.value = (sharedSession.logLines || ["[log] editor initialized"]).join("\n");
  consolePanel.append(consoleTitle, consoleArea);
  rightPanel.append(controlsTitle, controlsWrap, hint, footer, consolePanel);

  main.append(leftPanel, centerPanel, rightPanel);
  overlay.append(topbar, main);

  document.body.appendChild(overlay);
  overlay.tabIndex = -1;
  overlay.focus({ preventScroll: true });

  // THREE scene setup
  let three = null;
  let renderer = sharedSession.renderer;
  let camera = sharedSession.camera;
  let orbit = sharedSession.orbit;
  let transform = sharedSession.transform;
  let scene = sharedSession.scene;
  let OrbitControlsCls = null;
  let TransformControlsCls = null;
  let GLTFLoaderCls = null;
  let FBXLoaderCls = null;
  let OBJLoaderCls = null;
  let characters = Array.isArray(sharedSession.characters) ? sharedSession.characters : [];
  let activeCharacterId = sharedSession.activeCharacterId || null;
  let modelRoot = null;
  let bones = [];
  let boneNameMap = new Map();
  let groundPlane = sharedSession.groundPlane || null;
  let groundGrid = sharedSession.groundGrid || null;
  let originAxes = sharedSession.originAxes || null;
  let originDot = sharedSession.originDot || null;
  let selectedBone = null;
  let selectedBoneMarker = sharedSession.selectedBoneMarker || null;
  let selectedBoneParentLine = sharedSession.selectedBoneParentLine || null;
  let mirroredBone = null;
  let mirrorEnabled = false;
  let transformDragging = false;
  let savedPayload = null;
  let threeReady = false;
  let suppressHistory = false;
  let historyUndo = [];
  let historyRedo = [];
  let historyTimer = null;
  const HISTORY_LIMIT = 30;
  let cameraSlots = normalizeCameraSlots(sharedSession.cameraSlots);
  let activeCameraIndex = 0;

  const layoutRendererToFrame = () => {
    if (!renderer || !camera) return;
    const rect = viewport.getBoundingClientRect();
    const viewportW = Math.max(1, Math.round(rect.width));
    const viewportH = Math.max(1, Math.round(rect.height));
    const frameRect = getRenderFrameRect(viewportW, viewportH);
    renderer.setSize(Math.max(1, frameRect.width), Math.max(1, frameRect.height), false);
    const canvas = renderer.domElement;
    canvas.style.position = "absolute";
    canvas.style.left = `${frameRect.x}px`;
    canvas.style.top = `${frameRect.y}px`;
    canvas.style.width = `${Math.max(1, frameRect.width)}px`;
    canvas.style.height = `${Math.max(1, frameRect.height)}px`;
    canvas.style.right = "auto";
    canvas.style.bottom = "auto";
    camera.aspect = frameRect.aspect;
    camera.updateProjectionMatrix();
    return frameRect;
  };

  const resizeRenderer = () => {
    layoutRendererToFrame();
  };

  function appendLog(msg) {
    const line = String(msg ?? "");
    if (!line) return;
    if (!Array.isArray(sharedSession.logLines)) {
      sharedSession.logLines = [];
    }
    sharedSession.logLines.unshift(line);
    sharedSession.logLines = sharedSession.logLines.slice(0, 120);
    if (consoleArea) {
      consoleArea.value = sharedSession.logLines.join("\n");
    }
  }

  function setStatus(msg) {
    appendLog(msg);
  }

  function getActiveCameraSlot() {
    if (!Array.isArray(cameraSlots) || !cameraSlots.length) {
      cameraSlots = normalizeCameraSlots(null);
    }
    const idx = 0;
    return cameraSlots[idx] || cameraSlots[0];
  }

  function persistCameraSlotsToSession() {
    sharedSession.cameraSlots = cameraSlots;
    sharedSession.activeCameraIndex = 0;
  }

  function readCurrentCameraToSlot(idx = activeCameraIndex) {
    if (!camera || !orbit || !Array.isArray(cameraSlots)) return;
    const index = 0;
    const slot = normalizeCameraSlot(cameraSlots[index], "Camera");
    slot.fov = Number(camera.fov);
    slot.position = camera.position.toArray();
    slot.target = orbit.target.toArray();
    slot.near = Number(camera.near);
    slot.far = Number(camera.far);
    cameraSlots[index] = slot;
    persistCameraSlotsToSession();
  }

  function setCameraFromSlot(slot) {
    if (!camera || !orbit) return;
    const normalized = normalizeCameraSlot(slot, slot?.name || "Cam");
    camera.position.fromArray(normalized.position);
    camera.fov = Number(normalized.fov || 45);
    if (
      Number.isFinite(normalized.near) && Number.isFinite(normalized.far)
      && normalized.near > 0 && normalized.far > normalized.near
    ) {
      camera.near = Number(normalized.near);
      camera.far = Number(normalized.far);
    }
    camera.updateProjectionMatrix();
    orbit.target.fromArray(Array.isArray(normalized.target) ? normalized.target : [0, 0, 0]);
    orbit.update();
  }

  function applySlotToCamera(idx = activeCameraIndex) {
    if (!camera || !orbit || !Array.isArray(cameraSlots)) return;
    const index = 0;
    const slot = normalizeCameraSlot(cameraSlots[index], "Camera");
    cameraSlots[index] = slot;
    activeCameraIndex = index;
    setCameraFromSlot(slot);
    layoutRendererToFrame();
    updateFovSlider();
    updateOrbitDistanceLimits();
    persistCameraSlotsToSession();
  }

  function getRenderFrameRect(width, height, resolution = null) {
    const margin = 16;
    const availW = Math.max(1, width - margin * 2);
    const availH = Math.max(1, height - margin * 2);
    const slot = getActiveCameraSlot();
    const res = Array.isArray(resolution) ? resolution : slot?.resolution;
    const rw = Math.max(64, Number(res?.[0] || 1024));
    const rh = Math.max(64, Number(res?.[1] || 768));
    const aspect = rw / rh;
    let w = availW;
    let h = w / aspect;
    if (h > availH) {
      h = availH;
      w = h * aspect;
    }
    const x = Math.round((width - w) * 0.5);
    const y = Math.round((height - h) * 0.5);
    return { x, y, width: Math.round(w), height: Math.round(h), aspect };
  }

  function refreshCameraTabs() {
    return;
  }

  function syncResolutionInputsFromSlot() {
    const slot = getActiveCameraSlot();
    const w = Math.max(64, Math.min(8192, Math.round(Number(slot?.resolution?.[0] || 1024))));
    const h = Math.max(64, Math.min(8192, Math.round(Number(slot?.resolution?.[1] || 768))));
    camWidthInput.value = String(w);
    camHeightInput.value = String(h);
  }

  function applyResolutionInputsToSlot() {
    const slot = getActiveCameraSlot();
    const w = Math.max(64, Math.min(8192, Math.round(Number(camWidthInput.value) || 1024)));
    const h = Math.max(64, Math.min(8192, Math.round(Number(camHeightInput.value) || 768)));
    slot.resolution = [w, h];
    camWidthInput.value = String(w);
    camHeightInput.value = String(h);
    layoutRendererToFrame();
    persistCameraSlotsToSession();
    scheduleHistoryCapture(120);
  }

  function buildCameraTabs() {
    cameraSlots = normalizeCameraSlots(cameraSlots);
    activeCameraIndex = 0;
    syncResolutionInputsFromSlot();
    persistCameraSlotsToSession();
  }

  function sanitizeFilename(name, fallback = "pose") {
    const raw = String(name || "").trim();
    const safe = raw.replace(/[\\/:*?"<>|]+/g, "_").replace(/\s+/g, "_");
    return safe || fallback;
  }

  function downloadJsonFile(filename, payload) {
    const text = JSON.stringify(payload, null, 2);
    const blob = new Blob([text], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function pickJsonFile() {
    return new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".json,application/json";
      input.style.display = "none";
      input.addEventListener("change", async () => {
        try {
          const file = input.files?.[0];
          if (!file) {
            resolve(null);
            return;
          }
          const text = await file.text();
          const payload = JSON.parse(text);
          resolve(payload);
        } catch (err) {
          setStatus(`Failed to read JSON: ${err?.message || err}`);
          resolve(null);
        } finally {
          input.remove();
        }
      });
      document.body.appendChild(input);
      input.click();
    });
  }

  function pickImageFileAsDataUrl() {
    return new Promise((resolve) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = "image/png,image/jpeg,image/webp,image/bmp";
      input.style.display = "none";
      let settled = false;
      const settle = (value) => {
        if (settled) return;
        settled = true;
        try {
          window.removeEventListener("focus", onWindowFocus);
        } catch (_err) {}
        input.remove();
        resolve(value);
      };
      const onWindowFocus = () => {
        // Desktop/Electron can close file dialog without firing "change".
        setTimeout(() => {
          if (!settled && !(input.files && input.files.length > 0)) {
            settle(null);
          }
        }, 180);
      };
      window.addEventListener("focus", onWindowFocus);
      input.addEventListener("cancel", () => settle(null));
      input.addEventListener("change", async () => {
        try {
          const file = input.files?.[0];
          if (!file) {
            settle(null);
            return;
          }
          const dataUrl = await new Promise((ok, fail) => {
            const reader = new FileReader();
            reader.onload = () => ok(String(reader.result || ""));
            reader.onerror = (err) => fail(err);
            reader.readAsDataURL(file);
          });
          settle({
            name: String(file.name || "image"),
            dataUrl,
          });
        } catch (err) {
          setStatus(`Failed to read image: ${err?.message || err}`);
          settle(null);
        }
      });
      document.body.appendChild(input);
      input.click();
    });
  }

  async function requestPoseInitFromImage(imageDataUrl, requestId = null) {
    const rid = String(requestId || `ess_pose_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`);
    const payload = {
      image_base64: String(imageDataUrl || ""),
      confidence: 0.2,
      mode: "advanced",
      request_id: rid,
    };
    const response = await fetch("/ess/pose/init_from_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let data = {};
    try {
      data = await response.json();
    } catch (_err) {
      data = {};
    }
    if (!response.ok || !data?.ok) {
      const errorText = data?.error || `HTTP ${response.status}`;
      throw new Error(errorText);
    }
    return {
      ...data,
      request_id: String(data?.request_id || rid),
    };
  }

  async function fetchPoseInitProgress(requestId) {
    const rid = String(requestId || "");
    if (!rid) return null;
    const response = await fetch(`/ess/pose/init_progress?request_id=${encodeURIComponent(rid)}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      return null;
    }
    try {
      return await response.json();
    } catch (_err) {
      return null;
    }
  }

  function renderModelStructurePreview(payload, size = 720) {
    const persons = Array.isArray(payload?.persons) ? payload.persons : [];
    const imageW = Math.max(1, Number(payload?.image_width || persons[0]?.image_width || 1024));
    const imageH = Math.max(1, Number(payload?.image_height || persons[0]?.image_height || 1024));
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";

    ctx.fillStyle = "#020817";
    ctx.fillRect(0, 0, size, size);

    const pad = 24;
    const fitScale = Math.min((size - pad * 2) / imageW, (size - pad * 2) / imageH);
    const drawW = imageW * fitScale;
    const drawH = imageH * fitScale;
    const offX = (size - drawW) * 0.5;
    const offY = (size - drawH) * 0.5;

    ctx.fillStyle = "#061227";
    ctx.fillRect(offX, offY, drawW, drawH);
    ctx.strokeStyle = "rgba(96, 165, 250, 0.28)";
    ctx.lineWidth = 1;
    ctx.strokeRect(offX, offY, drawW, drawH);

    const toCanvas = (x, y) => ({
      x: offX + (Number(x) / imageW) * drawW,
      y: offY + (Number(y) / imageH) * drawH,
    });

    const getNamedUv = (person, name) => {
      const entry = person?.joints3d_named?.[name];
      if (!entry) return null;
      const u = Number(entry.u);
      const v = Number(entry.v);
      if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
      return toCanvas(u, v);
    };

    const connections = [
      ["head", "neck"],
      ["neck", "left_shoulder"],
      ["neck", "right_shoulder"],
      ["left_shoulder", "left_elbow"],
      ["left_elbow", "left_wrist"],
      ["right_shoulder", "right_elbow"],
      ["right_elbow", "right_wrist"],
      ["neck", "pelvis"],
      ["pelvis", "left_hip"],
      ["pelvis", "right_hip"],
      ["left_hip", "left_knee"],
      ["left_knee", "left_ankle"],
      ["right_hip", "right_knee"],
      ["right_knee", "right_ankle"],
      ["left_ankle", "left_foot"],
      ["right_ankle", "right_foot"],
    ];

    const palettes = [
      { line: "#22c55e", point: "#86efac", face: "#fbbf24", facePoint: "#f472b6" },
      { line: "#60a5fa", point: "#93c5fd", face: "#f59e0b", facePoint: "#f472b6" },
      { line: "#a78bfa", point: "#c4b5fd", face: "#fbbf24", facePoint: "#f472b6" },
    ];

    persons.forEach((person, idx) => {
      const colors = palettes[idx % palettes.length];
      const bbox = person?.bbox;
      if (bbox) {
        const p1 = toCanvas(Number(bbox.x1 || 0), Number(bbox.y1 || 0));
        const p2 = toCanvas(Number(bbox.x2 || 0), Number(bbox.y2 || 0));
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 2;
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      }

      ctx.strokeStyle = colors.line;
      ctx.lineWidth = 3;
      connections.forEach(([a, b]) => {
        const p0 = getNamedUv(person, a);
        const p1 = getNamedUv(person, b);
        if (!p0 || !p1) return;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();
      });

      const pointNames = [
        "head", "neck", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "pelvis", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot",
      ];
      ctx.fillStyle = colors.point;
      pointNames.forEach((name) => {
        const p = getNamedUv(person, name);
        if (!p) return;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });

      const face = person?.face;
      if (face?.bbox && Array.isArray(face.bbox) && face.bbox.length >= 4) {
        const p1 = toCanvas(face.bbox[0], face.bbox[1]);
        const p2 = toCanvas(face.bbox[2], face.bbox[3]);
        ctx.strokeStyle = colors.face;
        ctx.lineWidth = 2;
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      }
      if (Array.isArray(face?.kps)) {
        ctx.fillStyle = colors.facePoint;
        face.kps.forEach((pt) => {
          if (!Array.isArray(pt) || pt.length < 2) return;
          const p = toCanvas(Number(pt[0]), Number(pt[1]));
          ctx.beginPath();
          ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }
    });

    return canvas.toDataURL("image/png");
  }

  function createInitProgressDialog(fileName) {
    const shade = document.createElement("div");
    shade.style.position = "absolute";
    shade.style.inset = "0";
    shade.style.background = "rgba(3, 8, 18, 0.72)";
    shade.style.display = "flex";
    shade.style.alignItems = "center";
    shade.style.justifyContent = "center";
    shade.style.zIndex = "10020";

    const card = document.createElement("div");
    card.style.width = "1480px";
    card.style.maxWidth = "calc(100vw - 40px)";
    card.style.maxHeight = "calc(100vh - 36px)";
    card.style.background = "#061227";
    card.style.border = "1px solid #24436f";
    card.style.borderRadius = "12px";
    card.style.boxShadow = "0 18px 48px rgba(0, 0, 0, 0.45)";
    card.style.display = "flex";
    card.style.flexDirection = "column";
    card.style.overflow = "hidden";

    const header = document.createElement("div");
    header.style.padding = "14px 16px";
    header.style.borderBottom = "1px solid rgba(96, 165, 250, 0.22)";
    header.style.color = "#dbeafe";
    header.style.fontSize = "20px";
    header.style.fontWeight = "700";
    header.textContent = "Init From Image";

    const body = document.createElement("div");
    body.style.padding = "16px";
    body.style.display = "flex";
    body.style.flexDirection = "column";
    body.style.gap = "12px";
    body.style.minHeight = "0";

    const intro = document.createElement("div");
    intro.style.color = "#cbd5e1";
    intro.style.fontSize = "14px";
    intro.textContent = `Processing ${String(fileName || "image")}`;

    const statusBox = document.createElement("div");
    statusBox.style.background = "#020817";
    statusBox.style.border = "1px solid #27466f";
    statusBox.style.borderRadius = "10px";
    statusBox.style.padding = "14px";
    statusBox.style.color = "#e2e8f0";
    statusBox.style.fontSize = "16px";
    statusBox.style.minHeight = "56px";
    statusBox.style.display = "flex";
    statusBox.style.alignItems = "center";
    statusBox.textContent = "Preparing request";

    const hint = document.createElement("div");
    hint.style.color = "#93c5fd";
    hint.style.fontSize = "13px";
    hint.textContent = "First run may take several minutes while repos and model weights are downloaded.";

    const elapsed = document.createElement("div");
    elapsed.style.color = "#94a3b8";
    elapsed.style.fontSize = "12px";
    elapsed.textContent = "Elapsed: 0s";

    const previewWrap = document.createElement("div");
    previewWrap.style.display = "flex";
    previewWrap.style.flexDirection = "column";
    previewWrap.style.gap = "8px";
    previewWrap.style.minHeight = "0";

    const previewLabel = document.createElement("div");
    previewLabel.style.color = "#dbeafe";
    previewLabel.style.fontSize = "13px";
    previewLabel.textContent = "Preview";

    const previewGrid = document.createElement("div");
    previewGrid.style.display = "grid";
    previewGrid.style.gridTemplateColumns = "repeat(3, minmax(0, 1fr))";
    previewGrid.style.gap = "12px";

    const makePreviewPane = (labelText, altText) => {
      const wrap = document.createElement("div");
      wrap.style.display = "flex";
      wrap.style.flexDirection = "column";
      wrap.style.gap = "6px";
      wrap.style.minHeight = "0";

      const label = document.createElement("div");
      label.style.color = "#93c5fd";
      label.style.fontSize = "12px";
      label.textContent = labelText;

      const box = document.createElement("div");
      box.style.background = "#020817";
      box.style.border = "1px solid #27466f";
      box.style.borderRadius = "10px";
      box.style.padding = "10px";
      box.style.aspectRatio = "1 / 1";
      box.style.display = "flex";
      box.style.alignItems = "center";
      box.style.justifyContent = "center";
      box.style.overflow = "hidden";
      box.style.minWidth = "0";

      const image = document.createElement("img");
      image.style.width = "100%";
      image.style.height = "100%";
      image.style.objectFit = "contain";
      image.style.borderRadius = "6px";
      image.alt = altText;
      box.appendChild(image);
      wrap.append(label, box);
      return { wrap, image };
    };

    const sourcePane = makePreviewPane("Source", "Source preview");
    const modelPane = makePreviewPane("Model", "Model structure preview");
    const resultPane = makePreviewPane("Result", "Result preview");
    modelPane.image.style.opacity = "0.35";
    resultPane.image.style.opacity = "0.35";
    previewGrid.append(sourcePane.wrap, modelPane.wrap, resultPane.wrap);
    previewWrap.append(previewLabel, previewGrid);

    const logWrap = document.createElement("div");
    logWrap.style.display = "flex";
    logWrap.style.flexDirection = "column";
    logWrap.style.gap = "8px";
    logWrap.style.minHeight = "0";

    const logLabel = document.createElement("div");
    logLabel.style.color = "#dbeafe";
    logLabel.style.fontSize = "13px";
    logLabel.textContent = "Log";

    const logBox = document.createElement("div");
    logBox.style.background = "#020817";
    logBox.style.border = "1px solid #27466f";
    logBox.style.borderRadius = "10px";
    logBox.style.padding = "10px";
    logBox.style.minHeight = "140px";
    logBox.style.maxHeight = "220px";
    logBox.style.overflow = "auto";
    logBox.style.color = "#cbd5e1";
    logBox.style.fontSize = "12px";
    logBox.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, monospace";
    logBox.style.whiteSpace = "pre-wrap";

    logWrap.append(logLabel, logBox);

    const errorBox = document.createElement("div");
    errorBox.style.display = "none";
    errorBox.style.background = "rgba(127, 29, 29, 0.22)";
    errorBox.style.border = "1px solid rgba(248, 113, 113, 0.35)";
    errorBox.style.borderRadius = "10px";
    errorBox.style.padding = "12px";
    errorBox.style.color = "#fecaca";
    errorBox.style.fontSize = "13px";
    errorBox.style.whiteSpace = "pre-wrap";

    const footer = document.createElement("div");
    footer.style.display = "flex";
    footer.style.justifyContent = "flex-end";
    footer.style.gap = "8px";
    footer.style.padding = "0 16px 16px";

    const finish = document.createElement("button");
    finish.className = "save";
    finish.textContent = "Finish";
    finish.disabled = true;

    footer.append(finish);
    body.append(intro, statusBox, hint, elapsed, previewWrap, logWrap, errorBox);
    card.append(header, body, footer);
    shade.appendChild(card);
    overlay.appendChild(shade);

    const startedAt = Date.now();
    let removed = false;
    let elapsedTimer = null;
    let pollTimer = null;

    const stopTimers = () => {
      if (elapsedTimer) {
        clearInterval(elapsedTimer);
        elapsedTimer = null;
      }
      if (pollTimer) {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    };

    const remove = () => {
      if (removed) return;
      removed = true;
      stopTimers();
      shade.remove();
    };

    finish.addEventListener("click", () => remove());

    elapsedTimer = setInterval(() => {
      const sec = Math.max(0, Math.round((Date.now() - startedAt) / 1000));
      elapsed.textContent = `Elapsed: ${sec}s`;
    }, 500);

    const seenLogs = new Set();

    const appendLogs = (logs) => {
      if (!Array.isArray(logs)) return;
      logs.forEach((entry) => {
        const text = String(entry || "").trim();
        if (!text || seenLogs.has(text)) return;
        seenLogs.add(text);
        const line = document.createElement("div");
        line.textContent = text;
        logBox.appendChild(line);
      });
      logBox.scrollTop = logBox.scrollHeight;
    };

    return {
      setSourceImage(src) {
        const value = String(src || "");
        if (value) {
          sourcePane.image.src = value;
        }
      },
      setResultImage(src) {
        const value = String(src || "");
        if (value) {
          resultPane.image.src = value;
          resultPane.image.style.opacity = "1";
        }
      },
      setModelImage(src) {
        const value = String(src || "");
        if (value) {
          modelPane.image.src = value;
          modelPane.image.style.opacity = "1";
        }
      },
      setStage(message) {
        statusBox.textContent = String(message || "Working");
      },
      appendLogs,
      applyProgress(progress) {
        if (!progress || progress.ok === false) return;
        const stage = String(progress.stage || "");
        if (stage) {
          statusBox.textContent = stage;
        }
        if (progress.preview_image_base64) {
          sourcePane.image.src = String(progress.preview_image_base64);
        }
        appendLogs(progress.logs);
        if (progress.error) {
          errorBox.style.display = "";
          errorBox.textContent = String(progress.error);
          finish.disabled = false;
        }
        if (progress.done) {
          stopTimers();
          finish.disabled = false;
        }
      },
      setError(message) {
        errorBox.style.display = "";
        errorBox.textContent = String(message || "Unknown error");
        finish.disabled = false;
      },
      startPolling(requestId) {
        const rid = String(requestId || "");
        if (!rid) return;
        const poll = async () => {
          const progress = await fetchPoseInitProgress(rid);
          if (!progress || progress.ok === false) return;
          this.applyProgress(progress);
        };
        poll();
        pollTimer = setInterval(poll, 900);
      },
      complete(message) {
        stopTimers();
        statusBox.textContent = String(message || "Completed");
        finish.disabled = false;
      },
      fail(message) {
        stopTimers();
        statusBox.textContent = "Failed";
        errorBox.style.display = "";
        errorBox.textContent = String(message || "Unknown error");
        finish.disabled = false;
      },
      remove,
    };
  }

  function updateHistoryButtons() {
    if (undoBtn) undoBtn.disabled = historyUndo.length <= 1 || suppressHistory;
    if (redoBtn) redoBtn.disabled = historyRedo.length === 0 || suppressHistory;
  }

  function snapshotForHistory() {
    return serializeState(false);
  }

  function pushHistorySnapshot(snapshot, clearRedo = true) {
    const json = JSON.stringify(snapshot || {});
    const last = historyUndo.length ? historyUndo[historyUndo.length - 1] : null;
    if (last === json) {
      updateHistoryButtons();
      return;
    }
    historyUndo.push(json);
    if (historyUndo.length > HISTORY_LIMIT) {
      historyUndo.shift();
    }
    if (clearRedo) {
      historyRedo = [];
    }
    updateHistoryButtons();
  }

  function captureHistoryNow(clearRedo = true) {
    if (suppressHistory) return;
    if (historyTimer) {
      clearTimeout(historyTimer);
      historyTimer = null;
    }
    pushHistorySnapshot(snapshotForHistory(), clearRedo);
  }

  function scheduleHistoryCapture(delayMs = 240) {
    if (suppressHistory) return;
    if (historyTimer) clearTimeout(historyTimer);
    historyTimer = setTimeout(() => {
      historyTimer = null;
      pushHistorySnapshot(snapshotForHistory(), true);
    }, delayMs);
  }

  function clearCharactersFromScene() {
    if (!scene) return;
    characters.forEach((ch) => {
      if (ch?.modelRoot && ch.modelRoot.parent === scene) scene.remove(ch.modelRoot);
      if (ch?.skeletonHelper && ch.skeletonHelper.parent === scene) scene.remove(ch.skeletonHelper);
    });
    characters = [];
    sharedSession.characters = characters;
    applyActiveCharacter(null);
    refreshCharacterTabs();
    populateBones();
  }

  async function applyScenePayload(payload, options = {}) {
    const fromHistory = Boolean(options.fromHistory);
    const next = payload && typeof payload === "object" ? payload : {};
    suppressHistory = true;
    try {
      clearCharactersFromScene();
      savedPayload = next;

      if (Array.isArray(next.characters) && next.characters.length) {
        for (const ch of next.characters) {
          if (!ch?.mesh_b64) continue;
          const ext = ch.mesh_ext || "fbx";
          const name = ch.name || `cached.${ext}`;
          const buf = base64ToArrayBuffer(ch.mesh_b64);
          await loadModelFromFile(
            { name, arrayBuffer: async () => buf },
            buf,
            { id: ch.id, name, meshExt: ext },
          );
        }
      }

      if (next.active_character_id) {
        const active = characters.find((c) => c.id === next.active_character_id) || characters[0] || null;
        applyActiveCharacter(active);
      } else {
        applyActiveCharacter(characters[0] || null);
      }

      restoreStateAfterLoad();
      refreshCharacterTabs();
      populateBones();
      if (!characters.length) {
        setStatus("Scene loaded (empty).");
      } else {
        setStatus(`Scene loaded (${characters.length} character(s)).`);
      }
    } finally {
      suppressHistory = false;
      updateHistoryButtons();
    }
    if (!fromHistory) {
      captureHistoryNow(true);
    }
  }

  async function undoHistory() {
    if (historyUndo.length <= 1) return;
    const current = historyUndo.pop();
    historyRedo.push(current);
    const targetJson = historyUndo[historyUndo.length - 1];
    updateHistoryButtons();
    try {
      const payload = JSON.parse(targetJson);
      await applyScenePayload(payload, { fromHistory: true });
    } catch (err) {
      setStatus(`Undo failed: ${err?.message || err}`);
    }
  }

  async function redoHistory() {
    if (!historyRedo.length) return;
    const targetJson = historyRedo.pop();
    historyUndo.push(targetJson);
    updateHistoryButtons();
    try {
      const payload = JSON.parse(targetJson);
      await applyScenePayload(payload, { fromHistory: true });
    } catch (err) {
      setStatus(`Redo failed: ${err?.message || err}`);
    }
  }

  function getActiveCharacter() {
    if (!characters.length) return null;
    const found = characters.find((c) => c.id === activeCharacterId);
    return found || characters[0];
  }

  // Mirror across the character's left-right plane in local bone space.
  const MIRROR_ROT_SIGN = { x: 1, y: -1, z: -1 };
  const MIRROR_POS_SIGN = { x: -1, y: 1, z: 1 };

  function refreshSkinnedMeshes() {
    if (!modelRoot) return;
    modelRoot.traverse((child) => {
      if (child.isSkinnedMesh && child.skeleton) {
        child.skeleton.update();
      }
    });
  }

  function getBoneForOpenPoseIndex(character, openPoseIndex) {
    if (!character || !character.boneNameMap) return null;
    const name = OPENPOSE_NAMES[openPoseIndex];
    if (!name) return null;
    return findBoneByAliases(character.boneNameMap, OPENPOSE_TO_BONE[name] || []);
  }

  function getFootDriveBones(character, side) {
    if (!character?.boneNameMap) return { driveBone: null, endBone: null };
    const sideKey = String(side || "").toLowerCase();
    if (sideKey !== "left" && sideKey !== "right") {
      return { driveBone: null, endBone: null };
    }
    const title = sideKey === "left" ? "Left" : "Right";
    const driveBone = findBoneByAliases(character.boneNameMap, [
      `mixamorig${title}Foot`,
      `${title}Foot`,
    ]);
    const endBone = findBoneByAliases(character.boneNameMap, [
      `mixamorig${title}ToeBase`,
      `${title}ToeBase`,
      `mixamorig${title}Toe_End`,
      `${title}Toe_End`,
      `mixamorig${title}Toe`,
      `${title}Toe`,
    ]);
    return { driveBone, endBone };
  }

  function getHeadDriveBones(character) {
    if (!character?.boneNameMap) return { driveBone: null, endBone: null };
    const driveBone = findBoneByAliases(character.boneNameMap, OPENPOSE_TO_BONE.Neck || []);
    const endBone = findBoneByAliases(character.boneNameMap, [
      "mixamorigHead",
      "Head",
      "mixamorigHeadTop_End",
      "HeadTop_End",
      "HeadTop",
    ]);
    return { driveBone, endBone };
  }

  function getCharacterOpenPoseWorld(character) {
    const out = Array.from({ length: OPENPOSE_NAMES.length }, () => null);
    if (!character || !three || !camera) return out;
    OPENPOSE_NAMES.forEach((name, idx) => {
      const bone = findBoneByAliases(character.boneNameMap, OPENPOSE_TO_BONE[name] || []);
      if (!bone) return;
      const p = new three.Vector3();
      bone.getWorldPosition(p);
      out[idx] = p;
    });

    if (!out[1] && out[2] && out[5]) {
      out[1] = out[2].clone().lerp(out[5], 0.5);
    }

    const width = Math.max(64, Math.round(viewport?.clientWidth || 1024));
    const height = Math.max(64, Math.round(viewport?.clientHeight || 768));
    const headSynth = synthesizeHeadLandmarks(
      three,
      character.boneNameMap,
      camera,
      width,
      height,
      character.modelRoot,
      true,
    );
    if (headSynth?.world) {
      if (!out[0]) out[0] = headSynth.world.nose.clone();
      out[14] = headSynth.world.reye.clone();
      out[15] = headSynth.world.leye.clone();
      out[16] = headSynth.world.rear.clone();
      out[17] = headSynth.world.lear.clone();
    }

    return out;
  }

  function getCameraForwardVector() {
    if (!three || !camera) return null;
    return new three.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize();
  }

  function worldDepthOnCameraForward(worldPos) {
    const fwd = getCameraForwardVector();
    if (!fwd || !worldPos || !camera) return null;
    return worldPos.clone().sub(camera.position).dot(fwd);
  }

  function overlayPointToWorldAtDepth(overlayX, overlayY, frameRect, depth) {
    if (!three || !camera || !frameRect) return null;
    const fx = Number(frameRect.x || 0);
    const fy = Number(frameRect.y || 0);
    const fw = Math.max(1e-6, Number(frameRect.width || 1));
    const fh = Math.max(1e-6, Number(frameRect.height || 1));
    const nx = (Number(overlayX) - fx) / fw;
    const ny = (Number(overlayY) - fy) / fh;
    if (!Number.isFinite(nx) || !Number.isFinite(ny)) return null;
    const ndcX = nx * 2 - 1;
    const ndcY = -(ny * 2 - 1);
    const farPoint = new three.Vector3(ndcX, ndcY, 1).unproject(camera);
    const dir = farPoint.sub(camera.position);
    const dirLen = dir.length();
    if (!Number.isFinite(dirLen) || dirLen <= 1e-8) return null;
    dir.normalize();
    const forward = getCameraForwardVector();
    if (!forward) return null;
    const denom = dir.dot(forward);
    if (!Number.isFinite(denom) || Math.abs(denom) <= 1e-6) return null;
    const t = Number(depth) / denom;
    if (!Number.isFinite(t) || t <= 0.0) return null;
    return camera.position.clone().add(dir.multiplyScalar(t));
  }

  function detectionPointToOverlay(point, frameRect, imageWidth, imageHeight) {
    if (!point || !frameRect) return null;
    const xn = Number(point.xn);
    const yn = Number(point.yn);
    if (Number.isFinite(xn) && Number.isFinite(yn)) {
      return {
        x: frameRect.x + Math.max(0, Math.min(1, xn)) * frameRect.width,
        y: frameRect.y + Math.max(0, Math.min(1, yn)) * frameRect.height,
      };
    }
    const px = Number(point.x);
    const py = Number(point.y);
    const w = Math.max(1, Number(imageWidth || 1));
    const h = Math.max(1, Number(imageHeight || 1));
    if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
    return {
      x: frameRect.x + (px / w) * frameRect.width,
      y: frameRect.y + (py / h) * frameRect.height,
    };
  }

  function rotateBoneTowardWorldDirection(bone, currentEndWorld, targetDirWorld, maxAngle = null) {
    if (!bone || !currentEndWorld || !targetDirWorld || !three) return false;
    const start = new three.Vector3();
    bone.getWorldPosition(start);
    const currentDir = currentEndWorld.clone().sub(start);
    if (currentDir.lengthSq() <= 1e-10 || targetDirWorld.lengthSq() <= 1e-10) {
      return false;
    }
    currentDir.normalize();
    const targetDir = targetDirWorld.clone().normalize();
    const deltaQ = new three.Quaternion().setFromUnitVectors(currentDir, targetDir);
    if (!Number.isFinite(deltaQ.x + deltaQ.y + deltaQ.z + deltaQ.w)) return false;
    let appliedDelta = deltaQ;
    if (Number.isFinite(maxAngle) && maxAngle > 1e-6) {
      let angle = 2 * Math.acos(Math.max(-1, Math.min(1, deltaQ.w)));
      if (Number.isFinite(angle) && angle > 1e-6) {
        if (angle > Math.PI) angle = (2 * Math.PI) - angle;
        if (angle > maxAngle) {
          const axis = new three.Vector3(deltaQ.x, deltaQ.y, deltaQ.z);
          if (axis.lengthSq() > 1e-10) {
            axis.normalize();
            appliedDelta = new three.Quaternion().setFromAxisAngle(axis, maxAngle);
          }
        }
      }
    }

    const boneWorldQ = new three.Quaternion();
    bone.getWorldQuaternion(boneWorldQ);
    const newWorldQ = appliedDelta.multiply(boneWorldQ).normalize();

    const parentWorldQ = new three.Quaternion();
    if (bone.parent) {
      bone.parent.getWorldQuaternion(parentWorldQ);
    } else {
      parentWorldQ.identity();
    }
    const invParentQ = parentWorldQ.clone().invert();
    bone.quaternion.copy(invParentQ.multiply(newWorldQ)).normalize();
    bone.updateMatrixWorld(true);
    return true;
  }

  function setBoneWorldQuaternion(bone, worldQ) {
    if (!bone || !worldQ || !three) return false;
    const parentWorldQ = new three.Quaternion();
    if (bone.parent) {
      bone.parent.getWorldQuaternion(parentWorldQ);
    } else {
      parentWorldQ.identity();
    }
    const localQ = parentWorldQ.clone().invert().multiply(worldQ).normalize();
    if (!Number.isFinite(localQ.x + localQ.y + localQ.z + localQ.w)) return false;
    bone.quaternion.copy(localQ);
    bone.updateMatrixWorld(true);
    return true;
  }

  function applyPartialWorldRotationDelta(bone, deltaQ, fraction = 1.0, maxAngle = null) {
    if (!bone || !deltaQ || !three) return false;
    let angle = 2 * Math.acos(Math.max(-1, Math.min(1, deltaQ.w)));
    if (!Number.isFinite(angle) || angle <= 1e-6) return false;
    if (angle > Math.PI) angle = (2 * Math.PI) - angle;
    let axis = new three.Vector3(deltaQ.x, deltaQ.y, deltaQ.z);
    if (axis.lengthSq() <= 1e-10) return false;
    axis.normalize();
    const limitedAngle = Number.isFinite(maxAngle) && maxAngle > 1e-6
      ? Math.min(angle * Math.max(0, Number(fraction || 0)), maxAngle)
      : angle * Math.max(0, Number(fraction || 0));
    if (!Number.isFinite(limitedAngle) || limitedAngle <= 1e-6) return false;
    const partialDelta = new three.Quaternion().setFromAxisAngle(axis, limitedAngle);
    const currentWorldQ = new three.Quaternion();
    bone.getWorldQuaternion(currentWorldQ);
    const newWorldQ = partialDelta.multiply(currentWorldQ).normalize();
    return setBoneWorldQuaternion(bone, newWorldQ);
  }

  function getTorsoDriveBones(character) {
    if (!character?.boneNameMap) return { hipsBone: null, spineBone: null, neckBone: null };
    const hipsBone = findBoneByAliases(character.boneNameMap, [
      "mixamorigHips", "Hips", "mixamorigPelvis", "Pelvis",
    ]);
    const spineBone = findBoneByAliases(character.boneNameMap, [
      "mixamorigSpine2", "Spine2", "mixamorigUpperChest", "UpperChest",
      "mixamorigChest", "Chest", "mixamorigSpine1", "Spine1", "mixamorigSpine", "Spine",
    ]);
    const neckBone = findBoneByAliases(character.boneNameMap, OPENPOSE_TO_BONE.Neck || []);
    return { hipsBone, spineBone, neckBone };
  }

  function buildTargetFaceWorldPoints(person, frameRect, depth) {
    if (!person || !frameRect || !Number.isFinite(depth)) return null;
    const kps = Array.isArray(person?.face?.kps) ? person.face.kps : null;
    if (!kps || kps.length < 5) return null;
    const imageW = Math.max(1, Number(person?.image_width || 1));
    const imageH = Math.max(1, Number(person?.image_height || 1));
    const toWorld = (pt) => {
      if (!Array.isArray(pt) || pt.length < 2) return null;
      const overlay = detectionPointToOverlay({ x: Number(pt[0]), y: Number(pt[1]) }, frameRect, imageW, imageH);
      if (!overlay) return null;
      return overlayPointToWorldAtDepth(overlay.x, overlay.y, frameRect, depth);
    };
    // InsightFace 5-point key order is image-ordered:
    // [image-left eye, image-right eye, nose, image-left mouth, image-right mouth].
    // Our rig-side synthetic landmarks are subject-ordered:
    // [reye, leye, nose, ...] where "right" / "left" are the character's sides.
    // For a frontal face, subject-right appears on image-left, so we intentionally swap here.
    const reye = toWorld(kps[0]);
    const leye = toWorld(kps[1]);
    const nose = toWorld(kps[2]);
    const rmouth = toWorld(kps[3]);
    const lmouth = toWorld(kps[4]);
    const mouth = averageScenePoints([lmouth, rmouth]);
    if (!leye || !reye || !nose) return null;
    return { leye, reye, nose, mouth };
  }

  function buildTargetFaceOverlayPoints(person, frameRect) {
    if (!person || !frameRect) return null;
    const kps = Array.isArray(person?.face?.kps) ? person.face.kps : null;
    if (!kps || kps.length < 5) return null;
    const imageW = Math.max(1, Number(person?.image_width || 1));
    const imageH = Math.max(1, Number(person?.image_height || 1));
    const toOverlay = (pt) => {
      if (!Array.isArray(pt) || pt.length < 2) return null;
      return detectionPointToOverlay({ x: Number(pt[0]), y: Number(pt[1]) }, frameRect, imageW, imageH);
    };
    const leftEye = toOverlay(kps[0]);
    const rightEye = toOverlay(kps[1]);
    const nose = toOverlay(kps[2]);
    const leftMouth = toOverlay(kps[3]);
    const rightMouth = toOverlay(kps[4]);
    const mouth = averageScenePoints([leftMouth, rightMouth]);
    if (!leftEye || !rightEye || !nose) return null;
    return { leftEye, rightEye, nose, mouth };
  }

  function normalizeFaceOverlayPoints(face) {
    if (!face?.nose || !face?.leye || !face?.reye) return null;
    const leftEye = face.leye.x <= face.reye.x ? face.leye : face.reye;
    const rightEye = face.leye.x <= face.reye.x ? face.reye : face.leye;
    return {
      leftEye,
      rightEye,
      nose: face.nose,
      mouth: face.mouth || null,
    };
  }

  function applyFaceDrivenHeadOrientation(person, character) {
    if (!character || !three || !camera || !viewport) return false;
    if (!person?.face?.kps) return false;
    const width = Math.max(64, Math.round(viewport?.clientWidth || 1024));
    const height = Math.max(64, Math.round(viewport?.clientHeight || 768));
    const frameRect = getRenderFrameRect(width, height);
    const currentFace2dRaw = getCachedFaceLandmarks2d(character, width, height, frameRect);
    const currentFace2d = normalizeFaceOverlayPoints(currentFace2dRaw);
    const targetFace2d = buildTargetFaceOverlayPoints(person, frameRect);
    const headBone = findBoneByAliases(character.boneNameMap, ["mixamorigHead", "Head"]);
    const neckBone = findBoneByAliases(character.boneNameMap, OPENPOSE_TO_BONE.Neck || []);
    if (!headBone || !currentFace2d || !targetFace2d) return false;
    const currentEyeMid = averageScenePoints([currentFace2d.leftEye, currentFace2d.rightEye]);
    const targetEyeMid = averageScenePoints([targetFace2d.leftEye, targetFace2d.rightEye]);
    if (!currentEyeMid || !targetEyeMid) return false;

    const currentEyeVec = currentFace2d.rightEye.clone().sub(currentFace2d.leftEye);
    const targetEyeVec = targetFace2d.rightEye.clone().sub(targetFace2d.leftEye);
    const currentEyeDist = currentEyeVec.length();
    const targetEyeDist = targetEyeVec.length();
    if (!Number.isFinite(currentEyeDist) || !Number.isFinite(targetEyeDist) || currentEyeDist <= 1e-5 || targetEyeDist <= 1e-5) {
      return false;
    }

    const wrapAngle = (a) => {
      let out = a;
      while (out > Math.PI) out -= Math.PI * 2;
      while (out < -Math.PI) out += Math.PI * 2;
      return out;
    };

    const currentRoll = Math.atan2(currentEyeVec.y, currentEyeVec.x);
    const targetRoll = Math.atan2(targetEyeVec.y, targetEyeVec.x);
    const rollDelta = wrapAngle(targetRoll - currentRoll);

    const currentYawMetric = currentFace2d.nose && currentEyeDist > 1e-5
      ? currentFace2d.nose.clone().sub(currentEyeMid).dot(currentEyeVec.clone().normalize()) / currentEyeDist
      : 0;
    const targetYawMetric = targetFace2d.nose && targetEyeDist > 1e-5
      ? targetFace2d.nose.clone().sub(targetEyeMid).dot(targetEyeVec.clone().normalize()) / targetEyeDist
      : 0;
    const yawDelta = Math.max(-0.35, Math.min(0.35, (targetYawMetric - currentYawMetric) * 1.2));

    const currentWorldQ = new three.Quaternion();
    headBone.getWorldQuaternion(currentWorldQ);
    let applied = false;

    const cameraForward = getCameraForwardVector();
    if (cameraForward && Number.isFinite(rollDelta) && Math.abs(rollDelta) > 1e-4) {
      const clampedRoll = Math.max(-0.4, Math.min(0.4, rollDelta));
      const rollQ = new three.Quaternion().setFromAxisAngle(cameraForward, clampedRoll);
      if (neckBone) {
        applied = applyPartialWorldRotationDelta(neckBone, rollQ, 0.22, 0.12) || applied;
      }
      applied = applyPartialWorldRotationDelta(headBone, rollQ, 0.55, 0.24) || applied;
    }

    if (Number.isFinite(yawDelta) && Math.abs(yawDelta) > 1e-4) {
      const headUpAxis = new three.Vector3(0, 1, 0).applyQuaternion(currentWorldQ).normalize();
      const yawQ = new three.Quaternion().setFromAxisAngle(headUpAxis, yawDelta);
      if (neckBone) {
        applied = applyPartialWorldRotationDelta(neckBone, yawQ, 0.28, 0.14) || applied;
      }
      applied = applyPartialWorldRotationDelta(headBone, yawQ, 0.62, 0.28) || applied;
    }

    return applied;
  }

  function computeOverlayPoseStats(points2d) {
    const pts = Array.isArray(points2d) ? points2d.filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y)) : [];
    if (!pts.length) return null;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    return {
      minX, maxX, minY, maxY,
      centerX: (minX + maxX) * 0.5,
      centerY: (minY + maxY) * 0.5,
      width: maxX - minX,
      height: maxY - minY,
    };
  }

  function scenePointFromCvCoords(point) {
    if (!point || !three) return null;
    const x = Number(point.x);
    const y = Number(point.y);
    const z = Number(point.z);
    if (![x, y, z].every(Number.isFinite)) return null;
    // OpenCV camera coordinates: +x right, +y down, +z forward.
    // Three.js view space is more convenient here with +y up and camera looking along -Z.
    return new three.Vector3(x, -y, -z);
  }

  function namedJointScenePoint(namedJoints, name) {
    if (!namedJoints || typeof namedJoints !== "object") return null;
    return scenePointFromCvCoords(namedJoints[name]);
  }

  function namedJointOverlayPoint(namedJoints, name, frameRect, imageW = 1, imageH = 1) {
    if (!namedJoints || typeof namedJoints !== "object" || !frameRect) return null;
    const entry = namedJoints[name];
    if (!entry || typeof entry !== "object") return null;
    const un = Number(entry.un);
    const vn = Number(entry.vn);
    if (Number.isFinite(un) && Number.isFinite(vn)) {
      return {
        x: frameRect.x + un * frameRect.width,
        y: frameRect.y + vn * frameRect.height,
      };
    }
    const u = Number(entry.u);
    const v = Number(entry.v);
    if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
    return {
      x: frameRect.x + (u / Math.max(1, Number(imageW || 1))) * frameRect.width,
      y: frameRect.y + (v / Math.max(1, Number(imageH || 1))) * frameRect.height,
    };
  }

  function namedJointProjectedWorldPoint(person, jointName, frameRect, depth) {
    if (!person?.joints3d_named || !frameRect || !Number.isFinite(depth) || depth <= 0.0) {
      return namedJointScenePoint(person?.joints3d_named, jointName);
    }
    const imageW = Math.max(1, Number(person?.image_width || 1));
    const imageH = Math.max(1, Number(person?.image_height || 1));
    const overlay = namedJointOverlayPoint(person.joints3d_named, jointName, frameRect, imageW, imageH);
    if (!overlay) {
      return namedJointScenePoint(person.joints3d_named, jointName);
    }
    return overlayPointToWorldAtDepth(overlay.x, overlay.y, frameRect, depth)
      || namedJointScenePoint(person.joints3d_named, jointName);
  }

  function averageOverlayPoints(points) {
    const valid = Array.isArray(points) ? points.filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y)) : [];
    if (!valid.length) return null;
    return {
      x: valid.reduce((a, p) => a + p.x, 0) / valid.length,
      y: valid.reduce((a, p) => a + p.y, 0) / valid.length,
    };
  }

  function averageScenePoints(points) {
    const valid = Array.isArray(points) ? points.filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z)) : [];
    if (!valid.length || !three) return null;
    const out = new three.Vector3();
    valid.forEach((p) => out.add(p));
    out.multiplyScalar(1 / valid.length);
    return out;
  }

  function getCharacterReferencePose(character) {
    if (!character || !three) return null;
    const lHipBone = getBoneForOpenPoseIndex(character, 11);
    const rHipBone = getBoneForOpenPoseIndex(character, 8);
    const lShoulderBone = getBoneForOpenPoseIndex(character, 5);
    const rShoulderBone = getBoneForOpenPoseIndex(character, 2);
    const neckBone = getBoneForOpenPoseIndex(character, 1);
    const lAnkleBone = getBoneForOpenPoseIndex(character, 13);
    const rAnkleBone = getBoneForOpenPoseIndex(character, 10);
    const headBone = findBoneByAliases(character.boneNameMap, ["mixamorigHead", "Head"]);

    const worldPos = (bone) => {
      if (!bone) return null;
      const p = new three.Vector3();
      bone.getWorldPosition(p);
      return p;
    };

    const pelvis = averageScenePoints([worldPos(lHipBone), worldPos(rHipBone)]);
    const ankles = averageScenePoints([worldPos(lAnkleBone), worldPos(rAnkleBone)]);
    const shoulders = averageScenePoints([worldPos(lShoulderBone), worldPos(rShoulderBone)]);
    const leftHip = worldPos(lHipBone);
    const rightHip = worldPos(rHipBone);
    const leftShoulder = worldPos(lShoulderBone);
    const rightShoulder = worldPos(rShoulderBone);
    const neck = worldPos(neckBone);
    const head = worldPos(headBone);
    const shoulderWidth = (() => {
      return leftShoulder && rightShoulder ? leftShoulder.distanceTo(rightShoulder) : null;
    })();
    const hipWidth = (() => {
      return leftHip && rightHip ? leftHip.distanceTo(rightHip) : null;
    })();
    return {
      pelvis, ankles, shoulders, neck, head,
      leftHip, rightHip, leftShoulder, rightShoulder,
      shoulderWidth, hipWidth,
    };
  }

  function getCharacterReferencePose2d(character, width, height, frameRect = null) {
    const ref = getCharacterReferencePose(character);
    if (!ref?.pelvis) return null;
    const proj = (p) => (p ? projectWorldToOverlay(p, width, height, frameRect) : null);
    const pelvis = proj(ref.pelvis);
    const upper = averageOverlayPoints([
      proj(ref.neck),
      proj(ref.leftShoulder),
      proj(ref.rightShoulder),
      proj(ref.head),
    ]);
    if (!pelvis || !upper) return null;
    return { pelvis, upper };
  }

  function buildBasisQuaternion(rightVec, upVec, forwardHint = null) {
    if (!three || !rightVec || !upVec) return null;
    const right = rightVec.clone();
    const up = upVec.clone();
    if (right.lengthSq() <= 1e-10 || up.lengthSq() <= 1e-10) return null;
    right.normalize();
    up.normalize();
    let forward = new three.Vector3().crossVectors(right, up);
    if (forward.lengthSq() <= 1e-10) return null;
    forward.normalize();
    if (forwardHint && forward.dot(forwardHint) < 0) {
      forward.multiplyScalar(-1);
    }
    const orthoUp = new three.Vector3().crossVectors(forward, right).normalize();
    const mat = new three.Matrix4().makeBasis(right, orthoUp, forward);
    return new three.Quaternion().setFromRotationMatrix(mat).normalize();
  }

  function getCharacterTorsoBasis(character) {
    const ref = getCharacterReferencePose(character);
    if (!ref?.pelvis || !ref?.leftHip || !ref?.rightHip) return null;
    const rightVec = (() => {
      const shoulderRight = ref.rightShoulder && ref.leftShoulder
        ? ref.rightShoulder.clone().sub(ref.leftShoulder)
        : null;
      const hipRight = ref.rightHip.clone().sub(ref.leftHip);
      if (shoulderRight && shoulderRight.lengthSq() > 1e-10) {
        return shoulderRight.add(hipRight).multiplyScalar(0.5);
      }
      return hipRight;
    })();
    const upTarget = (ref.shoulders || ref.neck || ref.head);
    if (!upTarget) return null;
    const upVec = upTarget.clone().sub(ref.pelvis);
    const forwardHint = new three.Vector3(0, 0, 1).applyQuaternion(character.modelRoot.quaternion).normalize();
    return buildBasisQuaternion(rightVec, upVec, forwardHint);
  }

  function getTargetTorsoBasis(namedJoints) {
    if (!namedJoints || !three) return null;
    const leftHip = namedJointScenePoint(namedJoints, "left_hip");
    const rightHip = namedJointScenePoint(namedJoints, "right_hip");
    const leftShoulder = namedJointScenePoint(namedJoints, "left_shoulder");
    const rightShoulder = namedJointScenePoint(namedJoints, "right_shoulder");
    const pelvis = averageScenePoints([
      namedJointScenePoint(namedJoints, "pelvis"),
      leftHip,
      rightHip,
    ]);
    const upper = averageScenePoints([
      namedJointScenePoint(namedJoints, "neck"),
      leftShoulder,
      rightShoulder,
      namedJointScenePoint(namedJoints, "head"),
    ]);
    if (!pelvis || !upper || !leftHip || !rightHip) return null;
    const shoulderRight = leftShoulder && rightShoulder
      ? rightShoulder.clone().sub(leftShoulder)
      : null;
    const hipRight = rightHip.clone().sub(leftHip);
    const rightVec = shoulderRight && shoulderRight.lengthSq() > 1e-10
      ? shoulderRight.add(hipRight).multiplyScalar(0.5)
      : hipRight;
    const upVec = upper.clone().sub(pelvis);
    const forwardHint = new three.Vector3(0, 0, 1);
    return buildBasisQuaternion(rightVec, upVec, forwardHint);
  }

  function applyCharacterRootOrientation(character, namedJoints) {
    if (!character?.modelRoot || !three) return false;
    const currentBasis = getCharacterTorsoBasis(character);
    const targetBasis = getTargetTorsoBasis(namedJoints);
    if (!currentBasis || !targetBasis) return false;
    const currentForward = new three.Vector3(0, 0, 1).applyQuaternion(currentBasis);
    const targetForward = new three.Vector3(0, 0, 1).applyQuaternion(targetBasis);
    currentForward.y = 0;
    targetForward.y = 0;
    if (currentForward.lengthSq() <= 1e-10 || targetForward.lengthSq() <= 1e-10) return false;
    currentForward.normalize();
    targetForward.normalize();
    const cross = new three.Vector3().crossVectors(currentForward, targetForward);
    const dot = Math.max(-1, Math.min(1, currentForward.dot(targetForward)));
    const angle = Math.atan2(cross.y, dot);
    if (!Number.isFinite(angle) || Math.abs(angle) <= 1e-5) return false;
    const delta = new three.Quaternion().setFromAxisAngle(new three.Vector3(0, 1, 0), angle);
    if (!Number.isFinite(delta.x + delta.y + delta.z + delta.w)) return false;
    character.modelRoot.quaternion.premultiply(delta).normalize();
    character.modelRoot.updateMatrixWorld(true);
    return true;
  }

  function applyCharacterTorsoOrientation(character, namedJoints) {
    if (!character?.modelRoot || !three) return false;
    const currentBasis = getCharacterTorsoBasis(character);
    const targetBasis = getTargetTorsoBasis(namedJoints);
    if (!currentBasis || !targetBasis) return false;
    const delta = targetBasis.clone().multiply(currentBasis.clone().invert()).normalize();
    if (!Number.isFinite(delta.x + delta.y + delta.z + delta.w)) return false;
    const { hipsBone, spineBone, neckBone } = getTorsoDriveBones(character);
    let applied = false;
    if (hipsBone) {
      applied = applyPartialWorldRotationDelta(hipsBone, delta, 0.28, 0.22) || applied;
    }
    if (spineBone) {
      applied = applyPartialWorldRotationDelta(spineBone, delta, 0.72, 0.42) || applied;
    } else if (neckBone) {
      applied = applyPartialWorldRotationDelta(neckBone, delta, 0.35, 0.25) || applied;
    }
    if (applied) {
      character.modelRoot.updateMatrixWorld(true);
    }
    return applied;
  }

  function applyCharacterScreenLean(person, character) {
    if (!character?.modelRoot || !person?.joints3d_named || !three || !camera || !viewport) return false;
    const width = Math.max(64, Math.round(viewport?.clientWidth || 1024));
    const height = Math.max(64, Math.round(viewport?.clientHeight || 768));
    const frameRect = getRenderFrameRect(width, height);
    const current2d = getCharacterReferencePose2d(character, width, height, frameRect);
    if (!current2d?.pelvis || !current2d?.upper) return false;

    const joints = person.joints3d_named;
    const imageW = Math.max(1, Number(person?.image_width || 1));
    const imageH = Math.max(1, Number(person?.image_height || 1));
    const leftHip = namedJointOverlayPoint(joints, "left_hip", frameRect, imageW, imageH);
    const rightHip = namedJointOverlayPoint(joints, "right_hip", frameRect, imageW, imageH);
    const leftShoulder = namedJointOverlayPoint(joints, "left_shoulder", frameRect, imageW, imageH);
    const rightShoulder = namedJointOverlayPoint(joints, "right_shoulder", frameRect, imageW, imageH);
    const pelvis = averageOverlayPoints([
      namedJointOverlayPoint(joints, "pelvis", frameRect, imageW, imageH),
      leftHip,
      rightHip,
    ]);
    const upper = averageOverlayPoints([
      namedJointOverlayPoint(joints, "neck", frameRect, imageW, imageH),
      leftShoulder,
      rightShoulder,
      namedJointOverlayPoint(joints, "head", frameRect, imageW, imageH),
    ]);
    if (!pelvis || !upper) return false;

    const currentVec = new three.Vector2(current2d.upper.x - current2d.pelvis.x, current2d.upper.y - current2d.pelvis.y);
    const targetVec = new three.Vector2(upper.x - pelvis.x, upper.y - pelvis.y);
    if (currentVec.lengthSq() <= 1e-8 || targetVec.lengthSq() <= 1e-8) return false;
    const wrapAngle = (a) => {
      let out = a;
      while (out > Math.PI) out -= Math.PI * 2;
      while (out < -Math.PI) out += Math.PI * 2;
      return out;
    };
    const currentAngle = Math.atan2(currentVec.y, currentVec.x);
    const targetAngle = Math.atan2(targetVec.y, targetVec.x);
    const leanDelta = Math.max(-0.45, Math.min(0.45, wrapAngle(targetAngle - currentAngle)));
    if (!Number.isFinite(leanDelta) || Math.abs(leanDelta) <= 1e-4) return false;

    const cameraForward = getCameraForwardVector();
    if (!cameraForward) return false;
    const leanQ = new three.Quaternion().setFromAxisAngle(cameraForward, leanDelta);
    const { hipsBone, spineBone } = getTorsoDriveBones(character);
    let applied = false;
    if (hipsBone) {
      applied = applyPartialWorldRotationDelta(hipsBone, leanQ, 0.9, 0.34) || applied;
    }
    if (spineBone) {
      applied = applyPartialWorldRotationDelta(spineBone, leanQ, 0.28, 0.14) || applied;
    }
    if (applied) {
      character.modelRoot.updateMatrixWorld(true);
    }
    return applied;
  }

  function scaleCharacterToTargetBody(character, namedJoints, bodyMetrics = null) {
    if (!character || !namedJoints) return;
    const ref = getCharacterReferencePose(character);
    if (!ref?.pelvis || !ref?.ankles || !ref?.head) return;
    const targetPelvis = averageScenePoints([
      namedJointScenePoint(namedJoints, "left_hip"),
      namedJointScenePoint(namedJoints, "right_hip"),
      namedJointScenePoint(namedJoints, "pelvis"),
    ]);
    const targetAnkles = averageScenePoints([
      namedJointScenePoint(namedJoints, "left_ankle"),
      namedJointScenePoint(namedJoints, "right_ankle"),
    ]);
    const targetHead = averageScenePoints([
      namedJointScenePoint(namedJoints, "head"),
      namedJointScenePoint(namedJoints, "neck"),
    ]);
    const targetShoulderWidth = (() => {
      const l = namedJointScenePoint(namedJoints, "left_shoulder");
      const r = namedJointScenePoint(namedJoints, "right_shoulder");
      return l && r ? l.distanceTo(r) : null;
    })();
    const targetHipWidth = (() => {
      const l = namedJointScenePoint(namedJoints, "left_hip");
      const r = namedJointScenePoint(namedJoints, "right_hip");
      return l && r ? l.distanceTo(r) : null;
    })();
    if (!targetPelvis || !targetAnkles || !targetHead) return;
    const refHeight = ref.head.distanceTo(ref.ankles);
    const targetHeight = targetHead.distanceTo(targetAnkles);
    if (!Number.isFinite(refHeight) || !Number.isFinite(targetHeight) || refHeight <= 1e-6 || targetHeight <= 1e-6) return;
    const heightRatio = Math.max(0.3, Math.min(3.0, targetHeight / refHeight));

    let widthRatio = heightRatio;
    const widthCandidates = [];
    if (Number.isFinite(ref.shoulderWidth) && ref.shoulderWidth > 1e-6 && Number.isFinite(targetShoulderWidth) && targetShoulderWidth > 1e-6) {
      widthCandidates.push(targetShoulderWidth / ref.shoulderWidth);
    }
    if (Number.isFinite(ref.hipWidth) && ref.hipWidth > 1e-6 && Number.isFinite(targetHipWidth) && targetHipWidth > 1e-6) {
      widthCandidates.push(targetHipWidth / ref.hipWidth);
    }
    if (widthCandidates.length) {
      widthRatio = widthCandidates.reduce((a, b) => a + b, 0) / widthCandidates.length;
    }
    const metricsRatio = Number(bodyMetrics?.shoulder_hip_ratio || 0);
    if (Number.isFinite(metricsRatio) && metricsRatio > 0) {
      if (metricsRatio >= 1.18) {
        widthRatio *= 1.05;
      } else if (metricsRatio <= 1.02) {
        widthRatio *= 0.97;
      }
    }
    widthRatio = Math.max(0.35, Math.min(2.2, widthRatio));
    // Keep scaling isotropic to avoid visibly distorting mesh proportions.
    // Mesh selection should carry most body-shape variation; non-uniform root scaling
    // creates stretched feet/hands and skewed silhouettes.
    const uniformRatio = Math.max(0.3, Math.min(3.0, ((heightRatio * 2.0) + widthRatio) / 3.0));
    character.modelRoot.scale.set(uniformRatio, uniformRatio, uniformRatio);
    character.modelRoot.updateMatrixWorld(true);
  }

  function setCharacterPelvisTo(character, targetPelvis) {
    if (!character || !targetPelvis || !three) return;
    const ref = getCharacterReferencePose(character);
    if (!ref?.pelvis) return;
    const delta = targetPelvis.clone().sub(ref.pelvis);
    character.modelRoot.position.add(delta);
    character.modelRoot.updateMatrixWorld(true);
  }

  function applyAdvancedPoseToCharacter(person, character) {
    if (!person?.joints3d_named) {
      applyDetectedPoseToCharacter(person, character);
      return;
    }
    if (!character || !three) {
      throw new Error("Character is not ready for advanced pose application.");
    }

    const joints = person.joints3d_named;
    const width = Math.max(64, Math.round(viewport?.clientWidth || 1024));
    const height = Math.max(64, Math.round(viewport?.clientHeight || 768));
    const frameRect = getRenderFrameRect(width, height);
    scaleCharacterToTargetBody(character, joints, person?.body_metrics || null);
    applyCharacterRootOrientation(character, joints);

    const currentRef = getCharacterReferencePose(character);
    const pelvisDepth = currentRef?.pelvis ? worldDepthOnCameraForward(currentRef.pelvis) : null;
    const pelvisTarget = averageScenePoints([
      namedJointProjectedWorldPoint(person, "pelvis", frameRect, pelvisDepth),
      namedJointProjectedWorldPoint(person, "left_hip", frameRect, pelvisDepth),
      namedJointProjectedWorldPoint(person, "right_hip", frameRect, pelvisDepth),
    ]);
    if (pelvisTarget) {
      setCharacterPelvisTo(character, pelvisTarget);
    }
    applyCharacterTorsoOrientation(character, joints);
    applyCharacterScreenLean(person, character);

    const hasFaceDrivenHead = Array.isArray(person?.face?.kps) && person.face.kps.length >= 5;
    const segments = [
      { start: "right_shoulder", end: "right_elbow", driveIdx: 2, endIdx: 3 },
      { start: "right_elbow", end: "right_wrist", driveIdx: 3, endIdx: 4 },
      { start: "left_shoulder", end: "left_elbow", driveIdx: 5, endIdx: 6 },
      { start: "left_elbow", end: "left_wrist", driveIdx: 6, endIdx: 7 },
      { start: "right_hip", end: "right_knee", driveIdx: 8, endIdx: 9 },
      { start: "right_knee", end: "right_ankle", driveIdx: 9, endIdx: 10 },
      { start: "right_ankle", end: "right_foot", side: "right", maxAngle: 0.38 },
      { start: "left_hip", end: "left_knee", driveIdx: 11, endIdx: 12 },
      { start: "left_knee", end: "left_ankle", driveIdx: 12, endIdx: 13 },
      { start: "left_ankle", end: "left_foot", side: "left", maxAngle: 0.38 },
    ];
    if (!hasFaceDrivenHead) {
      segments.push({ start: "neck", end: "head", kind: "head", maxAngle: 0.45 });
    }

    let applied = 0;
    for (let iter = 0; iter < 4; iter += 1) {
      for (const seg of segments) {
        const { driveBone, endBone } = seg.kind === "head"
          ? getHeadDriveBones(character)
          : seg.side
            ? getFootDriveBones(character, seg.side)
            : {
                driveBone: getBoneForOpenPoseIndex(character, seg.driveIdx),
                endBone: getBoneForOpenPoseIndex(character, seg.endIdx),
              };
        if (!driveBone || !endBone) continue;
        const currentStart = new three.Vector3();
        const currentEnd = new three.Vector3();
        driveBone.getWorldPosition(currentStart);
        endBone.getWorldPosition(currentEnd);
        const startDepth = worldDepthOnCameraForward(currentStart);
        const endDepth = worldDepthOnCameraForward(currentEnd);
        const startTarget = namedJointProjectedWorldPoint(person, seg.start, frameRect, startDepth);
        const endTarget = namedJointProjectedWorldPoint(person, seg.end, frameRect, endDepth);
        if (!startTarget || !endTarget) continue;
        const targetDir = endTarget.clone().sub(startTarget);
        if (rotateBoneTowardWorldDirection(driveBone, currentEnd, targetDir, seg.maxAngle)) {
          applied += 1;
        }
      }
      character.modelRoot.updateMatrixWorld(true);
    }

    if (hasFaceDrivenHead && applyFaceDrivenHeadOrientation(person, character)) {
      applied += 1;
      character.modelRoot.updateMatrixWorld(true);
    }

    if (!applied) {
      throw new Error("Advanced 3D retarget produced no valid bone updates.");
    }

    refreshSkinnedMeshes();
    syncInputsFromBone();
  }

  function setSceneCameraFromAdvancedPayload(payload) {
    if (!camera || !orbit) return false;
    const K = Array.isArray(payload?.camera_intrinsics) ? payload.camera_intrinsics : null;
    const imageW = Math.max(1, Math.round(Number(payload?.image_width || 1024)));
    const imageH = Math.max(1, Math.round(Number(payload?.image_height || 768)));
    if (!K || !Array.isArray(K[1])) return false;
    const fy = Number(K[1][1]);
    if (!Number.isFinite(fy) || fy <= 1e-6) return false;

    const fov = 2 * Math.atan(imageH / (2 * fy)) * (180 / Math.PI);
    const slot = getActiveCameraSlot();
    slot.resolution = [imageW, imageH];
    slot.fov = Math.max(10, Math.min(140, fov));
    slot.position = [0, 0, 0];

    const pelvisPoints = [];
    const persons = Array.isArray(payload?.persons) ? payload.persons : [];
    persons.forEach((person) => {
      const pelvis = averageScenePoints([
        namedJointScenePoint(person?.joints3d_named, "pelvis"),
        namedJointScenePoint(person?.joints3d_named, "left_hip"),
        namedJointScenePoint(person?.joints3d_named, "right_hip"),
      ]);
      if (pelvis) pelvisPoints.push(pelvis);
    });
    const averagePelvis = averageScenePoints(pelvisPoints) || new three.Vector3(0, 1, -2);
    // Keep camera optical axis aligned with the original image.
    // Multi-HMR joints are already in camera space, so rotating the camera to look at the pelvis
    // destroys the original projection geometry.
    const target = new three.Vector3(0, 0, Number.isFinite(averagePelvis.z) ? averagePelvis.z : -2);
    slot.target = [target.x, target.y, target.z];
    setCameraFromSlot(slot);
    camera.position.set(0, 0, 0);
    camera.fov = slot.fov;
    camera.updateProjectionMatrix();
    orbit.target.copy(target);
    orbit.update();
    readCurrentCameraToSlot(activeCameraIndex);
    resizeRenderer();
    return true;
  }

  function applyCameraHintFromDetection(detectionPoints, frameRect, activeCharacter) {
    if (!camera || !orbit || !Array.isArray(detectionPoints) || !activeCharacter) return;
    const overlayW = Math.max(1, Math.round(viewport?.clientWidth || 1));
    const overlayH = Math.max(1, Math.round(viewport?.clientHeight || 1));
    const current2d = computeOpenPosePositions2d(overlayW, overlayH, frameRect, activeCharacter);
    const currentStats = computeOverlayPoseStats(current2d);
    const targetStats = computeOverlayPoseStats(detectionPoints);
    if (!currentStats || !targetStats) return;

    const forward = getCameraForwardVector();
    if (!forward) return;

    const camToTarget = orbit.target.clone().sub(camera.position);
    const baseDepth = Math.max(0.05, camToTarget.dot(forward));
    const worldCurrentCenter = overlayPointToWorldAtDepth(currentStats.centerX, currentStats.centerY, frameRect, baseDepth);
    const worldTargetCenter = overlayPointToWorldAtDepth(targetStats.centerX, targetStats.centerY, frameRect, baseDepth);
    if (worldCurrentCenter && worldTargetCenter) {
      const delta = worldCurrentCenter.clone().sub(worldTargetCenter);
      camera.position.add(delta);
      orbit.target.add(delta);
    }

    const currentHeight = Math.max(1e-6, Number(currentStats.height || 0));
    const targetHeight = Math.max(1e-6, Number(targetStats.height || 0));
    if (currentHeight > 0.0 && targetHeight > 0.0) {
      const scaleRatio = Math.max(0.45, Math.min(2.2, currentHeight / targetHeight));
      const dir = camera.position.clone().sub(orbit.target);
      const dist = dir.length();
      if (dist > 1e-5) {
        const newDist = Math.max(0.08, dist * scaleRatio);
        camera.position.copy(orbit.target.clone().add(dir.normalize().multiplyScalar(newDist)));
      }
    }

    camera.updateProjectionMatrix();
    orbit.update();
    readCurrentCameraToSlot(activeCameraIndex);
  }

  function applyDetectedPoseToCharacter(detectionPayload, targetCharacter = null) {
    const active = targetCharacter || getActiveCharacter();
    if (!active || !Array.isArray(active.bones) || !active.bones.length) {
      throw new Error("No active character loaded.");
    }
    const points = Array.isArray(detectionPayload?.keypoints_openpose18)
      ? detectionPayload.keypoints_openpose18
      : null;
    if (!points || !points.length) {
      throw new Error("Pose payload does not contain keypoints.");
    }

    syncProjectionState();
    const viewportW = Math.max(1, Math.round(viewport?.clientWidth || 1));
    const viewportH = Math.max(1, Math.round(viewport?.clientHeight || 1));
    const frameRect = getRenderFrameRect(viewportW, viewportH);
    const worldNow = getCharacterOpenPoseWorld(active);

    const depthValues = worldNow
      .map((p) => worldDepthOnCameraForward(p))
      .filter((v) => Number.isFinite(v) && v > 0.01);
    const defaultDepth = depthValues.length
      ? depthValues.reduce((a, b) => a + b, 0) / depthValues.length
      : Math.max(0.25, camera.position.distanceTo(orbit.target));

    const detectionOverlay = Array.from({ length: OPENPOSE_NAMES.length }, () => null);
    const targetWorld = Array.from({ length: OPENPOSE_NAMES.length }, () => null);
    for (let i = 0; i < OPENPOSE_NAMES.length; i += 1) {
      const point = points[i];
      const conf = Number(point?.confidence ?? 0);
      if (!point || !Number.isFinite(conf) || conf < 0.05) continue;
      const ov = detectionPointToOverlay(
        point,
        frameRect,
        Number(detectionPayload?.image_width || 0),
        Number(detectionPayload?.image_height || 0),
      );
      if (!ov) continue;
      detectionOverlay[i] = ov;
      const depth = Number.isFinite(worldDepthOnCameraForward(worldNow[i])) && worldDepthOnCameraForward(worldNow[i]) > 0.01
        ? worldDepthOnCameraForward(worldNow[i])
        : defaultDepth;
      const wp = overlayPointToWorldAtDepth(ov.x, ov.y, frameRect, depth);
      if (wp) {
        targetWorld[i] = wp;
      }
    }

    // Retarget major chains. Two passes improves convergence for longer chains.
    const segments = [
      { from: 2, to: 3, drive: 2 },
      { from: 3, to: 4, drive: 3 },
      { from: 5, to: 6, drive: 5 },
      { from: 6, to: 7, drive: 6 },
      { from: 8, to: 9, drive: 8 },
      { from: 9, to: 10, drive: 9 },
      { from: 11, to: 12, drive: 11 },
      { from: 12, to: 13, drive: 12 },
      { from: 1, to: 0, drive: 1 },
    ];

    let applied = 0;
    for (let iter = 0; iter < 2; iter += 1) {
      for (const seg of segments) {
        const startTarget = targetWorld[seg.from];
        const endTarget = targetWorld[seg.to];
        if (!startTarget || !endTarget) continue;
        const driveBone = getBoneForOpenPoseIndex(active, seg.drive);
        const toBone = getBoneForOpenPoseIndex(active, seg.to);
        if (!driveBone || !toBone) continue;
        const currentEnd = new three.Vector3();
        toBone.getWorldPosition(currentEnd);
        const targetDir = endTarget.clone().sub(startTarget);
        if (rotateBoneTowardWorldDirection(driveBone, currentEnd, targetDir)) {
          applied += 1;
        }
      }
      active.modelRoot?.updateMatrixWorld?.(true);
    }

    if (!applied) {
      throw new Error("Could not map detected pose to current rig.");
    }

    applyCameraHintFromDetection(detectionOverlay, frameRect, active);
    refreshSkinnedMeshes();
    syncInputsFromBone();
    scheduleHistoryCapture(90);
  }

  let riggedMeshListCache = null;
  const riggedMeshBufferCache = new Map();

  async function getRiggedMeshList(force = false) {
    if (!force && Array.isArray(riggedMeshListCache)) {
      return riggedMeshListCache.slice();
    }
    const listResp = await fetch("/ess/rigged/list", { cache: "no-store" });
    if (!listResp.ok) {
      throw new Error(`mesh list failed: ${listResp.status}`);
    }
    const payload = await listResp.json();
    const items = Array.isArray(payload?.items) ? payload.items.map((v) => String(v || "")) : [];
    riggedMeshListCache = items.filter(Boolean);
    return riggedMeshListCache.slice();
  }

  async function fetchRiggedMeshBuffer(meshName) {
    const key = String(meshName || "");
    if (!key) throw new Error("mesh name is empty");
    const cached = riggedMeshBufferCache.get(key);
    if (cached instanceof ArrayBuffer) {
      return cached.slice(0);
    }
    const fileResp = await fetch(`/ess/rigged/get?name=${encodeURIComponent(key)}`, {
      cache: "no-store",
    });
    if (!fileResp.ok) {
      throw new Error(`mesh load failed (${key}): ${fileResp.status}`);
    }
    const buf = await fileResp.arrayBuffer();
    riggedMeshBufferCache.set(key, buf.slice(0));
    return buf;
  }

  function chooseMeshForPerson(person, availableMeshes) {
    const items = Array.isArray(availableMeshes) ? availableMeshes.map((v) => String(v || "")) : [];
    if (!items.length) return null;
    const lc = new Set(items.map((n) => n.toLowerCase()));
    const has = (name) => (name && lc.has(String(name).toLowerCase())) ? items.find((m) => m.toLowerCase() === String(name).toLowerCase()) : null;

    const suggestion = has(person?.mesh_suggestion);
    if (suggestion) return suggestion;

    const gender = String(person?.gender || "unknown").toLowerCase();
    const tryList = [];
    if (gender === "male") {
      tryList.push("male_young.fbx");
    } else {
      tryList.push("female_young.fbx");
    }
    tryList.push("female_young.fbx", "male_young.fbx");
    for (const n of tryList) {
      const ok = has(n);
      if (ok) return ok;
    }
    return items[0];
  }

  function computeAllCharactersBounds() {
    if (!three || !Array.isArray(characters) || !characters.length) return null;
    let union = null;
    characters.forEach((ch) => {
      if (!ch?.modelRoot) return;
      const box = computeModelBounds(ch.modelRoot);
      if (!box) return;
      if (!union) {
        union = box.clone();
      } else {
        union.union(box);
      }
    });
    return union;
  }

  function placeCharactersByDetectionHints(persons) {
    if (!three || !Array.isArray(persons) || !persons.length || !Array.isArray(characters) || !characters.length) return;
    const pairs = [];
    for (let i = 0; i < Math.min(persons.length, characters.length); i += 1) {
      pairs.push({ person: persons[i], character: characters[i] });
    }
    if (!pairs.length) return;

    const widths = [];
    pairs.forEach(({ character }) => {
      const b = computeModelBounds(character.modelRoot);
      if (!b) return;
      const s = b.getSize(new three.Vector3());
      if (Number.isFinite(s.x) && s.x > 1e-5) widths.push(s.x);
    });
    const avgWidth = widths.length ? widths.reduce((a, b) => a + b, 0) / widths.length : 0.45;
    const spread = Math.max(avgWidth * Math.max(3.0, pairs.length * 1.2), 1.8);

    const validHeights = pairs
      .map((p) => Number(p.person?.scale_hint?.height || 0))
      .filter((v) => Number.isFinite(v) && v > 1e-4);
    const medianHeight = validHeights.length
      ? validHeights.slice().sort((a, b) => a - b)[Math.floor(validHeights.length * 0.5)]
      : 0.35;

    pairs.forEach(({ person, character }) => {
      const cxNorm = Number(person?.center_hint?.x);
      const hNorm = Number(person?.scale_hint?.height || medianHeight);
      const targetX = Number.isFinite(cxNorm) ? (cxNorm - 0.5) * spread : 0;
      const depthDelta = Number.isFinite(hNorm) && hNorm > 1e-5
        ? (medianHeight - hNorm) * Math.max(1.2, spread * 0.6)
        : 0.0;

      const bounds = computeModelBounds(character.modelRoot);
      if (!bounds) return;
      const center = bounds.getCenter(new three.Vector3());
      const minY = bounds.min.y;
      const shift = new three.Vector3(
        targetX - center.x,
        -minY,
        depthDelta - center.z * 0.25,
      );
      character.modelRoot.position.add(shift);
      character.modelRoot.updateMatrixWorld(true);
    });
  }

  function frameAllCharacters() {
    if (!camera || !orbit || !renderer) return;
    const box = computeAllCharactersBounds();
    if (!box) return;
    const size = box.getSize(new three.Vector3());
    const center = box.getCenter(new three.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(maxDim) || maxDim <= 0) return;
    const dist = maxDim * 2.0 / Math.tan((camera.fov * Math.PI) / 360);
    const dir = new three.Vector3(0, 0.35, 1).normalize();
    const newPos = center.clone().add(dir.multiplyScalar(Math.max(0.5, dist)));
    camera.position.copy(newPos);
    camera.lookAt(center);
    orbit.target.copy(center);
    orbit.update();
    updateOrbitDistanceLimits();
    updateCameraClipping();
    readCurrentCameraToSlot(activeCameraIndex);
    resizeRenderer();
  }

  function computeDetectionUnion2d(persons, frameRect) {
    if (!Array.isArray(persons) || !persons.length || !frameRect) return null;
    const points = [];
    persons.forEach((p) => {
      const kps = Array.isArray(p?.keypoints_openpose18) ? p.keypoints_openpose18 : [];
      kps.forEach((kp) => {
        const ov = detectionPointToOverlay(
          kp,
          frameRect,
          Number(p?.image_width || 0),
          Number(p?.image_height || 0),
        );
        if (ov) points.push(ov);
      });
    });
    return computeOverlayPoseStats(points);
  }

  function computeCurrentUnion2d(frameRect) {
    if (!Array.isArray(characters) || !characters.length) return null;
    const overlayW = Math.max(1, Math.round(viewport?.clientWidth || 1));
    const overlayH = Math.max(1, Math.round(viewport?.clientHeight || 1));
    const points = [];
    characters.forEach((ch) => {
      const pose2d = computeOpenPosePositions2d(overlayW, overlayH, frameRect, ch);
      pose2d.forEach((p) => {
        if (p) points.push(p);
      });
    });
    return computeOverlayPoseStats(points);
  }

  function alignCameraToDetectionUnion(persons, frameRect) {
    if (!camera || !orbit || !frameRect) return;
    const cur = computeCurrentUnion2d(frameRect);
    const tgt = computeDetectionUnion2d(persons, frameRect);
    if (!cur || !tgt) return;

    const forward = getCameraForwardVector();
    if (!forward) return;
    const camToTarget = orbit.target.clone().sub(camera.position);
    const baseDepth = Math.max(0.05, camToTarget.dot(forward));
    const worldCur = overlayPointToWorldAtDepth(cur.centerX, cur.centerY, frameRect, baseDepth);
    const worldTgt = overlayPointToWorldAtDepth(tgt.centerX, tgt.centerY, frameRect, baseDepth);
    if (worldCur && worldTgt) {
      const delta = worldCur.clone().sub(worldTgt);
      camera.position.add(delta);
      orbit.target.add(delta);
    }

    const curH = Math.max(1e-6, Number(cur.height || 0));
    const tgtH = Math.max(1e-6, Number(tgt.height || 0));
    if (curH > 0 && tgtH > 0) {
      const scaleRatio = Math.max(0.45, Math.min(2.2, curH / tgtH));
      const dir = camera.position.clone().sub(orbit.target);
      const dist = dir.length();
      if (dist > 1e-5) {
        camera.position.copy(orbit.target.clone().add(dir.normalize().multiplyScalar(Math.max(0.08, dist * scaleRatio))));
      }
    }
    camera.updateProjectionMatrix();
    orbit.update();
    readCurrentCameraToSlot(activeCameraIndex);
  }

  async function rebuildSceneFromDetectionPayload(payload) {
    const advancedPersons = Array.isArray(payload?.persons)
      ? payload.persons
          .filter((p) => p?.joints3d_named && typeof p.joints3d_named === "object")
          .map((p) => ({
            ...p,
            image_width: Number(payload?.image_width || p?.image_width || 0),
            image_height: Number(payload?.image_height || p?.image_height || 0),
          }))
      : [];
    if (String(payload?.engine || "") === "advanced_multihmr" && advancedPersons.length) {
      const available = await getRiggedMeshList();
      if (!available.length) {
        throw new Error("No rigged meshes found in meshes/human_rig.");
      }

      clearCharactersFromScene();
      for (let i = 0; i < advancedPersons.length; i += 1) {
        const person = advancedPersons[i];
        const meshName = chooseMeshForPerson(person, available);
        if (!meshName) continue;
        const buf = await fetchRiggedMeshBuffer(meshName);
        const character = await loadModelFromFile(
          { name: meshName, arrayBuffer: async () => buf },
          buf.slice(0),
          { name: meshName },
        );
        setSceneCameraFromAdvancedPayload(payload);
        applyAdvancedPoseToCharacter(person, character);
      }

      if (!characters.length) {
        throw new Error("No characters could be created from advanced detections.");
      }

      if (!setSceneCameraFromAdvancedPayload(payload)) {
        frameAllCharacters();
      }
      refreshCharacterTabs();
      applyActiveCharacter(characters[0] || null);
      populateBones();
      refreshSkinnedMeshes();
      scheduleHistoryCapture(120);
      return;
    }

    const personsRaw = Array.isArray(payload?.persons) && payload.persons.length
      ? payload.persons
      : [payload];
    const persons = personsRaw
      .filter((p) => Array.isArray(p?.keypoints_openpose18) && p.keypoints_openpose18.length)
      .map((p) => ({
        ...p,
        image_width: Number(payload?.image_width || p?.image_width || 0),
        image_height: Number(payload?.image_height || p?.image_height || 0),
      }));
    if (!persons.length) {
      throw new Error("No detectable people in image.");
    }

    const available = await getRiggedMeshList();
    if (!available.length) {
      throw new Error("No rigged meshes found in meshes/human_rig.");
    }

    clearCharactersFromScene();
    for (let i = 0; i < persons.length; i += 1) {
      const person = persons[i];
      const meshName = chooseMeshForPerson(person, available);
      if (!meshName) continue;
      const buf = await fetchRiggedMeshBuffer(meshName);
      const character = await loadModelFromFile(
        { name: meshName, arrayBuffer: async () => buf },
        buf.slice(0),
        { name: meshName },
      );
      applyDetectedPoseToCharacter(person, character);
    }

    if (!characters.length) {
      throw new Error("No characters could be created from detections.");
    }

    placeCharactersByDetectionHints(persons);
    frameAllCharacters();
    const frameRect = getRenderFrameRect(
      Math.max(1, Math.round(viewport?.clientWidth || 1)),
      Math.max(1, Math.round(viewport?.clientHeight || 1)),
    );
    alignCameraToDetectionUnion(persons, frameRect);
    refreshCharacterTabs();
    applyActiveCharacter(characters[0] || null);
    populateBones();
    refreshSkinnedMeshes();
    scheduleHistoryCapture(120);
  }

  function findMirroredBoneForSelection() {
    if (!selectedBone || !Array.isArray(bones) || !bones.length) return null;
    const selectedPath = getBonePath(selectedBone);
    const selectedPathCandidates = buildMirroredNameCandidates(selectedPath);
    if (selectedPathCandidates.length) {
      const byPath = new Map();
      bones.forEach((b) => byPath.set(getBonePath(b), b));
      for (const p of selectedPathCandidates) {
        const match = byPath.get(p);
        if (match && match.uuid !== selectedBone.uuid) {
          return match;
        }
      }
    }

    const nameCandidates = buildMirroredNameCandidates(selectedBone.name || "");
    if (!nameCandidates.length) return null;
    const lowerSet = new Set(nameCandidates.map((n) => n.toLowerCase()));

    const selectedDepth = selectedPath ? selectedPath.split("/").length : 0;
    const selectedRoot = selectedPath ? selectedPath.split("/")[0] : "";
    const selectedParent = selectedBone.parent?.isBone ? selectedBone.parent : null;
    const mirroredParent = selectedParent ? findMirroredBoneForName(bones, selectedParent.name) : null;

    const scored = [];
    bones.forEach((b) => {
      if (!b || b.uuid === selectedBone.uuid) return;
      const bName = String(b.name || "").toLowerCase();
      if (!lowerSet.has(bName)) return;
      const path = getBonePath(b);
      const depth = path ? path.split("/").length : 0;
      const root = path ? path.split("/")[0] : "";
      let score = 0;
      score += Math.abs(depth - selectedDepth) * 5;
      if (root === selectedRoot) score -= 8;
      if (mirroredParent && b.parent?.uuid === mirroredParent.uuid) score -= 20;
      scored.push({ bone: b, score });
    });

    if (!scored.length) return null;
    scored.sort((a, b) => a.score - b.score);
    return scored[0].bone || null;
  }

  function updateTransformContextUI() {
    if (!selectedBone) {
      selectedBoneInfo.textContent = "Selected: (none)";
      mirroredBone = null;
      mirrorEnabled = false;
      mirrorCheck.checked = false;
      mirrorWrap.style.display = "none";
      return;
    }
    selectedBoneInfo.textContent = `Selected: ${selectedBone.name || "(unnamed)"}`;
    const pair = findMirroredBoneForSelection();
    mirroredBone = pair;
    if (pair) {
      mirrorWrap.style.display = "flex";
      mirrorText.textContent = `Mirror with ${pair.name}`;
    } else {
      mirrorEnabled = false;
      mirrorCheck.checked = false;
      mirrorWrap.style.display = "none";
    }
  }

  function resetMirrorSelectionState() {
    mirrorEnabled = false;
    mirroredBone = null;
    mirrorCheck.checked = false;
    transformDragging = false;
  }

  function applyMirroredFromSelectedChannel(kind, axis = null) {
    if (!mirrorEnabled || !selectedBone || !mirroredBone) return;
    if (kind === "rotation") {
      if (axis === null) {
        const active = getActiveCharacter();
        const bindMap = active?.bindPoseByUuid instanceof Map ? active.bindPoseByUuid : null;
        const srcBind = bindMap?.get(selectedBone.uuid);
        const dstBind = bindMap?.get(mirroredBone.uuid);
        if (
          srcBind && dstBind
          && Array.isArray(srcBind.quaternion) && srcBind.quaternion.length === 4
          && Array.isArray(dstBind.quaternion) && dstBind.quaternion.length === 4
        ) {
          // Mirror rotation delta from bind-pose to handle asymmetric local axes.
          const qSrcBind = new three.Quaternion(
            srcBind.quaternion[0],
            srcBind.quaternion[1],
            srcBind.quaternion[2],
            srcBind.quaternion[3],
          );
          const qDstBind = new three.Quaternion(
            dstBind.quaternion[0],
            dstBind.quaternion[1],
            dstBind.quaternion[2],
            dstBind.quaternion[3],
          );
          const qSrcBindInv = qSrcBind.clone().invert();
          const qDelta = qSrcBindInv.multiply(selectedBone.quaternion.clone());
          const qDeltaMirrored = new three.Quaternion(qDelta.x, -qDelta.y, -qDelta.z, qDelta.w).normalize();
          const qFinal = qDstBind.clone().multiply(qDeltaMirrored).normalize();
          mirroredBone.quaternion.copy(qFinal);
        } else {
          mirroredBone.rotation.set(
            selectedBone.rotation.x * MIRROR_ROT_SIGN.x,
            selectedBone.rotation.y * MIRROR_ROT_SIGN.y,
            selectedBone.rotation.z * MIRROR_ROT_SIGN.z,
          );
        }
      } else {
        mirroredBone.rotation[axis] = selectedBone.rotation[axis] * (MIRROR_ROT_SIGN[axis] ?? 1);
      }
    } else if (kind === "position") {
      if (axis === null) {
        mirroredBone.position.set(
          selectedBone.position.x * MIRROR_POS_SIGN.x,
          selectedBone.position.y * MIRROR_POS_SIGN.y,
          selectedBone.position.z * MIRROR_POS_SIGN.z,
        );
      } else {
        mirroredBone.position[axis] = selectedBone.position[axis] * (MIRROR_POS_SIGN[axis] ?? 1);
      }
    } else if (kind === "scale") {
      if (axis === null || axis === "u") {
        mirroredBone.scale.copy(selectedBone.scale);
      } else {
        mirroredBone.scale[axis] = selectedBone.scale[axis];
      }
    }
    mirroredBone.updateMatrixWorld(true);
  }

  function applyMirroredFromSelectedAll() {
    applyMirroredFromSelectedChannel("rotation", null);
    applyMirroredFromSelectedChannel("position", null);
    applyMirroredFromSelectedChannel("scale", null);
    refreshSkinnedMeshes();
  }

  function applyMirroredForTransformMode() {
    if (!mirrorEnabled || !selectedBone || !mirroredBone) return;
    const mode = transform?.getMode?.() || "rotate";
    if (mode === "translate") {
      applyMirroredFromSelectedChannel("position", null);
    } else if (mode === "scale") {
      applyMirroredFromSelectedChannel("scale", null);
    } else {
      applyMirroredFromSelectedChannel("rotation", null);
    }
    refreshSkinnedMeshes();
  }

  function projectWorldToOverlay(worldPos, width, height, frameRect = null) {
    const pos = worldPos.clone().project(camera);
    const xNorm = (pos.x * 0.5 + 0.5);
    const yNorm = (-pos.y * 0.5 + 0.5);
    if (frameRect && Number.isFinite(frameRect.width) && Number.isFinite(frameRect.height)) {
      return {
        x: frameRect.x + xNorm * frameRect.width,
        y: frameRect.y + yNorm * frameRect.height,
      };
    }
    return {
      x: xNorm * width,
      y: yNorm * height,
    };
  }

  function syncProjectionState() {
    if (!camera || !scene) return;
    orbit?.update?.();
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    scene.updateMatrixWorld(true);
  }

  function ensureFaceLandmarkCache(character, width, height) {
    if (!character || !three || !camera) return null;
    if (character.faceLandmarkCache?.headUuid && character.faceLandmarkCache?.local) {
      return character.faceLandmarkCache;
    }
    const solved = synthesizeHeadLandmarks(
      three,
      character.boneNameMap,
      camera,
      width,
      height,
      character.modelRoot,
      true,
    );
    if (!solved?.headBone || !solved?.world) return null;

    const invHead = new three.Matrix4().copy(solved.headBone.matrixWorld).invert();
    character.faceLandmarkCache = {
      headUuid: solved.headBone.uuid,
      local: {
        nose: solved.world.nose.clone().applyMatrix4(invHead),
        reye: solved.world.reye.clone().applyMatrix4(invHead),
        leye: solved.world.leye.clone().applyMatrix4(invHead),
        rear: solved.world.rear.clone().applyMatrix4(invHead),
        lear: solved.world.lear.clone().applyMatrix4(invHead),
      },
    };
    return character.faceLandmarkCache;
  }

  function getCachedFaceLandmarksWorld(character, width, height) {
    if (!character || !three || !camera) return null;
    let cache = ensureFaceLandmarkCache(character, width, height);
    if (!cache) return null;

    let headBone = character.bones?.find?.((b) => b.uuid === cache.headUuid) || null;
    if (!headBone) {
      character.faceLandmarkCache = null;
      const refreshed = ensureFaceLandmarkCache(character, width, height);
      if (!refreshed) return null;
      cache = refreshed;
      headBone = character.bones?.find?.((b) => b.uuid === refreshed.headUuid) || null;
      if (!headBone) return null;
    }

    headBone.updateMatrixWorld(true);
    const m = headBone.matrixWorld;
    const world = {
      nose: cache.local.nose.clone().applyMatrix4(m),
      reye: cache.local.reye.clone().applyMatrix4(m),
      leye: cache.local.leye.clone().applyMatrix4(m),
      rear: cache.local.rear.clone().applyMatrix4(m),
      lear: cache.local.lear.clone().applyMatrix4(m),
    };
    const earTighten = 0.34;
    world.rear.lerp(world.reye, earTighten);
    world.lear.lerp(world.leye, earTighten);
    return world;
  }

  function getCachedFaceLandmarks2d(character, width, height, frameRect = null) {
    if (!character || !three || !camera) return null;
    const world = getCachedFaceLandmarksWorld(character, width, height);
    if (!world) return null;

    const points2d = {
      nose: projectWorldToOverlay(world.nose, width, height, frameRect),
      reye: projectWorldToOverlay(world.reye, width, height, frameRect),
      leye: projectWorldToOverlay(world.leye, width, height, frameRect),
      rear: projectWorldToOverlay(world.rear, width, height, frameRect),
      lear: projectWorldToOverlay(world.lear, width, height, frameRect),
    };

    return points2d;
  }

  function applyActiveCharacter(character) {
    const prevSelectedBoneUuid = selectedBone?.uuid || null;
    const ch = character || getActiveCharacter();
    if (!ch) {
      activeCharacterId = null;
      sharedSession.activeCharacterId = null;
      modelRoot = null;
      bones = [];
      boneNameMap = new Map();
      selectedBone = null;
      resetMirrorSelectionState();
      updateSelectedBoneVisuals();
      modelBadge.textContent = "No model loaded";
      boneList.innerHTML = "";
      transform?.detach?.();
      syncInputsFromBone();
      updateCharacterTabStyles();
      return;
    }
    activeCharacterId = ch.id;
    modelRoot = ch.modelRoot;
    bones = ch.bones || [];
    boneNameMap = ch.boneNameMap || new Map();
    selectedBone = ch.selectedBone || bones[0] || null;
    if ((selectedBone?.uuid || null) !== prevSelectedBoneUuid) {
      resetMirrorSelectionState();
    }
    if (selectedBone) {
      transform?.attach?.(selectedBone);
    } else {
      transform?.detach?.();
    }
    updateSelectedBoneVisuals();
    modelBadge.textContent = `${ch.name} (${bones.length} bones)`;
    sharedSession.activeCharacterId = activeCharacterId;
    updateOrbitDistanceLimits();
    syncInputsFromBone();
    updateCharacterTabStyles();
  }

  function updateCharacterTabStyles() {
    Array.from(characterTabs.querySelectorAll("button[data-char-id]")).forEach((tab) => {
      const isActive = tab.dataset.charId === String(activeCharacterId || "");
      tab.style.background = isActive ? "#29456a" : "#1a2230";
      tab.style.color = "#d6d6d6";
    });
  }

  function refreshCharacterTabs() {
    characterTabs.innerHTML = "";
    characters.forEach((ch, idx) => {
      const wrap = document.createElement("div");
      wrap.style.display = "flex";
      wrap.style.flexDirection = "column";
      wrap.style.alignItems = "stretch";
      wrap.style.border = "1px solid #2a303b";
      wrap.style.borderRadius = "6px";
      wrap.style.overflow = "hidden";

      const tab = document.createElement("button");
      tab.textContent = ch.name || `Char ${idx + 1}`;
      tab.title = ch.name || `Char ${idx + 1}`;
      tab.style.writingMode = "vertical-rl";
      tab.style.transform = "rotate(180deg)";
      tab.style.padding = "8px 4px";
      tab.style.border = "none";
      tab.style.cursor = "pointer";
      tab.dataset.charId = String(ch.id);
      tab.style.background = ch.id === activeCharacterId ? "#29456a" : "#1a2230";
      tab.style.color = "#d6d6d6";
      tab.addEventListener("click", () => {
        applyActiveCharacter(ch);
        updateCharacterTabStyles();
        populateBones();
      });

      const del = document.createElement("button");
      del.textContent = "x";
      del.style.border = "none";
      del.style.cursor = "pointer";
      del.style.background = "#3a1f28";
      del.style.color = "#f0c5d0";
      del.style.padding = "2px 0";
      del.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (characters.length <= 1) {
          characters = [];
        } else {
          characters = characters.filter((c) => c.id !== ch.id);
        }
        if (ch.modelRoot) scene?.remove?.(ch.modelRoot);
        if (ch.skeletonHelper) scene?.remove?.(ch.skeletonHelper);
        sharedSession.characters = characters;
        applyActiveCharacter(characters[0] || null);
        refreshCharacterTabs();
        populateBones();
        scheduleHistoryCapture(120);
      });

      wrap.append(tab, del);
      characterTabs.appendChild(wrap);
    });

    const addBtn = document.createElement("button");
    addBtn.textContent = "+";
    addBtn.title = "Add character from meshes/human_rig";
    addBtn.style.border = "1px solid #2a303b";
    addBtn.style.borderRadius = "6px";
    addBtn.style.background = "#1d2b40";
    addBtn.style.color = "#d6e8ff";
    addBtn.style.cursor = "pointer";
    addBtn.style.padding = "6px 0";
    addBtn.addEventListener("click", async () => {
      try {
        const items = await getRiggedMeshList();
        if (!items.length) {
          setStatus("No rigged meshes found in meshes/human_rig.");
          return;
        }
        const selectedName = await pickRiggedMeshName(items);
        if (!selectedName) return;
        if (!items.includes(selectedName)) {
          setStatus("Selected mesh is not in repository list.");
          return;
        }
        const buf = await fetchRiggedMeshBuffer(selectedName);
        await loadModelFromFile({ name: selectedName, arrayBuffer: async () => buf }, buf);
      } catch (err) {
        setStatus(`Failed repository load: ${err?.message || err}`);
      }
    });
    characterTabs.appendChild(addBtn);
    updateCharacterTabStyles();
  }

  function captureSkeletonPreviewForSlot(slotIdx) {
    if (!renderer || !camera || !orbit) return "";
    const idx = 0;
    const slot = normalizeCameraSlot(cameraSlots[idx], "Camera");
    cameraSlots[idx] = slot;

    const outW = Math.max(64, Math.round(Number(slot.resolution?.[0] || 1024)));
    const outH = Math.max(64, Math.round(Number(slot.resolution?.[1] || 768)));

    const prev = {
      position: camera.position.toArray(),
      target: orbit.target.toArray(),
      fov: camera.fov,
      near: camera.near,
      far: camera.far,
      aspect: camera.aspect,
    };

    setCameraFromSlot(slot);
    camera.aspect = outW / Math.max(1, outH);
    camera.updateProjectionMatrix();
    syncProjectionState();

    const canvas = document.createElement("canvas");
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, outW, outH);
    const strokeStyle = getOpenPoseStrokeStyle(outW, outH);
    const drawCharacters = getPoseDrawCharacters();
    if (drawCharacters.length) {
      drawCharacters.forEach((ch) => {
        const pose2d = computeOpenPosePositions2d(outW, outH, null, ch);
        drawOpenPoseSkeleton(ctx, pose2d, strokeStyle);
      });
    } else {
      const positionsOut = computeOpenPosePositions2d(outW, outH);
      drawOpenPoseSkeleton(ctx, positionsOut, strokeStyle);
    }

    camera.position.fromArray(prev.position);
    orbit.target.fromArray(prev.target);
    camera.fov = prev.fov;
    camera.near = prev.near;
    camera.far = prev.far;
    camera.aspect = prev.aspect;
    camera.updateProjectionMatrix();
    orbit.update();
    syncProjectionState();

    return canvas.toDataURL("image/png");
  }

  function captureAllSkeletonPreviews() {
    readCurrentCameraToSlot(activeCameraIndex);
    const previews = [captureSkeletonPreviewForSlot(0)];
    applySlotToCamera(activeCameraIndex);
    return previews;
  }

  function buildEdgeCanvasFromCanvas(sourceCanvas) {
    const w = Math.max(1, Number(sourceCanvas?.width || 0));
    const h = Math.max(1, Number(sourceCanvas?.height || 0));
    if (!w || !h) return null;
    const srcCtx = sourceCanvas.getContext("2d");
    if (!srcCtx) return null;
    const src = srcCtx.getImageData(0, 0, w, h);
    const outCanvas = document.createElement("canvas");
    outCanvas.width = w;
    outCanvas.height = h;
    const outCtx = outCanvas.getContext("2d");
    if (!outCtx) return null;
    const out = outCtx.createImageData(w, h);
    const lum = new Float32Array(w * h);
    for (let i = 0, p = 0; i < lum.length; i += 1, p += 4) {
      lum[i] = src.data[p] * 0.299 + src.data[p + 1] * 0.587 + src.data[p + 2] * 0.114;
    }
    for (let y = 1; y < h - 1; y += 1) {
      for (let x = 1; x < w - 1; x += 1) {
        const idx = y * w + x;
        const gx =
          -lum[idx - w - 1] + lum[idx - w + 1]
          - 2 * lum[idx - 1] + 2 * lum[idx + 1]
          - lum[idx + w - 1] + lum[idx + w + 1];
        const gy =
          -lum[idx - w - 1] - 2 * lum[idx - w] - lum[idx - w + 1]
          + lum[idx + w - 1] + 2 * lum[idx + w] + lum[idx + w + 1];
        const mag = Math.min(255, Math.hypot(gx, gy));
        const value = mag > 28 ? 255 : 0;
        const p = idx * 4;
        out.data[p] = value;
        out.data[p + 1] = value;
        out.data[p + 2] = value;
        out.data[p + 3] = 255;
      }
    }
    outCtx.putImageData(out, 0, 0);
    return outCanvas;
  }

  function captureScenePreviewDataUrl(options = {}) {
    if (!renderer || !camera || !scene) return "";
    const mode = String(options?.mode || "beauty");
    const explicitW = Number(options?.width || 0);
    const explicitH = Number(options?.height || 0);
    const maxWidth = Number(options?.maxWidth || 0);
    const maxHeight = Number(options?.maxHeight || 0);
    const slot = getActiveCameraSlot();
    const renderW = Math.max(64, Math.round(explicitW || Number(slot?.resolution?.[0] || 1024)));
    const renderH = Math.max(64, Math.round(explicitH || Number(slot?.resolution?.[1] || 768)));
    const prevTransformVisible = transform?.visible;
    const prevGroundPlaneVisible = groundPlane?.visible;
    const prevGroundGridVisible = groundGrid?.visible;
    const prevOriginAxesVisible = originAxes?.visible;
    const prevOriginDotVisible = originDot?.visible;
    const prevSelectedBoneMarkerVisible = selectedBoneMarker?.visible;
    const prevSelectedBoneParentLineVisible = selectedBoneParentLine?.visible;
    const materialStates = [];
    const helperStates = [];
    const prevOverrideMaterial = scene.overrideMaterial;
    const prevAspect = camera.aspect;
    const prevPixelRatio = renderer.getPixelRatio ? renderer.getPixelRatio() : 1;
    if (typeof prevTransformVisible === "boolean") transform.visible = false;
    if (typeof prevGroundPlaneVisible === "boolean") groundPlane.visible = false;
    if (typeof prevGroundGridVisible === "boolean") groundGrid.visible = false;
    if (typeof prevOriginAxesVisible === "boolean") originAxes.visible = false;
    if (typeof prevOriginDotVisible === "boolean") originDot.visible = false;
    if (typeof prevSelectedBoneMarkerVisible === "boolean") selectedBoneMarker.visible = false;
    if (typeof prevSelectedBoneParentLineVisible === "boolean") selectedBoneParentLine.visible = false;
    characters.forEach((character) => {
      if (character?.skeletonHelper) {
        helperStates.push([character.skeletonHelper, character.skeletonHelper.visible]);
        character.skeletonHelper.visible = false;
      }
      character?.modelRoot?.traverse?.((child) => {
        if (!child?.isMesh) return;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach((mat) => {
          if (!mat) return;
          materialStates.push([mat, mat.transparent, mat.opacity]);
          mat.transparent = false;
          mat.opacity = 1.0;
          mat.needsUpdate = true;
        });
      });
    });
    syncProjectionState();
    try {
      if (mode === "depth" || mode === "edges") {
        scene.overrideMaterial = new three.MeshDepthMaterial({
          depthPacking: three.BasicDepthPacking,
        });
      }
      if (renderer.setPixelRatio) {
        renderer.setPixelRatio(1);
      }
      renderer.setSize(renderW, renderH, false);
      camera.aspect = renderW / Math.max(1, renderH);
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
      const srcCanvas = renderer.domElement;
      if (!srcCanvas) return "";
      const srcW = Math.max(1, Number(srcCanvas.width || renderW));
      const srcH = Math.max(1, Number(srcCanvas.height || renderH));
      const scale = (maxWidth > 0 && maxHeight > 0)
        ? Math.min(maxWidth / srcW, maxHeight / srcH, 1)
        : 1;
      const outW = Math.max(1, Math.round(srcW * scale));
      const outH = Math.max(1, Math.round(srcH * scale));
      const canvas = document.createElement("canvas");
      canvas.width = outW;
      canvas.height = outH;
      const ctx = canvas.getContext("2d");
      if (!ctx) return "";
      ctx.drawImage(srcCanvas, 0, 0, outW, outH);
      if (mode === "edges") {
        const edgeCanvas = buildEdgeCanvasFromCanvas(canvas);
        return edgeCanvas ? edgeCanvas.toDataURL("image/png") : "";
      }
      return canvas.toDataURL("image/png");
    } finally {
      scene.overrideMaterial = prevOverrideMaterial || null;
      materialStates.forEach(([mat, transparent, opacity]) => {
        mat.transparent = transparent;
        mat.opacity = opacity;
        mat.needsUpdate = true;
      });
      helperStates.forEach(([helper, visible]) => {
        helper.visible = visible;
      });
      if (typeof prevTransformVisible === "boolean") transform.visible = prevTransformVisible;
      if (typeof prevGroundPlaneVisible === "boolean") groundPlane.visible = prevGroundPlaneVisible;
      if (typeof prevGroundGridVisible === "boolean") groundGrid.visible = prevGroundGridVisible;
      if (typeof prevOriginAxesVisible === "boolean") originAxes.visible = prevOriginAxesVisible;
      if (typeof prevOriginDotVisible === "boolean") originDot.visible = prevOriginDotVisible;
      if (typeof prevSelectedBoneMarkerVisible === "boolean") selectedBoneMarker.visible = prevSelectedBoneMarkerVisible;
      if (typeof prevSelectedBoneParentLineVisible === "boolean") selectedBoneParentLine.visible = prevSelectedBoneParentLineVisible;
      if (renderer.setPixelRatio) {
        renderer.setPixelRatio(prevPixelRatio);
      }
      camera.aspect = prevAspect;
      camera.updateProjectionMatrix();
      resizeRenderer();
      renderer.render(scene, camera);
    }
  }

  function captureViewportRenderPreview(maxWidth = 420, maxHeight = 420) {
    return captureScenePreviewDataUrl({ mode: "beauty", maxWidth, maxHeight });
  }

  function applyEuler(axis, deg) {
    if (!selectedBone) return;
    const euler = selectedBone.rotation;
    if (axis === "x") euler.x = degreesToRadians(deg);
    if (axis === "y") euler.y = degreesToRadians(deg);
    if (axis === "z") euler.z = degreesToRadians(deg);
    selectedBone.updateMatrixWorld(true);
    applyMirroredFromSelectedChannel("rotation", axis);
    refreshSkinnedMeshes();
    scheduleHistoryCapture();
    syncInputsFromBone();
  }

  function applyPosition(axis, value) {
    if (!selectedBone) return;
    selectedBone.position[axis] = value;
    selectedBone.updateMatrixWorld(true);
    applyMirroredFromSelectedChannel("position", axis);
    refreshSkinnedMeshes();
    scheduleHistoryCapture();
    syncInputsFromBone();
  }

  function applyScale(axis, value) {
    if (!selectedBone) return;
    if (axis === "u") {
      selectedBone.scale.set(value, value, value);
    } else {
      selectedBone.scale[axis] = value;
    }
    selectedBone.updateMatrixWorld(true);
    applyMirroredFromSelectedChannel("scale", axis);
    refreshSkinnedMeshes();
    scheduleHistoryCapture();
    syncInputsFromBone();
  }

  function syncInputsFromBone() {
    updateTransformContextUI();
    if (!selectedBone) return;
    rotX.setValue(radiansToDegrees(selectedBone.rotation.x));
    rotY.setValue(radiansToDegrees(selectedBone.rotation.y));
    rotZ.setValue(radiansToDegrees(selectedBone.rotation.z));
    posX.setValue(selectedBone.position.x);
    posY.setValue(selectedBone.position.y);
    posZ.setValue(selectedBone.position.z);
    scaleU.setValue(selectedBone.scale.x);
    scaleX.setValue(selectedBone.scale.x);
    scaleY.setValue(selectedBone.scale.y);
    scaleZ.setValue(selectedBone.scale.z);
  }

  function updateFovSlider() {
    if (!camera) return;
    fovRange.setValue(camera.fov);
  }

  function computeModelBounds(root) {
    if (!root || !three) return null;
    root.updateWorldMatrix?.(true, true);
    const box = new three.Box3();
    let hasAny = false;
    root.traverse((child) => {
      if (!child?.isMesh) return;
      const meshBox = new three.Box3().setFromObject(child);
      if (!Number.isFinite(meshBox.min.x) || meshBox.isEmpty()) return;
      if (!hasAny) {
        box.copy(meshBox);
        hasAny = true;
      } else {
        box.union(meshBox);
      }
    });
    if (!hasAny) {
      const fallback = new three.Box3().setFromObject(root);
      if (!fallback.isEmpty()) {
        return fallback;
      }
      return null;
    }
    return box;
  }

  function updateCameraClipping() {
    if (!camera || !three) return;
    const target = orbit?.target?.clone?.() || new three.Vector3(0, 0, 0);
    const dist = Math.max(0.0001, camera.position.distanceTo(target));
    let near = Math.max(0.00005, Math.min(0.02, dist * 0.005));
    let far = Math.max(500, dist * 140);
    const box = computeModelBounds(modelRoot);
    if (box) {
      const center = box.getCenter(new three.Vector3());
      const size = box.getSize(new three.Vector3());
      const radius = Math.max(0.001, size.length() * 0.5);
      const dCenter = camera.position.distanceTo(center);
      near = Math.min(near, Math.max(0.0001, (dCenter - radius) * 0.25));
      far = Math.max(far, dCenter + radius * 6.0);
    }
    if (Math.abs(camera.near - near) > 1e-6 || Math.abs(camera.far - far) > 1e-4) {
      camera.near = near;
      camera.far = far;
      camera.updateProjectionMatrix();
    }
  }

  function updateOrbitDistanceLimits() {
    if (!orbit || !three) return;
    const box = computeModelBounds(modelRoot);
    if (!box) {
      orbit.minDistance = 0.02;
      orbit.maxDistance = 300;
      return;
    }
    const size = box.getSize(new three.Vector3());
    const maxDim = Math.max(0.001, size.x, size.y, size.z);
    orbit.minDistance = Math.max(0.02, maxDim * 0.02);
    orbit.maxDistance = Math.max(120, maxDim * 220);
  }

  function frameModel() {
    if (!modelRoot || !camera || !orbit || !renderer) return;
    const box = computeModelBounds(modelRoot);
    if (!box) return;
    const size = box.getSize(new three.Vector3());
    const center = box.getCenter(new three.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    if (!Number.isFinite(maxDim) || maxDim <= 0) return;
    const dist = maxDim * 2.0 / Math.tan((camera.fov * Math.PI) / 360);
    const dir = new three.Vector3(0, 0.5, 1).normalize();
    const newPos = center.clone().add(dir.multiplyScalar(dist));
    camera.position.copy(newPos);
    camera.lookAt(center);
    orbit.target.copy(center);
    orbit.update();
    updateOrbitDistanceLimits();
    updateCameraClipping();
    readCurrentCameraToSlot(activeCameraIndex);
    resizeRenderer();
  }

  function ensureSelectedBoneVisuals() {
    if (!scene || !three) return;
    if (!selectedBoneMarker) {
      selectedBoneMarker = new three.Mesh(
        new three.SphereGeometry(0.03, 14, 14),
        new three.MeshBasicMaterial({
          color: 0xffd43b,
          depthTest: false,
          depthWrite: false,
          transparent: true,
          opacity: 0.95,
        }),
      );
      selectedBoneMarker.visible = false;
      selectedBoneMarker.renderOrder = 30;
    }
    if (!selectedBoneParentLine) {
      const geom = new three.BufferGeometry().setFromPoints([
        new three.Vector3(0, 0, 0),
        new three.Vector3(0, 0, 0),
      ]);
      selectedBoneParentLine = new three.Line(
        geom,
        new three.LineBasicMaterial({
          color: 0xffd43b,
          transparent: true,
          opacity: 0.85,
          depthTest: false,
          depthWrite: false,
        }),
      );
      selectedBoneParentLine.visible = false;
      selectedBoneParentLine.renderOrder = 29;
    }
    if (selectedBoneMarker.parent !== scene) {
      scene.add(selectedBoneMarker);
    }
    if (selectedBoneParentLine.parent !== scene) {
      scene.add(selectedBoneParentLine);
    }
    sharedSession.selectedBoneMarker = selectedBoneMarker;
    sharedSession.selectedBoneParentLine = selectedBoneParentLine;
  }

  function updateSelectedBoneVisuals() {
    ensureSelectedBoneVisuals();
    if (!selectedBone || !selectedBoneMarker || !selectedBoneParentLine) {
      if (selectedBoneMarker) selectedBoneMarker.visible = false;
      if (selectedBoneParentLine) selectedBoneParentLine.visible = false;
      return;
    }

    const pos = new three.Vector3();
    selectedBone.getWorldPosition(pos);
    selectedBoneMarker.position.copy(pos);
    selectedBoneMarker.visible = true;

    const parent = selectedBone.parent?.isBone ? selectedBone.parent : null;
    if (!parent) {
      selectedBoneParentLine.visible = false;
      return;
    }

    const parentPos = new three.Vector3();
    parent.getWorldPosition(parentPos);
    selectedBoneParentLine.geometry.setFromPoints([parentPos, pos]);
    selectedBoneParentLine.visible = true;
  }

  function ensureCharactersAttachedToScene() {
    if (!scene) return;
    characters.forEach((ch) => {
      if (ch?.modelRoot && ch.modelRoot.parent !== scene) {
        scene.add(ch.modelRoot);
      }
      if (ch?.skeletonHelper && ch.skeletonHelper.parent !== scene) {
        scene.add(ch.skeletonHelper);
      }
    });
    ensureSelectedBoneVisuals();
  }

  function clearBoneList() {
    boneList.innerHTML = "";
  }

  function selectBone(bone) {
    if (!bone) return;
    const prevSelectedBoneUuid = selectedBone?.uuid || null;
    selectedBone = bone;
    if ((selectedBone?.uuid || null) !== prevSelectedBoneUuid) {
      resetMirrorSelectionState();
    }
    const active = getActiveCharacter();
    if (active) {
      active.selectedBone = bone;
    }
    transform?.attach(bone);
    syncInputsFromBone();
    updateSelectedBoneVisuals();
    Array.from(boneList.querySelectorAll("button"))
      .filter((btn) => btn.dataset && btn.dataset.uuid !== undefined)
      .forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.uuid === bone.uuid);
      });
    const activeBtn = boneList.querySelector(`button[data-uuid="${bone.uuid}"]`);
    activeBtn?.scrollIntoView?.({ block: "nearest" });
  }

  function populateBones() {
    clearBoneList();
    if (!bones.length) return;

    const visited = new Set();
    const boneSet = new Set(bones.map((b) => b.uuid));
    const parentBoneByUuid = new Map();
    const childrenByParentUuid = new Map();
    bones.forEach((b) => childrenByParentUuid.set(b.uuid, []));

    bones.forEach((bone) => {
      let p = bone.parent;
      while (p && !boneSet.has(p.uuid)) {
        p = p.parent;
      }
      const parentBone = p && boneSet.has(p.uuid) ? p : null;
      parentBoneByUuid.set(bone.uuid, parentBone);
      if (parentBone) {
        childrenByParentUuid.get(parentBone.uuid).push(bone);
      }
    });

    const roots = bones.filter((b) => !parentBoneByUuid.get(b.uuid));
    if (!roots.length) {
      roots.push(...bones);
    }
    const expanded = new Set(roots.map((b) => b.uuid));

    const buildRow = (parentEl, bone, depth = 0) => {
      if (visited.has(bone.uuid)) {
        return;
      }
      visited.add(bone.uuid);
      const childBones = childrenByParentUuid.get(bone.uuid) || [];
      const hasChildren = childBones.length > 0;
      const row = document.createElement("div");
      row.className = "ess-pose-bone-row";
      row.style.paddingLeft = `${depth * 14 + 6}px`;

      if (hasChildren) {
        const toggle = document.createElement("button");
        toggle.className = "ess-pose-bone-toggle";
        toggle.textContent = expanded.has(bone.uuid) ? "[-]" : "[+]";
        toggle.addEventListener("click", (ev) => {
          ev.stopPropagation();
          const isOpen = expanded.has(bone.uuid);
          if (isOpen) expanded.delete(bone.uuid);
          else expanded.add(bone.uuid);
          toggle.textContent = isOpen ? "[+]" : "[-]";
          childrenWrap.style.display = isOpen ? "none" : "block";
        });
        row.appendChild(toggle);
      } else {
        const spacer = document.createElement("span");
        spacer.style.width = "22px";
        row.appendChild(spacer);
      }

      const btn = document.createElement("button");
      btn.className = "ess-pose-bone-btn";
      btn.textContent = bone.name || "(unnamed)";
      btn.dataset.name = bone.name;
      btn.dataset.uuid = bone.uuid;
      btn.addEventListener("click", () => selectBone(bone));

      row.appendChild(btn);
      parentEl.appendChild(row);

      const childrenWrap = document.createElement("div");
      childrenWrap.style.display = expanded.has(bone.uuid) ? "block" : "none";
      parentEl.appendChild(childrenWrap);

      childBones.forEach((child) => {
        buildRow(childrenWrap, child, depth + 1);
      });
    };

    roots.forEach((root) => buildRow(boneList, root, 0));
    selectBone(roots[0] || bones[0]);
  }

  function serializeState(includePreview = true) {
    const serializePose = (boneList) => boneList.map((bone) => ({
      name: bone.name,
      path: getBonePath(bone),
      position: [bone.position.x, bone.position.y, bone.position.z],
      rotation: [
        radiansToDegrees(bone.rotation.x),
        radiansToDegrees(bone.rotation.y),
        radiansToDegrees(bone.rotation.z),
      ],
      rotation_order: bone.rotation?.order || "XYZ",
      quaternion: [bone.quaternion.x, bone.quaternion.y, bone.quaternion.z, bone.quaternion.w],
      scale: [bone.scale.x, bone.scale.y, bone.scale.z],
      parent: bone.parent?.name || null,
      uuid: bone.uuid,
    }));

    const charactersPayload = characters.map((ch) => {
      const pose = serializePose(ch.bones || []);
      const charPayload = {
        id: ch.id,
        name: ch.name,
        pose,
        mesh_ext: ch.meshExt || "fbx",
      };
      if (ch.meshBuffer instanceof ArrayBuffer) {
        charPayload.mesh_b64 = bufferToBase64(ch.meshBuffer);
      }
      return charPayload;
    });

    const active = getActiveCharacter();
    const activePose = active ? serializePose(active.bones || []) : [];
    readCurrentCameraToSlot(0);
    const slot = normalizeCameraSlot(getActiveCameraSlot(), "Camera");
    const activeCam = {
      name: slot.name || "Camera",
      fov: Number(slot.fov || 45),
      position: Array.isArray(slot.position) ? slot.position.slice(0, 3) : [0.0, 1.45, 2.75],
      target: Array.isArray(slot.target) ? slot.target.slice(0, 3) : [0, 1.0, 0],
      near: Number(slot.near || 0.001),
      far: Number(slot.far || 3000),
      resolution: Array.isArray(slot.resolution) ? slot.resolution.slice(0, 2) : [1024, 768],
    };

    const payload = {
      pose: activePose,
      characters: charactersPayload,
      active_character_id: activeCharacterId,
      camera: activeCam,
      meta: {
        modelName: modelBadge.textContent || "",
        updated: Date.now(),
      },
    };

  if (includePreview && renderer) {
    try {
      const [width, height] = Array.isArray(activeCam.resolution) ? activeCam.resolution : [1024, 768];
      payload.preview_png = captureScenePreviewDataUrl({ mode: "beauty", width, height }) || "";
      payload.depth_png = captureScenePreviewDataUrl({ mode: "depth", width, height }) || "";
      payload.edges_png = captureScenePreviewDataUrl({ mode: "edges", width, height }) || "";
    } catch (err) {
      console.warn("Failed to capture preview", err);
    }
  }

    return payload;
  }

  function pushStateToWidget() {
    savedPayload = serializeState();
    const json = JSON.stringify(savedPayload);
    stateRef.value = json;
    widget.value = json;
    if (typeof widget.callback === "function") {
      widget.callback(json);
    }
    node?.graph?.setDirtyCanvas(true, true);
  }

  function getPoseDrawCharacters() {
    if (Array.isArray(characters) && characters.length) {
      return characters.filter((ch) => ch && ch.modelRoot);
    }
    const active = getActiveCharacter();
    return active ? [active] : [];
  }

  function distance2d(a, b) {
    if (!a || !b) return Infinity;
    const dx = Number(a.x || 0) - Number(b.x || 0);
    const dy = Number(a.y || 0) - Number(b.y || 0);
    return Math.hypot(dx, dy);
  }

  function sanitizeOpenPosePositions2d(positions2d) {
    if (!Array.isArray(positions2d) || !positions2d.length) return positions2d;
    const out = positions2d.slice();

    const neck = out[1];
    const rightHip = out[8];
    const leftHip = out[11];
    const hipCenter = (rightHip && leftHip)
      ? { x: (rightHip.x + leftHip.x) * 0.5, y: (rightHip.y + leftHip.y) * 0.5 }
      : (rightHip || leftHip || null);
    const torsoSize = (neck && hipCenter) ? distance2d(neck, hipCenter) : null;

    // Guard against synthesized head points drifting far from the body.
    if (torsoSize && Number.isFinite(torsoSize) && torsoSize > 1e-6 && neck) {
      const maxHeadDistFromNeck = torsoSize * 1.35;
      [0, 14, 15, 16, 17].forEach((idx) => {
        const p = out[idx];
        if (p && distance2d(p, neck) > maxHeadDistFromNeck) {
          out[idx] = null;
        }
      });
      const nose = out[0];
      if (nose) {
        if (out[14] && distance2d(out[14], nose) > torsoSize * 0.9) out[14] = null;
        if (out[15] && distance2d(out[15], nose) > torsoSize * 0.9) out[15] = null;
        if (out[16] && distance2d(out[16], nose) > torsoSize * 1.2) out[16] = null;
        if (out[17] && distance2d(out[17], nose) > torsoSize * 1.2) out[17] = null;
      }
    }

    // Remove isolated points that are not connected by any visible limb edge.
    const connected = new Set();
    OPENPOSE_EDGES.forEach(([a, b]) => {
      if (out[a] && out[b]) {
        connected.add(a);
        connected.add(b);
      }
    });
    out.forEach((p, idx) => {
      if (!p) return;
      if (!connected.has(idx)) {
        out[idx] = null;
      }
    });

    return out;
  }

  function computeOpenPosePositions2d(width, height, frameRect = null, character = null) {
    const targetCharacter = character || getActiveCharacter();
    const targetBoneMap = targetCharacter?.boneNameMap || boneNameMap;
    const positions2d = [];
    OPENPOSE_NAMES.forEach((name, idx) => {
      const bone = findBoneByAliases(targetBoneMap, OPENPOSE_TO_BONE[name] || []);
      if (!bone) {
        positions2d[idx] = null;
        return;
      }
      const pos = new three.Vector3();
      bone.getWorldPosition(pos);
      positions2d[idx] = projectWorldToOverlay(pos, width, height, frameRect);
    });

    const headSynth = targetCharacter
      ? getCachedFaceLandmarks2d(targetCharacter, width, height, frameRect)
      : null;
    if (headSynth) {
      // COCO-18 indexes: 0 nose, 14 right eye, 15 left eye, 16 right ear, 17 left ear.
      if (!positions2d[0]) positions2d[0] = headSynth.nose;
      // Always override face lateral points for consistency across rigs.
      positions2d[14] = headSynth.reye;
      positions2d[15] = headSynth.leye;
      positions2d[16] = headSynth.rear;
      positions2d[17] = headSynth.lear;
    }
    return sanitizeOpenPosePositions2d(positions2d);
  }

  function getOpenPoseStrokeStyle(width, height) {
    const minDim = Math.max(1, Math.min(Number(width || 0), Number(height || 0)));
    // Stronger adaptive scaling so thickness difference is clearly visible across resolutions.
    const lineWidth = Math.max(1.4, Math.min(8.0, minDim / 192));
    const pointRadius = Math.max(2.0, Math.min(10.0, lineWidth * 1.25));
    return { lineWidth, pointRadius };
  }

  function drawOpenPoseSkeleton(ctx, positions2d, options = {}) {
    const lineWidth = Number(options.lineWidth || 3);
    const pointRadius = Number(options.pointRadius || 4);

    OPENPOSE_EDGES.forEach(([a, b], edgeIdx) => {
      const pa = positions2d[a];
      const pb = positions2d[b];
      if (!pa || !pb) return;
      const color = OPENPOSE_COLORS[edgeIdx % OPENPOSE_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(pa.x, pa.y);
      ctx.lineTo(pb.x, pb.y);
      ctx.stroke();
    });

    positions2d.forEach((p, idx) => {
      if (!p) return;
      ctx.fillStyle = OPENPOSE_COLORS[idx % OPENPOSE_COLORS.length];
      ctx.beginPath();
      ctx.arc(p.x, p.y, pointRadius, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function drawOpenPoseOverlay() {
    if (!renderer || !camera || !scene) return;
    syncProjectionState();
    const rect = viewport.getBoundingClientRect();
    if (!overlayCanvas) {
      overlayCanvas = document.createElement("canvas");
      overlayCanvas.style.position = "absolute";
      overlayCanvas.style.pointerEvents = "none";
      overlayCanvas.style.inset = "0";
      overlayCanvas.width = rect.width;
      overlayCanvas.height = rect.height;
      viewport.appendChild(overlayCanvas);
    }
    if (overlayCanvas.width !== rect.width || overlayCanvas.height !== rect.height) {
      overlayCanvas.width = rect.width;
      overlayCanvas.height = rect.height;
    }
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    const frameRect = getRenderFrameRect(overlayCanvas.width, overlayCanvas.height);
    const strokeStyle = getOpenPoseStrokeStyle(frameRect.width, frameRect.height);

    // Shade outside active render region so camera framing is explicit.
    ctx.fillStyle = "rgba(2, 4, 8, 0.36)";
    ctx.fillRect(0, 0, overlayCanvas.width, frameRect.y);
    ctx.fillRect(0, frameRect.y + frameRect.height, overlayCanvas.width, overlayCanvas.height - (frameRect.y + frameRect.height));
    ctx.fillRect(0, frameRect.y, frameRect.x, frameRect.height);
    ctx.fillRect(frameRect.x + frameRect.width, frameRect.y, overlayCanvas.width - (frameRect.x + frameRect.width), frameRect.height);
    ctx.strokeStyle = "rgba(200, 220, 255, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(frameRect.x + 0.5, frameRect.y + 0.5, Math.max(1, frameRect.width - 1), Math.max(1, frameRect.height - 1));

    const drawCharacters = getPoseDrawCharacters();
    let activePositions = null;
    if (drawCharacters.length) {
      drawCharacters.forEach((ch, idx) => {
        const pose2d = computeOpenPosePositions2d(overlayCanvas.width, overlayCanvas.height, frameRect, ch);
        drawOpenPoseSkeleton(ctx, pose2d, strokeStyle);
        if (!activePositions && (ch.id === activeCharacterId || idx === 0)) {
          activePositions = pose2d;
        }
      });
    }
    if (!activePositions) {
      activePositions = computeOpenPosePositions2d(overlayCanvas.width, overlayCanvas.height, frameRect);
      drawOpenPoseSkeleton(ctx, activePositions, strokeStyle);
    }

    // Extra head contour for readability.
    const hp = [activePositions[16], activePositions[14], activePositions[0], activePositions[15], activePositions[17]];
    if (hp.every(Boolean)) {
      ctx.strokeStyle = "#8ecbff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(hp[0].x, hp[0].y);
      ctx.lineTo(hp[1].x, hp[1].y);
      ctx.lineTo(hp[2].x, hp[2].y);
      ctx.lineTo(hp[3].x, hp[3].y);
      ctx.lineTo(hp[4].x, hp[4].y);
      ctx.stroke();
    }

    // Highlight currently selected bone even if it is not one of openpose aliases.
    if (selectedBone) {
      const selPos = new three.Vector3();
      selectedBone.getWorldPosition(selPos);
      const s2 = projectWorldToOverlay(selPos, overlayCanvas.width, overlayCanvas.height, frameRect);
      const sx = s2.x;
      const sy = s2.y;

      if (selectedBone.parent?.isBone) {
        const parentPos = new three.Vector3();
        selectedBone.parent.getWorldPosition(parentPos);
        const p2 = projectWorldToOverlay(parentPos, overlayCanvas.width, overlayCanvas.height, frameRect);
        const px = p2.x;
        const py = p2.y;
        ctx.strokeStyle = "#ffd43b";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(sx, sy);
        ctx.stroke();
      }

      ctx.fillStyle = "#ffd43b";
      ctx.strokeStyle = "#11161d";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(sx, sy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  function ensureCharacterBindPose(character) {
    if (!character || !Array.isArray(character.bones)) return;
    if (character.bindPoseByUuid instanceof Map && character.bindPoseByUuid.size) return;
    character.bindPoseByUuid = new Map(
      character.bones.map((bone) => [
        bone.uuid,
        {
          position: [bone.position.x, bone.position.y, bone.position.z],
          quaternion: [bone.quaternion.x, bone.quaternion.y, bone.quaternion.z, bone.quaternion.w],
          scale: [bone.scale.x, bone.scale.y, bone.scale.z],
        },
      ]),
    );
  }

  function resetCharacterToBindPose(character) {
    if (!character || !Array.isArray(character.bones)) return;
    ensureCharacterBindPose(character);
    const bind = character.bindPoseByUuid;
    if (!(bind instanceof Map)) return;
    character.bones.forEach((bone) => {
      const base = bind.get(bone.uuid);
      if (!base) return;
      if (Array.isArray(base.position) && base.position.length === 3) {
        bone.position.set(base.position[0], base.position[1], base.position[2]);
      }
      if (Array.isArray(base.quaternion) && base.quaternion.length === 4) {
        bone.quaternion.set(base.quaternion[0], base.quaternion[1], base.quaternion[2], base.quaternion[3]);
      }
      if (Array.isArray(base.scale) && base.scale.length === 3) {
        bone.scale.set(base.scale[0], base.scale[1], base.scale[2]);
      }
      bone.updateMatrixWorld(false);
    });
  }

  function applyPoseToBones(targetBones, poseEntries) {
    if (!Array.isArray(targetBones) || !targetBones.length || !Array.isArray(poseEntries)) return;
    const poseByPath = new Map();
    const poseByName = new Map();
    poseEntries.forEach((p) => {
      if (p?.path) poseByPath.set(String(p.path), p);
      if (p?.name && !poseByName.has(String(p.name))) {
        poseByName.set(String(p.name), p);
      }
    });

    targetBones.forEach((bone) => {
      const p = poseByPath.get(getBonePath(bone)) || poseByName.get(String(bone.name || ""));
      if (!p) return;
      if (Array.isArray(p.position) && p.position.length === 3) {
        bone.position.set(p.position[0], p.position[1], p.position[2]);
      }
      // Prefer quaternion restore for FBX rigs to avoid Euler-order drift.
      if (Array.isArray(p.quaternion) && p.quaternion.length === 4) {
        bone.quaternion.set(p.quaternion[0], p.quaternion[1], p.quaternion[2], p.quaternion[3]);
      } else if (Array.isArray(p.rotation) && p.rotation.length === 3) {
        if (typeof p.rotation_order === "string" && p.rotation_order) {
          bone.rotation.order = p.rotation_order;
        }
        bone.rotation.set(
          degreesToRadians(p.rotation[0]),
          degreesToRadians(p.rotation[1]),
          degreesToRadians(p.rotation[2]),
        );
      }
      if (Array.isArray(p.scale) && p.scale.length === 3) {
        bone.scale.set(p.scale[0], p.scale[1], p.scale[2]);
      }
      bone.updateMatrixWorld(false);
    });
  }

  function restoreStateAfterLoad() {
    if (!savedPayload) return;
    if (Array.isArray(savedPayload.characters) && savedPayload.characters.length) {
      savedPayload.characters.forEach((savedCharacter) => {
        const target = characters.find((ch) => ch.id === savedCharacter.id)
          || characters.find((ch) => ch.name === savedCharacter.name);
        if (!target) return;
        resetCharacterToBindPose(target);
        applyPoseToBones(target.bones, savedCharacter.pose || []);
      });
    } else if (savedPayload.pose && bones.length) {
      const active = getActiveCharacter();
      if (active) {
        resetCharacterToBindPose(active);
      }
      applyPoseToBones(bones, savedPayload.pose || []);
    }
    refreshSkinnedMeshes();

    if (camera) {
      if (Array.isArray(savedPayload.cameras) && savedPayload.cameras.length) {
        cameraSlots = normalizeCameraSlots(savedPayload.cameras);
        activeCameraIndex = 0;
      } else if (savedPayload.camera && typeof savedPayload.camera === "object") {
        const legacy = normalizeCameraSlot(savedPayload.camera, "Camera");
        cameraSlots[0] = {
          ...cameraSlots[0],
          ...legacy,
          resolution: Array.isArray(cameraSlots[0]?.resolution)
            ? cameraSlots[0].resolution
            : [1024, 768],
        };
      }
      persistCameraSlotsToSession();
      buildCameraTabs();
      applySlotToCamera(0);
      updateOrbitDistanceLimits();
    }
    syncInputsFromBone();
  }

  function startRenderLoop() {
    if (!renderer || !camera || !scene) return;
    const render = () => {
      orbit?.update?.();
      updateCameraClipping();
      updateSelectedBoneVisuals();
      characters.forEach((character) => {
        character?.modelRoot?.traverse?.((child) => {
          if (child.isSkinnedMesh && child.skeleton) {
            child.skeleton.update();
          }
        });
      });
      drawOpenPoseOverlay();
      renderer.render(scene, camera);
      sharedSession.renderLoopHandle = requestAnimationFrame(render);
    };
    render();
  }

  function addGroundAndOrigin() {
    if (!scene || !three) return;

    if (groundPlane) scene.remove(groundPlane);
    if (groundGrid) scene.remove(groundGrid);
    if (originAxes) scene.remove(originAxes);
    if (originDot) scene.remove(originDot);

    const planeSize = 120;
    const groundGeo = new three.PlaneGeometry(planeSize, planeSize, 1, 1);
    const groundMat = new three.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: {
        uMaxDist: { value: planeSize * 0.5 },
      },
      vertexShader: `
        varying vec3 vWorldPos;
        void main() {
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vWorldPos = worldPos.xyz;
          gl_Position = projectionMatrix * viewMatrix * worldPos;
        }
      `,
      fragmentShader: `
        uniform float uMaxDist;
        varying vec3 vWorldPos;
        void main() {
          float d = length(vWorldPos.xz);
          float fade = smoothstep(uMaxDist, uMaxDist * 0.15, d);
          vec3 col = vec3(0.23, 0.26, 0.30);
          gl_FragColor = vec4(col, fade * 0.38);
        }
      `,
    });
    groundPlane = new three.Mesh(groundGeo, groundMat);
    groundPlane.rotation.x = -Math.PI * 0.5;
    groundPlane.position.y = -0.001;
    scene.add(groundPlane);

    groundGrid = new three.GridHelper(40, 80, 0x7b8ba3, 0x334052);
    groundGrid.position.y = 0.0;
    const gridMats = Array.isArray(groundGrid.material) ? groundGrid.material : [groundGrid.material];
    gridMats.forEach((m) => {
      m.transparent = true;
      m.opacity = 0.38;
      m.depthWrite = false;
    });
    scene.add(groundGrid);

    originAxes = new three.AxesHelper(0.6);
    originAxes.position.set(0, 0.01, 0);
    scene.add(originAxes);

    originDot = new three.Mesh(
      new three.SphereGeometry(0.025, 12, 12),
      new three.MeshBasicMaterial({ color: 0xffffff })
    );
    originDot.position.set(0, 0.02, 0);
    scene.add(originDot);

    sharedSession.groundPlane = groundPlane;
    sharedSession.groundGrid = groundGrid;
    sharedSession.originAxes = originAxes;
    sharedSession.originDot = originDot;
  }

  function setupThree() {
    const rect = viewport.getBoundingClientRect();
    renderer = new three.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(rect.width, rect.height, false);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    viewport.innerHTML = "";
    viewport.appendChild(renderer.domElement);

    scene = new three.Scene();
    scene.background = new three.Color(0x0a0d13);

    camera = new three.PerspectiveCamera(45, rect.width / Math.max(1, rect.height), 0.001, 3000);
    camera.position.set(0.0, 1.45, 2.75);
    updateFovSlider();

    orbit = new OrbitControlsCls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.07;
    orbit.target.set(0, 1.0, 0);
    orbit.minDistance = 0.02;
    orbit.maxDistance = 300;
    orbit.update();
    applySlotToCamera(activeCameraIndex);
    orbit.addEventListener("end", () => {
      readCurrentCameraToSlot(activeCameraIndex);
      scheduleHistoryCapture(120);
    });

    transform = new TransformControlsCls(camera, renderer.domElement);
    transform.setMode("rotate");
    transform.setSpace("local");
    transform.setSize(0.7);
    transform.addEventListener("change", () => {
      syncInputsFromBone();
    });
    transform.addEventListener("objectChange", () => {
      if (transformDragging) {
        applyMirroredForTransformMode();
      }
    });
    transform.addEventListener("dragging-changed", (event) => {
      transformDragging = Boolean(event.value);
      orbit.enabled = !event.value;
      if (!event.value) {
        scheduleHistoryCapture(80);
      }
    });
    scene.add(transform);

    renderer.outputEncoding = three.sRGBEncoding;
    renderer.toneMapping = three.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.setClearColor(0x0a0d13, 1);

    const hemi = new three.HemisphereLight(0xffffff, 0x445566, 0.6);
    scene.add(hemi);
    const dir = new three.DirectionalLight(0xffffff, 0.9);
    dir.position.set(3, 6, 4);
    dir.castShadow = false;
    scene.add(dir);

    addGroundAndOrigin();

    sharedSession.renderer = renderer;
    sharedSession.scene = scene;
    sharedSession.camera = camera;
    sharedSession.orbit = orbit;
    sharedSession.transform = transform;

    if (!sharedSession.renderLoopHandle) {
      startRenderLoop();
    }
    window.addEventListener("resize", resizeRenderer);
  }

  function rebuildControlsForViewport() {
    if (!scene || !camera || !renderer) return;
    sharedSession.orbit?.dispose?.();
    if (sharedSession.transform) {
      scene.remove(sharedSession.transform);
      sharedSession.transform.dispose?.();
    }

    orbit = new OrbitControlsCls(camera, renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.07;
    orbit.minDistance = 0.02;
    orbit.maxDistance = 300;
    applySlotToCamera(activeCameraIndex);
    orbit.addEventListener("end", () => {
      readCurrentCameraToSlot(activeCameraIndex);
      scheduleHistoryCapture(120);
    });

    transform = new TransformControlsCls(camera, renderer.domElement);
    transform.setMode("rotate");
    transform.setSpace("local");
    transform.setSize(0.7);
    transform.addEventListener("change", () => {
      syncInputsFromBone();
    });
    transform.addEventListener("objectChange", () => {
      if (transformDragging) {
        applyMirroredForTransformMode();
      }
    });
    transform.addEventListener("dragging-changed", (event) => {
      transformDragging = Boolean(event.value);
      orbit.enabled = !event.value;
      if (!event.value) {
        scheduleHistoryCapture(80);
      }
    });
    scene.add(transform);
    sharedSession.orbit = orbit;
    sharedSession.transform = transform;
  }

  async function ensureThree() {
    if (threeReady) return;
    setStatus("Loading three.js...");
    if (!three) {
      const bundle = await loadThreeBundle();
      three = bundle.THREE;
      OrbitControlsCls = bundle.OrbitControls;
      TransformControlsCls = bundle.TransformControls;
      GLTFLoaderCls = bundle.GLTFLoader;
      FBXLoaderCls = bundle.FBXLoader;
      OBJLoaderCls = bundle.OBJLoader;
    }

    if (!sharedSession.renderer || !sharedSession.scene || !sharedSession.camera) {
      setupThree();
    } else {
      renderer = sharedSession.renderer;
      scene = sharedSession.scene;
      camera = sharedSession.camera;
      viewport.innerHTML = "";
      viewport.appendChild(renderer.domElement);
      rebuildControlsForViewport();
      addGroundAndOrigin();
      resizeRenderer();
      updateFovSlider();
      if (!sharedSession.renderLoopHandle) {
        startRenderLoop();
      }
      window.addEventListener("resize", resizeRenderer);
    }
    setStatus("Ready. Use the + tab to add character(s) from meshes/human_rig.");
    threeReady = true;
  }

  async function loadModelFromFile(file, providedBuffer = null, options = {}) {
    await ensureThree();
    const sourceName = String(options?.name || file?.name || `Character ${characters.length + 1}`);
    const ext = String(options?.meshExt || sourceName.split(".").pop() || "fbx").toLowerCase();
    const rawBuffer = providedBuffer != null ? providedBuffer : await file.arrayBuffer();
    const arrayBuffer = toArrayBuffer(rawBuffer);
    setStatus(`Parsing ${sourceName}...`);

    const onModel = (obj) => {
      if (!obj) {
        throw new Error("Model parser returned an empty result.");
      }
      const character = {
        id: options?.id || `char_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
        name: sourceName,
        modelRoot: obj,
        skeletonHelper: null,
        bones: [],
        boneNameMap: new Map(),
        selectedBone: null,
        faceLandmarkCache: null,
        bindPoseByUuid: null,
        meshExt: ext,
        meshBuffer: cloneArrayBuffer(arrayBuffer),
      };

      scene.add(character.modelRoot);
      const helper = new three.SkeletonHelper(character.modelRoot);
      helper.visible = false;
      helper.material.linewidth = 2;
      scene.add(helper);
      character.skeletonHelper = helper;

      const seen = new Set();
      const localBones = [];
      const localBoneNameMap = new Map();
      const addBone = (bone) => {
        if (!bone || !bone.isBone || seen.has(bone.uuid)) return;
        seen.add(bone.uuid);
        localBones.push(bone);
        if (bone.name && !localBoneNameMap.has(bone.name)) {
          localBoneNameMap.set(bone.name, bone);
        }
      };

      character.modelRoot.traverse((child) => {
        if (child.isSkinnedMesh && child.skeleton?.bones) {
          child.skeleton.bones.forEach(addBone);
        }
        if (child.isBone) {
          addBone(child);
        }
      });

      character.modelRoot.traverse((child) => {
        if (!child.isMesh) return;
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach((m) => {
          if (!m) return;
          m.side = three.FrontSide;
          m.depthWrite = true;
          m.depthTest = true;
          if (!m.map && !m.vertexColors && m.color) {
            m.color.set(0x888888);
            if ("metalness" in m) m.metalness = 0.05;
            if ("roughness" in m) m.roughness = 0.8;
          }
          m.opacity = 1.0;
          m.transparent = false;
          m.needsUpdate = true;
        });
        if (child.isSkinnedMesh && child.geometry && !child.geometry.attributes.normal) {
          child.geometry.computeVertexNormals();
        }
      });

      character.bones = localBones;
      character.boneNameMap = localBoneNameMap;
      character.selectedBone = localBones[0] || null;
      character.bindPoseByUuid = new Map(
        localBones.map((bone) => [
          bone.uuid,
          {
            position: [bone.position.x, bone.position.y, bone.position.z],
            quaternion: [bone.quaternion.x, bone.quaternion.y, bone.quaternion.z, bone.quaternion.w],
            scale: [bone.scale.x, bone.scale.y, bone.scale.z],
          },
        ]),
      );
      characters.push(character);
      sharedSession.characters = characters;
      sharedSession.activeCharacterId = character.id;

      applyActiveCharacter(character);
      refreshCharacterTabs();
      populateBones();
      if (characters.length === 1) {
        frameModel();
      }
      syncProjectionState();
      ensureFaceLandmarkCache(
        character,
        Math.max(64, Math.round(viewport?.clientWidth || 1024)),
        Math.max(64, Math.round(viewport?.clientHeight || 768)),
      );
      updateFovSlider();
      setStatus(`Loaded ${character.name}`);
      scheduleHistoryCapture(120);
      if (!sharedSession.renderLoopHandle) {
        startRenderLoop();
      }
      return character;
    };

    try {
      let parsed;
      if (ext === "gltf" || ext === "glb") {
        parsed = await new Promise((resolve, reject) => {
          const loader = new GLTFLoaderCls();
          loader.parse(
            arrayBuffer,
            "",
            (gltf) => resolve(gltf.scene || gltf.scenes?.[0]),
            (err) => reject(err)
          );
        });
      } else if (ext === "obj") {
        const text = new TextDecoder().decode(new Uint8Array(arrayBuffer));
        const loader = new OBJLoaderCls();
        parsed = loader.parse(text);
      } else {
        const loader = new FBXLoaderCls();
        parsed = loader.parse(arrayBuffer, "");
      }
      return onModel(parsed);
    } catch (err) {
      setStatus(`Failed: ${err?.message || err}`);
      throw err;
    }
  }

  function teardown() {
    window.removeEventListener("resize", resizeRenderer);
    if (sharedSession.renderLoopHandle) {
      cancelAnimationFrame(sharedSession.renderLoopHandle);
      sharedSession.renderLoopHandle = null;
    }
    overlay.remove();
  }

  // Event wiring
  const fovOnChange = (val) => {
    if (!camera) return;
    camera.fov = val || 45;
    camera.updateProjectionMatrix();
    readCurrentCameraToSlot(activeCameraIndex);
    scheduleHistoryCapture();
  };
  fovRange.slider.addEventListener("input", () => fovOnChange(Number(fovRange.slider.value)));
  fovRange.number.addEventListener("input", () => fovOnChange(Number(fovRange.number.value)));
  camWidthInput.addEventListener("change", () => applyResolutionInputsToSlot());
  camHeightInput.addEventListener("change", () => applyResolutionInputsToSlot());
  mirrorCheck.addEventListener("change", () => {
    mirrorEnabled = Boolean(mirrorCheck.checked && mirroredBone);
    if (mirrorEnabled && selectedBone && mirroredBone) {
      setStatus(`Mirror ON: ${selectedBone.name} <-> ${mirroredBone.name}`);
    } else {
      setStatus("Mirror OFF");
    }
  });

  const updateNodeState = () => {
    pushStateToWidget();
    setStatus("Saved scene state to node.");
  };
  saveBtn.addEventListener("click", updateNodeState);
  updateNodeBtn.addEventListener("click", updateNodeState);

  undoBtn.addEventListener("click", async () => {
    await undoHistory();
  });
  redoBtn.addEventListener("click", async () => {
    await redoHistory();
  });

  savePoseBtn.addEventListener("click", () => {
    const active = getActiveCharacter();
    if (!active || !Array.isArray(active.bones) || !active.bones.length) {
      setStatus("No active character to save pose.");
      return;
    }
    const pose = active.bones.map((bone) => ({
      name: bone.name,
      path: getBonePath(bone),
      position: [bone.position.x, bone.position.y, bone.position.z],
      rotation: [
        radiansToDegrees(bone.rotation.x),
        radiansToDegrees(bone.rotation.y),
        radiansToDegrees(bone.rotation.z),
      ],
      rotation_order: bone.rotation?.order || "XYZ",
      quaternion: [bone.quaternion.x, bone.quaternion.y, bone.quaternion.z, bone.quaternion.w],
      scale: [bone.scale.x, bone.scale.y, bone.scale.z],
    }));
    const payload = {
      type: "ess_pose_v1",
      character_id: active.id,
      character_name: active.name,
      updated: Date.now(),
      pose,
    };
    const filename = `${sanitizeFilename(active.name, "character")}_pose.json`;
    downloadJsonFile(filename, payload);
    setStatus(`Pose saved to ${filename}`);
  });

  loadPoseBtn.addEventListener("click", async () => {
    const payload = await pickJsonFile();
    if (!payload) return;
    const active = getActiveCharacter();
    if (!active || !Array.isArray(active.bones) || !active.bones.length) {
      setStatus("No active character to load pose.");
      return;
    }
    const poseEntries = Array.isArray(payload?.pose) ? payload.pose : null;
    if (!poseEntries) {
      setStatus("Invalid pose file.");
      return;
    }
    resetCharacterToBindPose(active);
    applyPoseToBones(active.bones, poseEntries);
    refreshSkinnedMeshes();
    syncInputsFromBone();
    scheduleHistoryCapture(80);
    setStatus(`Loaded pose for ${active.name || "character"}.`);
  });

  saveSceneBtn.addEventListener("click", () => {
    const payload = serializeState(false);
    payload.type = "ess_scene_v1";
    payload.version = 1;
    payload.saved_at = Date.now();
    const filename = `scene_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    downloadJsonFile(filename, payload);
    setStatus(`Scene saved to ${filename}`);
  });

  loadSceneBtn.addEventListener("click", async () => {
    const payload = await pickJsonFile();
    if (!payload) return;
    await applyScenePayload(payload, { fromHistory: false });
    setStatus("Scene loaded from file.");
  });

  initFromImageBtn.addEventListener("click", async () => {
    let progressDialog = null;
    try {
      if (initFromImageBtn.disabled) return;
      const picked = await pickImageFileAsDataUrl();
      if (!picked?.dataUrl) return;
      initFromImageBtn.disabled = true;
      setStatus(`Detecting pose from image: ${picked.name}`);
      progressDialog = createInitProgressDialog(picked.name);
      progressDialog.setSourceImage(picked.dataUrl);
      progressDialog.appendLogs(["Uploading image and starting scene initialization"]);
      progressDialog.setStage("Uploading image and starting scene initialization");
      const requestId = `ess_pose_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
      progressDialog.startPolling(requestId);
      const requestPromise = requestPoseInitFromImage(picked.dataUrl, requestId);
      const payload = await requestPromise;
      const finalProgress = await fetchPoseInitProgress(requestId);
      if (progressDialog && finalProgress) {
        progressDialog.applyProgress(finalProgress);
      }
      if (progressDialog) {
        const modelPreview = renderModelStructurePreview(payload);
        if (modelPreview) {
          progressDialog.setModelImage(modelPreview);
        }
      }
      if (progressDialog) {
        progressDialog.setStage("Applying reconstructed scene");
      }
      await rebuildSceneFromDetectionPayload(payload);
      if (progressDialog) {
        const renderPreview = captureViewportRenderPreview();
        if (renderPreview) {
          progressDialog.setResultImage(renderPreview);
        }
      }
      const count = Number(payload?.person_count || (Array.isArray(payload?.persons) ? payload.persons.length : (payload?.detected_count ? 1 : 0)));
      setStatus(`Scene initialized from image (${Math.max(1, count)} character(s)).`);
      if (progressDialog) {
        progressDialog.complete(`Scene initialized from image (${Math.max(1, count)} character(s))`);
      }
    } catch (err) {
      setStatus(`Init from image failed: ${err?.message || err}`);
      if (progressDialog) {
        progressDialog.fail(err?.message || err);
      }
    } finally {
      initFromImageBtn.disabled = false;
    }
  });

  centerBtn.addEventListener("click", () => {
    frameModel();
    scheduleHistoryCapture(80);
  });
  closeBtn.addEventListener("click", () => {
    pushStateToWidget();
    teardown();
  });

  overlay.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && !ev.shiftKey && (ev.key === "z" || ev.key === "Z")) {
      ev.preventDefault();
      undoHistory();
      return;
    }
    if (((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === "z" || ev.key === "Z"))
      || ((ev.ctrlKey || ev.metaKey) && (ev.key === "y" || ev.key === "Y"))) {
      ev.preventDefault();
      redoHistory();
      return;
    }
    if (!transform) return;
    if (ev.key === "w" || ev.key === "W") {
      transform.setMode("translate");
    } else if (ev.key === "e" || ev.key === "E") {
      transform.setMode("rotate");
    } else if (ev.key === "r" || ev.key === "R") {
      transform.setMode("scale");
    } else if (ev.key === "Escape") {
      pushStateToWidget();
      teardown();
    }
  });

  buildCameraTabs();
  updateHistoryButtons();

  // Restore prior payload if present
  if (stateRef.value && hasExplicitNodeState()) {
    try {
      savedPayload = JSON.parse(stateRef.value);
    } catch (err) {
      console.warn("Failed to parse saved pose state", err);
    }
  }

  ensureThree()
    .then(async () => {
      suppressHistory = true;
      try {
        characters = Array.isArray(sharedSession.characters) ? sharedSession.characters : [];
        activeCharacterId = sharedSession.activeCharacterId || null;
        ensureCharactersAttachedToScene();
        refreshCharacterTabs();

        if (characters.length) {
          applyActiveCharacter(getActiveCharacter());
          populateBones();
          restoreStateAfterLoad();
          setStatus(`Loaded ${characters.length} character(s).`);
          return;
        }

        if (Array.isArray(savedPayload?.characters) && savedPayload.characters.length) {
          for (const ch of savedPayload.characters) {
            if (!ch?.mesh_b64) continue;
            const ext = ch.mesh_ext || "fbx";
            const name = ch.name || `cached.${ext}`;
            const buf = base64ToArrayBuffer(ch.mesh_b64);
            await loadModelFromFile(
              { name, arrayBuffer: async () => buf },
              buf,
              { id: ch.id, name, meshExt: ext },
            );
          }
          if (savedPayload.active_character_id) {
            const active = characters.find((c) => c.id === savedPayload.active_character_id) || characters[0];
            applyActiveCharacter(active || null);
          }
          if (characters.length) {
            restoreStateAfterLoad();
            refreshCharacterTabs();
            populateBones();
            setStatus(`Loaded ${characters.length} character(s).`);
            return;
          }
        }

        if (savedPayload?.mesh_b64) {
          const ext = savedPayload.mesh_ext || "fbx";
          const buf = base64ToArrayBuffer(savedPayload.mesh_b64);
          const name = savedPayload.meta?.modelName || `cached.${ext}`;
          setStatus(`Rehydrating ${name}...`);
          await loadModelFromFile({ name, arrayBuffer: async () => buf }, buf, { name, meshExt: ext });
          restoreStateAfterLoad();
          refreshCharacterTabs();
          populateBones();
          return;
        }

        applyActiveCharacter(null);
        refreshCharacterTabs();
        setStatus("Empty scene ready.");
      } finally {
        suppressHistory = false;
        captureHistoryNow(true);
        updateHistoryButtons();
      }
    })
    .catch((err) => {
      setStatus(`Failed to initialize editor: ${err?.message || err}`);
    });

  resizeRenderer();
}

app.registerExtension({
  name: "ess_pose_mesh_editor",
  async getCustomWidgets() {
    return {
      ESS_POSE_MESH_EDITOR(node, inputName, inputData) {
        ensureStyles();
        const config = Array.isArray(inputData) ? (inputData[1] || {}) : (inputData || {});
        const schemaDefault = typeof config.default === "string" ? config.default : "";
        const initialValue = (!Array.isArray(inputData) && inputData && inputData.value != null)
          ? String(inputData.value)
          : "";
        const stateRef = { value: initialValue };
        const container = document.createElement("div");
        container.className = "ess-pose-widget";

        const button = document.createElement("button");
        button.textContent = "Open Pose Mesh Editor";
        const label = document.createElement("span");
        label.textContent = stateRef.value ? "Pose loaded" : "No pose yet";

        container.append(button, label);

        const widget = node.addDOMWidget(inputName, "pose_mesh_editor", container, {
          getValue: () => stateRef.value,
          setValue: (val) => {
            const next = val ?? "";
            const hasWidgetValues = Array.isArray(node?.widgets_values) && node.widgets_values.length > 0;
            if (!hasWidgetValues && schemaDefault && next === schemaDefault) {
              stateRef.value = "";
              label.textContent = "No pose yet";
              return;
            }
            stateRef.value = next;
            label.textContent = next ? "Pose loaded" : "No pose yet";
          },
          getMinHeight: () => 36,
          getMaxHeight: () => 36,
          hideOnZoom: false,
          margin: 6,
        });
        widget.value = stateRef.value;

        button.addEventListener("click", () => buildOverlay(node, widget, stateRef));

        return {
          widget,
          minHeight: 36,
        };
      },
    };
  },
});


