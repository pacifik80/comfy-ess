import { app } from "../../scripts/app.js";

const STYLE = {
  panel: "#1b1e24",
  panelBorder: "#2e343d",
  panelInset: "#0f1115",
  previewBorder: "#a020a0",
  skeleton: "#f2f2f2",
  text: "#d6d6d6",
  textMuted: "#8b9098",
  sliderTrack: "#3a414c",
  sliderFill: "#8de6a7",
  sliderHandle: "#e6e6e6",
  sliderValueBg: "#2b2f36",
  sliderValueBorder: "#424955",
  clusterBg: "rgba(12, 14, 18, 0.8)",
  clusterBorder: "#3a414c",
};

const MIN_NODE_WIDTH = 1400;
const PREVIEW_MAX = 460;
const PANEL_GAP = 24;
const PADDING = 16;
const SLIDER_HEIGHT = 16;
const SLIDER_GAP = 4;
const CLUSTER_WIDTH = 132;
const CLUSTER_LABEL_HEIGHT = 14;
const ROW_LABEL_WIDTH = 70;
const ROW_HEIGHT = 28;
const ROW_GAP = 10;
const LEFT_RATIO = 0.7;
const CHARACTER_PANEL_MIN = 200;
const CHARACTER_PANEL_MAX = 240;

const BODY_25_NAMES = [
  "Nose",
  "Neck",
  "RShoulder",
  "RElbow",
  "RWrist",
  "LShoulder",
  "LElbow",
  "LWrist",
  "MidHip",
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
  "LBigToe",
  "LSmallToe",
  "LHeel",
  "RBigToe",
  "RSmallToe",
  "RHeel",
];

const BODY_25_EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [1, 5],
  [5, 6],
  [6, 7],
  [1, 8],
  [8, 9],
  [9, 10],
  [10, 11],
  [8, 12],
  [12, 13],
  [13, 14],
  [0, 15],
  [15, 17],
  [0, 16],
  [16, 18],
  [11, 24],
  [11, 22],
  [22, 23],
  [14, 21],
  [14, 19],
  [19, 20],
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function rotX(angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotY(angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

function rotZ(angleDeg) {
  const rad = (angleDeg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

function matMul(a, b) {
  const out = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

function rotXYZ(x, y, z) {
  return matMul(matMul(rotZ(z), rotY(y)), rotX(x));
}

function applyRot(rot, vec) {
  return [
    rot[0][0] * vec[0] + rot[0][1] * vec[1] + rot[0][2] * vec[2],
    rot[1][0] * vec[0] + rot[1][1] * vec[1] + rot[1][2] * vec[2],
    rot[2][0] * vec[0] + rot[2][1] * vec[1] + rot[2][2] * vec[2],
  ];
}

function addVec(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scaleVec(v, s) {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function subVec(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dotVec(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function crossVec(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalizeVec(v) {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-6) {
    return [0, 0, 0];
  }
  return [v[0] / len, v[1] / len, v[2] / len];
}

function getWidgetValue(node, name, fallback) {
  const widget = node.widgets?.find((w) => w.name === name);
  if (!widget) {
    return fallback;
  }
  const value = Number(widget.value);
  if (Number.isFinite(value)) {
    return value;
  }
  if (widget.value != null) {
    return widget.value;
  }
  return fallback;
}

function buildPose(params) {
  const rootRot = rotXYZ(params.root_rot_x, params.root_rot_y, params.root_rot_z);
  const spineRot = rotXYZ(params.spine_rot_x, params.spine_rot_y, params.spine_rot_z);
  const neckRot = rotXYZ(params.neck_rot_x, params.neck_rot_y, params.neck_rot_z);
  const headRot = rotXYZ(params.head_rot_x, params.head_rot_y, params.head_rot_z);

  const pelvis = [params.root_offset_x, params.root_offset_y, params.root_offset_z];
  const torsoDir = applyRot(rootRot, applyRot(spineRot, [0, 1, 0]));
  const neckBase = addVec(pelvis, scaleVec(torsoDir, params.torso_length));

  const shoulderOffset = applyRot(
    rootRot,
    applyRot(spineRot, [params.shoulder_width * 0.5, 0, 0])
  );
  const hipOffset = applyRot(rootRot, [params.hip_width * 0.5, 0, 0]);
  const rightShoulder = addVec(neckBase, shoulderOffset);
  const leftShoulder = addVec(neckBase, scaleVec(shoulderOffset, -1));
  const rightHip = addVec(pelvis, hipOffset);
  const leftHip = addVec(pelvis, scaleVec(hipOffset, -1));

  const neckBasis = matMul(matMul(rootRot, spineRot), neckRot);
  const headBasis = matMul(neckBasis, headRot);
  const neckUp = applyRot(neckBasis, [0, 1, 0]);
  const headUp = applyRot(headBasis, [0, 1, 0]);
  const headForward = applyRot(headBasis, [0, 0, 1]);
  const headRight = applyRot(headBasis, [1, 0, 0]);
  const headCenter = addVec(
    neckBase,
    scaleVec(neckUp, params.neck_length + params.head_size * 0.5)
  );
  const nose = addVec(headCenter, scaleVec(headForward, params.head_size * 0.6));
  const eyeOffset = addVec(
    scaleVec(headUp, params.head_size * 0.15),
    scaleVec(headForward, params.head_size * 0.45)
  );
  const rightEye = addVec(addVec(headCenter, eyeOffset), scaleVec(headRight, params.head_size * 0.2));
  const leftEye = addVec(addVec(headCenter, eyeOffset), scaleVec(headRight, -params.head_size * 0.2));
  const rightEar = addVec(
    addVec(headCenter, scaleVec(headRight, params.head_size * 0.45)),
    scaleVec(headUp, params.head_size * 0.05)
  );
  const leftEar = addVec(
    addVec(headCenter, scaleVec(headRight, -params.head_size * 0.45)),
    scaleVec(headUp, params.head_size * 0.05)
  );

  function armChain(prefix, shoulderPos) {
    const shoulderRot = rotXYZ(
      params[`${prefix}_shoulder_rot_x`],
      params[`${prefix}_shoulder_rot_y`],
      params[`${prefix}_shoulder_rot_z`]
    );
    const upperBasis = matMul(matMul(rootRot, spineRot), shoulderRot);
    const upperDir = applyRot(upperBasis, [0, -1, 0]);
    const elbow = addVec(shoulderPos, scaleVec(upperDir, params[`${prefix}_upper_arm_length`]));
    const elbowRot = rotXYZ(
      params[`${prefix}_elbow_rot_x`],
      params[`${prefix}_elbow_rot_y`],
      params[`${prefix}_elbow_rot_z`]
    );
    const lowerDir = applyRot(matMul(upperBasis, elbowRot), [0, -1, 0]);
    const wrist = addVec(elbow, scaleVec(lowerDir, params[`${prefix}_lower_arm_length`]));
    return [elbow, wrist];
  }

  function legChain(prefix, hipPos) {
    const hipRot = rotXYZ(
      params[`${prefix}_hip_rot_x`],
      params[`${prefix}_hip_rot_y`],
      params[`${prefix}_hip_rot_z`]
    );
    const upperBasis = matMul(rootRot, hipRot);
    const upperDir = applyRot(upperBasis, [0, -1, 0]);
    const knee = addVec(hipPos, scaleVec(upperDir, params[`${prefix}_upper_leg_length`]));
    const kneeRot = rotXYZ(
      params[`${prefix}_knee_rot_x`],
      params[`${prefix}_knee_rot_y`],
      params[`${prefix}_knee_rot_z`]
    );
    const lowerDir = applyRot(matMul(upperBasis, kneeRot), [0, -1, 0]);
    const ankle = addVec(knee, scaleVec(lowerDir, params[`${prefix}_lower_leg_length`]));
    const ankleRot = rotXYZ(
      params[`${prefix}_ankle_rot_x`],
      params[`${prefix}_ankle_rot_y`],
      params[`${prefix}_ankle_rot_z`]
    );
    const footBasis = matMul(matMul(upperBasis, kneeRot), ankleRot);
    const footForward = applyRot(footBasis, [0, 0, 1]);
    const footRight = applyRot(footBasis, [1, 0, 0]);
    const toeCenter = addVec(ankle, scaleVec(footForward, params.foot_length));
    const heel = addVec(ankle, scaleVec(footForward, -params.foot_length * 0.5));
    const toeSpread = params.foot_length * 0.25;
    const bigToe = addVec(toeCenter, scaleVec(footRight, toeSpread));
    const smallToe = addVec(toeCenter, scaleVec(footRight, -toeSpread));
    return [knee, ankle, bigToe, smallToe, heel];
  }

  const [rightElbow, rightWrist] = armChain("right", rightShoulder);
  const [leftElbow, leftWrist] = armChain("left", leftShoulder);
  const [rightKnee, rightAnkle, rightBigToe, rightSmallToe, rightHeel] = legChain("right", rightHip);
  const [leftKnee, leftAnkle, leftBigToe, leftSmallToe, leftHeel] = legChain("left", leftHip);

  return {
    Nose: nose,
    Neck: neckBase,
    HeadCenter: headCenter,
    RShoulder: rightShoulder,
    RElbow: rightElbow,
    RWrist: rightWrist,
    LShoulder: leftShoulder,
    LElbow: leftElbow,
    LWrist: leftWrist,
    MidHip: pelvis,
    RHip: rightHip,
    RKnee: rightKnee,
    RAnkle: rightAnkle,
    LHip: leftHip,
    LKnee: leftKnee,
    LAnkle: leftAnkle,
    REye: rightEye,
    LEye: leftEye,
    REar: rightEar,
    LEar: leftEar,
    LBigToe: leftBigToe,
    LSmallToe: leftSmallToe,
    LHeel: leftHeel,
    RBigToe: rightBigToe,
    RSmallToe: rightSmallToe,
    RHeel: rightHeel,
  };
}

function projectPoints(points, params, width, height) {
  const halfX = width * 0.5;
  const halfY = height * 0.5;
  const viewRot = rotXYZ(params.view_rot_x, params.view_rot_y, params.view_rot_z);
  const projected = {};
  for (const [name, point] of Object.entries(points)) {
    const p = applyRot(viewRot, point);
    const depth = Math.max(params.camera_distance - p[2], 0.1);
    const scale = params.camera_distance / depth;
    const x = p[0] * scale * params.view_zoom + halfX + params.view_offset_x;
    const y = -p[1] * scale * params.view_zoom + halfY + params.view_offset_y;
    projected[name] = [x, y];
  }
  return projected;
}

function projectPoint(point, params, width, height) {
  const halfX = width * 0.5;
  const halfY = height * 0.5;
  const viewRot = rotXYZ(params.view_rot_x, params.view_rot_y, params.view_rot_z);
  const p = applyRot(viewRot, point);
  const depth = Math.max(params.camera_distance - p[2], 0.1);
  const scale = params.camera_distance / depth;
  const x = p[0] * scale * params.view_zoom + halfX + params.view_offset_x;
  const y = -p[1] * scale * params.view_zoom + halfY + params.view_offset_y;
  return [x, y];
}

function renderPreview(ctx, params, width, height, colors) {
  const points3d = buildPose(params);
  const points2d = projectPoints(points3d, params, width, height);

  ctx.save();
  ctx.fillStyle = colors?.bg || STYLE.panelInset;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = colors?.line || STYLE.skeleton;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [start, end] of BODY_25_EDGES) {
    const a = points2d[BODY_25_NAMES[start]];
    const b = points2d[BODY_25_NAMES[end]];
    if (!a || !b) {
      continue;
    }
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  }
  ctx.stroke();

  ctx.fillStyle = colors?.joint || STYLE.skeleton;
  for (const name of BODY_25_NAMES) {
    const point = points2d[name];
    if (!point) {
      continue;
    }
    ctx.beginPath();
    ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
    ctx.fill();
  }

  if (params.render_mode === "mesh") {
    renderMesh(ctx, points3d, params, width, height);
  }

  ctx.restore();
}

function buildCylinder(a, b, radius, segments = 8) {
  const axis = normalizeVec(subVec(b, a));
  const ref = Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const tangent = normalizeVec(crossVec(axis, ref));
  const bitangent = normalizeVec(crossVec(axis, tangent));
  const triangles = [];
  for (let i = 0; i < segments; i++) {
    const a0 = (i / segments) * Math.PI * 2;
    const a1 = ((i + 1) / segments) * Math.PI * 2;
    const dir0 = addVec(scaleVec(tangent, Math.cos(a0)), scaleVec(bitangent, Math.sin(a0)));
    const dir1 = addVec(scaleVec(tangent, Math.cos(a1)), scaleVec(bitangent, Math.sin(a1)));
    const s0 = addVec(a, scaleVec(dir0, radius));
    const s1 = addVec(a, scaleVec(dir1, radius));
    const e0 = addVec(b, scaleVec(dir0, radius));
    const e1 = addVec(b, scaleVec(dir1, radius));
    triangles.push([s0, e0, e1]);
    triangles.push([s0, e1, s1]);
  }
  return triangles;
}

function buildBox(a, b, right, forward, widthA, widthB, depthA, depthB) {
  const r = normalizeVec(right);
  const f = normalizeVec(forward);
  const p0 = addVec(addVec(a, scaleVec(r, widthA * 0.5)), scaleVec(f, depthA * 0.5));
  const p1 = addVec(addVec(a, scaleVec(r, -widthA * 0.5)), scaleVec(f, depthA * 0.5));
  const p2 = addVec(addVec(a, scaleVec(r, -widthA * 0.5)), scaleVec(f, -depthA * 0.5));
  const p3 = addVec(addVec(a, scaleVec(r, widthA * 0.5)), scaleVec(f, -depthA * 0.5));
  const p4 = addVec(addVec(b, scaleVec(r, widthB * 0.5)), scaleVec(f, depthB * 0.5));
  const p5 = addVec(addVec(b, scaleVec(r, -widthB * 0.5)), scaleVec(f, depthB * 0.5));
  const p6 = addVec(addVec(b, scaleVec(r, -widthB * 0.5)), scaleVec(f, -depthB * 0.5));
  const p7 = addVec(addVec(b, scaleVec(r, widthB * 0.5)), scaleVec(f, -depthB * 0.5));
  const faces = [
    [p0, p1, p2, p3],
    [p4, p5, p6, p7],
    [p0, p4, p7, p3],
    [p1, p5, p6, p2],
    [p0, p1, p5, p4],
    [p3, p2, p6, p7],
  ];
  const triangles = [];
  for (const face of faces) {
    triangles.push([face[0], face[1], face[2]]);
    triangles.push([face[0], face[2], face[3]]);
  }
  return triangles;
}

function buildSphere(center, radius, rings = 6, segments = 8) {
  const triangles = [];
  for (let i = 0; i < rings; i++) {
    const lat0 = Math.PI * (i / rings - 0.5);
    const lat1 = Math.PI * ((i + 1) / rings - 0.5);
    const y0 = Math.sin(lat0);
    const y1 = Math.sin(lat1);
    const r0 = Math.cos(lat0);
    const r1 = Math.cos(lat1);
    for (let j = 0; j < segments; j++) {
      const lon0 = (j / segments) * Math.PI * 2;
      const lon1 = ((j + 1) / segments) * Math.PI * 2;
      const p0 = addVec(center, [r0 * Math.cos(lon0) * radius, y0 * radius, r0 * Math.sin(lon0) * radius]);
      const p1 = addVec(center, [r1 * Math.cos(lon0) * radius, y1 * radius, r1 * Math.sin(lon0) * radius]);
      const p2 = addVec(center, [r1 * Math.cos(lon1) * radius, y1 * radius, r1 * Math.sin(lon1) * radius]);
      const p3 = addVec(center, [r0 * Math.cos(lon1) * radius, y0 * radius, r0 * Math.sin(lon1) * radius]);
      triangles.push([p0, p1, p2]);
      triangles.push([p0, p2, p3]);
    }
  }
  return triangles;
}

function buildMesh(points, params) {
  const isFemale = params.mesh_gender === "female";
  const shoulderWidth = params.shoulder_width * (isFemale ? 0.92 : 1.0);
  const hipWidth = params.hip_width * (isFemale ? 1.1 : 0.95);
  const limbRadius = (isFemale ? 0.075 : 0.085) * (params.torso_length / 1.4);

  const rootRot = rotXYZ(params.root_rot_x, params.root_rot_y, params.root_rot_z);
  const right = applyRot(rootRot, [1, 0, 0]);
  const forward = applyRot(rootRot, [0, 0, 1]);
  const pelvis = points.MidHip;
  const neck = points.Neck;
  const headCenter = points.HeadCenter;
  const depthHip = hipWidth * 0.45;
  const depthShoulder = shoulderWidth * 0.4;

  const triangles = [];
  triangles.push(...buildBox(pelvis, neck, right, forward, hipWidth, shoulderWidth, depthHip, depthShoulder));
  triangles.push(...buildSphere(headCenter, params.head_size * 0.55, 6, 10));
  triangles.push(...buildCylinder(neck, headCenter, limbRadius * 0.9, 8));

  function addLimb(a, b, scale = 1) {
    triangles.push(...buildCylinder(points[a], points[b], limbRadius * scale, 8));
  }

  addLimb("RShoulder", "RElbow", 1.0);
  addLimb("RElbow", "RWrist", 0.9);
  addLimb("LShoulder", "LElbow", 1.0);
  addLimb("LElbow", "LWrist", 0.9);
  addLimb("RHip", "RKnee", 1.1);
  addLimb("RKnee", "RAnkle", 1.0);
  addLimb("LHip", "LKnee", 1.1);
  addLimb("LKnee", "LAnkle", 1.0);
  addLimb("RAnkle", "RBigToe", 0.7);
  addLimb("LAnkle", "LBigToe", 0.7);

  return triangles;
}

function renderMesh(ctx, points, params, width, height) {
  const triangles = buildMesh(points, params);
  const lightDir = normalizeVec([0.2, 0.6, 1.0]);
  const projected = triangles.map((tri) => {
    const pts = tri.map((p) => projectPoint(p, params, width, height));
    const depth = (tri[0][2] + tri[1][2] + tri[2][2]) / 3;
    const normal = normalizeVec(crossVec(subVec(tri[1], tri[0]), subVec(tri[2], tri[0])));
    const shade = 0.4 + 0.6 * Math.max(0, dotVec(normal, lightDir));
    return { pts, depth, shade };
  });

  projected.sort((a, b) => a.depth - b.depth);
  ctx.save();
  ctx.globalAlpha = 0.35;
  for (const tri of projected) {
    ctx.beginPath();
    ctx.moveTo(tri.pts[0][0], tri.pts[0][1]);
    ctx.lineTo(tri.pts[1][0], tri.pts[1][1]);
    ctx.lineTo(tri.pts[2][0], tri.pts[2][1]);
    ctx.closePath();
    const shade = clamp(tri.shade, 0, 1);
    ctx.fillStyle = `rgb(${Math.round(180 * shade)}, ${Math.round(190 * shade)}, ${Math.round(210 * shade)})`;
    ctx.fill();
  }
  ctx.restore();
}

function collectParams(node) {
  return {
    view_rot_x: getWidgetValue(node, "view_rot_x", 0),
    view_rot_y: getWidgetValue(node, "view_rot_y", 0),
    view_rot_z: getWidgetValue(node, "view_rot_z", 0),
    view_offset_x: getWidgetValue(node, "view_offset_x", 0),
    view_offset_y: getWidgetValue(node, "view_offset_y", 0),
    view_zoom: getWidgetValue(node, "view_zoom", 160),
    camera_distance: getWidgetValue(node, "camera_distance", 6),
    width: getWidgetValue(node, "width", 512),
    height: getWidgetValue(node, "height", 512),
    render_mode: getWidgetValue(node, "render_mode", "lines"),
    mesh_gender: getWidgetValue(node, "mesh_gender", "male"),
    root_rot_x: getWidgetValue(node, "root_rot_x", 0),
    root_rot_y: getWidgetValue(node, "root_rot_y", 0),
    root_rot_z: getWidgetValue(node, "root_rot_z", 0),
    root_offset_x: getWidgetValue(node, "root_offset_x", 0),
    root_offset_y: getWidgetValue(node, "root_offset_y", 0),
    root_offset_z: getWidgetValue(node, "root_offset_z", 0),
    torso_length: getWidgetValue(node, "torso_length", 1.4),
    neck_length: getWidgetValue(node, "neck_length", 0.3),
    head_size: getWidgetValue(node, "head_size", 0.5),
    shoulder_width: getWidgetValue(node, "shoulder_width", 0.8),
    hip_width: getWidgetValue(node, "hip_width", 0.6),
    left_upper_arm_length: getWidgetValue(node, "left_upper_arm_length", 0.9),
    left_lower_arm_length: getWidgetValue(node, "left_lower_arm_length", 0.8),
    right_upper_arm_length: getWidgetValue(node, "right_upper_arm_length", 0.9),
    right_lower_arm_length: getWidgetValue(node, "right_lower_arm_length", 0.8),
    left_upper_leg_length: getWidgetValue(node, "left_upper_leg_length", 1.1),
    left_lower_leg_length: getWidgetValue(node, "left_lower_leg_length", 1.0),
    right_upper_leg_length: getWidgetValue(node, "right_upper_leg_length", 1.1),
    right_lower_leg_length: getWidgetValue(node, "right_lower_leg_length", 1.0),
    foot_length: getWidgetValue(node, "foot_length", 0.4),
    spine_rot_x: getWidgetValue(node, "spine_rot_x", 0),
    spine_rot_y: getWidgetValue(node, "spine_rot_y", 0),
    spine_rot_z: getWidgetValue(node, "spine_rot_z", 0),
    neck_rot_x: getWidgetValue(node, "neck_rot_x", 0),
    neck_rot_y: getWidgetValue(node, "neck_rot_y", 0),
    neck_rot_z: getWidgetValue(node, "neck_rot_z", 0),
    head_rot_x: getWidgetValue(node, "head_rot_x", 0),
    head_rot_y: getWidgetValue(node, "head_rot_y", 0),
    head_rot_z: getWidgetValue(node, "head_rot_z", 0),
    left_shoulder_rot_x: getWidgetValue(node, "left_shoulder_rot_x", 0),
    left_shoulder_rot_y: getWidgetValue(node, "left_shoulder_rot_y", 0),
    left_shoulder_rot_z: getWidgetValue(node, "left_shoulder_rot_z", 0),
    left_elbow_rot_x: getWidgetValue(node, "left_elbow_rot_x", 0),
    left_elbow_rot_y: getWidgetValue(node, "left_elbow_rot_y", 0),
    left_elbow_rot_z: getWidgetValue(node, "left_elbow_rot_z", 0),
    right_shoulder_rot_x: getWidgetValue(node, "right_shoulder_rot_x", 0),
    right_shoulder_rot_y: getWidgetValue(node, "right_shoulder_rot_y", 0),
    right_shoulder_rot_z: getWidgetValue(node, "right_shoulder_rot_z", 0),
    right_elbow_rot_x: getWidgetValue(node, "right_elbow_rot_x", 0),
    right_elbow_rot_y: getWidgetValue(node, "right_elbow_rot_y", 0),
    right_elbow_rot_z: getWidgetValue(node, "right_elbow_rot_z", 0),
    left_hip_rot_x: getWidgetValue(node, "left_hip_rot_x", 0),
    left_hip_rot_y: getWidgetValue(node, "left_hip_rot_y", 0),
    left_hip_rot_z: getWidgetValue(node, "left_hip_rot_z", 0),
    left_knee_rot_x: getWidgetValue(node, "left_knee_rot_x", 0),
    left_knee_rot_y: getWidgetValue(node, "left_knee_rot_y", 0),
    left_knee_rot_z: getWidgetValue(node, "left_knee_rot_z", 0),
    left_ankle_rot_x: getWidgetValue(node, "left_ankle_rot_x", 0),
    left_ankle_rot_y: getWidgetValue(node, "left_ankle_rot_y", 0),
    left_ankle_rot_z: getWidgetValue(node, "left_ankle_rot_z", 0),
    right_hip_rot_x: getWidgetValue(node, "right_hip_rot_x", 0),
    right_hip_rot_y: getWidgetValue(node, "right_hip_rot_y", 0),
    right_hip_rot_z: getWidgetValue(node, "right_hip_rot_z", 0),
    right_knee_rot_x: getWidgetValue(node, "right_knee_rot_x", 0),
    right_knee_rot_y: getWidgetValue(node, "right_knee_rot_y", 0),
    right_knee_rot_z: getWidgetValue(node, "right_knee_rot_z", 0),
    right_ankle_rot_x: getWidgetValue(node, "right_ankle_rot_x", 0),
    right_ankle_rot_y: getWidgetValue(node, "right_ankle_rot_y", 0),
    right_ankle_rot_z: getWidgetValue(node, "right_ankle_rot_z", 0),
  };
}

function getWidgetMap(node) {
  const map = {};
  for (const widget of node.widgets || []) {
    if (widget?.name) {
      map[widget.name] = widget;
    }
  }
  return map;
}

function getWidgetRange(widget, fallbackMin, fallbackMax, fallbackStep) {
  const min = Number.isFinite(widget?.options?.min) ? widget.options.min : fallbackMin;
  const max = Number.isFinite(widget?.options?.max) ? widget.options.max : fallbackMax;
  const step = Number.isFinite(widget?.options?.step) ? widget.options.step : fallbackStep;
  return { min, max, step };
}

function formatValue(value, step, widget) {
  if (!Number.isFinite(value)) {
    return "0";
  }
  let decimals = 0;
  if (widget?.type === "FLOAT") {
    decimals = step && step < 1 ? Math.min(3, Math.max(1, Math.ceil(-Math.log10(step)))) : 2;
  }
  return value.toFixed(decimals);
}

function setWidgetValue(widget, value, node) {
  if (!widget) {
    return;
  }
  if (typeof value === "string") {
    widget.value = value;
    widget.callback?.(value, null, node, null, null);
    return;
  }
  const range = getWidgetRange(widget, 0, 1, 0);
  let next = Math.min(range.max, Math.max(range.min, value));
  if (widget.type === "INT") {
    next = Math.round(next);
  } else if (widget.type === "FLOAT" && range.step && range.step > 0 && range.step < 1) {
    next = Math.round((next - range.min) / range.step) * range.step + range.min;
  }
  widget.value = next;
  widget.callback?.(next, null, node, null, null);
}

function scaleParamsForPreview(params, baseSize) {
  const scale = baseSize > 0 ? baseSize / 512 : 1;
  return {
    ...params,
    view_zoom: params.view_zoom * scale,
    view_offset_x: params.view_offset_x * scale,
    view_offset_y: params.view_offset_y * scale,
  };
}

function drawPanel(ctx, rect) {
  ctx.save();
  ctx.fillStyle = STYLE.panel;
  ctx.strokeStyle = STYLE.panelBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(rect.x, rect.y, rect.w, rect.h, 10);
  } else {
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
  }
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function getViewportRect(squareSize, outputW, outputH) {
  const safeW = outputW > 0 ? outputW : squareSize;
  const safeH = outputH > 0 ? outputH : squareSize;
  const aspect = safeW / safeH;
  let w = squareSize;
  let h = squareSize;
  if (aspect > 1) {
    h = squareSize / aspect;
  } else if (aspect < 1) {
    w = squareSize * aspect;
  }
  return {
    x: (squareSize - w) / 2,
    y: (squareSize - h) / 2,
    w,
    h,
  };
}

function drawRenderPanel(ctx, rect, params) {
  drawPanel(ctx, rect);
  const square = Math.min(rect.w - 12, rect.h - 12);
  const squareX = rect.x + (rect.w - square) / 2;
  const squareY = rect.y + (rect.h - square) / 2;
  ctx.save();
  ctx.strokeStyle = STYLE.previewBorder;
  ctx.lineWidth = 2;
  ctx.strokeRect(squareX, squareY, square, square);

  const squareInner = square - 8;
  const viewport = getViewportRect(squareInner, params.width, params.height);
  const viewportX = squareX + 4 + viewport.x;
  const viewportY = squareY + 4 + viewport.y;

  ctx.fillStyle = STYLE.panelInset;
  ctx.fillRect(squareX + 4, squareY + 4, square - 8, square - 8);
  ctx.strokeStyle = STYLE.panelBorder;
  ctx.strokeRect(viewportX, viewportY, viewport.w, viewport.h);

  ctx.translate(viewportX, viewportY);
  ctx.beginPath();
  ctx.rect(0, 0, viewport.w, viewport.h);
  ctx.clip();
  const scaledParams = scaleParamsForPreview(params, squareInner);
  renderPreview(ctx, scaledParams, viewport.w, viewport.h, { bg: STYLE.panelInset });
  ctx.restore();
}

function drawReferencePanel(ctx, rect, params, poseRect) {
  drawPanel(ctx, rect);
  let size = poseRect?.w;
  let offsetX = poseRect?.x;
  let offsetY = poseRect?.y;
  if (size == null || offsetX == null || offsetY == null) {
    const side = Math.min(rect.w, rect.h);
    const inset = Math.max(110, Math.min(180, Math.floor(side * 0.32)));
    size = Math.max(200, side - inset * 2);
    offsetX = rect.x + (rect.w - size) / 2;
    offsetY = rect.y + (rect.h - size) / 2;
  }
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.beginPath();
  ctx.rect(0, 0, size, size);
  ctx.clip();
  const scaledParams = scaleParamsForPreview(params, size);
  renderPreview(ctx, scaledParams, size, size, {
    bg: STYLE.panelInset,
    line: STYLE.skeleton,
    joint: STYLE.skeleton,
  });
  ctx.restore();
  return { x: offsetX, y: offsetY, w: size, h: size };
}

function drawSlider(ctx, config, widget, hitAreas) {
  const { x, y, w, h, label } = config;
  const labelWidth = 22;
  const valueWidth = 36;
  const trackX = x + labelWidth;
  const trackW = w - labelWidth - valueWidth - 6;
  const trackY = y + Math.floor(h / 2) - 3;
  const trackH = 6;
  const valueX = x + w - valueWidth;
  const valueY = y;

  const range = getWidgetRange(widget, 0, 1, 0.01);
  const value = Number.isFinite(widget?.value) ? widget.value : range.min;
  const t = (value - range.min) / (range.max - range.min || 1);
  const fillW = Math.max(0, Math.min(trackW, trackW * t));

  ctx.save();
  ctx.fillStyle = STYLE.textMuted;
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + 2, y + h / 2 + 0.5);

  ctx.fillStyle = STYLE.sliderTrack;
  ctx.fillRect(trackX, trackY, trackW, trackH);
  ctx.fillStyle = STYLE.sliderFill;
  ctx.fillRect(trackX, trackY, fillW, trackH);
  ctx.fillStyle = STYLE.sliderHandle;
  ctx.fillRect(trackX + fillW - 2, trackY - 3, 4, trackH + 6);

  ctx.fillStyle = STYLE.sliderValueBg;
  ctx.strokeStyle = STYLE.sliderValueBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.rect(valueX, valueY, valueWidth, h);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = STYLE.text;
  ctx.textAlign = "center";
  ctx.fillText(formatValue(value, range.step, widget), valueX + valueWidth / 2, y + h / 2 + 0.5);
  ctx.restore();

  hitAreas.push({
    type: "slider",
    widget,
    x: trackX,
    y: y,
    w: trackW + valueWidth,
    h,
    trackX,
    trackW,
  });
}

function getToggleOptions(widget) {
  const values = widget?.options?.values;
  if (Array.isArray(values) && values.length >= 2) {
    return values.slice(0, 2);
  }
  return ["off", "on"];
}

function drawToggle(ctx, config, widget, hitAreas) {
  const { x, y, w, h, label } = config;
  const value = widget?.value;
  const options = getToggleOptions(widget);
  const leftValue = options[0];
  const rightValue = options[1];
  const leftActive = value === leftValue;
  const rightActive = value === rightValue;
  const buttonGap = 6;
  const buttonW = Math.floor((w - buttonGap) / 2);
  const buttonH = h;

  ctx.save();
  ctx.fillStyle = STYLE.textMuted;
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + 2, y - 6);

  const buttonY = y;
  const leftX = x;
  const rightX = x + buttonW + buttonGap;

  ctx.fillStyle = leftActive ? STYLE.sliderFill : STYLE.sliderTrack;
  ctx.strokeStyle = STYLE.sliderValueBorder;
  ctx.lineWidth = 1;
  ctx.fillRect(leftX, buttonY, buttonW, buttonH);
  ctx.strokeRect(leftX, buttonY, buttonW, buttonH);
  ctx.fillStyle = STYLE.text;
  ctx.textAlign = "center";
  ctx.fillText(String(leftValue).toUpperCase(), leftX + buttonW / 2, buttonY + buttonH / 2 + 0.5);

  ctx.fillStyle = rightActive ? STYLE.sliderFill : STYLE.sliderTrack;
  ctx.strokeStyle = STYLE.sliderValueBorder;
  ctx.fillRect(rightX, buttonY, buttonW, buttonH);
  ctx.strokeRect(rightX, buttonY, buttonW, buttonH);
  ctx.fillStyle = STYLE.text;
  ctx.fillText(String(rightValue).toUpperCase(), rightX + buttonW / 2, buttonY + buttonH / 2 + 0.5);
  ctx.restore();

  hitAreas.push({
    type: "toggle",
    widget,
    optionValue: leftValue,
    x: leftX,
    y: buttonY,
    w: buttonW,
    h: buttonH,
  });
  hitAreas.push({
    type: "toggle",
    widget,
    optionValue: rightValue,
    x: rightX,
    y: buttonY,
    w: buttonW,
    h: buttonH,
  });
}

function getClusterHeight(controlCount) {
  if (!controlCount) {
    return 0;
  }
  return CLUSTER_LABEL_HEIGHT + 6 + controlCount * SLIDER_HEIGHT + (controlCount - 1) * SLIDER_GAP;
}

function computeRowPositions(rect, count, clusterHeight) {
  if (count <= 1) {
    return [rect.y + rect.h / 2];
  }
  const minStep = clusterHeight + 12;
  const available = rect.h - 20;
  const step = Math.max(minStep, available / (count - 1));
  const total = step * (count - 1);
  const start = rect.y + (rect.h - total) / 2;
  return Array.from({ length: count }, (_, index) => start + index * step);
}

function drawCluster(ctx, group, widgetMap, hitAreas, bounds) {
  const controls = group.controls.map((control) => ({
    ...control,
    widget: widgetMap[control.name],
  })).filter((control) => control.widget);

  if (!controls.length) {
    return;
  }

  const height = getClusterHeight(controls.length);
  let x = group.x ?? group.anchorX;
  if (group.x == null) {
    if (group.align === "left") {
      x -= CLUSTER_WIDTH + 14;
    } else if (group.align === "right") {
      x += 14;
    } else {
      x -= CLUSTER_WIDTH / 2;
    }
  }
  let y = (group.y ?? group.anchorY) - height / 2;

  x = Math.max(bounds.x + 6, Math.min(bounds.x + bounds.w - CLUSTER_WIDTH - 6, x));
  y = Math.max(bounds.y + 6, Math.min(bounds.y + bounds.h - height - 6, y));

  ctx.save();
  ctx.fillStyle = STYLE.clusterBg;
  ctx.strokeStyle = STYLE.clusterBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, y, CLUSTER_WIDTH, height, 8);
  } else {
    ctx.rect(x, y, CLUSTER_WIDTH, height);
  }
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = STYLE.text;
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(group.label, x + 6, y + CLUSTER_LABEL_HEIGHT / 2 + 1);
  ctx.restore();

  let rowY = y + CLUSTER_LABEL_HEIGHT + 4;
  for (const control of controls) {
    drawSlider(
      ctx,
      {
        x: x + 6,
        y: rowY,
        w: CLUSTER_WIDTH - 12,
        h: SLIDER_HEIGHT,
        label: control.label,
      },
      control.widget,
      hitAreas
    );
    rowY += SLIDER_HEIGHT + SLIDER_GAP;
  }
}

function drawBottomRows(ctx, rows, widgetMap, rect, hitAreas) {
  ctx.save();
  ctx.fillStyle = STYLE.panel;
  ctx.strokeStyle = STYLE.panelBorder;
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(rect.x, rect.y, rect.w, rect.h, 10);
  } else {
    ctx.rect(rect.x, rect.y, rect.w, rect.h);
  }
  ctx.fill();
  ctx.stroke();
  ctx.restore();

  let rowY = rect.y + 10;
  for (const row of rows) {
    const controls = row.controls.map((control) => ({
      ...control,
      widget: widgetMap[control.name],
    })).filter((control) => control.widget);

    if (!controls.length) {
      rowY += ROW_HEIGHT + ROW_GAP;
      continue;
    }

    const sliderAreaX = rect.x + ROW_LABEL_WIDTH;
    const sliderAreaW = rect.w - ROW_LABEL_WIDTH - 12;
    const count = controls.length;
    const gap = 8;
    const sliderW = Math.max(80, Math.floor((sliderAreaW - gap * (count - 1)) / count));

    ctx.save();
    ctx.fillStyle = STYLE.text;
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(row.label, rect.x + 10, rowY + ROW_HEIGHT / 2 + 1);
    ctx.restore();

    let x = sliderAreaX;
    for (const control of controls) {
      if (control.kind === "toggle") {
        drawToggle(
          ctx,
          {
            x,
            y: rowY + 6,
            w: sliderW,
            h: SLIDER_HEIGHT,
            label: control.label,
          },
          control.widget,
          hitAreas
        );
      } else {
        drawSlider(
          ctx,
          {
            x,
            y: rowY + 6,
            w: sliderW,
            h: SLIDER_HEIGHT,
            label: control.label,
          },
          control.widget,
          hitAreas
        );
      }
      x += sliderW + gap;
    }
    rowY += ROW_HEIGHT + ROW_GAP;
  }
}

function drawCharacterPanel(ctx, controls, widgetMap, rect, hitAreas) {
  drawPanel(ctx, rect);
  const padding = 10;
  let y = rect.y + padding;
  const available = rect.h - padding * 2;
  const rowTotal = controls.length;
  const rowSpace = Math.max(SLIDER_HEIGHT + 6, Math.floor(available / rowTotal));

  for (const control of controls) {
    const widget = widgetMap[control.name];
    if (!widget) {
      y += rowSpace;
      continue;
    }
    drawSlider(
      ctx,
      {
        x: rect.x + padding,
        y: y + Math.floor((rowSpace - SLIDER_HEIGHT) / 2),
        w: rect.w - padding * 2,
        h: SLIDER_HEIGHT,
        label: control.label,
      },
      widget,
      hitAreas
    );
    y += rowSpace;
  }
}

function layoutHeight(width) {
  const usable = width - PADDING * 2 - PANEL_GAP;
  const leftWidth = usable * LEFT_RATIO;
  const rightWidth = usable - leftWidth;
  const characterWidth = Math.max(CHARACTER_PANEL_MIN, Math.min(CHARACTER_PANEL_MAX, leftWidth * 0.35));
  const referenceWidth = leftWidth - characterWidth - PANEL_GAP;
  const topHeight = Math.max(600, Math.max(referenceWidth, rightWidth));
  const bottomHeight = ROW_HEIGHT * RENDER_ROWS.length + ROW_GAP * (RENDER_ROWS.length - 1) + 24;
  return PADDING * 2 + topHeight + bottomHeight + 20;
}

const JOINT_GROUPS = [
  {
    id: "head",
    label: "HEAD",
    column: "center",
    row: 0,
    controls: [
      { name: "head_rot_x", label: "RX" },
      { name: "head_rot_y", label: "RY" },
      { name: "head_rot_z", label: "RZ" },
      { name: "head_size", label: "LEN" },
    ],
  },
  {
    id: "neck",
    label: "NECK",
    column: "center",
    row: 1,
    controls: [
      { name: "neck_rot_x", label: "RX" },
      { name: "neck_rot_y", label: "RY" },
      { name: "neck_rot_z", label: "RZ" },
      { name: "neck_length", label: "LEN" },
    ],
  },
  {
    id: "spine",
    label: "SPINE",
    column: "center",
    row: 2,
    controls: [
      { name: "spine_rot_x", label: "RX" },
      { name: "spine_rot_y", label: "RY" },
      { name: "spine_rot_z", label: "RZ" },
      { name: "torso_length", label: "LEN" },
    ],
  },
  {
    id: "left_shoulder",
    label: "L SHOULDER",
    column: "left",
    row: 0,
    controls: [
      { name: "left_shoulder_rot_x", label: "RX" },
      { name: "left_shoulder_rot_y", label: "RY" },
      { name: "left_shoulder_rot_z", label: "RZ" },
      { name: "left_upper_arm_length", label: "LEN" },
    ],
  },
  {
    id: "right_shoulder",
    label: "R SHOULDER",
    column: "right",
    row: 0,
    controls: [
      { name: "right_shoulder_rot_x", label: "RX" },
      { name: "right_shoulder_rot_y", label: "RY" },
      { name: "right_shoulder_rot_z", label: "RZ" },
      { name: "right_upper_arm_length", label: "LEN" },
    ],
  },
  {
    id: "left_elbow",
    label: "L ELBOW",
    column: "left",
    row: 1,
    controls: [
      { name: "left_elbow_rot_x", label: "RX" },
      { name: "left_elbow_rot_y", label: "RY" },
      { name: "left_elbow_rot_z", label: "RZ" },
      { name: "left_lower_arm_length", label: "LEN" },
    ],
  },
  {
    id: "right_elbow",
    label: "R ELBOW",
    column: "right",
    row: 1,
    controls: [
      { name: "right_elbow_rot_x", label: "RX" },
      { name: "right_elbow_rot_y", label: "RY" },
      { name: "right_elbow_rot_z", label: "RZ" },
      { name: "right_lower_arm_length", label: "LEN" },
    ],
  },
  {
    id: "pelvis",
    label: "PELVIS",
    column: "center",
    row: 3,
    controls: [
      { name: "root_rot_x", label: "RX" },
      { name: "root_rot_y", label: "RY" },
      { name: "root_rot_z", label: "RZ" },
      { name: "hip_width", label: "LEN" },
    ],
  },
  {
    id: "left_hip",
    label: "L HIP",
    column: "left",
    row: 2,
    controls: [
      { name: "left_hip_rot_x", label: "RX" },
      { name: "left_hip_rot_y", label: "RY" },
      { name: "left_hip_rot_z", label: "RZ" },
      { name: "left_upper_leg_length", label: "LEN" },
    ],
  },
  {
    id: "right_hip",
    label: "R HIP",
    column: "right",
    row: 2,
    controls: [
      { name: "right_hip_rot_x", label: "RX" },
      { name: "right_hip_rot_y", label: "RY" },
      { name: "right_hip_rot_z", label: "RZ" },
      { name: "right_upper_leg_length", label: "LEN" },
    ],
  },
  {
    id: "left_knee",
    label: "L KNEE",
    column: "left",
    row: 3,
    controls: [
      { name: "left_knee_rot_x", label: "RX" },
      { name: "left_knee_rot_y", label: "RY" },
      { name: "left_knee_rot_z", label: "RZ" },
      { name: "left_lower_leg_length", label: "LEN" },
    ],
  },
  {
    id: "right_knee",
    label: "R KNEE",
    column: "right",
    row: 3,
    controls: [
      { name: "right_knee_rot_x", label: "RX" },
      { name: "right_knee_rot_y", label: "RY" },
      { name: "right_knee_rot_z", label: "RZ" },
      { name: "right_lower_leg_length", label: "LEN" },
    ],
  },
  {
    id: "left_ankle",
    label: "L FOOT",
    column: "left",
    row: 4,
    controls: [
      { name: "left_ankle_rot_x", label: "RX" },
      { name: "left_ankle_rot_y", label: "RY" },
      { name: "left_ankle_rot_z", label: "RZ" },
      { name: "foot_length", label: "LEN" },
    ],
  },
  {
    id: "right_ankle",
    label: "R FOOT",
    column: "right",
    row: 4,
    controls: [
      { name: "right_ankle_rot_x", label: "RX" },
      { name: "right_ankle_rot_y", label: "RY" },
      { name: "right_ankle_rot_z", label: "RZ" },
      { name: "foot_length", label: "LEN" },
    ],
  },
];

const RENDER_ROWS = [
  {
    label: "VIEW",
    controls: [
      { name: "view_rot_x", label: "RX" },
      { name: "view_rot_y", label: "RY" },
      { name: "view_rot_z", label: "RZ" },
      { name: "view_offset_x", label: "X" },
    ],
  },
  {
    label: "VIEW",
    controls: [
      { name: "view_offset_y", label: "Y" },
      { name: "view_zoom", label: "ZOOM" },
      { name: "camera_distance", label: "DIST" },
      { name: "width", label: "W" },
    ],
  },
  {
    label: "RENDER",
    controls: [
      { name: "height", label: "H" },
      { name: "line_thickness", label: "LINE" },
      { name: "joint_radius", label: "JOINT" },
    ],
  },
  {
    label: "STYLE",
    controls: [
      { name: "render_mode", label: "MODE", kind: "toggle" },
      { name: "mesh_gender", label: "GENDER", kind: "toggle" },
    ],
  },
];

const CHARACTER_CONTROLS = [
  { name: "torso_length", label: "TORSO" },
  { name: "shoulder_width", label: "SHOULD" },
  { name: "hip_width", label: "HIP" },
  { name: "neck_length", label: "NECK" },
  { name: "head_size", label: "HEAD" },
  { name: "foot_length", label: "FOOT" },
  { name: "left_upper_arm_length", label: "L UP ARM" },
  { name: "left_lower_arm_length", label: "L LO ARM" },
  { name: "right_upper_arm_length", label: "R UP ARM" },
  { name: "right_lower_arm_length", label: "R LO ARM" },
  { name: "left_upper_leg_length", label: "L UP LEG" },
  { name: "left_lower_leg_length", label: "L LO LEG" },
  { name: "right_upper_leg_length", label: "R UP LEG" },
  { name: "right_lower_leg_length", label: "R LO LEG" },
  { name: "root_offset_x", label: "ROOT X" },
  { name: "root_offset_y", label: "ROOT Y" },
  { name: "root_offset_z", label: "ROOT Z" },
];

const REFERENCE_PARAMS = {
  view_rot_x: 0,
  view_rot_y: 0,
  view_rot_z: 0,
  view_offset_x: 0,
  view_offset_y: 0,
  view_zoom: 90,
  camera_distance: 6,
  width: 512,
  height: 512,
  root_rot_x: 0,
  root_rot_y: 0,
  root_rot_z: 0,
  root_offset_x: 0,
  root_offset_y: 0,
  root_offset_z: 0,
  torso_length: 1.4,
  neck_length: 0.3,
  head_size: 0.5,
  shoulder_width: 0.8,
  hip_width: 0.6,
  left_upper_arm_length: 0.9,
  left_lower_arm_length: 0.8,
  right_upper_arm_length: 0.9,
  right_lower_arm_length: 0.8,
  left_upper_leg_length: 1.1,
  left_lower_leg_length: 1.0,
  right_upper_leg_length: 1.1,
  right_lower_leg_length: 1.0,
  foot_length: 0.4,
  spine_rot_x: 0,
  spine_rot_y: 0,
  spine_rot_z: 0,
  neck_rot_x: 0,
  neck_rot_y: 0,
  neck_rot_z: 0,
  head_rot_x: 0,
  head_rot_y: 0,
  head_rot_z: 0,
  left_shoulder_rot_x: 0,
  left_shoulder_rot_y: 0,
  left_shoulder_rot_z: 0,
  left_elbow_rot_x: 0,
  left_elbow_rot_y: 0,
  left_elbow_rot_z: 0,
  right_shoulder_rot_x: 0,
  right_shoulder_rot_y: 0,
  right_shoulder_rot_z: 0,
  right_elbow_rot_x: 0,
  right_elbow_rot_y: 0,
  right_elbow_rot_z: 0,
  left_hip_rot_x: 0,
  left_hip_rot_y: 0,
  left_hip_rot_z: 0,
  left_knee_rot_x: 0,
  left_knee_rot_y: 0,
  left_knee_rot_z: 0,
  left_ankle_rot_x: 0,
  left_ankle_rot_y: 0,
  left_ankle_rot_z: 0,
  right_hip_rot_x: 0,
  right_hip_rot_y: 0,
  right_hip_rot_z: 0,
  right_knee_rot_x: 0,
  right_knee_rot_y: 0,
  right_knee_rot_z: 0,
  right_ankle_rot_x: 0,
  right_ankle_rot_y: 0,
  right_ankle_rot_z: 0,
};

app.registerExtension({
  name: "ess_pose_figure_editor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ESS/PoseFigureEditor") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      if (this.size && this.size[0] < MIN_NODE_WIDTH) {
        this.size[0] = MIN_NODE_WIDTH;
      }

      const layoutWidget = {
        type: "ESS_POSE_LAYOUT",
        name: "pose_layout",
        value: null,
        computeSize: (width) => [width, layoutHeight(width)],
        draw: (ctx, node, width, y, height) => {
          const params = collectParams(node);
          const referenceParams = {
            ...params,
            view_rot_x: 0,
            view_rot_y: 0,
            view_rot_z: 0,
            view_offset_x: 0,
            view_offset_y: 0,
            view_zoom: REFERENCE_PARAMS.view_zoom,
            camera_distance: REFERENCE_PARAMS.camera_distance,
            width: REFERENCE_PARAMS.width,
            height: REFERENCE_PARAMS.height,
          };
          const usable = width - PADDING * 2 - PANEL_GAP;
          const leftWidth = Math.floor(usable * LEFT_RATIO);
          const rightWidth = usable - leftWidth;
          const characterWidth = Math.max(
            CHARACTER_PANEL_MIN,
            Math.min(CHARACTER_PANEL_MAX, leftWidth * 0.35)
          );
          const referenceWidth = leftWidth - characterWidth - PANEL_GAP;
          const topHeight = Math.max(520, Math.max(referenceWidth, rightWidth));
          const bottomHeight = ROW_HEIGHT * RENDER_ROWS.length + ROW_GAP * (RENDER_ROWS.length - 1) + 24;

          const referenceRect = {
            x: PADDING,
            y: y + PADDING,
            w: referenceWidth,
            h: topHeight,
          };
          const characterRect = {
            x: PADDING + referenceWidth + PANEL_GAP,
            y: y + PADDING,
            w: characterWidth,
            h: topHeight,
          };
          const renderRect = {
            x: PADDING + leftWidth + PANEL_GAP,
            y: y + PADDING,
            w: rightWidth,
            h: topHeight,
          };

          const hitAreas = [];
          const widgetMap = getWidgetMap(node);

          const columnGap = 12;
          const poseWidth = Math.max(
            200,
            referenceRect.w - (CLUSTER_WIDTH * 3 + columnGap * 3 + 24)
          );
          const poseSize = Math.min(poseWidth, referenceRect.h - 40);
          const leftX = referenceRect.x + 12;
          const poseRect = {
            x: leftX + CLUSTER_WIDTH + columnGap,
            y: referenceRect.y + (referenceRect.h - poseSize) / 2,
            w: poseSize,
            h: poseSize,
          };
          const centerX = poseRect.x + poseRect.w + columnGap;
          const rightX = centerX + CLUSTER_WIDTH + columnGap;

          drawReferencePanel(ctx, referenceRect, referenceParams, poseRect);
          drawCharacterPanel(ctx, CHARACTER_CONTROLS, widgetMap, characterRect, hitAreas);
          drawRenderPanel(ctx, renderRect, params);

          const clusterBounds = {
            x: referenceRect.x + 8,
            y: referenceRect.y + 8,
            w: referenceRect.w - 16,
            h: referenceRect.h - 16,
          };

          const clusterHeight = getClusterHeight(4);
          const leftRightRows = computeRowPositions(poseRect, 5, clusterHeight);
          const centerRows = computeRowPositions(poseRect, 4, clusterHeight);

          const clampedLeftX = Math.max(clusterBounds.x, leftX);
          const clampedCenterX = Math.min(
            clusterBounds.x + clusterBounds.w - CLUSTER_WIDTH,
            Math.max(clusterBounds.x, centerX)
          );
          const clampedRightX = Math.min(
            clusterBounds.x + clusterBounds.w - CLUSTER_WIDTH,
            Math.max(clusterBounds.x, rightX)
          );

          for (const group of JOINT_GROUPS) {
            const column = group.column || "center";
            const rowIndex = group.row ?? 0;
            const x = column === "left"
              ? clampedLeftX
              : column === "right"
                ? clampedRightX
                : clampedCenterX;
            const y = column === "center"
              ? centerRows[Math.min(rowIndex, centerRows.length - 1)]
              : leftRightRows[Math.min(rowIndex, leftRightRows.length - 1)];
            drawCluster(
              ctx,
              {
                ...group,
                x,
                y,
              },
              widgetMap,
              hitAreas,
              clusterBounds
            );
          }

          const bottomRect = {
            x: PADDING + leftWidth + PANEL_GAP,
            y: y + PADDING + topHeight + 16,
            w: rightWidth,
            h: bottomHeight,
          };
          drawBottomRows(ctx, RENDER_ROWS, widgetMap, bottomRect, hitAreas);

          node._poseUI = node._poseUI || {};
          node._poseUI.hitAreas = hitAreas;
        },
        serializeValue: () => null,
      };

      this.addCustomWidget(layoutWidget);
      const widgetIndex = this.widgets?.indexOf(layoutWidget);
      if (widgetIndex != null && widgetIndex > 0) {
        this.widgets.splice(widgetIndex, 1);
        this.widgets.unshift(layoutWidget);
      }

      for (const widget of this.widgets || []) {
        if (widget === layoutWidget) {
          continue;
        }
        widget.hidden = true;
        widget.options = widget.options || {};
        widget.options.hidden = true;
        widget.computeSize = () => [0, 0];
      }

      this.setSize?.([this.size[0], layoutHeight(this.size[0])]);
      this.setDirtyCanvas?.(true, true);
      return result;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      if (this.size && this.size[0] < MIN_NODE_WIDTH) {
        this.size[0] = MIN_NODE_WIDTH;
      }
      this.size[1] = layoutHeight(this.size[0]);
      this.setDirtyCanvas?.(true, true);
      if (onResize) {
        onResize.apply(this, arguments);
      }
    };

    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (event, localPos, canvas) {
      const hitAreas = this._poseUI?.hitAreas || [];
      const hit = hitAreas.find((area) => (
        localPos[0] >= area.x &&
        localPos[0] <= area.x + area.w &&
        localPos[1] >= area.y &&
        localPos[1] <= area.y + area.h
      ));

      if (hit) {
        this._poseUI = this._poseUI || {};
        if (hit.type === "toggle") {
          setWidgetValue(hit.widget, hit.optionValue, this);
          this._poseUI.active = null;
        } else {
          this._poseUI.active = hit;
          const t = (localPos[0] - hit.trackX) / (hit.trackW || 1);
          const range = getWidgetRange(hit.widget, 0, 1, 0);
          const value = range.min + Math.max(0, Math.min(1, t)) * (range.max - range.min);
          setWidgetValue(hit.widget, value, this);
        }
        this.setDirtyCanvas?.(true, true);
        return true;
      }

      return onMouseDown ? onMouseDown.call(this, event, localPos, canvas) : false;
    };

    const onMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (event, localPos, canvas) {
      const active = this._poseUI?.active;
      if (active && active.type === "slider") {
        const t = (localPos[0] - active.trackX) / (active.trackW || 1);
        const range = getWidgetRange(active.widget, 0, 1, 0);
        const value = range.min + Math.max(0, Math.min(1, t)) * (range.max - range.min);
        setWidgetValue(active.widget, value, this);
        this.setDirtyCanvas?.(true, true);
        return true;
      }
      return onMouseMove ? onMouseMove.call(this, event, localPos, canvas) : false;
    };

    const onMouseUp = nodeType.prototype.onMouseUp;
    nodeType.prototype.onMouseUp = function (event, localPos, canvas) {
      if (this._poseUI?.active) {
        this._poseUI.active = null;
        return true;
      }
      return onMouseUp ? onMouseUp.call(this, event, localPos, canvas) : false;
    };
  },
});
