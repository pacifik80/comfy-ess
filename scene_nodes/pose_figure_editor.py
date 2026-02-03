import json
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch


_BODY_25_NAMES = [
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
]

_BODY_25_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (11, 24),
    (11, 22),
    (22, 23),
    (14, 21),
    (14, 19),
    (19, 20),
]


def _rot_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rot_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rot_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_xyz(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    return _rot_z(z_deg) @ _rot_y(y_deg) @ _rot_x(x_deg)


def _apply_rot(rot: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return rot @ vec


def _project_points(
    points: Dict[str, np.ndarray],
    view_rot: np.ndarray,
    width: int,
    height: int,
    view_zoom: float,
    view_offset_x: float,
    view_offset_y: float,
    camera_distance: float,
) -> Dict[str, Tuple[float, float]]:
    half_x = width * 0.5
    half_y = height * 0.5
    projected: Dict[str, Tuple[float, float]] = {}
    for name, point in points.items():
        p = _apply_rot(view_rot, point)
        depth = camera_distance - p[2]
        depth = max(depth, 0.1)
        scale = camera_distance / depth
        x = (p[0] * scale) * view_zoom + half_x + view_offset_x
        y = (-p[1] * scale) * view_zoom + half_y + view_offset_y
        projected[name] = (float(x), float(y))
    return projected


def _build_pose_points(params: Dict[str, float]) -> Dict[str, np.ndarray]:
    root_rot = _rot_xyz(params["root_rot_x"], params["root_rot_y"], params["root_rot_z"])
    spine_rot = _rot_xyz(params["spine_rot_x"], params["spine_rot_y"], params["spine_rot_z"])
    neck_rot = _rot_xyz(params["neck_rot_x"], params["neck_rot_y"], params["neck_rot_z"])
    head_rot = _rot_xyz(params["head_rot_x"], params["head_rot_y"], params["head_rot_z"])

    pelvis = np.array([params["root_offset_x"], params["root_offset_y"], params["root_offset_z"]], dtype=np.float32)

    torso_dir = _apply_rot(root_rot, _apply_rot(spine_rot, np.array([0.0, 1.0, 0.0], dtype=np.float32)))
    neck_base = pelvis + torso_dir * params["torso_length"]

    shoulder_offset = _apply_rot(root_rot, _apply_rot(spine_rot, np.array([params["shoulder_width"] * 0.5, 0.0, 0.0], dtype=np.float32)))
    hip_offset = _apply_rot(root_rot, np.array([params["hip_width"] * 0.5, 0.0, 0.0], dtype=np.float32))

    right_shoulder = neck_base + shoulder_offset
    left_shoulder = neck_base - shoulder_offset
    right_hip = pelvis + hip_offset
    left_hip = pelvis - hip_offset

    neck_basis = root_rot @ spine_rot @ neck_rot
    head_basis = neck_basis @ head_rot
    neck_up = _apply_rot(neck_basis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    head_up = _apply_rot(head_basis, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    head_forward = _apply_rot(head_basis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    head_right = _apply_rot(head_basis, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    head_center = neck_base + neck_up * (params["neck_length"] + params["head_size"] * 0.5)
    nose = head_center + head_forward * (params["head_size"] * 0.6)
    eye_offset = head_up * (params["head_size"] * 0.15) + head_forward * (params["head_size"] * 0.45)
    right_eye = head_center + eye_offset + head_right * (params["head_size"] * 0.2)
    left_eye = head_center + eye_offset - head_right * (params["head_size"] * 0.2)
    right_ear = head_center + head_right * (params["head_size"] * 0.45) + head_up * (params["head_size"] * 0.05)
    left_ear = head_center - head_right * (params["head_size"] * 0.45) + head_up * (params["head_size"] * 0.05)

    def arm_chain(side_prefix: str, shoulder_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shoulder_rot = _rot_xyz(
            params[f"{side_prefix}_shoulder_rot_x"],
            params[f"{side_prefix}_shoulder_rot_y"],
            params[f"{side_prefix}_shoulder_rot_z"],
        )
        upper_basis = root_rot @ spine_rot @ shoulder_rot
        upper_dir = _apply_rot(upper_basis, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        elbow = shoulder_pos + upper_dir * params[f"{side_prefix}_upper_arm_length"]

        elbow_rot = _rot_xyz(
            params[f"{side_prefix}_elbow_rot_x"],
            params[f"{side_prefix}_elbow_rot_y"],
            params[f"{side_prefix}_elbow_rot_z"],
        )
        lower_dir = _apply_rot(upper_basis @ elbow_rot, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        wrist = elbow + lower_dir * params[f"{side_prefix}_lower_arm_length"]
        return elbow, wrist

    def leg_chain(side_prefix: str, hip_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        hip_rot = _rot_xyz(
            params[f"{side_prefix}_hip_rot_x"],
            params[f"{side_prefix}_hip_rot_y"],
            params[f"{side_prefix}_hip_rot_z"],
        )
        upper_basis = root_rot @ hip_rot
        upper_dir = _apply_rot(upper_basis, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        knee = hip_pos + upper_dir * params[f"{side_prefix}_upper_leg_length"]

        knee_rot = _rot_xyz(
            params[f"{side_prefix}_knee_rot_x"],
            params[f"{side_prefix}_knee_rot_y"],
            params[f"{side_prefix}_knee_rot_z"],
        )
        lower_dir = _apply_rot(upper_basis @ knee_rot, np.array([0.0, -1.0, 0.0], dtype=np.float32))
        ankle = knee + lower_dir * params[f"{side_prefix}_lower_leg_length"]

        ankle_rot = _rot_xyz(
            params[f"{side_prefix}_ankle_rot_x"],
            params[f"{side_prefix}_ankle_rot_y"],
            params[f"{side_prefix}_ankle_rot_z"],
        )
        foot_basis = upper_basis @ knee_rot @ ankle_rot
        foot_forward = _apply_rot(foot_basis, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        foot_right = _apply_rot(foot_basis, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        toe_center = ankle + foot_forward * params["foot_length"]
        heel = ankle - foot_forward * (params["foot_length"] * 0.5)
        toe_spread = params["foot_length"] * 0.25
        big_toe = toe_center + foot_right * toe_spread
        small_toe = toe_center - foot_right * toe_spread
        return knee, ankle, big_toe, small_toe, heel

    right_elbow, right_wrist = arm_chain("right", right_shoulder)
    left_elbow, left_wrist = arm_chain("left", left_shoulder)
    right_knee, right_ankle, right_big_toe, right_small_toe, right_heel = leg_chain("right", right_hip)
    left_knee, left_ankle, left_big_toe, left_small_toe, left_heel = leg_chain("left", left_hip)

    return {
        "Nose": nose,
        "Neck": neck_base,
        "HeadCenter": head_center,
        "RShoulder": right_shoulder,
        "RElbow": right_elbow,
        "RWrist": right_wrist,
        "LShoulder": left_shoulder,
        "LElbow": left_elbow,
        "LWrist": left_wrist,
        "MidHip": pelvis,
        "RHip": right_hip,
        "RKnee": right_knee,
        "RAnkle": right_ankle,
        "LHip": left_hip,
        "LKnee": left_knee,
        "LAnkle": left_ankle,
        "REye": right_eye,
        "LEye": left_eye,
        "REar": right_ear,
        "LEar": left_ear,
        "LBigToe": left_big_toe,
        "LSmallToe": left_small_toe,
        "LHeel": left_heel,
        "RBigToe": right_big_toe,
        "RSmallToe": right_small_toe,
        "RHeel": right_heel,
    }


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return vec
    return vec / norm


def _build_cylinder(a: np.ndarray, b: np.ndarray, radius: float, segments: int = 8) -> List[np.ndarray]:
    axis = _normalize(b - a)
    if abs(axis[1]) < 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent = _normalize(np.cross(axis, ref))
    bitangent = _normalize(np.cross(axis, tangent))

    triangles = []
    for i in range(segments):
        angle0 = (i / segments) * math.tau
        angle1 = ((i + 1) / segments) * math.tau
        dir0 = tangent * math.cos(angle0) + bitangent * math.sin(angle0)
        dir1 = tangent * math.cos(angle1) + bitangent * math.sin(angle1)
        s0 = a + dir0 * radius
        s1 = a + dir1 * radius
        e0 = b + dir0 * radius
        e1 = b + dir1 * radius
        triangles.append(np.stack([s0, e0, e1], axis=0))
        triangles.append(np.stack([s0, e1, s1], axis=0))
    return triangles


def _build_box(
    a: np.ndarray,
    b: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    width_a: float,
    width_b: float,
    depth_a: float,
    depth_b: float,
) -> List[np.ndarray]:
    right = _normalize(right)
    forward = _normalize(forward)
    p0 = a + right * (width_a * 0.5) + forward * (depth_a * 0.5)
    p1 = a - right * (width_a * 0.5) + forward * (depth_a * 0.5)
    p2 = a - right * (width_a * 0.5) - forward * (depth_a * 0.5)
    p3 = a + right * (width_a * 0.5) - forward * (depth_a * 0.5)

    p4 = b + right * (width_b * 0.5) + forward * (depth_b * 0.5)
    p5 = b - right * (width_b * 0.5) + forward * (depth_b * 0.5)
    p6 = b - right * (width_b * 0.5) - forward * (depth_b * 0.5)
    p7 = b + right * (width_b * 0.5) - forward * (depth_b * 0.5)

    faces = [
        (p0, p1, p2, p3),  # bottom
        (p4, p5, p6, p7),  # top
        (p0, p4, p7, p3),  # side
        (p1, p5, p6, p2),  # side
        (p0, p1, p5, p4),  # front
        (p3, p2, p6, p7),  # back
    ]
    triangles = []
    for a0, a1, a2, a3 in faces:
        triangles.append(np.stack([a0, a1, a2], axis=0))
        triangles.append(np.stack([a0, a2, a3], axis=0))
    return triangles


def _build_sphere(center: np.ndarray, radius: float, rings: int = 6, segments: int = 8) -> List[np.ndarray]:
    triangles = []
    for i in range(rings):
        lat0 = math.pi * (i / rings - 0.5)
        lat1 = math.pi * ((i + 1) / rings - 0.5)
        y0 = math.sin(lat0)
        y1 = math.sin(lat1)
        r0 = math.cos(lat0)
        r1 = math.cos(lat1)
        for j in range(segments):
            lon0 = math.tau * (j / segments)
            lon1 = math.tau * ((j + 1) / segments)
            p0 = center + np.array([r0 * math.cos(lon0), y0, r0 * math.sin(lon0)], dtype=np.float32) * radius
            p1 = center + np.array([r1 * math.cos(lon0), y1, r1 * math.sin(lon0)], dtype=np.float32) * radius
            p2 = center + np.array([r1 * math.cos(lon1), y1, r1 * math.sin(lon1)], dtype=np.float32) * radius
            p3 = center + np.array([r0 * math.cos(lon1), y0, r0 * math.sin(lon1)], dtype=np.float32) * radius
            triangles.append(np.stack([p0, p1, p2], axis=0))
            triangles.append(np.stack([p0, p2, p3], axis=0))
    return triangles


def _mesh_triangles(points: Dict[str, np.ndarray], params: Dict[str, float], gender: str) -> List[np.ndarray]:
    male_factor = 1.0
    female_factor = 0.92
    is_female = gender == "female"

    shoulder_width = params["shoulder_width"] * (female_factor if is_female else male_factor)
    hip_width = params["hip_width"] * (1.1 if is_female else 0.95)
    limb_radius = (0.075 if is_female else 0.085) * (params["torso_length"] / 1.4)

    pelvis = points["MidHip"]
    neck_base = points["Neck"]
    head_center = points["HeadCenter"]

    root_rot = _rot_xyz(params["root_rot_x"], params["root_rot_y"], params["root_rot_z"])
    right = _apply_rot(root_rot, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    forward = _apply_rot(root_rot, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    depth_hip = hip_width * 0.45
    depth_shoulder = shoulder_width * 0.4

    triangles = []
    triangles.extend(_build_box(pelvis, neck_base, right, forward, hip_width, shoulder_width, depth_hip, depth_shoulder))
    triangles.extend(_build_sphere(head_center, params["head_size"] * 0.55, rings=6, segments=10))
    triangles.extend(_build_cylinder(neck_base, head_center, limb_radius * 0.9, segments=8))

    def add_limb(a_name: str, b_name: str, radius_scale: float = 1.0):
        triangles.extend(_build_cylinder(points[a_name], points[b_name], limb_radius * radius_scale, segments=8))

    add_limb("RShoulder", "RElbow", 1.0)
    add_limb("RElbow", "RWrist", 0.9)
    add_limb("LShoulder", "LElbow", 1.0)
    add_limb("LElbow", "LWrist", 0.9)
    add_limb("RHip", "RKnee", 1.1)
    add_limb("RKnee", "RAnkle", 1.0)
    add_limb("LHip", "LKnee", 1.1)
    add_limb("LKnee", "LAnkle", 1.0)

    add_limb("RAnkle", "RBigToe", 0.7)
    add_limb("LAnkle", "LBigToe", 0.7)

    return triangles


def _draw_mesh(
    canvas: np.ndarray,
    triangles: List[np.ndarray],
    width: int,
    height: int,
    view_rot: np.ndarray,
    view_zoom: float,
    view_offset_x: float,
    view_offset_y: float,
    camera_distance: float,
) -> None:
    light_dir = _normalize(np.array([0.2, 0.6, 1.0], dtype=np.float32))
    overlay = canvas.copy()
    for tri in sorted(triangles, key=lambda t: float(np.mean(t[:, 2])), reverse=False):
        pts_2d = _project_points(
            {"a": tri[0], "b": tri[1], "c": tri[2]},
            view_rot,
            width,
            height,
            view_zoom,
            view_offset_x,
            view_offset_y,
            camera_distance,
        )
        a = np.array(pts_2d["a"], dtype=np.float32)
        b = np.array(pts_2d["b"], dtype=np.float32)
        c = np.array(pts_2d["c"], dtype=np.float32)
        normal = _normalize(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
        shade = 0.4 + 0.6 * max(0.0, float(np.dot(normal, light_dir)))
        color = np.array([180, 190, 210], dtype=np.float32) * shade
        poly = np.array([a, b, c], dtype=np.int32)
        cv2.fillConvexPoly(overlay, poly, color.tolist(), lineType=cv2.LINE_AA)

    alpha = 0.35
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)


def _build_openpose_json(points_2d: Dict[str, Tuple[float, float]]) -> str:
    keypoints: List[float] = []
    for name in _BODY_25_NAMES:
        x, y = points_2d.get(name, (0.0, 0.0))
        keypoints.extend([x, y, 1.0])
    payload = {
        "version": 1.3,
        "people": [
            {
                "pose_keypoints_2d": keypoints,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
            }
        ],
    }
    return json.dumps(payload, separators=(",", ":"))


def _render_pose_image(
    points_2d: Dict[str, Tuple[float, float]],
    width: int,
    height: int,
    line_thickness: int,
    joint_radius: int,
) -> torch.Tensor:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for start_idx, end_idx in _BODY_25_EDGES:
        start_name = _BODY_25_NAMES[start_idx]
        end_name = _BODY_25_NAMES[end_idx]
        x1, y1 = points_2d[start_name]
        x2, y2 = points_2d[end_name]
        cv2.line(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 255, 255),
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )

    for name in _BODY_25_NAMES:
        x, y = points_2d[name]
        cv2.circle(
            canvas,
            (int(round(x)), int(round(y))),
            joint_radius,
            (255, 255, 255),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)
    return tensor


class PoseFigureEditor:
    CATEGORY = "ESS/Pose"
    FUNCTION = "render_pose"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "openpose_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "view": ("ESS_SEPARATOR", {"label": "view", "tooltip": "Viewport controls."}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "joint_radius": ("INT", {"default": 6, "min": 1, "max": 32, "step": 1}),
                "render_mode": (("lines", "mesh"), {"default": "lines"}),
                "mesh_gender": (("male", "female"), {"default": "male"}),
                "view_rot_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "view_rot_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "view_rot_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "view_offset_x": ("FLOAT", {"default": 0.0, "min": -512.0, "max": 512.0, "step": 1.0}),
                "view_offset_y": ("FLOAT", {"default": 0.0, "min": -512.0, "max": 512.0, "step": 1.0}),
                "view_zoom": ("FLOAT", {"default": 160.0, "min": 20.0, "max": 600.0, "step": 1.0}),
                "camera_distance": ("FLOAT", {"default": 6.0, "min": 2.0, "max": 20.0, "step": 0.1}),
                "body": ("ESS_SEPARATOR", {"label": "body proportions", "tooltip": "Overall body proportions."}),
                "root_rot_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "root_rot_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "root_rot_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "root_offset_x": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "root_offset_y": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "root_offset_z": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "torso_length": ("FLOAT", {"default": 1.4, "min": 0.5, "max": 3.0, "step": 0.05}),
                "neck_length": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "head_size": ("FLOAT", {"default": 0.5, "min": 0.2, "max": 1.2, "step": 0.05}),
                "shoulder_width": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 2.0, "step": 0.05}),
                "hip_width": ("FLOAT", {"default": 0.6, "min": 0.3, "max": 1.8, "step": 0.05}),
                "left_upper_arm_length": ("FLOAT", {"default": 0.9, "min": 0.3, "max": 2.0, "step": 0.05}),
                "left_lower_arm_length": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 2.0, "step": 0.05}),
                "right_upper_arm_length": ("FLOAT", {"default": 0.9, "min": 0.3, "max": 2.0, "step": 0.05}),
                "right_lower_arm_length": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 2.0, "step": 0.05}),
                "left_upper_leg_length": ("FLOAT", {"default": 1.1, "min": 0.4, "max": 2.5, "step": 0.05}),
                "left_lower_leg_length": ("FLOAT", {"default": 1.0, "min": 0.4, "max": 2.5, "step": 0.05}),
                "right_upper_leg_length": ("FLOAT", {"default": 1.1, "min": 0.4, "max": 2.5, "step": 0.05}),
                "right_lower_leg_length": ("FLOAT", {"default": 1.0, "min": 0.4, "max": 2.5, "step": 0.05}),
                "foot_length": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "spine": ("ESS_SEPARATOR", {"label": "spine & neck", "tooltip": "Spine and neck articulation."}),
                "spine_rot_x": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "spine_rot_y": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "spine_rot_z": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "neck_rot_x": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "neck_rot_y": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "neck_rot_z": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "head_rot_x": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "head_rot_y": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "head_rot_z": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "left_arm": ("ESS_SEPARATOR", {"label": "left arm", "tooltip": "Left arm articulation."}),
                "left_shoulder_rot_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "left_shoulder_rot_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "left_shoulder_rot_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "left_elbow_rot_x": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "left_elbow_rot_y": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "left_elbow_rot_z": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "right_arm": ("ESS_SEPARATOR", {"label": "right arm", "tooltip": "Right arm articulation."}),
                "right_shoulder_rot_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "right_shoulder_rot_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "right_shoulder_rot_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "right_elbow_rot_x": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "right_elbow_rot_y": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "right_elbow_rot_z": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "left_leg": ("ESS_SEPARATOR", {"label": "left leg", "tooltip": "Left leg articulation."}),
                "left_hip_rot_x": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 1.0}),
                "left_hip_rot_y": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 1.0}),
                "left_hip_rot_z": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 1.0}),
                "left_knee_rot_x": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "left_knee_rot_y": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "left_knee_rot_z": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "left_ankle_rot_x": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "left_ankle_rot_y": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "left_ankle_rot_z": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "right_leg": ("ESS_SEPARATOR", {"label": "right leg", "tooltip": "Right leg articulation."}),
                "right_hip_rot_x": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 1.0}),
                "right_hip_rot_y": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 1.0}),
                "right_hip_rot_z": ("FLOAT", {"default": 0.0, "min": -120.0, "max": 120.0, "step": 1.0}),
                "right_knee_rot_x": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "right_knee_rot_y": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "right_knee_rot_z": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "right_ankle_rot_x": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "right_ankle_rot_y": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "right_ankle_rot_z": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 1.0}),
            }
        }

    def render_pose(
        self,
        width: int,
        height: int,
        line_thickness: int,
        joint_radius: int,
        render_mode: str,
        mesh_gender: str,
        view_rot_x: float,
        view_rot_y: float,
        view_rot_z: float,
        view_offset_x: float,
        view_offset_y: float,
        view_zoom: float,
        camera_distance: float,
        root_rot_x: float,
        root_rot_y: float,
        root_rot_z: float,
        root_offset_x: float,
        root_offset_y: float,
        root_offset_z: float,
        torso_length: float,
        neck_length: float,
        head_size: float,
        shoulder_width: float,
        hip_width: float,
        left_upper_arm_length: float,
        left_lower_arm_length: float,
        right_upper_arm_length: float,
        right_lower_arm_length: float,
        left_upper_leg_length: float,
        left_lower_leg_length: float,
        right_upper_leg_length: float,
        right_lower_leg_length: float,
        foot_length: float,
        spine_rot_x: float,
        spine_rot_y: float,
        spine_rot_z: float,
        neck_rot_x: float,
        neck_rot_y: float,
        neck_rot_z: float,
        head_rot_x: float,
        head_rot_y: float,
        head_rot_z: float,
        left_shoulder_rot_x: float,
        left_shoulder_rot_y: float,
        left_shoulder_rot_z: float,
        left_elbow_rot_x: float,
        left_elbow_rot_y: float,
        left_elbow_rot_z: float,
        right_shoulder_rot_x: float,
        right_shoulder_rot_y: float,
        right_shoulder_rot_z: float,
        right_elbow_rot_x: float,
        right_elbow_rot_y: float,
        right_elbow_rot_z: float,
        left_hip_rot_x: float,
        left_hip_rot_y: float,
        left_hip_rot_z: float,
        left_knee_rot_x: float,
        left_knee_rot_y: float,
        left_knee_rot_z: float,
        left_ankle_rot_x: float,
        left_ankle_rot_y: float,
        left_ankle_rot_z: float,
        right_hip_rot_x: float,
        right_hip_rot_y: float,
        right_hip_rot_z: float,
        right_knee_rot_x: float,
        right_knee_rot_y: float,
        right_knee_rot_z: float,
        right_ankle_rot_x: float,
        right_ankle_rot_y: float,
        right_ankle_rot_z: float,
    ) -> Tuple[torch.Tensor, str]:
        params = {
            "view_rot_x": view_rot_x,
            "view_rot_y": view_rot_y,
            "view_rot_z": view_rot_z,
            "root_rot_x": root_rot_x,
            "root_rot_y": root_rot_y,
            "root_rot_z": root_rot_z,
            "root_offset_x": root_offset_x,
            "root_offset_y": root_offset_y,
            "root_offset_z": root_offset_z,
            "torso_length": torso_length,
            "neck_length": neck_length,
            "head_size": head_size,
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "left_upper_arm_length": left_upper_arm_length,
            "left_lower_arm_length": left_lower_arm_length,
            "right_upper_arm_length": right_upper_arm_length,
            "right_lower_arm_length": right_lower_arm_length,
            "left_upper_leg_length": left_upper_leg_length,
            "left_lower_leg_length": left_lower_leg_length,
            "right_upper_leg_length": right_upper_leg_length,
            "right_lower_leg_length": right_lower_leg_length,
            "foot_length": foot_length,
            "spine_rot_x": spine_rot_x,
            "spine_rot_y": spine_rot_y,
            "spine_rot_z": spine_rot_z,
            "neck_rot_x": neck_rot_x,
            "neck_rot_y": neck_rot_y,
            "neck_rot_z": neck_rot_z,
            "head_rot_x": head_rot_x,
            "head_rot_y": head_rot_y,
            "head_rot_z": head_rot_z,
            "left_shoulder_rot_x": left_shoulder_rot_x,
            "left_shoulder_rot_y": left_shoulder_rot_y,
            "left_shoulder_rot_z": left_shoulder_rot_z,
            "left_elbow_rot_x": left_elbow_rot_x,
            "left_elbow_rot_y": left_elbow_rot_y,
            "left_elbow_rot_z": left_elbow_rot_z,
            "right_shoulder_rot_x": right_shoulder_rot_x,
            "right_shoulder_rot_y": right_shoulder_rot_y,
            "right_shoulder_rot_z": right_shoulder_rot_z,
            "right_elbow_rot_x": right_elbow_rot_x,
            "right_elbow_rot_y": right_elbow_rot_y,
            "right_elbow_rot_z": right_elbow_rot_z,
            "left_hip_rot_x": left_hip_rot_x,
            "left_hip_rot_y": left_hip_rot_y,
            "left_hip_rot_z": left_hip_rot_z,
            "left_knee_rot_x": left_knee_rot_x,
            "left_knee_rot_y": left_knee_rot_y,
            "left_knee_rot_z": left_knee_rot_z,
            "left_ankle_rot_x": left_ankle_rot_x,
            "left_ankle_rot_y": left_ankle_rot_y,
            "left_ankle_rot_z": left_ankle_rot_z,
            "right_hip_rot_x": right_hip_rot_x,
            "right_hip_rot_y": right_hip_rot_y,
            "right_hip_rot_z": right_hip_rot_z,
            "right_knee_rot_x": right_knee_rot_x,
            "right_knee_rot_y": right_knee_rot_y,
            "right_knee_rot_z": right_knee_rot_z,
            "right_ankle_rot_x": right_ankle_rot_x,
            "right_ankle_rot_y": right_ankle_rot_y,
            "right_ankle_rot_z": right_ankle_rot_z,
        }

        points_3d = _build_pose_points(params)
        view_rot = _rot_xyz(view_rot_x, view_rot_y, view_rot_z)
        points_2d = _project_points(
            points_3d,
            view_rot,
            width,
            height,
            view_zoom,
            view_offset_x,
            view_offset_y,
            camera_distance,
        )
        image = _render_pose_image(points_2d, width, height, line_thickness, joint_radius)
        if render_mode == "mesh":
            frame = image[0].detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = (frame * 255.0).round().clip(0, 255).astype(np.uint8)
            triangles = _mesh_triangles(points_3d, params, mesh_gender)
            _draw_mesh(
                frame,
                triangles,
                width,
                height,
                view_rot,
                view_zoom,
                view_offset_x,
                view_offset_y,
                camera_distance,
            )
            image = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
        openpose_json = _build_openpose_json(points_2d)
        return (image, openpose_json)
