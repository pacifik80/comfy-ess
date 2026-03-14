from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .person_crop_to_size import (
    _list_ultralytics_models,
    _load_model,
    _resolve_device,
    _selector_from_list,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


_FACE_ANALYZER = None
_FACE_ANALYZER_READY = False
_PART_SPECS = (
    {"key": "nose", "label": "Nose", "paired": False, "default_enabled": True, "default_mode": "include", "default_weight": 40.0, "default_margin": 0.8},
    {"key": "eye", "label": "Eye", "paired": True, "default_enabled": True, "default_mode": "include", "default_weight": 100.0, "default_margin": 1.8},
    {"key": "eyebrow", "label": "Eyebrow", "paired": True, "default_enabled": False, "default_mode": "include", "default_weight": 35.0, "default_margin": 1.2},
    {"key": "ear", "label": "Ear", "paired": True, "default_enabled": True, "default_mode": "include", "default_weight": 20.0, "default_margin": 1.2},
    {"key": "shoulder", "label": "Shoulder", "paired": True, "default_enabled": True, "default_mode": "include", "default_weight": 35.0, "default_margin": 1.0},
    {"key": "elbow", "label": "Elbow", "paired": True, "default_enabled": True, "default_mode": "include", "default_weight": 25.0, "default_margin": 1.0},
    {"key": "wrist", "label": "Wrist", "paired": True, "default_enabled": True, "default_mode": "include", "default_weight": 20.0, "default_margin": 1.0},
    {"key": "hip", "label": "Hip", "paired": True, "default_enabled": True, "default_mode": "include", "default_weight": 55.0, "default_margin": 1.0},
    {"key": "knee", "label": "Knee", "paired": True, "default_enabled": False, "default_mode": "include", "default_weight": 0.0, "default_margin": 1.0},
    {"key": "ankle", "label": "Ankle", "paired": True, "default_enabled": False, "default_mode": "include", "default_weight": 0.0, "default_margin": 1.2},
    {"key": "face", "label": "Face", "paired": False, "default_enabled": True, "default_mode": "include", "default_weight": 80.0, "default_margin": 0.8},
    {"key": "mouth", "label": "Mouth", "paired": False, "default_enabled": True, "default_mode": "include", "default_weight": 70.0, "default_margin": 1.8},
    {"key": "mouth_corner", "label": "Mouth Corner", "paired": True, "default_enabled": False, "default_mode": "include", "default_weight": 45.0, "default_margin": 1.0},
    {"key": "chin", "label": "Chin", "paired": False, "default_enabled": False, "default_mode": "include", "default_weight": 30.0, "default_margin": 0.8},
    {"key": "full_person", "label": "Full", "paired": False, "default_enabled": True, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.4},
)
_PART_SPEC_MAP = {spec["key"]: spec for spec in _PART_SPECS}
_PAIR_SIDE_VALUES = ("both", "left", "right")
_MODE_VALUES = ("include", "exclude")
_POSE_MODEL_DEFAULTS = ("yolo11n-pose.pt", "yolov8n-pose.pt", "yolov8s-pose.pt")


def _list_pose_model_options() -> list[str]:
    models = list(_list_ultralytics_models())
    seen = {str(item).strip() for item in models}
    for name in _POSE_MODEL_DEFAULTS:
        if name not in seen:
            models.append(name)
            seen.add(name)
    return models


def _tensor_to_uint8_image(image: torch.Tensor) -> np.ndarray:
    if image.dim() != 4:
        raise ValueError(f"Expected image tensor in NHWC format, got shape {tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError("This node expects a single image (batch size 1).")
    img = image[0].detach().cpu().clamp(0.0, 1.0)
    return (img * 255.0).to(torch.uint8).numpy()


def _uint8_image_to_tensor(image_uint8: np.ndarray, device: torch.device | str) -> torch.Tensor:
    arr = image_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


def _clip_box(box: Tuple[float, float, float, float], image_w: int, image_h: int) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    x0 = max(0, min(image_w - 1, x0))
    y0 = max(0, min(image_h - 1, y0))
    x1 = max(0, min(image_w, x1))
    y1 = max(0, min(image_h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _box_from_points(points: List[Tuple[float, float]], image_w: int, image_h: int, pad_ratio: float = 0.0) -> Optional[Tuple[int, int, int, int]]:
    valid = [(float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if not valid:
        return None
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    pad_x = (x1 - x0) * float(pad_ratio)
    pad_y = (y1 - y0) * float(pad_ratio)
    return _clip_box((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), image_w, image_h)


def _box_from_center_and_size(
    center: Tuple[float, float],
    half_w: float,
    half_h: float,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    cx, cy = center
    return _clip_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), image_w, image_h)


def _expand_box(box: Tuple[int, int, int, int], margin_ratio: float, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    mx = w * float(margin_ratio)
    my = h * float(margin_ratio)
    return _clip_box((x0 - mx, y0 - my, x1 + mx, y1 + my), image_w, image_h) or box


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _box_area(box: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _coverage(a: Tuple[int, int, int, int], crop: Tuple[int, int, int, int]) -> float:
    area = _box_area(a)
    if area <= 1e-6:
        return 0.0
    return _intersection_area(a, crop) / area


def _get_face_analyzer() -> Any:
    global _FACE_ANALYZER, _FACE_ANALYZER_READY
    if _FACE_ANALYZER_READY:
        return _FACE_ANALYZER
    _FACE_ANALYZER_READY = True
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception:
        _FACE_ANALYZER = None
        return None
    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_thresh=0.30, det_size=(640, 640))
        _FACE_ANALYZER = analyzer
    except Exception:
        _FACE_ANALYZER = None
    return _FACE_ANALYZER


def _extract_faces(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return []
    out: List[Dict[str, Any]] = []
    try:
        faces = analyzer.get(image_bgr)
    except Exception:
        return out
    for face in faces or []:
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        age = getattr(face, "age", None)
        age_val = int(age) if age is not None and str(age).strip() else None
        gender_raw = getattr(face, "gender", None)
        gender = "unknown"
        if gender_raw == 0:
            gender = "female"
        elif gender_raw == 1:
            gender = "male"
        kps = getattr(face, "kps", None)
        landmark_106 = getattr(face, "landmark_2d_106", None)
        points = []
        try:
            if kps is not None:
                points = [[float(p[0]), float(p[1])] for p in kps]
        except Exception:
            points = []
        points_106 = []
        try:
            if landmark_106 is not None:
                points_106 = [[float(p[0]), float(p[1])] for p in landmark_106]
        except Exception:
            points_106 = []
        out.append(
            {
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                "age": age_val,
                "gender": gender,
                "kps": points,
                "landmark_106": points_106,
            }
        )
    return out


def _match_face(person_box: Tuple[int, int, int, int], faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces:
        return None
    x0, y0, x1, y1 = [float(v) for v in person_box]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    best = None
    best_rank = float("inf")
    for face in faces:
        fx0, fy0, fx1, fy1 = [float(v) for v in face["bbox"]]
        fcx, fcy = face["center"]
        inside = (x0 <= fcx <= x1) and (y0 <= fcy <= y1)
        rank = ((fcx - cx) ** 2 + (fcy - cy) ** 2) ** 0.5
        if not inside:
            rank += 1e6
        if rank < best_rank:
            best_rank = rank
            best = face
    return best


def _age_score(age: Optional[int], age_min: int, age_max: int) -> float:
    if age is None:
        return 0.25
    lo = min(age_min, age_max)
    hi = max(age_min, age_max)
    if lo <= age <= hi:
        return 1.0
    dist = min(abs(age - lo), abs(age - hi))
    return max(0.0, 1.0 - (dist / 25.0))


def _gender_score(preferred: str, gender: str) -> float:
    pref = str(preferred or "any").strip().lower()
    gen = str(gender or "unknown").strip().lower()
    if pref == "any":
        return 1.0
    if gen == pref:
        return 1.0
    if gen == "unknown":
        return 0.35
    return 0.0


def _estimate_stage(age: Optional[int]) -> str:
    if age is None:
        return "adult"
    if age < 4:
        return "baby"
    if age < 13:
        return "child"
    if age < 18:
        return "teen"
    return "adult"


def _extract_pose_people(
    image_bgr: np.ndarray,
    model: Any,
    device_choice: str,
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    results = model.predict(
        source=image_bgr,
        conf=float(confidence_threshold),
        device=0 if device_choice == "cuda" else device_choice,
        verbose=False,
    )
    people: List[Dict[str, Any]] = []
    if not results:
        return people
    result = results[0]
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or boxes.xyxy is None or keypoints.xy is None:
        return people
    xyxy = boxes.xyxy.detach().cpu().numpy()
    scores = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else None
    xy = keypoints.xy.detach().cpu().numpy()
    conf = keypoints.conf.detach().cpu().numpy() if getattr(keypoints, "conf", None) is not None else None
    for idx in range(min(len(xyxy), len(xy))):
        bbox = tuple(float(v) for v in xyxy[idx].tolist())
        kp_xy = xy[idx]
        kp_conf = conf[idx] if conf is not None and idx < len(conf) else None
        people.append(
            {
                "bbox": bbox,
                "score": float(scores[idx]) if scores is not None and idx < len(scores) else 0.0,
                "keypoints": [[float(p[0]), float(p[1])] for p in kp_xy],
                "keypoint_conf": [float(v) for v in kp_conf] if kp_conf is not None else [1.0] * len(kp_xy),
            }
        )
    return people


def _kp(person: Dict[str, Any], idx: int, min_conf: float = 0.15) -> Optional[Tuple[float, float]]:
    kps = person.get("keypoints") or []
    confs = person.get("keypoint_conf") or []
    if idx >= len(kps):
        return None
    conf = float(confs[idx]) if idx < len(confs) else 1.0
    if conf < min_conf:
        return None
    x, y = kps[idx]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (float(x), float(y))


def _set_region(
    regions: Dict[str, Dict[str, Tuple[int, int, int, int]]],
    name: str,
    side: str,
    box: Optional[Tuple[int, int, int, int]],
) -> None:
    if box is None:
        return
    regions.setdefault(name, {})[side] = box


def _select_region_boxes(
    region_entry: Dict[str, Tuple[int, int, int, int]],
    side: str,
    paired: bool,
) -> List[Tuple[int, int, int, int]]:
    if not region_entry:
        return []
    if not paired:
        single = region_entry.get("single")
        return [single] if single is not None else []
    side_mode = str(side or "both").strip().lower()
    if side_mode == "left":
        left = region_entry.get("left")
        return [left] if left is not None else []
    if side_mode == "right":
        right = region_entry.get("right")
        return [right] if right is not None else []
    out: List[Tuple[int, int, int, int]] = []
    if region_entry.get("left") is not None:
        out.append(region_entry["left"])
    if region_entry.get("right") is not None:
        out.append(region_entry["right"])
    return out


def _mean_point(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    arr = np.asarray(points, dtype=np.float32)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def _extreme_point(points: List[Tuple[float, float]], axis: int, pick_max: bool) -> Optional[Tuple[float, float]]:
    if not points:
        return None
    return max(points, key=lambda p: p[axis]) if pick_max else min(points, key=lambda p: p[axis])


def _nearest_point(points: List[Tuple[float, float]], target: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not points or target is None:
        return None
    tx, ty = target
    return min(points, key=lambda p: (p[0] - tx) ** 2 + (p[1] - ty) ** 2)


def _derive_regions(
    person: Dict[str, Any],
    face: Optional[Dict[str, Any]],
    image_w: int,
    image_h: int,
    facial_source_policy: str = "auto",
) -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
    regions: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
    bbox = _clip_box(person["bbox"], image_w, image_h)
    if bbox is not None:
        _set_region(regions, "full_person", "single", bbox)
    body_w = max(1.0, float((bbox[2] - bbox[0]) if bbox is not None else image_w * 0.2))
    body_h = max(1.0, float((bbox[3] - bbox[1]) if bbox is not None else image_h * 0.4))

    l_eye = _kp(person, 1)
    r_eye = _kp(person, 2)
    nose = _kp(person, 0)
    l_ear = _kp(person, 3)
    r_ear = _kp(person, 4)
    l_sh = _kp(person, 5)
    r_sh = _kp(person, 6)
    l_elbow = _kp(person, 7)
    r_elbow = _kp(person, 8)
    l_wrist = _kp(person, 9)
    r_wrist = _kp(person, 10)
    l_hip = _kp(person, 11)
    r_hip = _kp(person, 12)
    l_knee = _kp(person, 13)
    r_knee = _kp(person, 14)
    l_ankle = _kp(person, 15)
    r_ankle = _kp(person, 16)

    if face is not None:
        face_box = _clip_box(face["bbox"], image_w, image_h)
        if face_box is not None:
            _set_region(regions, "face", "single", face_box)
        face_kps = face.get("kps") or []
        face_lm106 = face.get("landmark_106") or []
        if len(face_kps) >= 5:
            mouth = _box_from_points(
                [(face_kps[3][0], face_kps[3][1]), (face_kps[4][0], face_kps[4][1])],
                image_w,
                image_h,
                pad_ratio=0.22,
            )
            _set_region(regions, "mouth", "single", mouth)
        if len(face_lm106) >= 106:
            contour = [(float(p[0]), float(p[1])) for p in face_lm106[0:33]]
            brows = [(float(p[0]), float(p[1])) for p in face_lm106[33:51]]
            nose_points = [(float(p[0]), float(p[1])) for p in face_lm106[51:63]]
            eye_points = [(float(p[0]), float(p[1])) for p in face_lm106[63:87]]
            mouth_points = [(float(p[0]), float(p[1])) for p in face_lm106[87:106]]
            if contour:
                contour_box = _box_from_points(contour, image_w, image_h, pad_ratio=0.06)
                if contour_box is not None:
                    _set_region(regions, "face", "single", contour_box)

            center_x = float(np.mean([p[0] for p in face_lm106]))
            eye_left_cluster = [p for p in eye_points if p[0] <= center_x]
            eye_right_cluster = [p for p in eye_points if p[0] > center_x]
            brow_left_cluster = [p for p in brows if p[0] <= center_x]
            brow_right_cluster = [p for p in brows if p[0] > center_x]
            mouth_left_anchor = (float(face_kps[3][0]), float(face_kps[3][1])) if len(face_kps) >= 5 else None
            mouth_right_anchor = (float(face_kps[4][0]), float(face_kps[4][1])) if len(face_kps) >= 5 else None

            face_eye_left = _mean_point(eye_left_cluster)
            face_eye_right = _mean_point(eye_right_cluster)
            face_brow_left = _mean_point(brow_left_cluster)
            face_brow_right = _mean_point(brow_right_cluster)
            face_nose = _extreme_point(nose_points, axis=1, pick_max=True)
            face_mouth_left = _nearest_point(mouth_points, mouth_left_anchor) or _extreme_point(mouth_points, axis=0, pick_max=False)
            face_mouth_right = _nearest_point(mouth_points, mouth_right_anchor) or _extreme_point(mouth_points, axis=0, pick_max=True)
            face_chin = _extreme_point(contour, axis=1, pick_max=True)
            face_mouth_box = _box_from_points(mouth_points, image_w, image_h, pad_ratio=0.14)

            face_point_sources = {
                "nose": {"single": face_nose},
                "eye": {"left": face_eye_left, "right": face_eye_right},
                "eyebrow": {"left": face_brow_left, "right": face_brow_right},
                "mouth_corner": {"left": face_mouth_left, "right": face_mouth_right},
                "chin": {"single": face_chin},
            }
            face_box_sources = {
                "mouth": {"single": face_mouth_box},
            }
        else:
            face_point_sources = {}
            face_box_sources = {}
    else:
        face_point_sources = {}
        face_box_sources = {}

    if not regions.get("face"):
        face_box = _box_from_points([p for p in [nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh] if p is not None], image_w, image_h, pad_ratio=0.18)
        _set_region(regions, "face", "single", face_box)

    face_entry = regions.get("face") or {}
    face_box = face_entry.get("single")
    face_w = max(1.0, float((face_box[2] - face_box[0]) if face_box is not None else body_w * 0.28))
    face_h = max(1.0, float((face_box[3] - face_box[1]) if face_box is not None else body_h * 0.18))

    nose_half_w = max(4.0, face_w * 0.11)
    nose_half_h = max(4.0, face_h * 0.12)
    eye_half_w = max(4.0, face_w * 0.16)
    eye_half_h = max(3.0, face_h * 0.10)
    brow_half_w = max(5.0, face_w * 0.18)
    brow_half_h = max(3.0, face_h * 0.08)
    ear_half_w = max(4.0, face_w * 0.12)
    ear_half_h = max(4.0, face_h * 0.14)
    mouth_corner_half_w = max(4.0, face_w * 0.10)
    mouth_corner_half_h = max(3.0, face_h * 0.08)
    chin_half_w = max(5.0, face_w * 0.12)
    chin_half_h = max(4.0, face_h * 0.08)
    shoulder_half_w = max(6.0, body_w * 0.06)
    shoulder_half_h = max(6.0, body_h * 0.03)
    elbow_half_w = max(6.0, body_w * 0.05)
    elbow_half_h = max(6.0, body_h * 0.03)
    wrist_half_w = max(5.0, body_w * 0.045)
    wrist_half_h = max(5.0, body_h * 0.025)
    hip_half_w = max(6.0, body_w * 0.055)
    hip_half_h = max(6.0, body_h * 0.03)
    knee_half_w = max(6.0, body_w * 0.05)
    knee_half_h = max(6.0, body_h * 0.028)
    ankle_half_w = max(6.0, body_w * 0.06)
    ankle_half_h = max(4.0, body_h * 0.02)

    facial_source_policy = str(facial_source_policy or "auto").strip().lower()

    def choose_point(face_point: Optional[Tuple[float, float]], pose_point: Optional[Tuple[float, float]], prefer_pose_only: bool = False) -> Optional[Tuple[float, float]]:
        if prefer_pose_only:
            return pose_point
        if facial_source_policy == "prefer_pose":
            return pose_point or face_point
        if facial_source_policy == "prefer_face_landmarks":
            return face_point or pose_point
        return face_point or pose_point

    def choose_box(face_box_value: Optional[Tuple[int, int, int, int]], fallback_box_value: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if facial_source_policy == "prefer_pose":
            return fallback_box_value or face_box_value
        return face_box_value or fallback_box_value

    face_nose = face_point_sources.get("nose", {}).get("single")
    face_eye_left = face_point_sources.get("eye", {}).get("left")
    face_eye_right = face_point_sources.get("eye", {}).get("right")
    face_brow_left = face_point_sources.get("eyebrow", {}).get("left")
    face_brow_right = face_point_sources.get("eyebrow", {}).get("right")
    face_mouth_left = face_point_sources.get("mouth_corner", {}).get("left")
    face_mouth_right = face_point_sources.get("mouth_corner", {}).get("right")
    face_chin = face_point_sources.get("chin", {}).get("single")

    _set_region(regions, "nose", "single", _box_from_center_and_size(choose_point(face_nose, nose), nose_half_w, nose_half_h, image_w, image_h) if choose_point(face_nose, nose) is not None else None)
    _set_region(regions, "eye", "left", _box_from_center_and_size(choose_point(face_eye_left, l_eye), eye_half_w, eye_half_h, image_w, image_h) if choose_point(face_eye_left, l_eye) is not None else None)
    _set_region(regions, "eye", "right", _box_from_center_and_size(choose_point(face_eye_right, r_eye), eye_half_w, eye_half_h, image_w, image_h) if choose_point(face_eye_right, r_eye) is not None else None)
    _set_region(regions, "ear", "left", _box_from_center_and_size(l_ear, ear_half_w, ear_half_h, image_w, image_h) if l_ear is not None else None)
    _set_region(regions, "ear", "right", _box_from_center_and_size(r_ear, ear_half_w, ear_half_h, image_w, image_h) if r_ear is not None else None)
    _set_region(regions, "eyebrow", "left", _box_from_center_and_size(face_brow_left, brow_half_w, brow_half_h, image_w, image_h) if face_brow_left is not None else None)
    _set_region(regions, "eyebrow", "right", _box_from_center_and_size(face_brow_right, brow_half_w, brow_half_h, image_w, image_h) if face_brow_right is not None else None)
    _set_region(regions, "shoulder", "left", _box_from_center_and_size(l_sh, shoulder_half_w, shoulder_half_h, image_w, image_h) if l_sh is not None else None)
    _set_region(regions, "shoulder", "right", _box_from_center_and_size(r_sh, shoulder_half_w, shoulder_half_h, image_w, image_h) if r_sh is not None else None)
    _set_region(regions, "elbow", "left", _box_from_center_and_size(l_elbow, elbow_half_w, elbow_half_h, image_w, image_h) if l_elbow is not None else None)
    _set_region(regions, "elbow", "right", _box_from_center_and_size(r_elbow, elbow_half_w, elbow_half_h, image_w, image_h) if r_elbow is not None else None)
    _set_region(regions, "wrist", "left", _box_from_center_and_size(l_wrist, wrist_half_w, wrist_half_h, image_w, image_h) if l_wrist is not None else None)
    _set_region(regions, "wrist", "right", _box_from_center_and_size(r_wrist, wrist_half_w, wrist_half_h, image_w, image_h) if r_wrist is not None else None)
    _set_region(regions, "hip", "left", _box_from_center_and_size(l_hip, hip_half_w, hip_half_h, image_w, image_h) if l_hip is not None else None)
    _set_region(regions, "hip", "right", _box_from_center_and_size(r_hip, hip_half_w, hip_half_h, image_w, image_h) if r_hip is not None else None)
    _set_region(regions, "knee", "left", _box_from_center_and_size(l_knee, knee_half_w, knee_half_h, image_w, image_h) if l_knee is not None else None)
    _set_region(regions, "knee", "right", _box_from_center_and_size(r_knee, knee_half_w, knee_half_h, image_w, image_h) if r_knee is not None else None)
    _set_region(regions, "ankle", "left", _box_from_center_and_size(l_ankle, ankle_half_w, ankle_half_h, image_w, image_h) if l_ankle is not None else None)
    _set_region(regions, "ankle", "right", _box_from_center_and_size(r_ankle, ankle_half_w, ankle_half_h, image_w, image_h) if r_ankle is not None else None)
    _set_region(regions, "mouth_corner", "left", _box_from_center_and_size(face_mouth_left, mouth_corner_half_w, mouth_corner_half_h, image_w, image_h) if face_mouth_left is not None else None)
    _set_region(regions, "mouth_corner", "right", _box_from_center_and_size(face_mouth_right, mouth_corner_half_w, mouth_corner_half_h, image_w, image_h) if face_mouth_right is not None else None)
    _set_region(
        regions,
        "mouth",
        "single",
        choose_box(face_box_sources.get("mouth", {}).get("single"), regions.get("mouth", {}).get("single")),
    )
    _set_region(regions, "chin", "single", _box_from_center_and_size(face_chin, chin_half_w, chin_half_h, image_w, image_h) if face_chin is not None else None)
    return regions


def _subject_score(
    person: Dict[str, Any],
    face: Optional[Dict[str, Any]],
    preferred_gender: str,
    age_min: int,
    age_max: int,
    image_w: int,
    image_h: int,
) -> float:
    bbox = _clip_box(person["bbox"], image_w, image_h)
    if bbox is None:
        return -1e9
    box_area = _box_area(bbox) / max(1.0, image_w * image_h)
    center_x, center_y = _box_center(bbox)
    dx = (center_x / max(1.0, image_w)) - 0.5
    dy = (center_y / max(1.0, image_h)) - 0.5
    center_bias = max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5) * 1.65)
    age = face.get("age") if face else None
    gender = face.get("gender") if face else "unknown"
    age_match = _age_score(age, age_min, age_max)
    gender_match = _gender_score(preferred_gender, gender)
    pose_quality = float(person.get("score", 0.0))
    face_visible = 1.0 if face else 0.0
    return (
        4.0 * age_match
        + 3.5 * gender_match
        + 1.25 * box_area
        + 0.9 * face_visible
        + 0.8 * pose_quality
        + 0.6 * center_bias
    )


def _fit_crop_from_center(
    cx: float,
    cy: float,
    crop_w: int,
    crop_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    crop_w = max(1, min(int(crop_w), image_w))
    crop_h = max(1, min(int(crop_h), image_h))
    x0 = int(round(cx - crop_w / 2.0))
    y0 = int(round(cy - crop_h / 2.0))
    x0 = max(0, min(image_w - crop_w, x0))
    y0 = max(0, min(image_h - crop_h, y0))
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _normalize_crop_to_aspect(
    crop: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    aspect = float(target_w) / max(1.0, float(target_h))
    current = crop_w / max(1.0, float(crop_h))
    if abs(current - aspect) <= 1e-4:
        return crop

    cx, cy = _box_center(crop)
    if current < aspect:
        new_w = min(image_w, max(crop_w, int(np.ceil(crop_h * aspect))))
        new_h = min(image_h, max(1, int(round(new_w / aspect))))
        if new_h < crop_h:
            new_h = crop_h
            new_w = min(image_w, max(new_w, int(round(new_h * aspect))))
    else:
        new_h = min(image_h, max(crop_h, int(np.ceil(crop_w / aspect))))
        new_w = min(image_w, max(1, int(round(new_h * aspect))))
        if new_w < crop_w:
            new_w = crop_w
            new_h = min(image_h, max(new_h, int(round(new_w / aspect))))

    new_w = max(1, min(image_w, int(new_w)))
    new_h = max(1, min(image_h, int(new_h)))
    return _fit_crop_from_center(cx, cy, new_w, new_h, image_w, image_h)


def _inscribe_crop_to_aspect(
    crop: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    aspect = float(target_w) / max(1.0, float(target_h))
    current = crop_w / max(1.0, float(crop_h))
    cx, cy = _box_center(crop)
    if current > aspect:
        new_h = crop_h
        new_w = max(1, int(round(new_h * aspect)))
    else:
        new_w = crop_w
        new_h = max(1, int(round(new_w / aspect)))
    new_w = max(1, min(new_w, crop_w, image_w))
    new_h = max(1, min(new_h, crop_h, image_h))
    return _fit_crop_from_center(cx, cy, new_w, new_h, image_w, image_h)


def _expand_crop_to_aspect(
    crop: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    crop_w = max(1, int(x1 - x0))
    crop_h = max(1, int(y1 - y0))
    aspect = float(target_w) / max(1.0, float(target_h))
    current = crop_w / max(1.0, float(crop_h))
    cx, cy = _box_center(crop)
    if current < aspect:
        new_h = crop_h
        new_w = max(1, int(np.ceil(new_h * aspect)))
    else:
        new_w = crop_w
        new_h = max(1, int(np.ceil(new_w / aspect)))
    new_x0 = int(round(cx - new_w / 2.0))
    new_y0 = int(round(cy - new_h / 2.0))
    return (new_x0, new_y0, new_x0 + new_w, new_y0 + new_h)


def _parse_hex_color(color_value: str) -> Tuple[int, int, int]:
    raw = str(color_value or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return (0, 0, 0)
    try:
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
    except Exception:
        return (0, 0, 0)


def _render_crop_region(
    image_uint8: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    fill_mode: str,
    fill_color: str,
) -> np.ndarray:
    image_h, image_w = image_uint8.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)

    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x1 - image_w)
    bottom = max(0, y1 - image_h)

    if left == 0 and top == 0 and right == 0 and bottom == 0:
        return image_uint8[y0:y1, x0:x1].copy()

    if fill_mode == "fill_color" or cv2 is None:
        canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        canvas[:, :] = np.array(_parse_hex_color(fill_color), dtype=np.uint8)
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(image_w, x1)
        src_y1 = min(image_h, y1)
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        canvas[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = image_uint8[src_y0:src_y1, src_x0:src_x1]
        return canvas

    padded = cv2.copyMakeBorder(
        image_uint8,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REPLICATE,
    )
    blur_kernel = max(9, int(round(min(crop_w, crop_h) * 0.08)))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(padded, (blur_kernel, blur_kernel), 0)
    blurred[top:top + image_h, left:left + image_w] = image_uint8
    start_x = x0 + left
    start_y = y0 + top
    return blurred[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()


def _resize_crop_exact(
    crop_image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    crop_h, crop_w = crop_image.shape[:2]
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("Invalid crop image size.")
    if cv2 is not None:
        return cv2.resize(crop_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(crop_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(tensor, size=(target_h, target_w), mode="bicubic", align_corners=False)
    return (resized.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def _min_crop_size_for_box(box: Tuple[int, int, int, int], aspect: float) -> Tuple[int, int]:
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    if (w / h) > aspect:
        return w, int(np.ceil(w / aspect))
    return int(np.ceil(h * aspect)), h


def _union_boxes(boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not boxes:
        return None
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _choose_crop(
    regions: List[Dict[str, Any]],
    fallback_box: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    aspect = float(target_w) / max(1.0, float(target_h))
    include_boxes = [r["box"] for r in regions if r["mode"] == "include" and float(r["weight"]) > 0.0]
    base_box = _union_boxes(include_boxes) or fallback_box
    preferred_box = _union_boxes(include_boxes) or fallback_box
    min_w, min_h = _min_crop_size_for_box(base_box, aspect)
    pref_w, pref_h = _min_crop_size_for_box(preferred_box, aspect)
    weighted_cx = 0.0
    weighted_cy = 0.0
    total_w = 0.0
    candidate_centers: List[Tuple[float, float]] = [_box_center(fallback_box)]
    for region in regions:
        if region["mode"] != "include":
            continue
        rcx, rcy = _box_center(region["box"])
        candidate_centers.append((rcx, rcy))
        weighted_cx += rcx * region["weight"]
        weighted_cy += rcy * region["weight"]
        total_w += region["weight"]
    if total_w > 1e-6:
        candidate_centers.append((weighted_cx / total_w, weighted_cy / total_w))

    scale_candidates = [1.0, 1.05, 1.1, 1.18, 1.28, 1.4, 1.6, 1.85]
    if pref_w > min_w or pref_h > min_h:
        scale_candidates.extend([
            max(pref_w / max(1.0, min_w), pref_h / max(1.0, min_h)),
        ])

    best_crop = _fit_crop_from_center(*candidate_centers[0], min_w, min_h, image_w, image_h)
    best_score = -1e18
    for cx, cy in candidate_centers:
        for scale in scale_candidates:
            crop_w = min(image_w, int(np.ceil(min_w * scale)))
            crop_h = min(image_h, int(np.ceil(min_h * scale)))
            crop = _fit_crop_from_center(cx, cy, crop_w, crop_h, image_w, image_h)
            score = 0.0
            for region in regions:
                cov = _coverage(region["box"], crop)
                if region["mode"] == "exclude":
                    score -= region["weight"] * cov
                else:
                    score += region["weight"] * cov
            score -= 2.5 * (_box_area(crop) / max(1.0, image_w * image_h))
            if score > best_score:
                best_score = score
                best_crop = crop
    return best_crop


def _draw_box(image: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 2) -> None:
    if cv2 is None:
        return
    x0, y0, x1, y1 = box
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)


def _draw_text(image: np.ndarray, text: str, pos: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x, max(0, y - th - 6)), (x + tw + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_point(image: np.ndarray, point: Tuple[float, float], color: Tuple[int, int, int]) -> None:
    if cv2 is None:
        return
    cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 3, color, -1, cv2.LINE_AA)


def _render_overlay(
    image_uint8: np.ndarray,
    people: List[Dict[str, Any]],
    selected_index: int,
    regions: Dict[str, List[Tuple[int, int, int, int]]],
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray:
    overlay = image_uint8.copy()
    person_colors = {
        True: (64, 255, 96),
        False: (64, 196, 255),
    }
    region_colors = {
        "nose": (255, 0, 255),
        "eye": (255, 180, 0),
        "eyebrow": (255, 220, 120),
        "ear": (255, 210, 80),
        "shoulder": (0, 200, 255),
        "elbow": (0, 150, 255),
        "wrist": (0, 255, 220),
        "hip": (255, 128, 0),
        "knee": (255, 255, 0),
        "ankle": (180, 255, 0),
        "face": (64, 196, 255),
        "mouth": (255, 0, 180),
        "mouth_corner": (255, 80, 200),
        "chin": (180, 80, 255),
        "full_person": (64, 255, 96),
    }
    for idx, person in enumerate(people):
        bbox = person.get("bbox")
        if bbox is None:
            continue
        selected = idx == selected_index
        color = person_colors[selected]
        _draw_box(overlay, bbox, color, thickness=3 if selected else 2)
        label = f"person {idx + 1}: {person.get('gender','unknown')}, {person.get('age') if person.get('age') is not None else person.get('life_stage','adult')}, s={person.get('subject_score',0.0):.2f}"
        _draw_text(overlay, label, (bbox[0], max(16, bbox[1])), color)
    for name, boxes in regions.items():
        color = region_colors.get(name, (255, 255, 255))
        for box in boxes:
            _draw_box(overlay, box, color, thickness=1)
            _draw_point(overlay, _box_center(box), color)
    clipped_crop = _clip_box(crop_box, image_uint8.shape[1], image_uint8.shape[0])
    if clipped_crop is not None:
        _draw_box(overlay, clipped_crop, (255, 0, 0), thickness=3)
        _draw_text(overlay, "crop", (clipped_crop[0], max(18, clipped_crop[1])), (255, 0, 0))
    return overlay


class CompositionCrop:
    CATEGORY = "ESS/Image"
    FUNCTION = "crop"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "debug_overlay", "crop_box_json", "debug_json")

    @classmethod
    def INPUT_TYPES(cls):
        models = _list_pose_model_options()
        pose_selector = _selector_from_list(models, default="yolo11n-pose.pt")
        optional = {
            "pose_model_name": pose_selector,
            "pose_model_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional absolute path to pose .pt"}),
            "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
            "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
            "framing_mode": (("crop", "expand"), {"default": "crop"}),
            "expand_fill_mode": (("border_fill", "fill_color"), {"default": "border_fill"}),
            "expand_fill_color": ("STRING", {"default": "#000000", "multiline": False}),
            "facial_source_policy": (("auto", "prefer_face_landmarks", "prefer_pose"), {"default": "auto"}),
            "preferred_gender": (("any", "female", "male"), {"default": "any"}),
            "target_age_min": ("INT", {"default": 18, "min": 0, "max": 120, "step": 1}),
            "target_age_max": ("INT", {"default": 35, "min": 0, "max": 120, "step": 1}),
        }
        for spec in _PART_SPECS:
            key = spec["key"]
            optional[f"enabled_{key}"] = ("BOOLEAN", {"default": bool(spec["default_enabled"])})
            optional[f"mode_{key}"] = (_MODE_VALUES, {"default": spec["default_mode"]})
            if spec["paired"]:
                optional[f"side_{key}"] = (_PAIR_SIDE_VALUES, {"default": "both"})
            optional[f"weight_{key}"] = ("FLOAT", {"default": spec["default_weight"], "min": 0.0, "max": 100.0, "step": 1.0})
            optional[f"margin_{key}"] = ("FLOAT", {"default": spec["default_margin"], "min": 0.0, "max": 10.0, "step": 0.1})
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 768, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": optional,
        }

    def crop(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        pose_model_name: str = "yolo11n-pose.pt",
        pose_model_path: str = "",
        device: str = "auto",
        confidence_threshold: float = 0.25,
        framing_mode: str = "crop",
        expand_fill_mode: str = "border_fill",
        expand_fill_color: str = "#000000",
        facial_source_policy: str = "auto",
        preferred_gender: str = "any",
        target_age_min: int = 18,
        target_age_max: int = 35,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        image_uint8 = _tensor_to_uint8_image(image)
        image_bgr = image_uint8[..., ::-1].copy() if cv2 is not None else image_uint8.copy()
        image_h, image_w = image_uint8.shape[:2]
        device_choice = _resolve_device(device)
        pose_model = _load_model(pose_model_name, pose_model_path or None, device_choice)
        people = _extract_pose_people(image_bgr, pose_model, device_choice, confidence_threshold)
        if not people:
            raise RuntimeError("No people detected by the pose model.")

        faces = _extract_faces(image_bgr)
        for person in people:
            bbox = _clip_box(person["bbox"], image_w, image_h)
            if bbox is None:
                continue
            person["bbox"] = bbox
            face = _match_face(bbox, faces)
            person["face"] = face
            person["gender"] = face.get("gender", "unknown") if face else "unknown"
            person["age"] = face.get("age") if face else None
            person["life_stage"] = _estimate_stage(person["age"])
            person["subject_score"] = _subject_score(person, face, preferred_gender, target_age_min, target_age_max, image_w, image_h)

        people = sorted(people, key=lambda p: float(p.get("subject_score", 0.0)), reverse=True)
        selected = people[0]
        selected_idx = 0
        regions_all = _derive_regions(selected, selected.get("face"), image_w, image_h, facial_source_policy=facial_source_policy)
        part_settings: Dict[str, Dict[str, Any]] = {}
        for spec in _PART_SPECS:
            key = spec["key"]
            enabled = bool(kwargs.get(f"enabled_{key}", spec["default_enabled"]))
            mode = str(kwargs.get(f"mode_{key}", spec["default_mode"]) or spec["default_mode"]).strip().lower()
            if mode not in _MODE_VALUES:
                mode = spec["default_mode"]
            side = "single"
            if spec["paired"]:
                side = str(kwargs.get(f"side_{key}", "both") or "both").strip().lower()
                if side not in _PAIR_SIDE_VALUES:
                    side = "both"
            weight = max(0.0, min(100.0, float(kwargs.get(f"weight_{key}", spec["default_weight"]))))
            margin = max(0.0, min(10.0, float(kwargs.get(f"margin_{key}", spec["default_margin"]))))
            part_settings[key] = {"enabled": enabled, "mode": mode, "side": side, "weight": weight, "margin": margin}

        regions_for_solver: List[Dict[str, Any]] = []
        debug_regions: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for spec in _PART_SPECS:
            name = spec["key"]
            region_entry = regions_all.get(name) or {}
            boxes = list(region_entry.values())
            if not boxes:
                continue
            debug_regions[name] = boxes
            settings = part_settings[name]
            if (not settings["enabled"]) or settings["weight"] <= 0.0:
                continue
            selected_boxes = _select_region_boxes(region_entry, settings["side"], bool(spec["paired"]))
            if not selected_boxes:
                continue
            per_box_weight = settings["weight"] / max(1, len(selected_boxes))
            for box in selected_boxes:
                regions_for_solver.append(
                    {
                        "name": name,
                        "weight": per_box_weight,
                        "mode": settings["mode"],
                        "box": _expand_box(box, settings["margin"], image_w, image_h),
                    }
                )

        fallback_box = (regions_all.get("full_person") or {}).get("single") or selected["bbox"]
        crop_box = _choose_crop(regions_for_solver, fallback_box, width, height, image_w, image_h)
        framing_mode = str(framing_mode or "crop").strip().lower()
        if framing_mode == "expand":
            crop_box = _expand_crop_to_aspect(crop_box, width, height)
        else:
            crop_box = _inscribe_crop_to_aspect(crop_box, width, height, image_w, image_h)

        crop_image = _render_crop_region(
            image_uint8,
            crop_box,
            str(expand_fill_mode or "border_fill").strip().lower(),
            expand_fill_color,
        )
        resized_uint8 = _resize_crop_exact(crop_image, int(width), int(height))
        output = _uint8_image_to_tensor(resized_uint8, image.device)

        overlay = _render_overlay(image_uint8, people, selected_idx, debug_regions, crop_box)
        overlay_tensor = _uint8_image_to_tensor(overlay, image.device)

        x0, y0, x1, y1 = [int(v) for v in crop_box]
        crop_box_json = json.dumps({"x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1), "width": int(x1 - x0), "height": int(y1 - y0)})
        debug_people = []
        for idx, person in enumerate(people):
            bbox = person.get("bbox")
            debug_people.append(
                {
                    "index": idx,
                    "selected": idx == selected_idx,
                    "bbox": [int(v) for v in bbox] if bbox else None,
                    "gender": person.get("gender", "unknown"),
                    "age": person.get("age"),
                    "life_stage": person.get("life_stage", "adult"),
                    "subject_score": float(person.get("subject_score", 0.0)),
                    "pose_score": float(person.get("score", 0.0)),
                    "regions": {
                        k: {side: list(box) for side, box in boxes.items()}
                        for k, boxes in (_derive_regions(person, person.get("face"), image_w, image_h, facial_source_policy=facial_source_policy)).items()
                    },
                }
            )
        debug_json = json.dumps(
            {
                "selected_index": selected_idx,
                "preferred_gender": preferred_gender,
                "framing_mode": framing_mode,
                "expand_fill_mode": expand_fill_mode,
                "expand_fill_color": expand_fill_color,
                "facial_source_policy": facial_source_policy,
                "target_age_min": int(target_age_min),
                "target_age_max": int(target_age_max),
                "crop_box": json.loads(crop_box_json),
                "people": debug_people,
            },
            ensure_ascii=True,
        )
        return (output, overlay_tensor, crop_box_json, debug_json)
