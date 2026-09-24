from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

_PREVIEW_COUNTER = itertools.count()
_PREVIEW_COUNTER_LOCK = threading.Lock()

import numpy as np
import torch

from ._crop_common import (
    _box_area,
    _box_center,
    _box_from_center_and_size,
    _box_from_points,
    _clip_box,
    _coverage,
    _crop_fill_mask,
    _expand_box,
    _expand_crop_to_aspect,
    _extract_pose_people,
    _fit_crop_from_center,
    _get_face_analyzer,
    _inscribe_crop_to_aspect,
    _intersection_area,
    _min_crop_size_for_box,
    _parse_hex_color,
    _render_crop_region,
    _resize_crop_exact,
    _tensor_to_uint8_image,
    _uint8_image_to_tensor,
    _union_boxes,
    _load_model,
    _resolve_device,
)

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None


_DEFAULT_POSE_MODEL = "yolo11s-pose.pt"


_ZONE_GROUPS: Tuple[Dict[str, Any], ...] = (
    {
        "key": "composite",
        "label": "Composites",
        "zones": (
            {"key": "full",        "label": "Full",        "paired": False, "default_enabled": True,  "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "head_only",   "label": "Head only",   "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "bust",        "label": "Bust",        "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "upper_body",  "label": "Upper body",  "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "half_body",   "label": "Half body",   "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
        ),
    },
    {
        "key": "head",
        "label": "Head",
        "zones": (
            {"key": "face",         "label": "Face",         "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "eyebrow",      "label": "Eyebrow",      "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "eye",          "label": "Eye",          "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "ear",          "label": "Ear",          "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "nose",         "label": "Nose",         "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "cheek",        "label": "Cheek",        "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "mouth",        "label": "Mouth",        "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "mouth_corner", "label": "Mouth corner", "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "chin",         "label": "Chin",         "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "jaw",          "label": "Jaw",          "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
        ),
    },
    {
        "key": "torso",
        "label": "Torso",
        "zones": (
            {"key": "neck",     "label": "Neck",     "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "shoulder", "label": "Shoulder", "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "chest",    "label": "Chest",    "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "belly",    "label": "Belly",    "paired": False, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "hip",      "label": "Hip",      "paired": True,  "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
        ),
    },
    {
        "key": "arms",
        "label": "Arms",
        "zones": (
            {"key": "upper_arm", "label": "Upper arm", "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "elbow",     "label": "Elbow",     "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "forearm",   "label": "Forearm",   "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "wrist",     "label": "Wrist",     "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
        ),
    },
    {
        "key": "legs",
        "label": "Legs",
        "zones": (
            {"key": "thigh", "label": "Thigh", "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "knee",  "label": "Knee",  "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "shin",  "label": "Shin",  "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
            {"key": "ankle", "label": "Ankle", "paired": True, "default_enabled": False, "default_weight": 100.0, "default_margin": 0.0, "default_mode": "include"},
        ),
    },
)

_ZONE_SPECS: Tuple[Dict[str, Any], ...] = tuple(
    {**spec, "group": group["key"]}
    for group in _ZONE_GROUPS
    for spec in group["zones"]
)
_ZONE_SPEC_BY_KEY: Dict[str, Dict[str, Any]] = {spec["key"]: spec for spec in _ZONE_SPECS}
_MODE_VALUES = ("include", "exclude")
_SIDE_VALUES = ("both", "left", "right")


_GROUP_COLORS: Dict[str, Tuple[int, int, int]] = {
    "composite": (96, 220, 120),
    "head":      (96, 196, 255),
    "torso":     (255, 196, 64),
    "arms":      (255, 128, 200),
    "legs":      (180, 128, 255),
}


_DETECTION_CACHE_LOCK = threading.Lock()
_DETECTION_CACHE: Dict[str, Dict[str, Any]] = {}
_DETECTION_CACHE_MAX = 16
_NODE_IMAGE_CACHE: Dict[str, np.ndarray] = {}
_NODE_IMAGE_CACHE_LOCK = threading.Lock()
_NODE_IMAGE_CACHE_MAX = 32


def _image_digest(image_uint8: np.ndarray) -> str:
    h, w = image_uint8.shape[:2]
    return hashlib.sha1(image_uint8.tobytes()).hexdigest() + f"_{h}x{w}"


def _detection_cache_key(image_digest: str, *, confidence_threshold: float, device_choice: str) -> str:
    payload = json.dumps(
        {"img": image_digest, "conf": round(float(confidence_threshold), 4), "dev": device_choice},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _detection_cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _DETECTION_CACHE_LOCK:
        entry = _DETECTION_CACHE.get(key)
        if entry is None:
            return None
        entry["touched"] = time.time()
        return entry


def _detection_cache_put(key: str, value: Dict[str, Any]) -> None:
    with _DETECTION_CACHE_LOCK:
        value["touched"] = time.time()
        _DETECTION_CACHE[key] = value
        if len(_DETECTION_CACHE) > _DETECTION_CACHE_MAX:
            oldest = sorted(_DETECTION_CACHE.items(), key=lambda kv: kv[1].get("touched", 0.0))
            for stale_key, _ in oldest[: max(1, len(_DETECTION_CACHE) - _DETECTION_CACHE_MAX)]:
                _DETECTION_CACHE.pop(stale_key, None)


def _store_node_image(node_id: Optional[str], image_uint8: np.ndarray) -> None:
    if not node_id:
        return
    with _NODE_IMAGE_CACHE_LOCK:
        _NODE_IMAGE_CACHE[str(node_id)] = image_uint8
        if len(_NODE_IMAGE_CACHE) > _NODE_IMAGE_CACHE_MAX:
            keys = list(_NODE_IMAGE_CACHE.keys())
            for k in keys[: len(_NODE_IMAGE_CACHE) - _NODE_IMAGE_CACHE_MAX]:
                _NODE_IMAGE_CACHE.pop(k, None)


def _get_node_image(node_id: Optional[str]) -> Optional[np.ndarray]:
    if not node_id:
        return None
    with _NODE_IMAGE_CACHE_LOCK:
        return _NODE_IMAGE_CACHE.get(str(node_id))


def _extract_faces_ibug(image_bgr: np.ndarray) -> List[Dict[str, Any]]:
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return []
    try:
        faces = analyzer.get(image_bgr)
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for face in faces or []:
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        det_score = float(getattr(face, "det_score", 0.0) or 0.0)
        age = getattr(face, "age", None)
        age_val = int(age) if age is not None and str(age).strip() else None
        gender_raw = getattr(face, "gender", None)
        gender = "unknown"
        if gender_raw == 0:
            gender = "female"
        elif gender_raw == 1:
            gender = "male"
        kps_raw = getattr(face, "kps", None)
        kps_points: List[List[float]] = []
        if kps_raw is not None:
            try:
                kps_points = [[float(p[0]), float(p[1])] for p in kps_raw]
            except Exception:
                kps_points = []
        lm68_raw = getattr(face, "landmark_3d_68", None)
        lm68_points: List[List[float]] = []
        if lm68_raw is not None:
            try:
                lm68_points = [[float(p[0]), float(p[1])] for p in lm68_raw]
            except Exception:
                lm68_points = []
        head_pose: Optional[Dict[str, float]] = None
        pose_raw = getattr(face, "pose", None)
        if pose_raw is not None:
            try:
                vals = [float(v) for v in pose_raw]
                if len(vals) >= 3:
                    head_pose = {"pitch": vals[0], "yaw": vals[1], "roll": vals[2]}
            except Exception:
                head_pose = None
        out.append(
            {
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) * 0.5, (y1 + y2) * 0.5),
                "kps": kps_points,
                "landmark_68": lm68_points,
                "head_pose": head_pose,
                "age": age_val,
                "gender": gender,
                "det_score": det_score,
            }
        )
    return out


def _match_face_to_person(person_bbox: Tuple[int, int, int, int], faces: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not faces:
        return None
    x0, y0, x1, y1 = [float(v) for v in person_bbox]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    best = None
    best_rank = float("inf")
    for face in faces:
        fcx, fcy = face["center"]
        inside = (x0 <= fcx <= x1) and (y0 <= fcy <= y1)
        rank = ((fcx - cx) ** 2 + (fcy - cy) ** 2) ** 0.5
        if not inside:
            rank += 1e6
        if rank < best_rank:
            best_rank = rank
            best = face
    return best


def _kp_filtered(person: Dict[str, Any], idx: int, min_conf: float) -> Optional[Tuple[float, float, float]]:
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
    return (float(x), float(y), float(conf))


def _pair_by_image_x(
    pt_a: Optional[Tuple[float, float, float]],
    pt_b: Optional[Tuple[float, float, float]],
) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    if pt_a is None and pt_b is None:
        return None, None
    if pt_a is None:
        return None, pt_b
    if pt_b is None:
        return pt_a, None
    if pt_a[0] <= pt_b[0]:
        return pt_a, pt_b
    return pt_b, pt_a


def _cluster_by_image_x(
    cluster_a: List[Tuple[float, float]],
    cluster_b: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    ma = float(np.mean([p[0] for p in cluster_a])) if cluster_a else None
    mb = float(np.mean([p[0] for p in cluster_b])) if cluster_b else None
    if ma is None and mb is None:
        return [], []
    if ma is None:
        return [], cluster_b
    if mb is None:
        return cluster_a, []
    if ma <= mb:
        return cluster_a, cluster_b
    return cluster_b, cluster_a


def _midpoint(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _segment_box(
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    half_thickness: float,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    x0 = min(a[0], b[0]) - half_thickness
    y0 = min(a[1], b[1]) - half_thickness
    x1 = max(a[0], b[0]) + half_thickness
    y1 = max(a[1], b[1]) + half_thickness
    return _clip_box((x0, y0, x1, y1), image_w, image_h)


def _box_around_cluster(
    cluster: List[Tuple[float, float]],
    image_w: int,
    image_h: int,
    pad_ratio: float = 0.15,
) -> Optional[Tuple[int, int, int, int]]:
    return _box_from_points(cluster, image_w, image_h, pad_ratio=pad_ratio)


def _zone(
    name: str,
    side: str,
    group: str,
    source: str,
    box: Optional[Tuple[int, int, int, int]],
    confidence: float,
) -> Optional[Dict[str, Any]]:
    if box is None:
        return None
    return {
        "name": name,
        "side": side,
        "group": group,
        "source": source,
        "box": (int(box[0]), int(box[1]), int(box[2]), int(box[3])),
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


def _derive_anatomy_zones(
    person: Dict[str, Any],
    face: Optional[Dict[str, Any]],
    image_w: int,
    image_h: int,
    keypoint_min_conf: float = 0.15,
) -> List[Dict[str, Any]]:
    zones: List[Dict[str, Any]] = []

    person_box = _clip_box(person["bbox"], image_w, image_h)
    if person_box is not None:
        zones.append(_zone("full", "single", "composite", "pose", person_box, float(person.get("score", 0.0)) or 0.5))
        body_w = max(1.0, float(person_box[2] - person_box[0]))
        body_h = max(1.0, float(person_box[3] - person_box[1]))
    else:
        body_w = max(1.0, image_w * 0.2)
        body_h = max(1.0, image_h * 0.4)

    kp_nose = _kp_filtered(person, 0, keypoint_min_conf)
    kp_eye_a = _kp_filtered(person, 1, keypoint_min_conf)
    kp_eye_b = _kp_filtered(person, 2, keypoint_min_conf)
    kp_ear_a = _kp_filtered(person, 3, keypoint_min_conf)
    kp_ear_b = _kp_filtered(person, 4, keypoint_min_conf)
    kp_sh_a = _kp_filtered(person, 5, keypoint_min_conf)
    kp_sh_b = _kp_filtered(person, 6, keypoint_min_conf)
    kp_elbow_a = _kp_filtered(person, 7, keypoint_min_conf)
    kp_elbow_b = _kp_filtered(person, 8, keypoint_min_conf)
    kp_wrist_a = _kp_filtered(person, 9, keypoint_min_conf)
    kp_wrist_b = _kp_filtered(person, 10, keypoint_min_conf)
    kp_hip_a = _kp_filtered(person, 11, keypoint_min_conf)
    kp_hip_b = _kp_filtered(person, 12, keypoint_min_conf)
    kp_knee_a = _kp_filtered(person, 13, keypoint_min_conf)
    kp_knee_b = _kp_filtered(person, 14, keypoint_min_conf)
    kp_ankle_a = _kp_filtered(person, 15, keypoint_min_conf)
    kp_ankle_b = _kp_filtered(person, 16, keypoint_min_conf)

    img_eye_l, img_eye_r = _pair_by_image_x(kp_eye_a, kp_eye_b)
    img_ear_l, img_ear_r = _pair_by_image_x(kp_ear_a, kp_ear_b)
    img_sh_l, img_sh_r = _pair_by_image_x(kp_sh_a, kp_sh_b)
    img_elbow_l, img_elbow_r = _pair_by_image_x(kp_elbow_a, kp_elbow_b)
    img_wrist_l, img_wrist_r = _pair_by_image_x(kp_wrist_a, kp_wrist_b)
    img_hip_l, img_hip_r = _pair_by_image_x(kp_hip_a, kp_hip_b)
    img_knee_l, img_knee_r = _pair_by_image_x(kp_knee_a, kp_knee_b)
    img_ankle_l, img_ankle_r = _pair_by_image_x(kp_ankle_a, kp_ankle_b)

    face_box: Optional[Tuple[int, int, int, int]] = None
    face_w = body_w * 0.28
    face_h = body_h * 0.18
    face_conf = 0.0
    if face is not None:
        face_box = _clip_box(face["bbox"], image_w, image_h)
        face_conf = float(face.get("det_score", 0.0) or 0.0)
        if face_box is not None:
            face_w = max(1.0, float(face_box[2] - face_box[0]))
            face_h = max(1.0, float(face_box[3] - face_box[1]))
            zones.append(_zone("face", "single", "head", "insightface", face_box, face_conf or 0.6))

    lm68 = face.get("landmark_68") if face else None
    if lm68 and len(lm68) >= 68:
        lm = [(float(p[0]), float(p[1])) for p in lm68]
        contour = lm[0:17]
        brow_a = lm[17:22]
        brow_b = lm[22:27]
        nose_bridge = lm[27:31]
        nose_base = lm[31:36]
        eye_a = lm[36:42]
        eye_b = lm[42:48]
        mouth_outer = lm[48:60]

        if not zones or zones[-1].get("name") != "face" or face_box is None:
            contour_box = _box_around_cluster(contour, image_w, image_h, pad_ratio=0.05)
            if contour_box is not None:
                face_box = contour_box
                face_w = max(1.0, float(face_box[2] - face_box[0]))
                face_h = max(1.0, float(face_box[3] - face_box[1]))
                zones.append(_zone("face", "single", "head", "ibug68", face_box, face_conf or 0.7))

        brow_left_img, brow_right_img = _cluster_by_image_x(brow_a, brow_b)
        eye_left_img, eye_right_img = _cluster_by_image_x(eye_a, eye_b)
        zones.append(_zone("eyebrow", "left",  "head", "ibug68", _box_around_cluster(brow_left_img,  image_w, image_h, 0.20), face_conf or 0.7))
        zones.append(_zone("eyebrow", "right", "head", "ibug68", _box_around_cluster(brow_right_img, image_w, image_h, 0.20), face_conf or 0.7))
        zones.append(_zone("eye", "left",  "head", "ibug68", _box_around_cluster(eye_left_img,  image_w, image_h, 0.20), face_conf or 0.8))
        zones.append(_zone("eye", "right", "head", "ibug68", _box_around_cluster(eye_right_img, image_w, image_h, 0.20), face_conf or 0.8))

        nose_pts = nose_bridge + nose_base
        zones.append(_zone("nose", "single", "head", "ibug68", _box_around_cluster(nose_pts, image_w, image_h, 0.10), face_conf or 0.75))
        zones.append(_zone("mouth", "single", "head", "ibug68", _box_around_cluster(mouth_outer, image_w, image_h, 0.10), face_conf or 0.75))

        mouth_corner_a = mouth_outer[0] if mouth_outer else None
        mouth_corner_b = mouth_outer[6] if len(mouth_outer) >= 7 else None
        if mouth_corner_a is not None and mouth_corner_b is not None:
            ma = (mouth_corner_a[0], mouth_corner_a[1], 1.0)
            mb = (mouth_corner_b[0], mouth_corner_b[1], 1.0)
            mc_l, mc_r = _pair_by_image_x(ma, mb)
            half_w = max(4.0, face_w * 0.10)
            half_h = max(3.0, face_h * 0.08)
            if mc_l is not None:
                zones.append(_zone("mouth_corner", "left", "head", "ibug68", _box_from_center_and_size((mc_l[0], mc_l[1]), half_w, half_h, image_w, image_h), face_conf or 0.7))
            if mc_r is not None:
                zones.append(_zone("mouth_corner", "right", "head", "ibug68", _box_from_center_and_size((mc_r[0], mc_r[1]), half_w, half_h, image_w, image_h), face_conf or 0.7))

        chin_pt = lm[8] if len(lm) > 8 else None
        if chin_pt is not None:
            half_w = max(5.0, face_w * 0.14)
            half_h = max(4.0, face_h * 0.10)
            zones.append(_zone("chin", "single", "head", "ibug68", _box_from_center_and_size(chin_pt, half_w, half_h, image_w, image_h), face_conf or 0.7))

        cheek_left_idxs = [1, 2, 3]
        cheek_right_idxs = [13, 14, 15]
        cheek_a_pts = [lm[i] for i in cheek_left_idxs if i < len(lm)]
        cheek_b_pts = [lm[i] for i in cheek_right_idxs if i < len(lm)]
        cheek_l, cheek_r = _cluster_by_image_x(cheek_a_pts, cheek_b_pts)
        zones.append(_zone("cheek", "left",  "head", "ibug68", _box_around_cluster(cheek_l, image_w, image_h, 0.10), face_conf or 0.65))
        zones.append(_zone("cheek", "right", "head", "ibug68", _box_around_cluster(cheek_r, image_w, image_h, 0.10), face_conf or 0.65))

        jaw_left_idxs = [4, 5, 6, 7]
        jaw_right_idxs = [9, 10, 11, 12]
        jaw_a_pts = [lm[i] for i in jaw_left_idxs if i < len(lm)]
        jaw_b_pts = [lm[i] for i in jaw_right_idxs if i < len(lm)]
        jaw_l, jaw_r = _cluster_by_image_x(jaw_a_pts, jaw_b_pts)
        zones.append(_zone("jaw", "left",  "head", "ibug68", _box_around_cluster(jaw_l, image_w, image_h, 0.10), face_conf or 0.65))
        zones.append(_zone("jaw", "right", "head", "ibug68", _box_around_cluster(jaw_r, image_w, image_h, 0.10), face_conf or 0.65))

    if face_box is None:
        head_pts = [pt for pt in (kp_nose, kp_eye_a, kp_eye_b, kp_ear_a, kp_ear_b) if pt is not None]
        if head_pts:
            cluster_xy = [(p[0], p[1]) for p in head_pts]
            face_box = _box_around_cluster(cluster_xy, image_w, image_h, pad_ratio=0.45)
            if face_box is not None:
                face_w = max(1.0, float(face_box[2] - face_box[0]))
                face_h = max(1.0, float(face_box[3] - face_box[1]))
                zones.append(_zone("face", "single", "head", "pose", face_box, 0.45))

    if not any(z and z["name"] == "nose" for z in zones) and kp_nose is not None:
        zones.append(_zone(
            "nose", "single", "head", "pose",
            _box_from_center_and_size((kp_nose[0], kp_nose[1]), max(4.0, face_w * 0.12), max(4.0, face_h * 0.13), image_w, image_h),
            kp_nose[2],
        ))
    if not any(z and z["name"] == "eye" and z["side"] == "left" for z in zones) and img_eye_l is not None:
        zones.append(_zone(
            "eye", "left", "head", "pose",
            _box_from_center_and_size((img_eye_l[0], img_eye_l[1]), max(4.0, face_w * 0.16), max(3.0, face_h * 0.10), image_w, image_h),
            img_eye_l[2],
        ))
    if not any(z and z["name"] == "eye" and z["side"] == "right" for z in zones) and img_eye_r is not None:
        zones.append(_zone(
            "eye", "right", "head", "pose",
            _box_from_center_and_size((img_eye_r[0], img_eye_r[1]), max(4.0, face_w * 0.16), max(3.0, face_h * 0.10), image_w, image_h),
            img_eye_r[2],
        ))

    ear_half_w = max(5.0, face_w * 0.14)
    ear_half_h = max(5.0, face_h * 0.16)
    if img_ear_l is not None:
        zones.append(_zone("ear", "left", "head", "pose",
            _box_from_center_and_size((img_ear_l[0], img_ear_l[1]), ear_half_w, ear_half_h, image_w, image_h), img_ear_l[2]))
    if img_ear_r is not None:
        zones.append(_zone("ear", "right", "head", "pose",
            _box_from_center_and_size((img_ear_r[0], img_ear_r[1]), ear_half_w, ear_half_h, image_w, image_h), img_ear_r[2]))

    sh_half_w = max(6.0, body_w * 0.06)
    sh_half_h = max(6.0, body_h * 0.035)
    if img_sh_l is not None:
        zones.append(_zone("shoulder", "left", "torso", "pose",
            _box_from_center_and_size((img_sh_l[0], img_sh_l[1]), sh_half_w, sh_half_h, image_w, image_h), img_sh_l[2]))
    if img_sh_r is not None:
        zones.append(_zone("shoulder", "right", "torso", "pose",
            _box_from_center_and_size((img_sh_r[0], img_sh_r[1]), sh_half_w, sh_half_h, image_w, image_h), img_sh_r[2]))

    if kp_sh_a is not None and kp_sh_b is not None:
        neck_x, neck_y = _midpoint(kp_sh_a, kp_sh_b)
        head_anchor_y = kp_nose[1] if kp_nose is not None else neck_y - body_h * 0.06
        neck_cy = (neck_y + head_anchor_y) * 0.5
        neck_half_w = max(8.0, body_w * 0.06)
        neck_half_h = max(8.0, abs(neck_y - head_anchor_y) * 0.6 + body_h * 0.02)
        zones.append(_zone("neck", "single", "torso", "pose",
            _box_from_center_and_size((neck_x, neck_cy), neck_half_w, neck_half_h, image_w, image_h),
            min(kp_sh_a[2], kp_sh_b[2])))

    if kp_sh_a is not None and kp_sh_b is not None and kp_hip_a is not None and kp_hip_b is not None:
        sh_mid = _midpoint(kp_sh_a, kp_sh_b)
        hip_mid = _midpoint(kp_hip_a, kp_hip_b)
        torso_h = max(8.0, abs(hip_mid[1] - sh_mid[1]))
        torso_w_full = max(8.0, abs(kp_sh_a[0] - kp_sh_b[0])) * 0.95
        chest_cx = (sh_mid[0] * 0.6 + hip_mid[0] * 0.4)
        chest_cy = sh_mid[1] + torso_h * 0.28
        zones.append(_zone("chest", "single", "torso", "pose",
            _box_from_center_and_size((chest_cx, chest_cy), torso_w_full * 0.45, torso_h * 0.22, image_w, image_h),
            min(kp_sh_a[2], kp_sh_b[2], kp_hip_a[2], kp_hip_b[2])))
        belly_cx = (sh_mid[0] * 0.35 + hip_mid[0] * 0.65)
        belly_cy = sh_mid[1] + torso_h * 0.72
        zones.append(_zone("belly", "single", "torso", "pose",
            _box_from_center_and_size((belly_cx, belly_cy), torso_w_full * 0.40, torso_h * 0.20, image_w, image_h),
            min(kp_sh_a[2], kp_sh_b[2], kp_hip_a[2], kp_hip_b[2])))

    hip_half_w = max(7.0, body_w * 0.06)
    hip_half_h = max(7.0, body_h * 0.035)
    if img_hip_l is not None:
        zones.append(_zone("hip", "left", "torso", "pose",
            _box_from_center_and_size((img_hip_l[0], img_hip_l[1]), hip_half_w, hip_half_h, image_w, image_h), img_hip_l[2]))
    if img_hip_r is not None:
        zones.append(_zone("hip", "right", "torso", "pose",
            _box_from_center_and_size((img_hip_r[0], img_hip_r[1]), hip_half_w, hip_half_h, image_w, image_h), img_hip_r[2]))

    elbow_half = max(6.0, body_w * 0.05)
    wrist_half = max(6.0, body_w * 0.045)
    if img_elbow_l is not None:
        zones.append(_zone("elbow", "left", "arms", "pose",
            _box_from_center_and_size((img_elbow_l[0], img_elbow_l[1]), elbow_half, elbow_half, image_w, image_h), img_elbow_l[2]))
    if img_elbow_r is not None:
        zones.append(_zone("elbow", "right", "arms", "pose",
            _box_from_center_and_size((img_elbow_r[0], img_elbow_r[1]), elbow_half, elbow_half, image_w, image_h), img_elbow_r[2]))
    if img_wrist_l is not None:
        zones.append(_zone("wrist", "left", "arms", "pose",
            _box_from_center_and_size((img_wrist_l[0], img_wrist_l[1]), wrist_half, wrist_half, image_w, image_h), img_wrist_l[2]))
    if img_wrist_r is not None:
        zones.append(_zone("wrist", "right", "arms", "pose",
            _box_from_center_and_size((img_wrist_r[0], img_wrist_r[1]), wrist_half, wrist_half, image_w, image_h), img_wrist_r[2]))

    seg_half = max(6.0, body_w * 0.045)
    if img_sh_l is not None and img_elbow_l is not None:
        zones.append(_zone("upper_arm", "left", "arms", "pose",
            _segment_box(img_sh_l, img_elbow_l, seg_half, image_w, image_h), min(img_sh_l[2], img_elbow_l[2])))
    if img_sh_r is not None and img_elbow_r is not None:
        zones.append(_zone("upper_arm", "right", "arms", "pose",
            _segment_box(img_sh_r, img_elbow_r, seg_half, image_w, image_h), min(img_sh_r[2], img_elbow_r[2])))
    if img_elbow_l is not None and img_wrist_l is not None:
        zones.append(_zone("forearm", "left", "arms", "pose",
            _segment_box(img_elbow_l, img_wrist_l, seg_half, image_w, image_h), min(img_elbow_l[2], img_wrist_l[2])))
    if img_elbow_r is not None and img_wrist_r is not None:
        zones.append(_zone("forearm", "right", "arms", "pose",
            _segment_box(img_elbow_r, img_wrist_r, seg_half, image_w, image_h), min(img_elbow_r[2], img_wrist_r[2])))

    knee_half = max(6.0, body_w * 0.05)
    ankle_half = max(6.0, body_w * 0.05)
    if img_knee_l is not None:
        zones.append(_zone("knee", "left", "legs", "pose",
            _box_from_center_and_size((img_knee_l[0], img_knee_l[1]), knee_half, knee_half, image_w, image_h), img_knee_l[2]))
    if img_knee_r is not None:
        zones.append(_zone("knee", "right", "legs", "pose",
            _box_from_center_and_size((img_knee_r[0], img_knee_r[1]), knee_half, knee_half, image_w, image_h), img_knee_r[2]))
    if img_ankle_l is not None:
        zones.append(_zone("ankle", "left", "legs", "pose",
            _box_from_center_and_size((img_ankle_l[0], img_ankle_l[1]), ankle_half, ankle_half, image_w, image_h), img_ankle_l[2]))
    if img_ankle_r is not None:
        zones.append(_zone("ankle", "right", "legs", "pose",
            _box_from_center_and_size((img_ankle_r[0], img_ankle_r[1]), ankle_half, ankle_half, image_w, image_h), img_ankle_r[2]))

    leg_half = max(6.0, body_w * 0.05)
    if img_hip_l is not None and img_knee_l is not None:
        zones.append(_zone("thigh", "left", "legs", "pose",
            _segment_box(img_hip_l, img_knee_l, leg_half, image_w, image_h), min(img_hip_l[2], img_knee_l[2])))
    if img_hip_r is not None and img_knee_r is not None:
        zones.append(_zone("thigh", "right", "legs", "pose",
            _segment_box(img_hip_r, img_knee_r, leg_half, image_w, image_h), min(img_hip_r[2], img_knee_r[2])))
    if img_knee_l is not None and img_ankle_l is not None:
        zones.append(_zone("shin", "left", "legs", "pose",
            _segment_box(img_knee_l, img_ankle_l, leg_half, image_w, image_h), min(img_knee_l[2], img_ankle_l[2])))
    if img_knee_r is not None and img_ankle_r is not None:
        zones.append(_zone("shin", "right", "legs", "pose",
            _segment_box(img_knee_r, img_ankle_r, leg_half, image_w, image_h), min(img_knee_r[2], img_ankle_r[2])))

    if face_box is not None and (kp_sh_a is not None or kp_sh_b is not None):
        face_top = face_box[1]
        sh_pts = [pt for pt in (kp_sh_a, kp_sh_b) if pt is not None]
        sh_y = max(pt[1] for pt in sh_pts) if sh_pts else face_box[3]
        sh_xs = [pt[0] for pt in sh_pts] if sh_pts else [face_box[0], face_box[2]]
        sh_left = min(sh_xs)
        sh_right = max(sh_xs)
        head_only_box = _clip_box((face_box[0], face_top, face_box[2], sh_y - body_h * 0.01), image_w, image_h)
        if head_only_box is not None:
            zones.append(_zone("head_only", "single", "composite", "derived", head_only_box, 0.7))
        bust_box = _clip_box((min(sh_left, face_box[0]) - body_w * 0.02, face_top, max(sh_right, face_box[2]) + body_w * 0.02, sh_y + body_h * 0.08), image_w, image_h)
        if bust_box is not None:
            zones.append(_zone("bust", "single", "composite", "derived", bust_box, 0.7))
        if kp_hip_a is not None or kp_hip_b is not None:
            hip_pts = [pt for pt in (kp_hip_a, kp_hip_b) if pt is not None]
            hip_y = max(pt[1] for pt in hip_pts)
            ub_box = _clip_box((min(sh_left, face_box[0]) - body_w * 0.02, face_top, max(sh_right, face_box[2]) + body_w * 0.02, hip_y + body_h * 0.02), image_w, image_h)
            if ub_box is not None:
                zones.append(_zone("upper_body", "single", "composite", "derived", ub_box, 0.7))
            if kp_knee_a is not None or kp_knee_b is not None:
                knee_pts = [pt for pt in (kp_knee_a, kp_knee_b) if pt is not None]
                mid_thigh_y = (hip_y + min(pt[1] for pt in knee_pts)) * 0.5
                hb_box = _clip_box((min(sh_left, face_box[0]) - body_w * 0.02, face_top, max(sh_right, face_box[2]) + body_w * 0.02, mid_thigh_y), image_w, image_h)
                if hb_box is not None:
                    zones.append(_zone("half_body", "single", "composite", "derived", hb_box, 0.7))

    return [z for z in zones if z is not None]


def _select_zones_for_solver(
    all_zones: List[Dict[str, Any]],
    part_settings: Dict[str, Dict[str, Any]],
    image_w: int,
    image_h: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for zone in all_zones:
        spec = _ZONE_SPEC_BY_KEY.get(zone["name"])
        if spec is None:
            continue
        settings = part_settings.get(zone["name"])
        if not settings or not settings["enabled"] or settings["weight"] <= 0.0:
            continue
        if spec["paired"]:
            side_filter = settings.get("side", "both")
            if side_filter == "left" and zone["side"] != "left":
                continue
            if side_filter == "right" and zone["side"] != "right":
                continue
            if side_filter == "both" and zone["side"] not in ("left", "right"):
                continue
        elif zone["side"] != "single":
            continue
        eff_weight = float(settings["weight"]) * float(zone["confidence"])
        if eff_weight <= 0.0:
            continue
        out.append({
            "name": zone["name"],
            "side": zone["side"],
            "weight": eff_weight,
            "mode": settings["mode"],
            "box": _expand_box(zone["box"], float(settings["margin"]), image_w, image_h),
            "raw_box": zone["box"],
            "confidence": zone["confidence"],
            "source": zone["source"],
        })
    return out


def _exclude_repulsion(box: Tuple[int, int, int, int], crop: Tuple[int, int, int, int]) -> float:
    cov = _coverage(box, crop)
    cx, cy = _box_center(box)
    cropcx, cropcy = _box_center(crop)
    crop_w = max(1.0, float(crop[2] - crop[0]))
    crop_h = max(1.0, float(crop[3] - crop[1]))
    dx = abs(cx - cropcx) / crop_w
    dy = abs(cy - cropcy) / crop_h
    proximity = max(0.0, 1.0 - min(1.5, max(dx, dy)))
    return float(cov) + 0.25 * proximity


def _choose_crop_zones(
    solver_zones: List[Dict[str, Any]],
    fallback_box: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    aspect = float(target_w) / max(1.0, float(target_h))
    include_zones = [z for z in solver_zones if z["mode"] == "include" and z["weight"] > 0.0]
    exclude_zones = [z for z in solver_zones if z["mode"] == "exclude" and z["weight"] > 0.0]
    include_boxes = [z["box"] for z in include_zones]
    base_box = _union_boxes(include_boxes) or fallback_box
    min_w, min_h = _min_crop_size_for_box(base_box, aspect)

    candidate_centers: List[Tuple[float, float]] = [_box_center(fallback_box), _box_center(base_box)]
    weighted_cx = 0.0
    weighted_cy = 0.0
    total_w = 0.0
    for z in include_zones:
        rcx, rcy = _box_center(z["box"])
        candidate_centers.append((rcx, rcy))
        weighted_cx += rcx * z["weight"]
        weighted_cy += rcy * z["weight"]
        total_w += z["weight"]
    if total_w > 1e-6:
        candidate_centers.append((weighted_cx / total_w, weighted_cy / total_w))

    for z_inc in include_zones:
        ic_cx, ic_cy = _box_center(z_inc["box"])
        inc_w = float(z_inc["box"][2] - z_inc["box"][0])
        inc_h = float(z_inc["box"][3] - z_inc["box"][1])
        inc_dim = max(inc_w, inc_h)
        for z_exc in exclude_zones:
            ec_cx, ec_cy = _box_center(z_exc["box"])
            dx = ic_cx - ec_cx
            dy = ic_cy - ec_cy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1.0:
                continue
            exc_strength = float(z_exc["weight"]) / 100.0
            for push_factor in (0.2, 0.4, 0.65, 0.9):
                push = inc_dim * push_factor * exc_strength
                candidate_centers.append((ic_cx + dx / dist * push, ic_cy + dy / dist * push))

    scale_candidates = [0.35, 0.45, 0.55, 0.7, 0.85, 0.95, 1.0, 1.05, 1.12, 1.22, 1.35, 1.55, 1.8, 2.15]

    best_crop = _fit_crop_from_center(candidate_centers[0][0], candidate_centers[0][1], min_w, min_h, image_w, image_h)
    best_score = -1e18
    for cx, cy in candidate_centers:
        for scale in scale_candidates:
            crop_w = min(image_w, int(np.ceil(min_w * scale)))
            crop_h = min(image_h, int(np.ceil(min_h * scale)))
            crop = _fit_crop_from_center(cx, cy, crop_w, crop_h, image_w, image_h)
            score = 0.0
            for z in solver_zones:
                if z["mode"] == "exclude":
                    score -= z["weight"] * _exclude_repulsion(z["box"], crop)
                else:
                    score += z["weight"] * _coverage(z["box"], crop)
            score -= 2.0 * (_box_area(crop) / max(1.0, image_w * image_h))
            if score > best_score:
                best_score = score
                best_crop = crop
    return best_crop


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
    box_area_norm = _box_area(bbox) / max(1.0, image_w * image_h)
    center_x, center_y = _box_center(bbox)
    dx = (center_x / max(1.0, image_w)) - 0.5
    dy = (center_y / max(1.0, image_h)) - 0.5
    center_bias = max(0.0, 1.0 - ((dx * dx + dy * dy) ** 0.5) * 1.65)
    age = face.get("age") if face else None
    gender = face.get("gender") if face else "unknown"
    det_score = float(face.get("det_score", 0.0) or 0.0) if face else 0.0

    pref = str(preferred_gender or "any").strip().lower()
    gen = str(gender or "unknown").strip().lower()
    if pref == "any":
        gender_match = 1.0
    elif gen == pref:
        gender_match = 1.0
    elif gen == "unknown":
        gender_match = 0.45
    else:
        gender_match = 0.0

    if age is None:
        age_match = 0.5
    else:
        lo = min(age_min, age_max)
        hi = max(age_min, age_max)
        if lo <= age <= hi:
            age_match = 1.0
        else:
            age_match = max(0.0, 1.0 - (min(abs(age - lo), abs(age - hi)) / 25.0))

    pose_quality = float(person.get("score", 0.0))
    return (
        3.5 * age_match
        + 3.0 * gender_match
        + 1.5 * box_area_norm
        + 1.0 * det_score
        + 0.8 * pose_quality
        + 0.6 * center_bias
    )


def _detect_people_and_zones(
    image_uint8: np.ndarray,
    *,
    pose_model_name: str,
    device_choice: str,
    confidence_threshold: float,
    keypoint_min_conf: float = 0.15,
) -> List[Dict[str, Any]]:
    image_bgr = image_uint8[..., ::-1].copy() if cv2 is not None else image_uint8.copy()
    image_h, image_w = image_uint8.shape[:2]
    pose_model = _load_model(pose_model_name, None, device_choice)
    people = _extract_pose_people(image_bgr, pose_model, device_choice, confidence_threshold)
    if not people:
        return []
    faces = _extract_faces_ibug(image_bgr)
    for person in people:
        bbox = _clip_box(person["bbox"], image_w, image_h)
        if bbox is None:
            person["zones"] = []
            continue
        person["bbox"] = bbox
        face = _match_face_to_person(bbox, faces)
        person["face"] = face
        person["gender"] = face.get("gender", "unknown") if face else "unknown"
        person["age"] = face.get("age") if face else None
        person["head_pose"] = face.get("head_pose") if face else None
        person["zones"] = _derive_anatomy_zones(person, face, image_w, image_h, keypoint_min_conf=keypoint_min_conf)
    return people


def _run_detection_cached(
    image_uint8: np.ndarray,
    *,
    confidence_threshold: float,
    device_choice: str,
) -> Dict[str, Any]:
    digest = _image_digest(image_uint8)
    key = _detection_cache_key(digest, confidence_threshold=confidence_threshold, device_choice=device_choice)
    cached = _detection_cache_get(key)
    if cached is not None:
        return cached
    people = _detect_people_and_zones(
        image_uint8,
        pose_model_name=_DEFAULT_POSE_MODEL,
        device_choice=device_choice,
        confidence_threshold=confidence_threshold,
    )
    entry = {
        "people": people,
        "image_digest": digest,
        "confidence_threshold": float(confidence_threshold),
        "device_choice": device_choice,
        "image_size": (int(image_uint8.shape[1]), int(image_uint8.shape[0])),
    }
    _detection_cache_put(key, entry)
    return entry


def _draw_box_outline(image: np.ndarray, box: Tuple[int, int, int, int], color: Tuple[int, int, int], thickness: int = 2) -> None:
    if cv2 is None:
        return
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)


def _render_overlay(
    image_uint8: np.ndarray,
    people: List[Dict[str, Any]],
    selected_idx: int,
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray:
    overlay = image_uint8.copy()
    if cv2 is None:
        return overlay
    for idx, person in enumerate(people):
        bbox = person.get("bbox")
        if bbox is None:
            continue
        is_selected = idx == selected_idx
        color = (64, 255, 96) if is_selected else (160, 160, 160)
        _draw_box_outline(overlay, bbox, color, thickness=2 if is_selected else 1)
        for zone in person.get("zones", []):
            zc = _GROUP_COLORS.get(zone["group"], (200, 200, 200))
            _draw_box_outline(overlay, zone["box"], zc, thickness=1)
    clipped = _clip_box(crop_box, image_uint8.shape[1], image_uint8.shape[0])
    if clipped is not None:
        _draw_box_outline(overlay, clipped, (255, 0, 0), thickness=3)
    return overlay


def _save_preview_png(image_uint8: np.ndarray, prefix: str = "recrop") -> Optional[Dict[str, str]]:
    if Image is None or folder_paths is None:
        return None
    try:
        temp_dir = folder_paths.get_temp_directory()
    except Exception:
        return None
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception:
        return None
    with _PREVIEW_COUNTER_LOCK:
        seq = next(_PREVIEW_COUNTER)
    filename = f"{prefix}_{os.getpid()}_{time.time_ns()}_{seq}.png"
    full_path = os.path.join(temp_dir, filename)
    try:
        Image.fromarray(image_uint8).save(full_path)
    except Exception:
        return None
    return {"filename": filename, "subfolder": "", "type": "temp"}


def _zones_payload(people: List[Dict[str, Any]], selected_idx: int, image_w: int, image_h: int) -> Dict[str, Any]:
    payload_people = []
    for idx, person in enumerate(people):
        bbox = person.get("bbox")
        payload_people.append({
            "index": idx,
            "selected": idx == selected_idx,
            "bbox": [int(v) for v in bbox] if bbox else None,
            "gender": person.get("gender", "unknown"),
            "age": person.get("age"),
            "subject_score": float(person.get("subject_score", 0.0)),
            "head_pose": person.get("head_pose"),
            "zones": [
                {
                    "name": z["name"],
                    "side": z["side"],
                    "group": z["group"],
                    "source": z["source"],
                    "box": list(z["box"]),
                    "confidence": float(z["confidence"]),
                }
                for z in person.get("zones", [])
            ],
        })
    return {
        "image_size": [int(image_w), int(image_h)],
        "people": payload_people,
        "selected_index": int(selected_idx),
    }


def _build_part_settings(kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    settings: Dict[str, Dict[str, Any]] = {}
    for spec in _ZONE_SPECS:
        key = spec["key"]
        enabled = bool(kwargs.get(f"enabled_{key}", spec["default_enabled"]))
        mode = str(kwargs.get(f"mode_{key}", spec["default_mode"]) or spec["default_mode"]).strip().lower()
        if mode not in _MODE_VALUES:
            mode = spec["default_mode"]
        side = "single"
        if spec["paired"]:
            side = str(kwargs.get(f"side_{key}", "both") or "both").strip().lower()
            if side not in _SIDE_VALUES:
                side = "both"
        weight = max(0.0, min(100.0, float(kwargs.get(f"weight_{key}", spec["default_weight"]))))
        margin = max(0.0, min(10.0, float(kwargs.get(f"margin_{key}", spec["default_margin"]))))
        settings[key] = {"enabled": enabled, "mode": mode, "side": side, "weight": weight, "margin": margin}
    return settings


def _zero_mask(height: int, width: int, device: Any) -> torch.Tensor:
    """All-black MASK (nothing to repair), shaped [1, H, W] to match its IMAGE."""
    return torch.zeros((1, int(height), int(width)), dtype=torch.float32, device=device)


def _fill_mask_tensor(
    crop_box: Tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    target_w: int,
    target_h: int,
    device: Any,
    feather_px: int = 0,
) -> torch.Tensor:
    """MASK marking the synthetic fill region, aligned + resized to the output image.

    The synthetic (out-of-image) region is ALWAYS solid 1.0 so a downstream model fully
    regenerates it. ``feather_px`` only ramps the mask 1.0 -> 0.0 *inward* across that many
    output pixels of real content, giving the model a blend overlap at the seam without
    ever weakening coverage of the invented pixels.
    """
    hard = _crop_fill_mask(crop_box, image_w, image_h)
    tw, th = int(target_w), int(target_h)

    # Resize the hard mask to output size WITHOUT softening the seam (nearest, not linear).
    if cv2 is not None:
        hard_t = cv2.resize(hard, (tw, th), interpolation=cv2.INTER_NEAREST)
    else:
        import torch.nn.functional as F
        hard_t = F.interpolate(torch.from_numpy(hard)[None, None], size=(th, tw), mode="nearest")[0, 0].numpy()
    hard_t = hard_t.astype(np.float32)

    feather_px = max(0, int(feather_px))
    if feather_px > 0 and cv2 is not None and float(hard_t.max()) > 0.0:
        # Distance (in output px) from each real pixel to the nearest synthetic pixel.
        real = (hard_t < 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(real, cv2.DIST_L2, 3)
        ramp = np.clip(1.0 - dist / float(feather_px), 0.0, 1.0).astype(np.float32)
        # max(): inward ramp only; synthetic region stays fully 1.0.
        mask = np.maximum(hard_t, ramp)
    else:
        mask = hard_t

    arr = np.ascontiguousarray(mask, dtype=np.float32)
    return torch.from_numpy(arr).clamp_(0.0, 1.0).unsqueeze(0).to(device)


def _parse_manual_crop(value: Any) -> Optional[Tuple[int, int, int, int]]:
    """Parse a manual crop box 'x0,y0,x1,y1' (image px; negatives / out-of-bounds allowed).
    Returns None if empty or malformed."""
    if not value:
        return None
    try:
        parts = [int(round(float(v))) for v in str(value).replace(";", ",").split(",") if v.strip() != ""]
    except Exception:
        return None
    if len(parts) != 4:
        return None
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _compute_crop_box(selected, part_settings, framing_mode, width, height,
                      image_w, image_h, crop_mode, manual_box):
    """Resolve the crop box for the selected person: a verbatim manual box (out-of-bounds
    allowed) in manual mode, else the auto solver + aspect reconcile. Returns (box, mode)."""
    if crop_mode == "manual" and manual_box is not None:
        return manual_box, "manual"
    solver_zones = _select_zones_for_solver(selected.get("zones", []), part_settings, image_w, image_h)
    full_zone = next((z for z in selected.get("zones", []) if z["name"] == "full"), None)
    fallback_box = full_zone["box"] if full_zone else _clip_box(selected["bbox"], image_w, image_h) or (0, 0, image_w, image_h)
    crop_box = _choose_crop_zones(solver_zones, fallback_box, int(width), int(height), image_w, image_h)
    fm = str(framing_mode or "crop").strip().lower()
    if fm == "expand":
        crop_box = _expand_crop_to_aspect(crop_box, int(width), int(height))
    else:
        crop_box = _inscribe_crop_to_aspect(crop_box, int(width), int(height), image_w, image_h)
    return crop_box, fm


class Recrop:
    CATEGORY = "ESS/Image"
    FUNCTION = "recrop"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "debug_overlay", "fill_mask")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "width": ("INT", {"default": 768, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
        }
        optional: Dict[str, Any] = {
            "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
            "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
            "framing_mode": (("crop", "expand"), {"default": "crop"}),
            "expand_fill_mode": (("border_fill", "fill_color", "reflect", "inpaint"), {"default": "border_fill"}),
            "expand_fill_color": ("STRING", {"default": "#000000", "multiline": False}),
            "preferred_gender": (("any", "female", "male"), {"default": "any"}),
            "target_age_min": ("INT", {"default": 18, "min": 0, "max": 120, "step": 1}),
            "target_age_max": ("INT", {"default": 35, "min": 0, "max": 120, "step": 1}),
            "detect_only": ("BOOLEAN", {"default": False}),
        }
        for spec in _ZONE_SPECS:
            key = spec["key"]
            optional[f"enabled_{key}"] = ("BOOLEAN", {"default": bool(spec["default_enabled"])})
            optional[f"mode_{key}"] = (_MODE_VALUES, {"default": spec["default_mode"]})
            if spec["paired"]:
                optional[f"side_{key}"] = (_SIDE_VALUES, {"default": "both"})
            optional[f"weight_{key}"] = ("FLOAT", {"default": spec["default_weight"], "min": 0.0, "max": 100.0, "step": 1.0})
            optional[f"margin_{key}"] = ("FLOAT", {"default": spec["default_margin"], "min": 0.0, "max": 10.0, "step": 0.1})
        # New inputs must be appended at the END so existing nodes' positional
        # widgets_values stay aligned (inserting mid-list shifts every later widget).
        optional["mask_feather"] = ("INT", {"default": 16, "min": 0, "max": 512, "step": 1})
        optional["crop_mode"] = (("auto", "manual"), {"default": "auto"})
        optional["manual_crop"] = ("STRING", {"default": "", "multiline": False})
        return {"required": required, "optional": optional, "hidden": {"node_id": "UNIQUE_ID"}}

    def recrop(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        device: str = "auto",
        confidence_threshold: float = 0.25,
        framing_mode: str = "crop",
        expand_fill_mode: str = "border_fill",
        expand_fill_color: str = "#000000",
        mask_feather: int = 16,
        preferred_gender: str = "any",
        target_age_min: int = 18,
        target_age_max: int = 35,
        detect_only: bool = False,
        node_id: Optional[str] = None,
        **kwargs: Any,
    ):
        image_uint8 = _tensor_to_uint8_image(image)
        image_h, image_w = image_uint8.shape[:2]
        device_choice = _resolve_device(device)

        _store_node_image(node_id, image_uint8)

        cache_entry = _run_detection_cached(
            image_uint8,
            confidence_threshold=confidence_threshold,
            device_choice=device_choice,
        )
        people: List[Dict[str, Any]] = list(cache_entry["people"])

        if detect_only:
            for person in people:
                person["subject_score"] = _subject_score(
                    person, person.get("face"), preferred_gender, target_age_min, target_age_max, image_w, image_h
                )
            people.sort(key=lambda p: float(p.get("subject_score", 0.0)), reverse=True)
            selected_idx = 0 if people else -1
            preview_info = _save_preview_png(image_uint8, prefix="recrop_detect")
            zones_payload = _zones_payload(people, selected_idx, image_w, image_h)
            zones_payload["target_size"] = [int(width), int(height)]
            zones_payload["device_choice"] = device_choice
            zones_payload["confidence_threshold"] = float(confidence_threshold)
            zones_payload["detect_only"] = True
            # Compute the crop box now so the frame shows on Detect (same logic as a full run).
            if people:
                crop_mode = str(kwargs.get("crop_mode", "auto") or "auto").strip().lower()
                manual_box = _parse_manual_crop(kwargs.get("manual_crop", ""))
                crop_box, fm = _compute_crop_box(
                    people[0], _build_part_settings(kwargs), framing_mode,
                    width, height, image_w, image_h, crop_mode, manual_box,
                )
                zones_payload["crop_box"] = [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])]
                zones_payload["framing_mode"] = fm
            if preview_info is not None:
                zones_payload["preview"] = preview_info
            ui_payload: Dict[str, Any] = {"zones": [json.dumps(zones_payload)]}
            passthrough = _uint8_image_to_tensor(image_uint8, image.device)
            empty_mask = _zero_mask(image_h, image_w, image.device)
            return {
                "ui": ui_payload,
                "result": (passthrough, passthrough, empty_mask),
            }

        if not people:
            blank_overlay = _uint8_image_to_tensor(image_uint8, image.device)
            preview_info = _save_preview_png(image_uint8, prefix="recrop_empty")
            empty_payload = _zones_payload([], -1, image_w, image_h)
            if preview_info is not None:
                empty_payload["preview"] = preview_info
            ui_payload = {"zones": [json.dumps(empty_payload)]}
            return {
                "ui": ui_payload,
                "result": (image, blank_overlay, _zero_mask(image_h, image_w, image.device)),
            }

        for person in people:
            person["subject_score"] = _subject_score(
                person, person.get("face"), preferred_gender, target_age_min, target_age_max, image_w, image_h
            )
        people.sort(key=lambda p: float(p.get("subject_score", 0.0)), reverse=True)
        selected = people[0]
        selected_idx = 0

        part_settings = _build_part_settings(kwargs)
        crop_mode = str(kwargs.get("crop_mode", "auto") or "auto").strip().lower()
        manual_box = _parse_manual_crop(kwargs.get("manual_crop", ""))
        # Manual mode uses the user box verbatim (edges may sit outside the image; the fill
        # modes synthesize those pixels). Auto runs the solver + aspect reconcile.
        crop_box, framing_mode_norm = _compute_crop_box(
            selected, part_settings, framing_mode, width, height,
            image_w, image_h, crop_mode, manual_box,
        )

        crop_image = _render_crop_region(
            image_uint8,
            crop_box,
            str(expand_fill_mode or "border_fill").strip().lower(),
            expand_fill_color,
        )
        resized = _resize_crop_exact(crop_image, int(width), int(height))
        output_tensor = _uint8_image_to_tensor(resized, image.device)

        overlay_uint8 = _render_overlay(image_uint8, people, selected_idx, crop_box)
        overlay_tensor = _uint8_image_to_tensor(overlay_uint8, image.device)

        preview_info = _save_preview_png(image_uint8, prefix="recrop")
        zones_payload = _zones_payload(people, selected_idx, image_w, image_h)
        zones_payload["crop_box"] = [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])]
        zones_payload["framing_mode"] = framing_mode_norm
        zones_payload["target_size"] = [int(width), int(height)]
        zones_payload["device_choice"] = device_choice
        zones_payload["confidence_threshold"] = float(confidence_threshold)
        if preview_info is not None:
            zones_payload["preview"] = preview_info

        ui_payload: Dict[str, Any] = {"zones": [json.dumps(zones_payload)]}

        fill_mask = _fill_mask_tensor(crop_box, image_w, image_h, int(width), int(height), image.device, mask_feather)

        return {
            "ui": ui_payload,
            "result": (output_tensor, overlay_tensor, fill_mask),
        }


def detect_on_cached_image(
    node_id: str,
    *,
    confidence_threshold: float,
    device_choice: str,
) -> Optional[Dict[str, Any]]:
    image_uint8 = _get_node_image(node_id)
    if image_uint8 is None:
        return None
    dev = _resolve_device(device_choice)
    entry = _run_detection_cached(image_uint8, confidence_threshold=confidence_threshold, device_choice=dev)
    image_h, image_w = image_uint8.shape[:2]
    preview_info = _save_preview_png(image_uint8, prefix="recrop_detect")
    selected_idx = 0 if entry["people"] else -1
    payload = _zones_payload(entry["people"], selected_idx, image_w, image_h)
    return {"zones": payload, "preview": preview_info}


def register_recrop_routes(prompt_server, web_module) -> None:
    if prompt_server is None or web_module is None:
        return

    @prompt_server.instance.routes.post("/ess/recrop/detect")
    async def ess_recrop_detect(request):
        try:
            payload = await request.json()
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": f"invalid json: {exc}"}, status=400)
        node_id = str(payload.get("node_id") or "").strip()
        if not node_id:
            return web_module.json_response({"ok": False, "error": "node_id required"}, status=400)
        try:
            confidence_threshold = float(payload.get("confidence_threshold", 0.25))
        except Exception:
            confidence_threshold = 0.25
        device_choice = str(payload.get("device", "auto") or "auto")
        try:
            import asyncio  # local import; aiohttp already pulls it
            result = await asyncio.to_thread(
                detect_on_cached_image,
                node_id,
                confidence_threshold=confidence_threshold,
                device_choice=device_choice,
            )
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)
        if result is None:
            return web_module.json_response(
                {"ok": False, "error": "no cached image; run the workflow once before pressing Detect"},
                status=404,
            )
        return web_module.json_response({"ok": True, **result})


def list_zone_groups() -> List[Dict[str, Any]]:
    return [
        {
            "key": group["key"],
            "label": group["label"],
            "zones": [
                {
                    "key": z["key"],
                    "label": z["label"],
                    "paired": bool(z["paired"]),
                    "default_enabled": bool(z["default_enabled"]),
                }
                for z in group["zones"]
            ],
        }
        for group in _ZONE_GROUPS
    ]
