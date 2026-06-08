from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .composition_crop import (
    _PART_SPECS,
    _PAIR_SIDE_VALUES,
    _MODE_VALUES,
    _tensor_to_uint8_image,
    _uint8_image_to_tensor,
    _clip_box,
    _box_from_points,
    _box_from_center_and_size,
    _expand_box,
    _box_center,
    _set_region,
    _select_region_boxes,
    _extract_pose_people,
    _extract_faces,
    _match_face,
    _kp,
    _estimate_stage,
    _subject_score,
    _choose_crop,
    _inscribe_crop_to_aspect,
    _expand_crop_to_aspect,
    _render_crop_region,
    _resize_crop_exact,
    _draw_box,
    _draw_text,
    _draw_point,
)
from .person_crop_to_size import _load_model, _resolve_device

_DEFAULT_POSE_MODEL = "yolo11n-pose.pt"

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


_V2_PART_SPECS = (
    {"key": "full_person",  "label": "Full",         "paired": False, "default_enabled": True,  "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "face",         "label": "Face",         "paired": False, "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "eyebrow",      "label": "Eyebrow",      "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "eye",          "label": "Eye",          "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "ear",          "label": "Ear",          "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "nose",         "label": "Nose",         "paired": False, "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "mouth_corner", "label": "Mouth Corner", "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "mouth",        "label": "Mouth",        "paired": False, "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "chin",         "label": "Chin",         "paired": False, "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "shoulder",     "label": "Shoulder",     "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "elbow",        "label": "Elbow",        "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "wrist",        "label": "Wrist",        "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "hip",          "label": "Hip",          "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "knee",         "label": "Knee",         "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
    {"key": "ankle",        "label": "Ankle",        "paired": True,  "default_enabled": False, "default_mode": "include", "default_weight": 100.0, "default_margin": 0.0},
)


def _viewer_pair(
    a: Optional[Tuple[float, float]],
    b: Optional[Tuple[float, float]],
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if a is None and b is None:
        return None, None
    if a is None:
        return None, b
    if b is None:
        return a, None
    if a[0] <= b[0]:
        return a, b
    return b, a


def _safe_kp_pair(person: Dict[str, Any], idx_a: int, idx_b: int) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    return _viewer_pair(_kp(person, idx_a), _kp(person, idx_b))


def _kps_point(face: Optional[Dict[str, Any]], idx: int) -> Optional[Tuple[float, float]]:
    if not face:
        return None
    points = face.get("kps") or []
    if idx >= len(points):
        return None
    x, y = points[idx]
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return (float(x), float(y))


def _derive_regions_v2(
    person: Dict[str, Any],
    face: Optional[Dict[str, Any]],
    image_w: int,
    image_h: int,
) -> Dict[str, Dict[str, Tuple[int, int, int, int]]]:
    """Build region boxes for one person using only well-defined anchors.

    Why: the original V1 implementation split the InsightFace 2d_106 landmark
    array at hand-picked indices (33-51 brows, 51-63 nose, 63-87 eyes,
    87-106 mouth). That layout is not documented in the installed insightface
    package and the slicing produces wrong regions for several labels.

    V2 anchors every face-derived region on the well-defined 5-point
    `face.kps` (viewer-relative: 0=left-eye, 1=right-eye, 2=nose,
    3=left-mouth, 4=right-mouth) plus `face.bbox`. For paired pose parts the
    left/right assignment is taken from image-x order so "left" always means
    "image-left" regardless of which way the subject is facing.
    """
    regions: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}

    person_box = _clip_box(person["bbox"], image_w, image_h)
    if person_box is not None:
        _set_region(regions, "full_person", "single", person_box)
    body_w = max(1.0, float(person_box[2] - person_box[0]) if person_box else image_w * 0.2)
    body_h = max(1.0, float(person_box[3] - person_box[1]) if person_box else image_h * 0.4)

    eye_l_v, eye_r_v = _safe_kp_pair(person, 1, 2)
    ear_l_v, ear_r_v = _safe_kp_pair(person, 3, 4)
    sh_l_v, sh_r_v = _safe_kp_pair(person, 5, 6)
    el_l_v, el_r_v = _safe_kp_pair(person, 7, 8)
    wr_l_v, wr_r_v = _safe_kp_pair(person, 9, 10)
    hp_l_v, hp_r_v = _safe_kp_pair(person, 11, 12)
    kn_l_v, kn_r_v = _safe_kp_pair(person, 13, 14)
    an_l_v, an_r_v = _safe_kp_pair(person, 15, 16)
    nose_pose = _kp(person, 0)

    face_box = None
    nose_face = None
    eye_l_face = None
    eye_r_face = None
    mouth_l = None
    mouth_r = None
    mouth_center = None
    if face is not None:
        face_box = _clip_box(face["bbox"], image_w, image_h)
        if face_box is not None:
            _set_region(regions, "face", "single", face_box)
        eye_l_face = _kps_point(face, 0)
        eye_r_face = _kps_point(face, 1)
        nose_face = _kps_point(face, 2)
        mouth_l = _kps_point(face, 3)
        mouth_r = _kps_point(face, 4)
        if mouth_l is not None and mouth_r is not None:
            mouth_center = ((mouth_l[0] + mouth_r[0]) * 0.5, (mouth_l[1] + mouth_r[1]) * 0.5)
        elif mouth_l is not None:
            mouth_center = mouth_l
        elif mouth_r is not None:
            mouth_center = mouth_r

    if face_box is None:
        anchor_pts = [p for p in (nose_pose, eye_l_v, eye_r_v, ear_l_v, ear_r_v, sh_l_v, sh_r_v) if p is not None]
        fallback_face = _box_from_points(anchor_pts, image_w, image_h, pad_ratio=0.22)
        if fallback_face is not None:
            _set_region(regions, "face", "single", fallback_face)
            face_box = fallback_face

    face_w = max(1.0, float((face_box[2] - face_box[0]) if face_box is not None else body_w * 0.28))
    face_h = max(1.0, float((face_box[3] - face_box[1]) if face_box is not None else body_h * 0.18))

    nose_half_w = max(4.0, face_w * 0.11)
    nose_half_h = max(4.0, face_h * 0.12)
    eye_half_w = max(4.0, face_w * 0.16)
    eye_half_h = max(3.0, face_h * 0.10)
    brow_half_w = max(5.0, face_w * 0.18)
    brow_half_h = max(3.0, face_h * 0.08)
    ear_half_w = max(4.0, face_w * 0.14)
    ear_half_h = max(4.0, face_h * 0.16)
    mouth_corner_half_w = max(4.0, face_w * 0.10)
    mouth_corner_half_h = max(3.0, face_h * 0.08)
    chin_half_w = max(5.0, face_w * 0.20)
    chin_half_h = max(4.0, face_h * 0.10)
    mouth_half_w = max(6.0, face_w * 0.22)
    mouth_half_h = max(4.0, face_h * 0.10)
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

    nose_center = nose_face if nose_face is not None else nose_pose
    if nose_center is not None:
        _set_region(regions, "nose", "single",
                    _box_from_center_and_size(nose_center, nose_half_w, nose_half_h, image_w, image_h))

    eye_left_pt = eye_l_face if eye_l_face is not None else eye_l_v
    eye_right_pt = eye_r_face if eye_r_face is not None else eye_r_v
    if eye_left_pt is not None:
        _set_region(regions, "eye", "left",
                    _box_from_center_and_size(eye_left_pt, eye_half_w, eye_half_h, image_w, image_h))
    if eye_right_pt is not None:
        _set_region(regions, "eye", "right",
                    _box_from_center_and_size(eye_right_pt, eye_half_w, eye_half_h, image_w, image_h))

    if eye_left_pt is not None:
        brow_left = (eye_left_pt[0], eye_left_pt[1] - face_h * 0.14)
        _set_region(regions, "eyebrow", "left",
                    _box_from_center_and_size(brow_left, brow_half_w, brow_half_h, image_w, image_h))
    if eye_right_pt is not None:
        brow_right = (eye_right_pt[0], eye_right_pt[1] - face_h * 0.14)
        _set_region(regions, "eyebrow", "right",
                    _box_from_center_and_size(brow_right, brow_half_w, brow_half_h, image_w, image_h))

    if ear_l_v is not None:
        _set_region(regions, "ear", "left",
                    _box_from_center_and_size(ear_l_v, ear_half_w, ear_half_h, image_w, image_h))
    if ear_r_v is not None:
        _set_region(regions, "ear", "right",
                    _box_from_center_and_size(ear_r_v, ear_half_w, ear_half_h, image_w, image_h))

    if mouth_center is not None:
        _set_region(regions, "mouth", "single",
                    _box_from_center_and_size(mouth_center, mouth_half_w, mouth_half_h, image_w, image_h))
    if mouth_l is not None:
        _set_region(regions, "mouth_corner", "left",
                    _box_from_center_and_size(mouth_l, mouth_corner_half_w, mouth_corner_half_h, image_w, image_h))
    if mouth_r is not None:
        _set_region(regions, "mouth_corner", "right",
                    _box_from_center_and_size(mouth_r, mouth_corner_half_w, mouth_corner_half_h, image_w, image_h))

    chin_anchor = None
    if mouth_center is not None and face_box is not None:
        chin_anchor = (mouth_center[0], face_box[3] - face_h * 0.08)
    elif face_box is not None:
        chin_anchor = ((face_box[0] + face_box[2]) * 0.5, face_box[3] - face_h * 0.08)
    if chin_anchor is not None:
        _set_region(regions, "chin", "single",
                    _box_from_center_and_size(chin_anchor, chin_half_w, chin_half_h, image_w, image_h))

    pose_paired = (
        ("shoulder", sh_l_v, sh_r_v, shoulder_half_w, shoulder_half_h),
        ("elbow",    el_l_v, el_r_v, elbow_half_w,    elbow_half_h),
        ("wrist",    wr_l_v, wr_r_v, wrist_half_w,    wrist_half_h),
        ("hip",      hp_l_v, hp_r_v, hip_half_w,      hip_half_h),
        ("knee",     kn_l_v, kn_r_v, knee_half_w,     knee_half_h),
        ("ankle",    an_l_v, an_r_v, ankle_half_w,    ankle_half_h),
    )
    for name, left_pt, right_pt, hw, hh in pose_paired:
        if left_pt is not None:
            _set_region(regions, name, "left",
                        _box_from_center_and_size(left_pt, hw, hh, image_w, image_h))
        if right_pt is not None:
            _set_region(regions, name, "right",
                        _box_from_center_and_size(right_pt, hw, hh, image_w, image_h))

    return regions


_REGION_COLORS = {
    "nose":         (255,  64, 220),
    "eye":          (255, 200,  40),
    "eyebrow":      (255, 220, 120),
    "ear":          (255, 180,  80),
    "shoulder":     ( 60, 200, 255),
    "elbow":        ( 40, 150, 255),
    "wrist":        ( 40, 255, 220),
    "hip":          (255, 128,  40),
    "knee":         (255, 255,  60),
    "ankle":        (180, 255,  60),
    "face":         ( 60, 196, 255),
    "mouth":        (255,  40, 180),
    "mouth_corner": (255,  90, 200),
    "chin":         (180,  90, 255),
    "full_person":  ( 60, 255,  96),
}


def _render_overlay_v2(
    image_uint8: np.ndarray,
    people: List[Dict[str, Any]],
    selected_index: int,
    debug_regions: Dict[str, Dict[str, Tuple[int, int, int, int]]],
    effective_solver_boxes: List[Dict[str, Any]],
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray:
    overlay = image_uint8.copy()
    for idx, person in enumerate(people):
        bbox = person.get("bbox")
        if bbox is None:
            continue
        selected = idx == selected_index
        color = (64, 255, 96) if selected else (64, 196, 255)
        _draw_box(overlay, bbox, color, thickness=3 if selected else 2)
        label = f"person {idx + 1}: {person.get('gender','unknown')}, {person.get('age') if person.get('age') is not None else person.get('life_stage','adult')}, s={person.get('subject_score',0.0):.2f}"
        _draw_text(overlay, label, (bbox[0], max(16, bbox[1])), color)

    for name, sides in debug_regions.items():
        color = _REGION_COLORS.get(name, (255, 255, 255))
        for side, box in sides.items():
            _draw_box(overlay, box, color, thickness=1)
            _draw_point(overlay, _box_center(box), color)
            side_suffix = "" if side == "single" else f" ({side[0].upper()})"
            _draw_text(overlay, f"{name}{side_suffix}", (box[0], max(14, box[1])), color)

    for entry in effective_solver_boxes:
        color = _REGION_COLORS.get(entry["name"], (255, 255, 255))
        if cv2 is not None:
            x0, y0, x1, y1 = entry["box"]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)

    clipped_crop = _clip_box(crop_box, image_uint8.shape[1], image_uint8.shape[0])
    if clipped_crop is not None:
        _draw_box(overlay, clipped_crop, (255, 0, 0), thickness=3)
        _draw_text(overlay, "crop", (clipped_crop[0], max(18, clipped_crop[1])), (255, 0, 0))

    return overlay


class CompositionCropV2:
    CATEGORY = "ESS/Image"
    FUNCTION = "crop"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "debug_overlay")

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
            "confidence_threshold": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
            "framing_mode": (("crop", "expand"), {"default": "crop"}),
            "expand_fill_mode": (("border_fill", "fill_color"), {"default": "border_fill"}),
            "expand_fill_color": ("STRING", {"default": "#000000", "multiline": False}),
            "preferred_gender": (("any", "female", "male"), {"default": "any"}),
            "target_age_min": ("INT", {"default": 18, "min": 0, "max": 120, "step": 1}),
            "target_age_max": ("INT", {"default": 35, "min": 0, "max": 120, "step": 1}),
        }
        for spec in _V2_PART_SPECS:
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
        device: str = "auto",
        confidence_threshold: float = 0.25,
        framing_mode: str = "crop",
        expand_fill_mode: str = "border_fill",
        expand_fill_color: str = "#000000",
        preferred_gender: str = "any",
        target_age_min: int = 18,
        target_age_max: int = 35,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image_uint8 = _tensor_to_uint8_image(image)
        image_bgr = image_uint8[..., ::-1].copy() if cv2 is not None else image_uint8.copy()
        image_h, image_w = image_uint8.shape[:2]
        device_choice = _resolve_device(device)
        pose_model = _load_model(_DEFAULT_POSE_MODEL, None, device_choice)
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
            person["regions_v2"] = _derive_regions_v2(person, face, image_w, image_h)

        people = sorted(people, key=lambda p: float(p.get("subject_score", 0.0)), reverse=True)
        selected = people[0]
        selected_idx = 0
        regions_all = selected["regions_v2"]

        part_settings: Dict[str, Dict[str, Any]] = {}
        for spec in _V2_PART_SPECS:
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
        effective_solver_boxes: List[Dict[str, Any]] = []
        debug_regions: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
        for spec in _V2_PART_SPECS:
            name = spec["key"]
            region_entry = regions_all.get(name) or {}
            if not region_entry:
                continue
            debug_regions[name] = dict(region_entry)
            settings = part_settings[name]
            if (not settings["enabled"]) or settings["weight"] <= 0.0:
                continue
            selected_boxes = _select_region_boxes(region_entry, settings["side"], bool(spec["paired"]))
            if not selected_boxes:
                continue
            per_box_weight = settings["weight"] / max(1, len(selected_boxes))
            for box in selected_boxes:
                effective_box = _expand_box(box, settings["margin"], image_w, image_h)
                regions_for_solver.append(
                    {
                        "name": name,
                        "weight": per_box_weight,
                        "mode": settings["mode"],
                        "box": effective_box,
                    }
                )
                effective_solver_boxes.append({"name": name, "box": effective_box})

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

        overlay = _render_overlay_v2(
            image_uint8,
            people,
            selected_idx,
            debug_regions,
            effective_solver_boxes,
            crop_box,
        )
        overlay_tensor = _uint8_image_to_tensor(overlay, image.device)
        return (output, overlay_tensor)
