import sys
import json
import base64
import asyncio
import time
import uuid
import threading
from urllib.parse import unquote
from pathlib import Path
from typing import Any

try:
    from aiohttp import web
except Exception:  # pragma: no cover - optional in some test contexts
    web = None

FaceSwapInSwapperNode = None
FaceSwapSimSwapNode = None
FaceSwapFaceFusionNode = None
_face_swapper_error = None
PersonCropToSize = None
_person_crop_error = None
CompositionCrop = None
_composition_crop_error = None

try:
    from .nodes.prompt_builder.replacements_dictionary import ReplaceDict
    from .nodes.image_processing.image_adjustments import ImageAdjustmentsNode
    from .nodes.image.segmentation_detailer import SegmentationDetailerNode
    from .nodes.detailer.ess_face_detailer import ESSFaceDetailer
    from .nodes.utils.group_reroute import GroupReroute
    from .nodes.utils.string_concat import StringConcatenate
    from .nodes.utils.label_note import LabelNote
    try:
        from .nodes.face_swapping.face_swapping import FaceSwapInSwapperNode, FaceSwapSimSwapNode, FaceSwapFaceFusionNode
    except Exception as exc:
        _face_swapper_error = exc
    from .nodes.pose.pose_mesh_editor import PoseMeshEditor
    from .nodes.utils.prefix_generator import PrefixGenerator
    from .nodes.prompt_builder.text_prompt_generator import TextPromptGenerator
    from .nodes.prompt_builder.text_prompt_replacer import TextPromptReplacer
    from .nodes.prompt_builder.prompt_template_editor import PromptTemplateEditor
    from .nodes.prompt_builder.prompt_template_editor_multi import PromptTemplateEditorMulti
    from .nodes.prompt_builder.prompt_conditioning_builder import PromptConditioningBuilder
    from .nodes.prompt_builder.scene_flow_editor import SceneFlowEditor
    try:
        from .nodes.image.person_crop_to_size import PersonCropToSize
    except Exception as exc:
        _person_crop_error = exc
    try:
        from .nodes.image.composition_crop import CompositionCrop
    except Exception as exc:
        _composition_crop_error = exc
except ImportError as exc:
    if "attempted relative import" not in str(exc):
        raise
    import importlib
    import importlib.util

    package_dir = Path(__file__).resolve().parent
    nodes_dir = package_dir / "nodes"
    local_nodes_pkg = "comfyui_ess_local_nodes"
    if local_nodes_pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            local_nodes_pkg,
            nodes_dir / "__init__.py",
            submodule_search_locations=[str(nodes_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load local nodes package from: {nodes_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[local_nodes_pkg] = module
        spec.loader.exec_module(module)

    def _import_node_attr(module_name: str, attr_name: str):
        module = importlib.import_module(f"{local_nodes_pkg}.{module_name}")
        return getattr(module, attr_name)

    ReplaceDict = _import_node_attr("prompt_builder.replacements_dictionary", "ReplaceDict")
    ImageAdjustmentsNode = _import_node_attr("image_processing.image_adjustments", "ImageAdjustmentsNode")
    SegmentationDetailerNode = _import_node_attr("image.segmentation_detailer", "SegmentationDetailerNode")
    ESSFaceDetailer = _import_node_attr("detailer.ess_face_detailer", "ESSFaceDetailer")
    GroupReroute = _import_node_attr("utils.group_reroute", "GroupReroute")
    StringConcatenate = _import_node_attr("utils.string_concat", "StringConcatenate")
    LabelNote = _import_node_attr("utils.label_note", "LabelNote")
    try:
        FaceSwapInSwapperNode = _import_node_attr("face_swapping.face_swapping", "FaceSwapInSwapperNode")
        FaceSwapSimSwapNode = _import_node_attr("face_swapping.face_swapping", "FaceSwapSimSwapNode")
        FaceSwapFaceFusionNode = _import_node_attr("face_swapping.face_swapping", "FaceSwapFaceFusionNode")
    except Exception as exc:
        _face_swapper_error = exc
    PoseMeshEditor = _import_node_attr("pose.pose_mesh_editor", "PoseMeshEditor")
    PrefixGenerator = _import_node_attr("utils.prefix_generator", "PrefixGenerator")
    TextPromptGenerator = _import_node_attr("prompt_builder.text_prompt_generator", "TextPromptGenerator")
    TextPromptReplacer = _import_node_attr("prompt_builder.text_prompt_replacer", "TextPromptReplacer")
    PromptTemplateEditor = _import_node_attr("prompt_builder.prompt_template_editor", "PromptTemplateEditor")
    PromptTemplateEditorMulti = _import_node_attr("prompt_builder.prompt_template_editor_multi", "PromptTemplateEditorMulti")
    PromptConditioningBuilder = _import_node_attr("prompt_builder.prompt_conditioning_builder", "PromptConditioningBuilder")
    SceneFlowEditor = _import_node_attr("prompt_builder.scene_flow_editor", "SceneFlowEditor")
    try:
        PersonCropToSize = _import_node_attr("image.person_crop_to_size", "PersonCropToSize")
    except Exception as exc:
        _person_crop_error = exc
    try:
        CompositionCrop = _import_node_attr("image.composition_crop", "CompositionCrop")
    except Exception as exc:
        _composition_crop_error = exc


_NODE_PREFIX = "ESS/"

_BASE_NODE_CLASS_MAPPINGS = {
    "ReplaceDict": ReplaceDict,
    "ImageAdjustments": ImageAdjustmentsNode,
    "SegmentationDetailer": SegmentationDetailerNode,
    "ESSFaceDetailer": ESSFaceDetailer,
    "GroupReroute": GroupReroute,
    "StringConcatenate": StringConcatenate,
    "LabelNote": LabelNote,
    "PoseMeshEditor": PoseMeshEditor,
    "PrefixGenerator": PrefixGenerator,
    "TextPromptGenerator": TextPromptGenerator,
    "TextPromptReplacer": TextPromptReplacer,
    "PromptTemplateEditor": PromptTemplateEditor,
    "PromptTemplateEditorMulti": PromptTemplateEditorMulti,
    "PromptConditioningBuilder": PromptConditioningBuilder,
    "SceneFlowEditor": SceneFlowEditor,
}

_BASE_NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplaceDict": "Replace Dict",
    "ImageAdjustments": "Image Adjustments",
    "SegmentationDetailer": "Segmentation Detailer",
    "ESSFaceDetailer": "Face Detailer (ESS)",
    "GroupReroute": "Group Reroute",
    "StringConcatenate": "String Concatenate",
    "LabelNote": "Label Note",
    "PoseMeshEditor": "Pose Mesh Editor",
    "PrefixGenerator": "Prefix Generator",
    "TextPromptGenerator": "Text Prompt Generator",
    "TextPromptReplacer": "Text Prompt Replacer",
    "PromptTemplateEditor": "Prompt Template Editor",
    "PromptTemplateEditorMulti": "Prompt Template Editor - Multioutput",
    "PromptConditioningBuilder": "Prompt Conditioning Builder",
    "SceneFlowEditor": "Scene Flow Editor",
}

NODE_CLASS_MAPPINGS = {
    f"{_NODE_PREFIX}{key}": value for key, value in _BASE_NODE_CLASS_MAPPINGS.items()
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"{_NODE_PREFIX}{key}": f"ESS - {value}"
    for key, value in _BASE_NODE_DISPLAY_NAME_MAPPINGS.items()
}

if FaceSwapInSwapperNode is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}FaceSwapInSwapper"] = FaceSwapInSwapperNode
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}FaceSwapInSwapper"] = "ESS - Face Swap (InSwapper)"
if FaceSwapSimSwapNode is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}FaceSwapSimSwap"] = FaceSwapSimSwapNode
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}FaceSwapSimSwap"] = "ESS - Face Swap (SimSwap)"
if FaceSwapFaceFusionNode is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}FaceSwapFaceFusion"] = FaceSwapFaceFusionNode
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}FaceSwapFaceFusion"] = "ESS - Face Swap (FaceFusion)"
elif _face_swapper_error:
    print(f"[comfyui-ess] FaceSwap node disabled: {_face_swapper_error}", file=sys.stderr)

if PersonCropToSize is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}PersonCropToSize"] = PersonCropToSize
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}PersonCropToSize"] = "ESS - Person Crop To Size"
elif _person_crop_error:
    print(f"[comfyui-ess] PersonCropToSize node disabled: {_person_crop_error}", file=sys.stderr)

if CompositionCrop is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}CompositionCrop"] = CompositionCrop
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}CompositionCrop"] = "ESS - Composition Crop"
elif _composition_crop_error:
    print(f"[comfyui-ess] CompositionCrop node disabled: {_composition_crop_error}", file=sys.stderr)

# Define categories
# Processing: ColorField
# Image Processing: ImageAdjustments


def get_custom_types():
    return {
        "SEGMENTATION_MODEL": {
            "input": object,
            "output": lambda x: callable(x) or any(callable(getattr(x, attr, None)) for attr in ("predict_mask", "segment", "predict"))
        }
    }


WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _get_rigged_meshes_dir() -> Path:
    return Path(__file__).resolve().parent / "meshes" / "human_rig"


def _list_rigged_meshes() -> list[str]:
    meshes_dir = _get_rigged_meshes_dir()
    if not meshes_dir.exists():
        return []
    allowed = {".fbx", ".obj", ".glb", ".gltf"}
    return sorted(
        [
            p.name
            for p in meshes_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed
        ]
    )


_POSE_ESTIMATOR_MODEL = None
_POSE_ESTIMATOR_NAME = None
_FACE_ANALYZER = None
_FACE_ANALYZER_READY = False


def _decode_b64_image_to_bgr(image_b64: str):
    if not isinstance(image_b64, str):
        raise ValueError("image_base64 must be a string")
    payload = image_b64.strip()
    if "base64," in payload:
        payload = payload.split("base64,", 1)[1]
    if not payload:
        raise ValueError("empty image payload")
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"invalid base64 image: {exc}") from exc
    try:
        import cv2  # local import: optional dependency in some test contexts
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"opencv/numpy unavailable: {exc}") from exc
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode image")
    return img


def _get_pose_estimator():
    global _POSE_ESTIMATOR_MODEL, _POSE_ESTIMATOR_NAME
    if _POSE_ESTIMATOR_MODEL is not None:
        return _POSE_ESTIMATOR_MODEL, _POSE_ESTIMATOR_NAME
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            f"ultralytics not available ({exc}). Install it or disable image-to-pose init."
        ) from exc

    # Prefer smallest pose models to keep first-run load time acceptable.
    candidates = ["yolo11n-pose.pt", "yolov8n-pose.pt"]
    last_exc = None
    for name in candidates:
        try:
            model = YOLO(name)
            _POSE_ESTIMATOR_MODEL = model
            _POSE_ESTIMATOR_NAME = name
            return _POSE_ESTIMATOR_MODEL, _POSE_ESTIMATOR_NAME
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"unable to load YOLO pose model: {last_exc}")


def _get_face_analyzer():
    global _FACE_ANALYZER, _FACE_ANALYZER_READY
    if _FACE_ANALYZER_READY:
        return _FACE_ANALYZER

    _FACE_ANALYZER_READY = True
    try:
        import torch
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception:
        _FACE_ANALYZER = None
        return None

    try:
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_thresh=0.30, det_size=(640, 640))
        _FACE_ANALYZER = analyzer
    except Exception:
        _FACE_ANALYZER = None
    return _FACE_ANALYZER


def _select_best_pose_instance(result: Any):
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return None
    xy = getattr(keypoints, "xy", None)
    conf = getattr(keypoints, "conf", None)
    if xy is None:
        return None
    # Convert to numpy once; works for torch tensors and ndarray.
    try:
        xy_np = xy.cpu().numpy() if hasattr(xy, "cpu") else xy
    except Exception:
        xy_np = xy
    try:
        conf_np = conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu") else conf
    except Exception:
        conf_np = conf
    xyxy = getattr(boxes, "xyxy", None)
    try:
        xyxy_np = xyxy.cpu().numpy() if xyxy is not None and hasattr(xyxy, "cpu") else xyxy
    except Exception:
        xyxy_np = xyxy
    score = getattr(boxes, "conf", None)
    try:
        score_np = score.cpu().numpy() if score is not None and hasattr(score, "cpu") else score
    except Exception:
        score_np = score

    if xy_np is None or len(xy_np) == 0:
        return None

    best_i = 0
    best_rank = float("-inf")
    for i in range(len(xy_np)):
        area = 0.0
        if xyxy_np is not None and i < len(xyxy_np):
            x1, y1, x2, y2 = [float(v) for v in xyxy_np[i]]
            area = max(0.0, (x2 - x1) * (y2 - y1))
        det_conf = float(score_np[i]) if score_np is not None and i < len(score_np) else 0.0
        kp_conf = 0.0
        if conf_np is not None and i < len(conf_np):
            row = conf_np[i]
            try:
                kp_conf = float(sum(float(v) for v in row)) / max(1, len(row))
            except Exception:
                kp_conf = 0.0
        rank = (det_conf * 0.5) + (kp_conf * 0.5) + (area * 1e-6)
        if rank > best_rank:
            best_rank = rank
            best_i = i

    row_xy = xy_np[best_i]
    row_conf = conf_np[best_i] if conf_np is not None and best_i < len(conf_np) else None
    row_box = xyxy_np[best_i] if xyxy_np is not None and best_i < len(xyxy_np) else None
    return {
        "xy": row_xy,
        "conf": row_conf,
        "box": row_box,
    }


def _extract_pose_instances(result: Any):
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None:
        return []
    xy = getattr(keypoints, "xy", None)
    conf = getattr(keypoints, "conf", None)
    if xy is None:
        return []
    try:
        xy_np = xy.cpu().numpy() if hasattr(xy, "cpu") else xy
    except Exception:
        xy_np = xy
    try:
        conf_np = conf.cpu().numpy() if conf is not None and hasattr(conf, "cpu") else conf
    except Exception:
        conf_np = conf
    xyxy = getattr(boxes, "xyxy", None)
    try:
        xyxy_np = xyxy.cpu().numpy() if xyxy is not None and hasattr(xyxy, "cpu") else xyxy
    except Exception:
        xyxy_np = xyxy
    score = getattr(boxes, "conf", None)
    try:
        score_np = score.cpu().numpy() if score is not None and hasattr(score, "cpu") else score
    except Exception:
        score_np = score

    if xy_np is None:
        return []

    out = []
    for i in range(len(xy_np)):
        row_xy = xy_np[i]
        row_conf = conf_np[i] if conf_np is not None and i < len(conf_np) else None
        row_box = xyxy_np[i] if xyxy_np is not None and i < len(xyxy_np) else None
        det_conf = float(score_np[i]) if score_np is not None and i < len(score_np) else 0.0
        out.append({
            "xy": row_xy,
            "conf": row_conf,
            "box": row_box,
            "det_conf": det_conf,
        })
    return out


def _coco17_to_openpose18(instance: dict, image_w: int, image_h: int):
    # COCO-17: nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, ...
    # OpenPose-18 order expected by editor.
    coco_to_openpose = {
        0: 0,   # nose
        6: 2,   # r_shoulder
        8: 3,   # r_elbow
        10: 4,  # r_wrist
        5: 5,   # l_shoulder
        7: 6,   # l_elbow
        9: 7,   # l_wrist
        12: 8,  # r_hip
        14: 9,  # r_knee
        16: 10, # r_ankle
        11: 11, # l_hip
        13: 12, # l_knee
        15: 13, # l_ankle
        2: 14,  # r_eye
        1: 15,  # l_eye
        4: 16,  # r_ear
        3: 17,  # l_ear
    }

    out = [None for _ in range(18)]
    xy = instance.get("xy", [])
    conf = instance.get("conf", None)
    w = max(1.0, float(image_w))
    h = max(1.0, float(image_h))

    def add_point(openpose_idx: int, x: float, y: float, c: float):
        out[openpose_idx] = {
            "x": float(x),
            "y": float(y),
            "xn": float(x) / w,
            "yn": float(y) / h,
            "confidence": float(c),
        }

    for coco_idx, openpose_idx in coco_to_openpose.items():
        if coco_idx >= len(xy):
            continue
        px, py = xy[coco_idx]
        c = float(conf[coco_idx]) if conf is not None and coco_idx < len(conf) else 1.0
        if not (px is not None and py is not None):
            continue
        add_point(openpose_idx, float(px), float(py), c)

    # Synthesize neck (OpenPose index 1) from shoulders.
    l_sh = out[5]
    r_sh = out[2]
    if l_sh and r_sh:
        add_point(
            1,
            (l_sh["x"] + r_sh["x"]) * 0.5,
            (l_sh["y"] + r_sh["y"]) * 0.5,
            min(float(l_sh["confidence"]), float(r_sh["confidence"])),
        )
    elif l_sh:
        add_point(1, l_sh["x"], l_sh["y"], float(l_sh["confidence"]) * 0.6)
    elif r_sh:
        add_point(1, r_sh["x"], r_sh["y"], float(r_sh["confidence"]) * 0.6)

    return out


def _estimate_pose_from_image_b64(image_b64: str, conf_threshold: float = 0.20):
    image = _decode_b64_image_to_bgr(image_b64)
    h, w = image.shape[:2]

    model, model_name = _get_pose_estimator()
    results = model.predict(
        source=image,
        verbose=False,
        conf=max(0.01, min(0.95, float(conf_threshold))),
        imgsz=max(256, min(1280, int(max(h, w)))),
    )
    if not results:
        raise RuntimeError("pose detector returned no results")

    instances = _extract_pose_instances(results[0])
    if not instances:
        raise RuntimeError("no person pose detected")

    face_analyzer = _get_face_analyzer()
    faces = []
    if face_analyzer is not None:
        try:
            face_objs = face_analyzer.get(image)
            for f in face_objs or []:
                bbox = getattr(f, "bbox", None)
                if bbox is None or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                raw_gender = getattr(f, "gender", None)
                gender = "unknown"
                if raw_gender == 0:
                    gender = "female"
                elif raw_gender == 1:
                    gender = "male"
                age = getattr(f, "age", None)
                age_val = int(age) if age is not None and str(age).strip() != "" else None
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "gender": gender,
                    "age": age_val,
                })
        except Exception:
            faces = []

    def match_face_for_person(pb):
        if not faces or pb is None or len(pb) != 4:
            return None
        px1, py1, px2, py2 = [float(v) for v in pb]
        pcx = (px1 + px2) * 0.5
        pcy = (py1 + py2) * 0.5
        best = None
        best_rank = float("inf")
        for face in faces:
            fx1, fy1, fx2, fy2 = face["bbox"]
            fcx, fcy = face["center"]
            inside = (px1 <= fcx <= px2) and (py1 <= fcy <= py2)
            dx = fcx - pcx
            dy = fcy - pcy
            dist = (dx * dx + dy * dy) ** 0.5
            rank = dist if inside else dist + 1e6
            if rank < best_rank:
                best_rank = rank
                best = face
        return best

    def age_to_stage(age_val):
        if age_val is None:
            return "adult"
        if age_val < 4:
            return "baby"
        if age_val < 13:
            return "child"
        if age_val < 18:
            return "teen"
        return "adult"

    def suggest_mesh(gender: str, stage: str):
        if gender == "male":
            return "male_young.fbx"
        return "female_young.fbx"

    persons = []
    for inst_idx, inst in enumerate(instances):
        openpose = _coco17_to_openpose18(inst, w, h)
        detected = sum(1 for p in openpose if p and float(p.get("confidence", 0.0)) > 0.05)
        if detected < 6:
            continue

        box = inst.get("box")
        bbox = None
        if box is not None and len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            bbox = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "xn1": x1 / max(1.0, float(w)),
                "yn1": y1 / max(1.0, float(h)),
                "xn2": x2 / max(1.0, float(w)),
                "yn2": y2 / max(1.0, float(h)),
            }

        valid_xy = [(p["x"], p["y"]) for p in openpose if p and p.get("confidence", 0.0) > 0.05]
        center_hint = None
        scale_hint = None
        if valid_xy:
            xs = [xy[0] for xy in valid_xy]
            ys = [xy[1] for xy in valid_xy]
            cx = (min(xs) + max(xs)) * 0.5
            cy = (min(ys) + max(ys)) * 0.5
            center_hint = {"x": cx / max(1.0, float(w)), "y": cy / max(1.0, float(h))}
            scale_hint = {
                "width": (max(xs) - min(xs)) / max(1.0, float(w)),
                "height": (max(ys) - min(ys)) / max(1.0, float(h)),
            }

        matched_face = match_face_for_person(inst.get("box"))
        gender = matched_face["gender"] if matched_face else "unknown"
        age_val = matched_face["age"] if matched_face else None
        stage = age_to_stage(age_val)
        mesh_name = suggest_mesh(gender, stage)

        persons.append({
            "id": f"person_{inst_idx + 1}",
            "det_confidence": float(inst.get("det_conf", 0.0)),
            "keypoints_openpose18": openpose,
            "detected_count": detected,
            "bbox": bbox,
            "center_hint": center_hint,
            "scale_hint": scale_hint,
            "gender": gender,
            "age": age_val,
            "life_stage": stage,
            "mesh_suggestion": mesh_name,
        })

    if not persons:
        raise RuntimeError("insufficient keypoints detected")

    # Left-to-right ordering tends to be better for scene placement.
    persons.sort(key=lambda p: float((p.get("center_hint") or {}).get("x", 0.5)))
    primary = persons[0]

    return {
        "ok": True,
        "model": model_name,
        "image_width": int(w),
        "image_height": int(h),
        # Backward-compatible single-person fields (primary person)
        "keypoints_openpose18": primary.get("keypoints_openpose18"),
        "bbox": primary.get("bbox"),
        "center_hint": primary.get("center_hint"),
        "scale_hint": primary.get("scale_hint"),
        "detected_count": int(primary.get("detected_count", 0)),
        "gender": primary.get("gender", "unknown"),
        "age": primary.get("age"),
        "life_stage": primary.get("life_stage", "adult"),
        "mesh_suggestion": primary.get("mesh_suggestion", "female_young.fbx"),
        # New multi-person payload
        "persons": persons,
        "person_count": len(persons),
    }


if web is not None:
    try:
        from server import PromptServer

        _POSE_INIT_PROGRESS_LOCK = threading.Lock()
        _POSE_INIT_PROGRESS: dict[str, dict[str, Any]] = {}

        def _pose_progress_set(request_id: str, **values):
            rid = str(request_id or "").strip()
            if not rid:
                return
            with _POSE_INIT_PROGRESS_LOCK:
                state = dict(_POSE_INIT_PROGRESS.get(rid) or {})
                if "log_entry" in values:
                    entry = str(values.pop("log_entry") or "").strip()
                    logs = list(state.get("logs") or [])
                    if entry:
                        if not logs or logs[-1] != entry:
                            logs.append(entry)
                    state["logs"] = logs[-120:]
                state.update(values)
                state["request_id"] = rid
                state["updated_at"] = time.time()
                _POSE_INIT_PROGRESS[rid] = state

        def _pose_progress_cb(request_id: str):
            def _cb(message: str, **extra):
                _pose_progress_set(
                    request_id,
                    active=True,
                    done=False,
                    stage=str(message or ""),
                    error="",
                    log_entry=str(message or ""),
                    **extra,
                )
            return _cb

        def _run_scene_flow_test_payload(payload):
            if not isinstance(payload, dict):
                payload = {}

            flow_script = str(payload.get("flow_script", "") or "")

            raw_seed = payload.get("seed", 0)
            try:
                seed = int(raw_seed)
            except Exception:
                seed = 0
            if seed < 0:
                seed = 0

            raw_parse = payload.get("parse_templates", True)
            if isinstance(raw_parse, str):
                parse_templates = raw_parse.strip().lower() in {"1", "true", "yes", "on", "parse"}
            else:
                parse_templates = bool(raw_parse)

            generator = SceneFlowEditor()
            positive, negative, debug_json = generator.generate(
                flow_script=flow_script,
                seed=seed,
                parse_templates=parse_templates,
            )
            debug_obj = {}
            try:
                parsed_debug = json.loads(debug_json) if isinstance(debug_json, str) else {}
                if isinstance(parsed_debug, dict):
                    debug_obj = parsed_debug
            except Exception:
                debug_obj = {"raw": str(debug_json)}

            return {
                "ok": True,
                "positive": positive,
                "negative": negative,
                "debug": debug_obj,
            }

        @PromptServer.instance.routes.get("/ess/rigged/list")
        async def ess_rigged_list(_request):
            return web.json_response({"items": _list_rigged_meshes()})

        @PromptServer.instance.routes.get("/ess/rigged/get")
        async def ess_rigged_get(request):
            raw_name = request.rel_url.query.get("name", "")
            name = unquote(raw_name).strip()
            if not name:
                return web.json_response({"error": "missing name"}, status=400)

            meshes_dir = _get_rigged_meshes_dir().resolve()
            target = (meshes_dir / name).resolve()
            if not str(target).startswith(str(meshes_dir)):
                return web.json_response({"error": "invalid name"}, status=400)
            if not target.exists() or not target.is_file():
                return web.json_response({"error": "not found"}, status=404)

            return web.Response(
                body=target.read_bytes(),
                headers={"X-Mesh-Name": target.name},
                content_type="application/octet-stream",
            )

        @PromptServer.instance.routes.post("/ess/scene_flow/test")
        async def ess_scene_flow_test(request):
            try:
                payload = await request.json()
            except Exception:
                payload = {}

            try:
                return web.json_response(_run_scene_flow_test_payload(payload))
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=500)

        @PromptServer.instance.routes.get("/ess/scene_flow/test")
        async def ess_scene_flow_test_get(request):
            payload = {
                "flow_script": request.rel_url.query.get("flow_script", ""),
                "seed": request.rel_url.query.get("seed", 0),
                "parse_templates": request.rel_url.query.get("parse_templates", True),
            }
            try:
                return web.json_response(_run_scene_flow_test_payload(payload))
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=500)

        @PromptServer.instance.routes.post("/ess/pose/init_from_image")
        async def ess_pose_init_from_image(request):
            try:
                payload = await request.json()
            except Exception:
                payload = {}

            image_b64 = ""
            if isinstance(payload, dict):
                image_b64 = str(payload.get("image_base64", "") or "")
            if not image_b64:
                return web.json_response({"ok": False, "error": "image_base64 is required"}, status=400)

            conf_threshold = 0.20
            try:
                conf_threshold = float(payload.get("confidence", 0.20))
            except Exception:
                conf_threshold = 0.20

            request_id = str(payload.get("request_id", "") or "").strip() or uuid.uuid4().hex
            _pose_progress_set(
                request_id,
                active=True,
                done=False,
                stage="Request received",
                error="",
                logs=["Request received"],
                started_at=time.time(),
            )

            mode = str(payload.get("mode", "advanced") or "advanced").strip().lower()
            try:
                if mode in {"advanced", "multihmr", "wildcamera"}:
                    from .nodes.pose.advanced_scene_init import infer_scene_from_image

                    result = await asyncio.to_thread(
                        infer_scene_from_image,
                        image_b64,
                        conf_threshold,
                        _pose_progress_cb(request_id),
                    )
                    _pose_progress_set(
                        request_id,
                        active=False,
                        done=True,
                        stage="Completed",
                        error="",
                    )
                    if isinstance(result, dict):
                        result["request_id"] = request_id
                    return web.json_response(result)
            except Exception as exc:
                if mode in {"advanced", "multihmr", "wildcamera"}:
                    _pose_progress_set(
                        request_id,
                        active=False,
                        done=True,
                        stage="Failed",
                        error=str(exc),
                    )
                    return web.json_response({"ok": False, "error": str(exc)}, status=500)

            try:
                _pose_progress_set(request_id, stage="Running fallback pose detector")
                result = await asyncio.to_thread(
                    _estimate_pose_from_image_b64,
                    image_b64,
                    conf_threshold,
                )
                _pose_progress_set(
                    request_id,
                    active=False,
                    done=True,
                    stage="Completed",
                    error="",
                )
                if isinstance(result, dict):
                    result["request_id"] = request_id
                return web.json_response(result)
            except Exception as exc:
                _pose_progress_set(
                    request_id,
                    active=False,
                    done=True,
                    stage="Failed",
                    error=str(exc),
                )
                return web.json_response({"ok": False, "error": str(exc)}, status=500)

        @PromptServer.instance.routes.get("/ess/pose/init_progress")
        async def ess_pose_init_progress(request):
            request_id = str(request.rel_url.query.get("request_id", "") or "").strip()
            if not request_id:
                return web.json_response({"ok": False, "error": "request_id is required"}, status=400)
            with _POSE_INIT_PROGRESS_LOCK:
                state = dict(_POSE_INIT_PROGRESS.get(request_id) or {})
            if not state:
                return web.json_response({
                    "ok": True,
                    "request_id": request_id,
                    "active": False,
                    "done": False,
                    "stage": "",
                    "error": "",
                })
            return web.json_response({"ok": True, **state})
    except Exception as exc:  # pragma: no cover - route registration is optional
        print(f"[comfyui-ess] Mesh routes unavailable: {exc}", file=sys.stderr)
