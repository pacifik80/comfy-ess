from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
try:
    from ultralytics import YOLO  # type: ignore
    _ULTRA_IMPORT_ERROR: Optional[str] = None
except Exception as _ultra_exc:  # pragma: no cover
    YOLO = None
    try:
        _ULTRA_IMPORT_ERROR = f"{type(_ultra_exc).__name__}: {_ultra_exc}"
    except Exception:
        _ULTRA_IMPORT_ERROR = "unknown import error"
try:
    import folder_paths  # type: ignore
except ImportError:  # pragma: no cover
    folder_paths = None
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
_ULTRALYTICS_CACHE: dict[Tuple[str, str], Any] = {}
_DEFAULT_MODELS = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "yolo11n-pose.pt",
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
)

# --- Ultralytics model loading -------------------------------------------------
def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "ultralytics"
def _ultralytics_search_roots() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    default_root = _default_model_dir()
    roots.append(default_root)
    seen.add(str(default_root).lower())

    if folder_paths is not None:
        try:
            for item in folder_paths.get_folder_paths("ultralytics"):
                try:
                    path_obj = Path(item)
                except Exception:
                    continue
                key = str(path_obj).lower()
                if key in seen:
                    continue
                seen.add(key)
                roots.append(path_obj)
        except Exception:
            pass

        models_dir = getattr(folder_paths, "models_dir", None)
        if models_dir:
            try:
                model_root = Path(models_dir) / "ultralytics"
                key = str(model_root).lower()
                if key not in seen:
                    seen.add(key)
                    roots.append(model_root)
            except Exception:
                pass

    return roots
def _list_ultralytics_models() -> list[str]:
    models: set[str] = set()
    for root in _ultralytics_search_roots():
        if not root.exists():
            continue
        for file in root.rglob("*.pt"):
            try:
                rel = file.relative_to(root)
                models.add(rel.as_posix())
            except Exception:
                models.add(file.name)
    if not models:
        models.update(_DEFAULT_MODELS)
    return sorted(models)
def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
def _candidate_model_paths(model_name: str, custom_path: Optional[str]) -> list[Path]:
    candidates: list[Path] = []
    if custom_path:
        try:
            candidates.append(Path(custom_path))
        except Exception:
            pass

    for root in _ultralytics_search_roots():
        candidates.append(root / model_name)

    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("ultralytics", model_name)
            if path:
                candidates.append(Path(path))
        except Exception:
            pass

    candidates.append(Path(model_name))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique
def _load_model(model_name: str, model_path: Optional[str], device: str) -> Any:
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is required for PersonCropToSize. Install 'ultralytics'. "
            f"Details: {_ULTRA_IMPORT_ERROR or 'no import details'}"
        )

    cache_key = (model_name or "", device)
    if model_path:
        cache_key = (model_path, device)
    cached = _ULTRALYTICS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    errors: list[str] = []
    paths_tried: list[str] = []
    for candidate in _candidate_model_paths(model_name, model_path):
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        paths_tried.append(str(resolved))
        try:
            model = YOLO(str(resolved))
        except Exception as exc:
            errors.append(f"{resolved}: {exc}")
            continue
        _ULTRALYTICS_CACHE[cache_key] = model
        return model

    if model_name and not model_path:
        try:
            model = YOLO(str(model_name))
            _ULTRALYTICS_CACHE[cache_key] = model
            return model
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    available = ', '.join(_list_ultralytics_models())
    detail = '; '.join(errors) if errors else 'no additional information'
    raise RuntimeError(
        f"Ultralytics model '{model_name}' could not be loaded. Paths tried: {paths_tried}. "
        f"Available models: {available}. Details: {detail}"
    )
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None
_FACE_ANALYZER = None
_FACE_ANALYZER_READY = False

# --- Geometry / detection / crop-render helpers --------------------------------
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

    start_x = x0 + left
    start_y = y0 + top

    # Mirror real content outward: continues textures/gradients far more
    # naturally than replicate, with no flat band or smear.
    if fill_mode == "reflect":
        padded = cv2.copyMakeBorder(
            image_uint8, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101
        )
        return padded[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()

    # Content-aware soft fill: extend edges/colors into the empty area, then
    # progressively defocus + grain outward so the synthetic region reads as
    # out-of-focus background that blends into the real photo.
    if fill_mode == "inpaint":
        return _inpaint_extend_region(
            image_uint8, top, bottom, left, right, start_x, start_y, crop_w, crop_h
        )

    # Default ("border_fill"): edge-replicate + global blur of the border band.
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
    return blurred[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()
def _inpaint_extend_region(
    image_uint8: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    start_x: int,
    start_y: int,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    """Outpaint the out-of-bounds border with cv2 inpainting, feathered + defocused."""
    image_h, image_w = image_uint8.shape[:2]

    # Replicate seed gives inpaint a stable colour at the real/synthetic seam.
    padded = cv2.copyMakeBorder(
        image_uint8, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE
    )
    pad_h, pad_w = padded.shape[:2]

    # Synthetic region = everything outside the original image rectangle.
    synth = np.full((pad_h, pad_w), 255, dtype=np.uint8)
    synth[top:top + image_h, left:left + image_w] = 0

    radius = max(3, int(round(min(pad_w, pad_h) * 0.02)))
    inpainted = cv2.inpaint(padded, synth, radius, cv2.INPAINT_TELEA)

    # Distance from the real region (0 at the seam -> 1 deep in the fill) drives
    # how much we defocus and grain, so the seam stays sharp and the far edge
    # dissolves into soft background.
    dist = cv2.distanceTransform((synth > 0).astype(np.uint8), cv2.DIST_L2, 3)
    max_span = max(1.0, float(min(image_w, image_h)) * 0.5)
    alpha = np.clip(dist / max_span, 0.0, 1.0)[..., None].astype(np.float32)

    blur_kernel = max(9, int(round(min(pad_w, pad_h) * 0.05)))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(inpainted, (blur_kernel, blur_kernel), 0)

    src = inpainted.astype(np.float32)
    soft = src * (1.0 - alpha) + blurred.astype(np.float32) * alpha

    # Faint grain only where defocused, to avoid a too-clean synthetic patch.
    grain = np.random.normal(0.0, 4.0, soft.shape).astype(np.float32) * alpha
    soft = np.clip(soft + grain, 0.0, 255.0).astype(np.uint8)

    # Keep the real pixels exactly intact.
    soft[top:top + image_h, left:left + image_w] = image_uint8
    return soft[start_y:start_y + crop_h, start_x:start_x + crop_w].copy()


def _crop_fill_mask(
    crop_box: Tuple[int, int, int, int],
    image_w: int,
    image_h: int,
) -> np.ndarray:
    """Hard float32 mask (crop_h x crop_w): 1.0 = synthetic out-of-bounds fill, 0.0 = real.

    White (1.0) marks invented pixels that fall outside the source image and should be
    fully regenerated by a downstream inpaint/outpaint model; black (0.0) is real content
    to keep. Purely geometric and *unfeathered* on purpose: any feather is applied later
    at output resolution and only ever bleeds inward into real content, so the synthetic
    region always stays solidly 1.0 (see _fill_mask_tensor).
    """
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)

    mask = np.ones((crop_h, crop_w), dtype=np.float32)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, image_w), min(y1, image_h)
    if sx1 > sx0 and sy1 > sy0:
        mask[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = 0.0
    return mask


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
