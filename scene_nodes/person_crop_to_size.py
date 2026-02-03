from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

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


_ULTRALYTICS_CACHE: dict[Tuple[str, str], Any] = {}
_DEFAULT_MODELS = (
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
)


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "ultralytics"


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

    available = ', '.join(_list_ultralytics_models())
    detail = '; '.join(errors) if errors else 'no additional information'
    raise RuntimeError(
        f"Ultralytics model '{model_name}' could not be loaded. Paths tried: {paths_tried}. "
        f"Available models: {available}. Details: {detail}"
    )


def _select_best_bbox(
    candidates: list[Tuple[float, float, float, float]],
    image_w: int,
    image_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not candidates:
        return None
    center_x = image_w / 2.0
    center_y = image_h / 2.0
    best = None
    best_area = -1.0
    best_dist = 0.0
    for bbox in candidates:
        x0, y0, x1, y1 = bbox
        area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
        if area <= 0:
            continue
        mid_x = (x0 + x1) / 2.0
        mid_y = (y0 + y1) / 2.0
        dist = (mid_x - center_x) ** 2 + (mid_y - center_y) ** 2
        if area > best_area or (area == best_area and dist < best_dist):
            best = bbox
            best_area = area
            best_dist = dist
    if best is None:
        return None
    x0, y0, x1, y1 = best
    return int(x0), int(y0), int(x1), int(y1)


def _apply_margin_percent(
    bbox: Tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    margin_percent: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    margin_x = int(round(width * margin_percent))
    margin_y = int(round(height * margin_percent))

    x0 = max(0, x0 - margin_x)
    y0 = max(0, y0 - margin_y)
    x1 = min(image_w, x1 + margin_x)
    y1 = min(image_h, y1 + margin_y)
    return x0, y0, x1, y1


def _crop_to_aspect(
    bbox: Tuple[int, int, int, int],
    target_w: int,
    target_h: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)
    target_ratio = target_w / target_h
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_w = max(1, int(round(height * target_ratio)))
        if new_w > width:
            new_w = width
        mid_x = (x0 + x1) / 2.0
        new_x0 = int(round(mid_x - new_w / 2.0))
        new_x1 = new_x0 + new_w
        if new_x0 < x0:
            new_x0 = x0
            new_x1 = x0 + new_w
        if new_x1 > x1:
            new_x1 = x1
            new_x0 = x1 - new_w
        return new_x0, y0, new_x1, y1
    if current_ratio < target_ratio:
        new_h = max(1, int(round(width / target_ratio)))
        if new_h > height:
            new_h = height
        mid_y = (y0 + y1) / 2.0
        new_y0 = int(round(mid_y - new_h / 2.0))
        new_y1 = new_y0 + new_h
        if new_y0 < y0:
            new_y0 = y0
            new_y1 = y0 + new_h
        if new_y1 > y1:
            new_y1 = y1
            new_y0 = y1 - new_h
        return x0, new_y0, x1, new_y1
    return bbox


def _selector_from_list(options: list[str], *, default: str = "", allow_empty: bool = True):
    if options:
        default_value = default if default and default in options else options[0]
        return (tuple(options), {"default": default_value})
    if allow_empty:
        return ("STRING", {"default": default})
    return ("STRING", {"default": default, "tooltip": "Provide a file name."})


class PersonCropToSize:
    """
    Detect the main person via an Ultralytics model and crop to a target size.
    """

    CATEGORY = "ESS/Image"
    FUNCTION = "crop_person"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "debug_overlay")

    @classmethod
    def INPUT_TYPES(cls):
        available_models = _list_ultralytics_models()
        model_selector = _selector_from_list(available_models, default="yolov8s.pt")
        head_model_selector = _selector_from_list(available_models, default="", allow_empty=True)
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 480, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 832, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                "model_name": model_selector,
                "model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional absolute path to .pt",
                    },
                ),
                "head_model_name": head_model_selector,
                "head_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Optional absolute path to head .pt",
                    },
                ),
                "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
                "confidence_threshold": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01}),
                "head_confidence_threshold": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 0.99, "step": 0.01}),
                "person_class_name": ("STRING", {"default": "person"}),
                "head_class_names": ("STRING", {"default": "head,face"}),
                "use_person_box": ("BOOLEAN", {"default": True}),
                "use_head_box": ("BOOLEAN", {"default": True}),
                "person_margin_percent": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "head_margin_percent": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "resize_mode": (("nearest", "bilinear", "bicubic", "area"), {"default": "bicubic"}),
                "align_corners": ("BOOLEAN", {"default": False}),
            },
        }

    def crop_person(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        model_name: str = "yolov8s.pt",
        model_path: str = "",
        head_model_name: str = "",
        head_model_path: str = "",
        device: str = "auto",
        confidence_threshold: float = 0.35,
        head_confidence_threshold: float = 0.35,
        person_class_name: str = "person",
        head_class_names: str = "head,face",
        use_person_box: bool = True,
        use_head_box: bool = True,
        person_margin_percent: float = 0.05,
        head_margin_percent: float = 0.05,
        resize_mode: str = "bicubic",
        align_corners: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.dim() != 4:
            raise ValueError(f"Expected image tensor in NHWC format, got shape {tuple(image.shape)}")
        if image.shape[0] != 1:
            raise ValueError("This node expects a single image (batch size 1).")

        if not model_name and not model_path:
            model_name = "yolov8s.pt"

        device_choice = _resolve_device(device)
        model = _load_model(model_name, model_path or None, device_choice)

        img = image[0].detach().cpu().clamp(0.0, 1.0)
        img_np = (img * 255.0).to(torch.uint8).numpy()

        person_boxes = []
        resolved_person_id = None
        if use_person_box:
            results = model.predict(
                source=img_np,
                conf=float(confidence_threshold),
                device=0 if device_choice == "cuda" else device_choice,
                verbose=False,
            )
            if results:
                result = results[0]
                detected = getattr(result, "boxes", None)
                if detected is not None and detected.xyxy is not None:
                    classes = detected.cls.detach().cpu().numpy().astype(int)
                    xyxy = detected.xyxy.detach().cpu().numpy()
                    names = getattr(result, "names", None) or getattr(model, "names", None)
                    if names and person_class_name:
                        for cls_id, name in names.items():
                            if isinstance(name, str) and name.lower() == person_class_name.lower():
                                resolved_person_id = int(cls_id)
                                break
                    if resolved_person_id is None:
                        resolved_person_id = 0
                    for idx, cls_id in enumerate(classes):
                        if cls_id != resolved_person_id:
                            continue
                        x0, y0, x1, y1 = xyxy[idx].tolist()
                        person_boxes.append((x0, y0, x1, y1))

        head_boxes = []
        if use_head_box and (head_model_name or head_model_path):
            head_model = _load_model(head_model_name, head_model_path or None, device_choice)
            head_results = head_model.predict(
                source=img_np,
                conf=float(head_confidence_threshold),
                device=0 if device_choice == "cuda" else device_choice,
                verbose=False,
            )
            if head_results:
                head_result = head_results[0]
                detected = getattr(head_result, "boxes", None)
                if detected is not None and detected.xyxy is not None:
                    classes = detected.cls.detach().cpu().numpy().astype(int)
                    xyxy = detected.xyxy.detach().cpu().numpy()
                    names = getattr(head_result, "names", None) or getattr(head_model, "names", None)
                    resolved_head_ids: set[int] = set()
                    if names and head_class_names:
                        for entry in head_class_names.split(","):
                            entry = entry.strip().lower()
                            if not entry:
                                continue
                            for cls_id, name in names.items():
                                if isinstance(name, str) and name.lower() == entry:
                                    resolved_head_ids.add(int(cls_id))
                                    break
                    if not resolved_head_ids:
                        resolved_head_ids.add(0)
                    for idx, cls_id in enumerate(classes):
                        if cls_id not in resolved_head_ids:
                            continue
                        x0, y0, x1, y1 = xyxy[idx].tolist()
                        head_boxes.append((x0, y0, x1, y1))

        _, img_h, img_w, _ = image.shape
        chosen_bbox = _select_best_bbox(person_boxes, img_w, img_h) if use_person_box else None
        if chosen_bbox is None and use_person_box:
            chosen_bbox = (0, 0, img_w, img_h)
        if chosen_bbox is not None and use_person_box:
            chosen_bbox = _apply_margin_percent(
                chosen_bbox,
                img_w,
                img_h,
                person_margin_percent,
            )

        head_bbox = _select_best_bbox(head_boxes, img_w, img_h) if use_head_box else None
        if head_bbox is not None and use_head_box:
            head_bbox = _apply_margin_percent(
                head_bbox,
                img_w,
                img_h,
                head_margin_percent,
            )

        crop_bbox = _crop_to_aspect_with_constraints(chosen_bbox, head_bbox, width, height, img_w, img_h)
        x0, y0, x1, y1 = crop_bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid crop region computed from detection and target size.")

        cropped = image[:, y0:y1, x0:x1, :]
        cropped_nchw = cropped.permute(0, 3, 1, 2)

        if resize_mode in ("bilinear", "bicubic"):
            resized = F.interpolate(
                cropped_nchw,
                size=(height, width),
                mode=resize_mode,
                align_corners=align_corners,
            )
        else:
            resized = F.interpolate(
                cropped_nchw,
                size=(height, width),
                mode=resize_mode,
            )

        output = resized.permute(0, 2, 3, 1)
        overlay = _render_overlay(img_np, chosen_bbox, head_bbox, crop_bbox)
        overlay_tensor = torch.from_numpy(overlay.astype(np.float32) / 255.0).unsqueeze(0).to(image.device)
        return (output, overlay_tensor)


def _crop_to_aspect_with_constraints(
    person_bbox: Optional[Tuple[int, int, int, int]],
    head_bbox: Optional[Tuple[int, int, int, int]],
    target_w: int,
    target_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    target_ratio = target_w / target_h
    if person_bbox is None and head_bbox is None:
        if image_w / image_h > target_ratio:
            crop_h = image_h
            crop_w = max(1, int(round(crop_h * target_ratio)))
        else:
            crop_w = image_w
            crop_h = max(1, int(round(crop_w / target_ratio)))
        crop_w = min(crop_w, image_w)
        crop_h = min(crop_h, image_h)
        x0 = max(0, int(round((image_w - crop_w) / 2.0)))
        y0 = max(0, int(round((image_h - crop_h) / 2.0)))
        return x0, y0, x0 + crop_w, y0 + crop_h

    base_bbox = head_bbox if person_bbox is None else person_bbox
    px0, py0, px1, py1 = base_bbox
    pw = max(px1 - px0, 1)
    ph = max(py1 - py0, 1)
    current_ratio = pw / ph

    if current_ratio > target_ratio:
        crop_w = pw
        crop_h = max(1, int(round(pw / target_ratio)))
    elif current_ratio < target_ratio:
        crop_h = ph
        crop_w = max(1, int(round(ph * target_ratio)))
    else:
        crop_w = pw
        crop_h = ph

    if crop_w > image_w or crop_h > image_h:
        if image_w / image_h > target_ratio:
            crop_h = image_h
            crop_w = max(1, int(round(crop_h * target_ratio)))
        else:
            crop_w = image_w
            crop_h = max(1, int(round(crop_w / target_ratio)))

    crop_w = min(crop_w, image_w)
    crop_h = min(crop_h, image_h)

    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(min(value, hi), lo)

    person_min_x = None
    person_max_x = None
    person_min_y = None
    person_max_y = None
    if person_bbox is not None:
        bx0, by0, bx1, by1 = person_bbox
        person_min_x = bx1 - crop_w
        person_max_x = bx0
        person_min_y = by1 - crop_h
        person_max_y = by0

    desired_x0 = int(round((px0 + px1 - crop_w) / 2.0))
    desired_y0 = int(round((py0 + py1 - crop_h) / 2.0))

    if head_bbox is not None:
        hx0, hy0, hx1, hy1 = head_bbox
        head_min_x = hx1 - crop_w
        head_max_x = hx0
        head_min_y = hy1 - crop_h
        head_max_y = hy0

        head_min_x = max(0, int(round(head_min_x)))
        head_max_x = min(image_w - crop_w, int(round(head_max_x)))
        head_min_y = max(0, int(round(head_min_y)))
        head_max_y = min(image_h - crop_h, int(round(head_max_y)))

        if head_min_x > head_max_x:
            head_min_x = head_max_x = max(0, min(image_w - crop_w, int(round((hx0 + hx1 - crop_w) / 2.0))))
        if head_min_y > head_max_y:
            head_min_y = head_max_y = max(0, min(image_h - crop_h, int(round((hy0 + hy1 - crop_h) / 2.0))))

        if person_min_x is not None:
            person_min_x = max(0, int(round(person_min_x)))
            person_max_x = min(image_w - crop_w, int(round(person_max_x)))
            person_min_y = max(0, int(round(person_min_y)))
            person_max_y = min(image_h - crop_h, int(round(person_max_y)))

            inter_min_x = max(head_min_x, person_min_x)
            inter_max_x = min(head_max_x, person_max_x)
            inter_min_y = max(head_min_y, person_min_y)
            inter_max_y = min(head_max_y, person_max_y)

            if inter_min_x <= inter_max_x:
                x0 = _clamp(desired_x0, inter_min_x, inter_max_x)
            else:
                x0 = _clamp(desired_x0, head_min_x, head_max_x)

            if inter_min_y <= inter_max_y:
                y0 = _clamp(desired_y0, inter_min_y, inter_max_y)
            else:
                y0 = _clamp(desired_y0, head_min_y, head_max_y)
        else:
            x0 = _clamp(desired_x0, head_min_x, head_max_x)
            y0 = _clamp(desired_y0, head_min_y, head_max_y)
    else:
        min_x0 = max(0, int(round(person_min_x))) if person_min_x is not None else 0
        max_x0 = min(image_w - crop_w, int(round(person_max_x))) if person_max_x is not None else image_w - crop_w
        min_y0 = max(0, int(round(person_min_y))) if person_min_y is not None else 0
        max_y0 = min(image_h - crop_h, int(round(person_max_y))) if person_max_y is not None else image_h - crop_h

        if min_x0 > max_x0:
            min_x0, max_x0 = 0, max(0, image_w - crop_w)
        if min_y0 > max_y0:
            min_y0, max_y0 = 0, max(0, image_h - crop_h)

        x0 = _clamp(desired_x0, min_x0, max_x0)
        y0 = _clamp(desired_y0, min_y0, max_y0)

    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return int(x0), int(y0), int(x1), int(y1)


def _render_overlay(
    image_uint8: np.ndarray,
    person_bbox: Optional[Tuple[int, int, int, int]],
    head_bbox: Optional[Tuple[int, int, int, int]],
    crop_bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    overlay = image_uint8.copy()
    if person_bbox is not None:
        _draw_box(overlay, person_bbox, (0, 255, 0), thickness=2)
    if head_bbox is not None:
        _draw_box(overlay, head_bbox, (0, 128, 255), thickness=2)
    _draw_box(overlay, crop_bbox, (255, 0, 0), thickness=2)
    return overlay


def _draw_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x0, y0, x1, y1 = bbox
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return
    for t in range(thickness):
        top = y0 + t
        bottom = y1 - 1 - t
        left = x0 + t
        right = x1 - 1 - t
        if top <= bottom:
            image[top, left:right + 1] = color
            image[bottom, left:right + 1] = color
        if left <= right:
            image[top:bottom + 1, left] = color
            image[top:bottom + 1, right] = color
