"""Face swapping nodes based on ONNX InsightFace swappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import inspect

import numpy as np
import torch
import sys

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import insightface  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore
    from insightface.model_zoo import get_model as insight_get_model  # type: ignore
    from insightface.model_zoo.inswapper import INSwapper  # type: ignore
    from insightface.utils import face_align  # type: ignore
except ImportError:  # pragma: no cover
    insightface = None
    FaceAnalysis = None
    insight_get_model = None
    INSwapper = None
    face_align = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None

try:
    from gfpgan import GFPGANer  # type: ignore
except Exception:  # pragma: no cover
    GFPGANer = None

try:
    import folder_paths  # type: ignore
except ImportError:  # pragma: no cover
    folder_paths = None


@dataclass
class DetectedFace:
    bbox: np.ndarray
    face: object


_MODEL_FOLDER_KEYS = ("insightface", "face_restore")
_ROPE_DEFAULT_DIR = Path("C:/content/roop")
_DEFAULT_MODEL_SUBDIR = Path("ess") / "faceswap_models"
_DEFAULT_RESTORE_SUBDIR = Path("ess") / "face_restore"
_MIN_MODEL_BYTES = 1_000_000

_INSIGHTFACE_PROFILES = ("buffalo_l", "buffalo_m", "buffalo_s", "antelopev2")
_DEFAULT_SWAP_MODELS = (
    "inswapper_128.onnx",
    "inswapper_128.fp16.onnx",
    "inswapper_128_fp16.onnx",
    "reswapper_256.onnx",
)
_SWAP_NAME_HINTS = ("inswapper", "reswapper", "hyperswap", "ghost", "blendswap", "hififace", "uniface", "simswap", "swapper")
_DETECTOR_CACHE: dict[Tuple[str, str, float, int, int], object] = {}
_SWAPPER_CACHE: dict[Tuple[str, str, str], object] = {}
_SESSION_CACHE: dict[Tuple[str, str], object] = {}
_RESTORER_CACHE: dict[Tuple[str, str, int, str, int], object] = {}
_DEBUG_LOGGED_MODELS: set[str] = set()
_RESTORE_WARNING_LOGGED = False

# Model registry based on the user-provided research list.
_INSWAPPER_REGISTRY = {
    "inswapper_128": "inswapper_128.onnx",
    "inswapper_128_fp16": "inswapper_128.fp16.onnx",
    "reswapper_256": "reswapper_256.onnx",
}

_FACEFUSION_REGISTRY = {
    "hyperswap_1a_256": "hyperswap_1a_256.onnx",
    "hyperswap_1b_256": "hyperswap_1b_256.onnx",
    "hyperswap_1c_256": "hyperswap_1c_256.onnx",
    "ghost_1_256": "ghost_1_256.onnx",
    "ghost_2_256": "ghost_2_256.onnx",
    "ghost_3_256": "ghost_3_256.onnx",
    "blendswap_256": "blendswap_256.onnx",
    "hififace_unofficial_256": "hififace_unofficial_256.onnx",
    "uniface_256": "uniface_256.onnx",
}

_FACEFUSION_MODEL_META = {
    "blendswap_256.onnx": {
        "type": "blendface",
        "template": "ffhq",
        "size": (512, 256),
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
    },
}
for _name in _FACEFUSION_REGISTRY.values():
    _FACEFUSION_MODEL_META.setdefault(
        _name,
        {
            "type": "blendface",
            "template": "ffhq",
            "size": (512, 256),
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0],
        },
    )

_FACEFUSION_TEMPLATES = {
    "arcface_v1": np.array(
        [
            [39.7300, 51.1380],
            [72.2700, 51.1380],
            [56.0000, 68.4930],
            [42.4630, 87.0100],
            [69.5370, 87.0100],
        ],
        dtype=np.float32,
    ),
    "arcface_v2": np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    ),
    "ffhq": np.array(
        [
            [192.98138, 239.94708],
            [318.90277, 240.1936],
            [256.63416, 314.01935],
            [201.26117, 371.41043],
            [313.08905, 371.15118],
        ],
        dtype=np.float32,
    ),
}

_SIMSWAP_REGISTRY = {
    "simswap_256": "simswap_256.onnx",
    "simswap_unofficial_512": "simswap_unofficial_512.onnx",
}

_DEFAULT_RESTORE_MODELS = (
    "GFPGANv1.4.pth",
    "GFPGANv1.3.pth",
    "GFPGANv1.2.pth",
)

_MODEL_URLS: dict[str, List[str]] = {
    "inswapper_128.onnx": [
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx",
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx?download=true",
    ],
    "inswapper_128.fp16.onnx": [
        "https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx?download=true",
    ],
    "inswapper_128_fp16.onnx": [
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128_fp16.onnx",
    ],
    "blendswap_256.onnx": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/blendswap_256.onnx",
    ],
    "ghost_1_256.onnx": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_1_256.onnx?download=true",
    ],
    "ghost_2_256.onnx": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_2_256.onnx?download=true",
    ],
    "ghost_3_256.onnx": [
        "https://huggingface.co/facefusion/models-3.0.0/resolve/main/ghost_3_256.onnx",
    ],
    "hififace_unofficial_256.onnx": [
        "https://huggingface.co/facefusion/models-3.1.0/resolve/main/hififace_unofficial_256.onnx",
    ],
    "hyperswap_1a_256.onnx": [
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
    ],
    "hyperswap_1b_256.onnx": [
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1b_256.onnx?download=true",
    ],
    "hyperswap_1c_256.onnx": [
        "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx?download=true",
    ],
    "reswapper_256.onnx": [
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/reswapper_256.onnx",
    ],
    "uniface_256.onnx": [
        "https://huggingface.co/netrunner-exe/Insight-Swap-models-onnx/resolve/main/uniface_256.onnx",
    ],
    "simswap_256.onnx": [
        "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/simswap_256.onnx",
        "https://huggingface.co/netrunner-exe/SimSwap-models/resolve/1afe43249c4d4b5d856cdd1a3708edf43fa830fd/simswap_256.onnx?download=true",
    ],
    "simswap_unofficial_512.onnx": [
        "https://huggingface.co/LPDoctor/simswap_official/resolve/main/simswap_unofficial_512.onnx",
    ],
    "GFPGANv1.4.pth": [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth",
    ],
    "GFPGANv1.3.pth": [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
    ],
    "GFPGANv1.2.pth": [
        "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.2.pth",
    ],
}


class FaceSwapInSwapperNode:
    CATEGORY = "ESS/FaceSwapping"
    FUNCTION = "swap"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("swapped", "restored", "debug_source", "debug_target")

    @classmethod
    def INPUT_TYPES(cls):
        detector_options = list(_INSIGHTFACE_PROFILES)
        registry_models = _registry_model_names(_INSWAPPER_REGISTRY)
        swap_selector = tuple(registry_models) if registry_models else ("inswapper_128",)
        default_swap = _default_model_name(swap_selector)
        restore_models = _list_restore_models()
        restore_selector = tuple(restore_models) if restore_models else ("GFPGANv1.4.pth",)
        default_restore = _default_restore_model(restore_selector)

        return {
            "required": {
                "target_image": ("IMAGE", {"tooltip": "Target image that receives the swapped face."}),
                "source_image": ("IMAGE", {"tooltip": "Source image that provides the face identity."}),
            },
            "optional": {
                "detector_profile": (
                    tuple(detector_options),
                    {"default": detector_options[0], "tooltip": "InsightFace detector profile (buffalo/antelope variants)."},
                ),
                "swap_model": (swap_selector, {"default": default_swap, "tooltip": "InSwapper ONNX model to use."}),
                "device": (
                    ("auto", "cuda", "cpu"),
                    {"default": "auto", "tooltip": "Device for detection/swapper. Auto uses CUDA when available."},
                ),
                "det_thresh": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.99,
                        "step": 0.01,
                        "tooltip": "Detector confidence threshold. Higher finds fewer faces.",
                    },
                ),
                "det_size": (
                    "INT",
                    {
                        "default": 640,
                        "min": 128,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Detector input size (square). Larger helps small faces but is slower.",
                    },
                ),
                "det_max_num": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Max faces to detect (0 means no limit if supported).",
                    },
                ),
                "gender": (("any", "male", "female"), {"default": "any", "tooltip": "Filter faces by gender metadata."}),
                "source_face_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Index of source face after sorting by size (and gender).",
                    },
                ),
                "target_face_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Index of target face after sorting by size (and gender).",
                    },
                ),
                "blend_mode": (
                    ("swapper", "mask"),
                    {"default": "swapper", "tooltip": "Use model paste-back or a custom mask blend."},
                ),
                "mask_type": (
                    ("landmarks", "ellipse"),
                    {"default": "landmarks", "tooltip": "Mask shape for blend_mode=mask."},
                ),
                "mask_expand": (
                    "INT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Expand mask outward in pixels.",
                    },
                ),
                "mask_blur": (
                    "INT",
                    {
                        "default": 13,
                        "min": 1,
                        "max": 199,
                        "step": 2,
                        "tooltip": "Mask blur radius (odd values only).",
                    },
                ),
                "swap_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Swap blend strength. Lower mixes with the original.",
                    },
                ),
                "face_restore": ("BOOLEAN", {"default": False, "tooltip": "Enable GFPGAN face restoration."}),
                "face_restore_model": (
                    restore_selector,
                    {"default": default_restore, "tooltip": "GFPGAN model weights to use."},
                ),
                "face_restore_visibility": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend restored face back into the swapped result.",
                    },
                ),
                "face_restore_upscale": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "GFPGAN upscale factor.",
                    },
                ),
                "face_restore_arch": (
                    ("clean", "original"),
                    {"default": "clean", "tooltip": "GFPGAN architecture variant."},
                ),
                "face_restore_channel_multiplier": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "GFPGAN channel multiplier; higher uses more VRAM.",
                    },
                ),
                "face_restore_only_center": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Restore only the most central face."},
                ),
            },
        }

    def swap(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        detector_profile: str = "buffalo_l",
        swap_model: str = "inswapper_128",
        device: str = "auto",
        det_thresh: float = 0.5,
        det_size: int = 640,
        det_max_num: int = 0,
        gender: str = "any",
        source_face_index: int = 0,
        target_face_index: int = 0,
        blend_mode: str = "swapper",
        mask_type: str = "landmarks",
        mask_expand: int = 8,
        mask_blur: int = 13,
        swap_strength: float = 1.0,
        face_restore: bool = False,
        face_restore_model: str = "GFPGANv1.4.pth",
        face_restore_visibility: float = 1.0,
        face_restore_upscale: int = 1,
        face_restore_arch: str = "clean",
        face_restore_channel_multiplier: int = 2,
        face_restore_only_center: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cv2 is None:
            raise RuntimeError("FaceSwapInSwapperNode requires OpenCV (cv2). Please install opencv-python.")
        if insightface is None or FaceAnalysis is None or insight_get_model is None:
            raise RuntimeError("FaceSwapInSwapperNode requires the insightface package to be installed.")
        if target_image.dim() != 4 or source_image.dim() != 4:
            raise ValueError("FaceSwapInSwapperNode expects batched IMAGE tensors (B,H,W,C).")

        device_choice = _resolve_device(device)
        providers = _providers_for_device(device_choice)

        detector = _load_detector(detector_profile, device_choice, det_thresh, det_size, det_max_num, providers)
        if swap_model not in _INSWAPPER_REGISTRY:
            available = ", ".join(sorted(_INSWAPPER_REGISTRY.keys()))
            raise RuntimeError(
                f"Unsupported InSwapper model '{swap_model}'. Available: {available}. "
                "Use ESS/FaceSwapFaceFusion for HyperSwap/Ghost/BlendSwap/HiFiFace/UniFace."
            )
        resolved_name = _resolve_swap_model_name(swap_model, _INSWAPPER_REGISTRY)
        swapper = _load_swapper(resolved_name, str(_default_model_dir()), device_choice, providers)

        batch = target_image.shape[0]
        sources = source_image.shape[0]
        outputs: List[torch.Tensor] = []
        restored_outputs: List[torch.Tensor] = []
        debug_sources: List[torch.Tensor] = []
        debug_targets: List[torch.Tensor] = []
        for batch_index in range(batch):
            source_index = min(batch_index, sources - 1)
            target_img = _tensor_to_image(target_image[batch_index])
            source_img = _tensor_to_image(source_image[source_index])

            target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)

            target_face = _detect_face(detector, target_bgr, target_face_index, gender)
            source_face = _detect_face(detector, source_bgr, source_face_index, gender)

            if target_face is None or source_face is None:
                outputs.append(target_image[batch_index])
                restored_outputs.append(target_image[batch_index])
                debug_sources.append(source_image[source_index])
                debug_targets.append(target_image[batch_index])
                continue

            swapped_bgr = _swapper_get(swapper, target_bgr, target_face.face, source_face.face)
            debug_mask = None
            if blend_mode == "mask":
                mask = _face_mask_from_detected(target_face, target_bgr.shape[:2], mask_expand, mask_blur, mask_type)
                if mask is not None:
                    if swap_strength < 1.0:
                        mask = mask * swap_strength
                    swapped_bgr = _blend_with_mask(target_bgr, swapped_bgr, mask)
                    debug_mask = mask
                elif swap_strength < 1.0:
                    swapped_bgr = _blend_with_strength(target_bgr, swapped_bgr, swap_strength)
            elif swap_strength < 1.0:
                swapped_bgr = _blend_with_strength(target_bgr, swapped_bgr, swap_strength)

            pre_restore_bgr = swapped_bgr.copy()
            if face_restore:
                swapped_bgr = _restore_faces(
                    swapped_bgr,
                    face_restore_model,
                    face_restore_visibility,
                    face_restore_upscale,
                    face_restore_arch,
                    face_restore_channel_multiplier,
                    face_restore_only_center,
                    device_choice,
                )

            swapped_rgb = cv2.cvtColor(pre_restore_bgr, cv2.COLOR_BGR2RGB)
            outputs.append(_image_to_tensor(swapped_rgb, device=target_image.device))
            restored_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)
            restored_outputs.append(_image_to_tensor(restored_rgb, device=target_image.device))

            source_mask = None
            if blend_mode == "mask":
                source_mask = _face_mask_from_detected(source_face, source_bgr.shape[:2], mask_expand, mask_blur, mask_type)
            debug_source = _draw_face_debug(source_bgr, source_face, source_mask)
            debug_target = _draw_face_debug(target_bgr, target_face, debug_mask)
            debug_sources.append(_image_to_tensor(cv2.cvtColor(debug_source, cv2.COLOR_BGR2RGB), device=target_image.device))
            debug_targets.append(_image_to_tensor(cv2.cvtColor(debug_target, cv2.COLOR_BGR2RGB), device=target_image.device))

        return (
            torch.stack(outputs, dim=0),
            torch.stack(restored_outputs, dim=0),
            torch.stack(debug_sources, dim=0),
            torch.stack(debug_targets, dim=0),
        )


class FaceSwapSimSwapNode:
    CATEGORY = "ESS/FaceSwapping"
    FUNCTION = "swap"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("swapped", "restored", "debug_source", "debug_target")

    @classmethod
    def INPUT_TYPES(cls):
        detector_options = list(_INSIGHTFACE_PROFILES)
        registry_models = _registry_model_names(_SIMSWAP_REGISTRY)
        swap_selector = tuple(registry_models) if registry_models else ("simswap_256",)
        default_swap = swap_selector[0]
        restore_models = _list_restore_models()
        restore_selector = tuple(restore_models) if restore_models else ("GFPGANv1.4.pth",)
        default_restore = _default_restore_model(restore_selector)
        return {
            "required": {
                "target_image": ("IMAGE", {"tooltip": "Target image that receives the swapped face."}),
                "source_image": ("IMAGE", {"tooltip": "Source image that provides the face identity."}),
            },
            "optional": {
                "detector_profile": (
                    tuple(detector_options),
                    {"default": detector_options[0], "tooltip": "InsightFace detector profile (buffalo/antelope variants)."},
                ),
                "swap_model": (swap_selector, {"default": default_swap, "tooltip": "SimSwap ONNX model to use."}),
                "device": (
                    ("auto", "cuda", "cpu"),
                    {"default": "auto", "tooltip": "Device for detection/onnxruntime. Auto uses CUDA when available."},
                ),
                "det_thresh": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.99,
                        "step": 0.01,
                        "tooltip": "Detector confidence threshold. Higher finds fewer faces.",
                    },
                ),
                "det_size": (
                    "INT",
                    {
                        "default": 640,
                        "min": 128,
                        "max": 2048,
                        "step": 32,
                        "tooltip": "Detector input size (square). Larger helps small faces but is slower.",
                    },
                ),
                "det_max_num": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Max faces to detect (0 means no limit if supported).",
                    },
                ),
                "gender": (("any", "male", "female"), {"default": "any", "tooltip": "Filter faces by gender metadata."}),
                "source_face_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Index of source face after sorting."},
                ),
                "target_face_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Index of target face after sorting."},
                ),
                "mask_blur": (
                    "INT",
                    {"default": 15, "min": 1, "max": 199, "step": 2, "tooltip": "Mask blur radius (odd values)."},
                ),
                "mask_expand": (
                    "INT",
                    {"default": 6, "min": 0, "max": 200, "step": 1, "tooltip": "Expand mask outward in pixels."},
                ),
                "swap_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Swap blend strength. Lower mixes with the original.",
                    },
                ),
                "face_restore": ("BOOLEAN", {"default": False, "tooltip": "Enable GFPGAN face restoration."}),
                "face_restore_model": (
                    restore_selector,
                    {"default": default_restore, "tooltip": "GFPGAN model weights to use."},
                ),
                "face_restore_visibility": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Blend restored face back into the swapped result.",
                    },
                ),
                "face_restore_upscale": (
                    "INT",
                    {"default": 1, "min": 1, "max": 4, "step": 1, "tooltip": "GFPGAN upscale factor."},
                ),
                "face_restore_arch": (
                    ("clean", "original"),
                    {"default": "clean", "tooltip": "GFPGAN architecture variant."},
                ),
                "face_restore_channel_multiplier": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "tooltip": "GFPGAN channel multiplier; higher uses more VRAM.",
                    },
                ),
                "face_restore_only_center": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Restore only the most central face."},
                ),
            },
        }

    def swap(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        detector_profile: str = "buffalo_l",
        swap_model: str = "simswap_256",
        device: str = "auto",
        det_thresh: float = 0.5,
        det_size: int = 640,
        det_max_num: int = 0,
        gender: str = "any",
        source_face_index: int = 0,
        target_face_index: int = 0,
        mask_blur: int = 15,
        mask_expand: int = 6,
        swap_strength: float = 1.0,
        face_restore: bool = False,
        face_restore_model: str = "GFPGANv1.4.pth",
        face_restore_visibility: float = 1.0,
        face_restore_upscale: int = 1,
        face_restore_arch: str = "clean",
        face_restore_channel_multiplier: int = 2,
        face_restore_only_center: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cv2 is None:
            raise RuntimeError("FaceSwapSimSwapNode requires OpenCV (cv2). Please install opencv-python.")
        if insightface is None or FaceAnalysis is None:
            raise RuntimeError("FaceSwapSimSwapNode requires the insightface package to be installed.")
        if ort is None:
            raise RuntimeError("FaceSwapSimSwapNode requires onnxruntime to be installed.")
        if target_image.dim() != 4 or source_image.dim() != 4:
            raise ValueError("FaceSwapSimSwapNode expects batched IMAGE tensors (B,H,W,C).")

        device_choice = _resolve_device(device)
        providers = _providers_for_device(device_choice)
        detector = _load_detector(detector_profile, device_choice, det_thresh, det_size, det_max_num, providers)

        resolved_name = _resolve_swap_model_name(swap_model, _SIMSWAP_REGISTRY)
        model_path = _resolve_model_path(resolved_name, str(_default_model_dir()))

        session = ort.InferenceSession(str(model_path), providers=providers)
        input_meta = session.get_inputs()

        batch = target_image.shape[0]
        sources = source_image.shape[0]
        outputs: List[torch.Tensor] = []
        restored_outputs: List[torch.Tensor] = []
        debug_sources: List[torch.Tensor] = []
        debug_targets: List[torch.Tensor] = []
        for batch_index in range(batch):
            source_index = min(batch_index, sources - 1)
            target_img = _tensor_to_image(target_image[batch_index])
            source_img = _tensor_to_image(source_image[source_index])

            target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)

            target_face = _detect_face(detector, target_bgr, target_face_index, gender)
            source_face = _detect_face(detector, source_bgr, source_face_index, gender)
            if target_face is None or source_face is None:
                outputs.append(target_image[batch_index])
                restored_outputs.append(target_image[batch_index])
                debug_sources.append(source_image[source_index])
                debug_targets.append(target_image[batch_index])
                continue

            output_size = _infer_simswap_size(input_meta)
            src_aligned, _ = _align_face(source_bgr, source_face.face, output_size)
            tgt_aligned, inv_m = _align_face(target_bgr, target_face.face, output_size)

            feed = _build_simswap_feed(input_meta, tgt_aligned, src_aligned, source_face.face)
            outs = session.run(None, feed)
            swapped_out = _select_swap_output(outs)
            swapped = _tensor_to_bgr(swapped_out)

            if swapped.shape[:2] == target_bgr.shape[:2]:
                pasted = swapped
                debug_mask = _ellipse_mask_from_bbox(target_face.bbox, target_bgr.shape[:2], mask_expand, mask_blur)
            else:
                pasted, debug_mask = _paste_back(target_bgr, swapped, inv_m, mask_expand, mask_blur)
            if swap_strength < 1.0:
                pasted = _blend_with_strength(target_bgr, pasted, swap_strength)

            pre_restore_bgr = pasted.copy()
            if face_restore:
                pasted = _restore_faces(
                    pasted,
                    face_restore_model,
                    face_restore_visibility,
                    face_restore_upscale,
                    face_restore_arch,
                    face_restore_channel_multiplier,
                    face_restore_only_center,
                    device_choice,
                )

            swapped_rgb = cv2.cvtColor(pre_restore_bgr, cv2.COLOR_BGR2RGB)
            outputs.append(_image_to_tensor(swapped_rgb, device=target_image.device))
            restored_rgb = cv2.cvtColor(pasted, cv2.COLOR_BGR2RGB)
            restored_outputs.append(_image_to_tensor(restored_rgb, device=target_image.device))

            debug_source = _draw_face_debug(source_bgr, source_face, None)
            debug_target = _draw_face_debug(target_bgr, target_face, debug_mask)
            debug_sources.append(_image_to_tensor(cv2.cvtColor(debug_source, cv2.COLOR_BGR2RGB), device=target_image.device))
            debug_targets.append(_image_to_tensor(cv2.cvtColor(debug_target, cv2.COLOR_BGR2RGB), device=target_image.device))

        return (
            torch.stack(outputs, dim=0),
            torch.stack(restored_outputs, dim=0),
            torch.stack(debug_sources, dim=0),
            torch.stack(debug_targets, dim=0),
        )


class FaceSwapFaceFusionNode:
    CATEGORY = "ESS/FaceSwapping"
    FUNCTION = "swap"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        detector_options = list(_INSIGHTFACE_PROFILES)
        registry_models = _registry_model_names(_FACEFUSION_REGISTRY)
        swap_selector = tuple(registry_models) if registry_models else ("hyperswap_1a_256",)
        default_swap = swap_selector[0]
        return {
            "required": {
                "target_image": ("IMAGE",),
                "source_image": ("IMAGE",),
            },
            "optional": {
                "detector_profile": (tuple(detector_options), {"default": detector_options[0]}),
                "swap_model": (swap_selector, {"default": default_swap}),
                "device": (("auto", "cuda", "cpu"), {"default": "auto"}),
                "det_thresh": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.99, "step": 0.01}),
                "gender": (("any", "male", "female"), {"default": "any"}),
                "source_face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "target_face_index": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
                "blend_mode": (("swapper", "mask"), {"default": "swapper"}),
                "mask_expand": ("INT", {"default": 8, "min": 0, "max": 200, "step": 1}),
                "mask_blur": ("INT", {"default": 13, "min": 1, "max": 199, "step": 2}),
            },
        }

    def swap(
        self,
        target_image: torch.Tensor,
        source_image: torch.Tensor,
        detector_profile: str = "buffalo_l",
        swap_model: str = "hyperswap_1a_256",
        device: str = "auto",
        det_thresh: float = 0.5,
        gender: str = "any",
        source_face_index: int = 0,
        target_face_index: int = 0,
        blend_mode: str = "swapper",
        mask_expand: int = 8,
        mask_blur: int = 13,
    ) -> Tuple[torch.Tensor]:
        if cv2 is None:
            raise RuntimeError("FaceSwapFaceFusionNode requires OpenCV (cv2). Please install opencv-python.")
        if insightface is None or FaceAnalysis is None:
            raise RuntimeError("FaceSwapFaceFusionNode requires the insightface package to be installed.")
        if ort is None:
            raise RuntimeError("FaceSwapFaceFusionNode requires onnxruntime to be installed.")
        if target_image.dim() != 4 or source_image.dim() != 4:
            raise ValueError("FaceSwapFaceFusionNode expects batched IMAGE tensors (B,H,W,C).")

        device_choice = _resolve_device(device)
        providers = _providers_for_device(device_choice)
        detector = _load_detector(detector_profile, device_choice, det_thresh, providers=providers)

        resolved_name = _resolve_swap_model_name(swap_model, _FACEFUSION_REGISTRY)
        model_path = _resolve_model_path(resolved_name, str(_default_model_dir()))
        session = _load_facefusion_session(str(model_path), providers)
        input_meta = session.get_inputs()
        output_meta = session.get_outputs()
        outputs_desc = ", ".join(f"{meta.name}:{getattr(meta, 'shape', None)}" for meta in output_meta)
        inputs_desc = ", ".join(f"{meta.name}:{getattr(meta, 'shape', None)}" for meta in input_meta)
        print(f"[comfyui-ess] FaceFusion model '{resolved_name}' inputs: {inputs_desc}", flush=True)
        print(f"[comfyui-ess] FaceFusion model '{resolved_name}' outputs: {outputs_desc}", flush=True)

        batch = target_image.shape[0]
        sources = source_image.shape[0]
        outputs: List[torch.Tensor] = []
        for batch_index in range(batch):
            source_index = min(batch_index, sources - 1)
            target_img = _tensor_to_image(target_image[batch_index])
            source_img = _tensor_to_image(source_image[source_index])

            target_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
            source_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)

            target_face = _detect_face(detector, target_bgr, target_face_index, gender)
            source_face = _detect_face(detector, source_bgr, source_face_index, gender)
            if target_face is None or source_face is None:
                outputs.append(target_image[batch_index])
                continue

            model_meta = _FACEFUSION_MODEL_META.get(resolved_name, _FACEFUSION_MODEL_META["blendswap_256.onnx"])
            model_template = model_meta["template"]
            model_size = model_meta["size"]
            model_type = model_meta["type"]
            model_mean = model_meta["mean"]
            model_std = model_meta["std"]

            crop_frame, affine_matrix = _warp_face_ff(target_bgr, target_face.face.kps, model_template, model_size)
            crop_frame = _prepare_facefusion_crop(crop_frame, model_mean, model_std)

            frame_inputs = {}
            for frame_input in input_meta:
                if frame_input.name == "source":
                    if model_type == "blendface":
                        frame_inputs[frame_input.name] = _prepare_facefusion_source_frame(source_bgr, source_face.face)
                    else:
                        frame_inputs[frame_input.name] = _prepare_facefusion_source_embedding(source_face.face)
                elif frame_input.name == "target":
                    frame_inputs[frame_input.name] = crop_frame

            out = session.run(None, frame_inputs)
            swapped = _select_facefusion_output(out, output_meta, model_size[1])
            if swapped.ndim == 4:
                swapped = swapped[0]
            swapped = _normalize_facefusion_crop(swapped)

            blur, padding = _facefusion_mask_settings(mask_blur, mask_expand, model_size[1])
            pasted = _paste_back_ff(target_bgr, swapped, affine_matrix, blur, padding)
            swapped_rgb = cv2.cvtColor(pasted, cv2.COLOR_BGR2RGB)
            outputs.append(_image_to_tensor(swapped_rgb, device=target_image.device))

        return (torch.stack(outputs, dim=0),)


def _tensor_to_image(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().cpu().numpy()
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def _image_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.clip(frame.astype(np.float32) / 255.0, 0.0, 1.0))
    return tensor.to(device)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _providers_for_device(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _insightface_root() -> Optional[str]:
    if not folder_paths:
        return None
    for key in _MODEL_FOLDER_KEYS:
        try:
            paths = folder_paths.get_folder_paths(key)
        except Exception:
            continue
        if paths:
            return str(Path(paths[0]))
    return None


def _insightface_search_roots() -> List[Path]:
    roots: List[Path] = []
    seen: set[str] = set()
    if folder_paths is not None:
        for key in _MODEL_FOLDER_KEYS:
            try:
                paths = folder_paths.get_folder_paths(key)
            except Exception:
                continue
            for item in paths:
                try:
                    path_obj = Path(item)
                except Exception:
                    continue
                norm = str(path_obj.resolve()).lower() if path_obj.exists() else str(path_obj).lower()
                if norm in seen:
                    continue
                seen.add(norm)
                roots.append(path_obj)
    root = _insightface_root()
    if root:
        root_path = Path(root)
        norm_root = str(root_path.resolve()).lower() if root_path.exists() else str(root_path).lower()
        if norm_root not in seen:
            roots.append(root_path)
    return roots


def _load_detector(
    profile: str,
    device: str,
    det_thresh: float,
    det_size: int = 640,
    det_max_num: int = 0,
    providers: List[str] = None,
):
    if providers is None:
        providers = _providers_for_device(device)
    key = (profile, device, round(det_thresh, 3), int(det_size), int(det_max_num))
    cached = _DETECTOR_CACHE.get(key)
    if cached is not None:
        return cached

    roots = _insightface_search_roots()
    root_path = roots[0] if roots else None
    if root_path:
        app = FaceAnalysis(name=profile, root=str(root_path), providers=providers)
    else:
        app = FaceAnalysis(name=profile, providers=providers)
    ctx_id = 0 if device == "cuda" else -1
    prepare_kwargs = {
        "ctx_id": ctx_id,
        "det_thresh": det_thresh,
        "det_size": (int(det_size), int(det_size)),
    }
    try:
        supports_det_max = "det_max_num" in inspect.signature(app.prepare).parameters
    except (TypeError, ValueError):
        supports_det_max = False
    if supports_det_max and int(det_max_num) > 0:
        prepare_kwargs["det_max_num"] = int(det_max_num)
    try:
        app.prepare(**prepare_kwargs)
    except TypeError:
        prepare_kwargs.pop("det_max_num", None)
        app.prepare(**prepare_kwargs)
    _DETECTOR_CACHE[key] = app
    return app


def _candidate_model_paths(model_name: str, model_dir: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if model_dir:
        try:
            candidates.append(Path(model_dir) / model_name)
        except Exception:
            pass

    for root in _insightface_search_roots():
        for subdir in [root, root / "models", root / "weights", root / "model"]:
            candidates.append(subdir / model_name)
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("insightface", model_name)
            if path:
                candidates.append(Path(path))
        except Exception:
            pass
    candidates.append(Path(model_name))

    unique: List[Path] = []
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


def _resolve_model_path(model_name: str, model_dir: Optional[str]) -> Path:
    if model_dir:
        try:
            _ensure_model_available(model_name, model_dir)
        except Exception:
            pass
    for candidate in _candidate_model_paths(model_name, model_dir):
        try:
            return candidate.resolve(strict=True)
        except Exception:
            continue
    available = ", ".join(_list_swap_models())
    raise RuntimeError(f"Swap model '{model_name}' could not be found. Available models: {available}")


def _load_inswapper_like(model_path: str, providers: List[str]):
    if INSwapper is None or ort is None:
        return None
    filename = Path(model_path).name
    if filename not in set(_INSWAPPER_REGISTRY.values()):
        return None
    session = ort.InferenceSession(model_path, providers=providers)
    return INSwapper(model_file=model_path, session=session)


def _load_swapper(model_name: str, model_dir: Optional[str], device: str, providers: List[str]):
    cache_key = (model_name, device, model_dir or "")
    cached = _SWAPPER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    errors: List[str] = []
    paths_tried: List[str] = []
    retry_download = True

    if model_dir:
        try:
            _ensure_model_available(model_name, model_dir)
        except Exception as exc:
            errors.append(str(exc))

    for candidate in _candidate_model_paths(model_name, model_dir):
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        paths_tried.append(str(resolved))
        try:
            swapper = _load_inswapper_like(str(resolved), providers) or insight_get_model(str(resolved), providers=providers)
        except Exception as exc:
            errors.append(f"{resolved}: {exc}")
            if retry_download and model_dir and _MODEL_URLS.get(model_name):
                retry_download = False
                try:
                    if resolved.exists():
                        resolved.unlink()
                except Exception:
                    pass
                try:
                    _ensure_model_available(model_name, model_dir)
                except Exception as download_exc:
                    errors.append(f"download retry failed: {download_exc}")
            continue
        _SWAPPER_CACHE[cache_key] = swapper
        return swapper

    available = ", ".join(_list_swap_models())
    detail = "; ".join(errors) if errors else "no additional information"
    raise RuntimeError(
        f"InsightFace swap model '{model_name}' could not be loaded. Paths tried: {paths_tried}. "
        f"Available models: {available}. Details: {detail}"
    )


def _load_facefusion_session(model_path: str, providers: List[str]):
    key = (model_path, ",".join(providers))
    cached = _SESSION_CACHE.get(key)
    if cached is not None:
        return cached
    session = ort.InferenceSession(model_path, providers=providers)
    _SESSION_CACHE[key] = session
    return session


def _detect_face(detector, image_bgr: np.ndarray, face_index: int, gender: str) -> Optional[DetectedFace]:
    faces = detector.get(image_bgr)
    if not faces:
        return None

    def gender_score(face_obj) -> int:
        if gender == "any":
            return 0
        face_gender = getattr(face_obj, "gender", None)
        if face_gender is None:
            return 1
        if gender == "male" and face_gender == 1:
            return 0
        if gender == "female" and face_gender == 0:
            return 0
        return 2

    sorted_faces = sorted(
        faces,
        key=lambda f: (
            gender_score(f),
            -((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        ),
    )
    index = min(max(face_index, 0), len(sorted_faces) - 1)
    selected = sorted_faces[index]
    if gender != "any":
        face_gender = getattr(selected, "gender", None)
        if gender == "male" and face_gender != 1:
            return None
        if gender == "female" and face_gender != 0:
            return None
    bbox = np.array(selected.bbox, dtype=np.float32)
    return DetectedFace(bbox=bbox, face=selected)


def _looks_like_swap_model(name: str) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in _SWAP_NAME_HINTS)


def _default_swap_model(options: Tuple[str, ...]) -> str:
    for preferred in _DEFAULT_SWAP_MODELS:
        if preferred in options:
            return preferred
    for name in options:
        if _looks_like_swap_model(name):
            return name
    return options[0]


def _registry_model_names(registry: dict[str, str]) -> List[str]:
    return sorted(registry.keys())


def _default_model_name(options: Tuple[str, ...]) -> str:
    preferred = "inswapper_128_fp16"
    if preferred in options:
        return preferred
    return options[0]


def _resolve_swap_model_name(model_key: str, registry: dict[str, str]) -> str:
    resolved = registry.get(model_key)
    if resolved:
        return resolved
    return model_key


def _list_swap_models() -> List[str]:
    models: set[str] = set()
    if _ROPE_DEFAULT_DIR.exists():
        for file in _ROPE_DEFAULT_DIR.glob("*.onnx"):
            models.add(file.name)
    default_dir = _default_model_dir()
    if default_dir.exists():
        for file in default_dir.glob("*.onnx"):
            models.add(file.name)
    for root in _insightface_search_roots():
        for subdir in [root, root / "models", root / "weights", root / "model"]:
            if not subdir.exists():
                continue
            for file in subdir.glob("*.onnx"):
                models.add(file.name)
    if folder_paths is not None:
        try:
            files = folder_paths.get_filename_list("insightface")
            for item in files:
                models.add(item)
        except Exception:
            pass
    filtered = [m for m in models if _looks_like_swap_model(m)]
    if filtered:
        return sorted(filtered)
    if not models:
        models.update(_DEFAULT_SWAP_MODELS)
    return sorted(models)


def _restore_search_roots() -> List[Path]:
    roots: List[Path] = []
    default_dir = _default_restore_dir()
    if default_dir:
        roots.append(default_dir)
    if folder_paths is not None:
        try:
            paths = folder_paths.get_folder_paths("face_restore")
        except Exception:
            paths = []
        for item in paths:
            try:
                roots.append(Path(item))
            except Exception:
                continue
    return roots


def _candidate_restore_paths(model_name: str, model_dir: Optional[str]) -> List[Path]:
    candidates: List[Path] = []
    if model_dir:
        try:
            candidates.append(Path(model_dir) / model_name)
        except Exception:
            pass

    for root in _restore_search_roots():
        for subdir in [root, root / "models", root / "weights", root / "model"]:
            candidates.append(subdir / model_name)
    if folder_paths is not None:
        try:
            path = folder_paths.get_full_path("face_restore", model_name)
            if path:
                candidates.append(Path(path))
        except Exception:
            pass
    candidates.append(Path(model_name))

    unique: List[Path] = []
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


def _resolve_restore_model_path(model_name: str, model_dir: Optional[str]) -> Path:
    if model_dir:
        try:
            _ensure_restore_available(model_name, model_dir)
        except Exception:
            pass
    for candidate in _candidate_restore_paths(model_name, model_dir):
        try:
            return candidate.resolve(strict=True)
        except Exception:
            continue
    available = ", ".join(_list_restore_models())
    raise RuntimeError(f"Face restore model '{model_name}' could not be found. Available models: {available}")


def _list_restore_models() -> List[str]:
    models: set[str] = set()
    default_dir = _default_restore_dir()
    if default_dir.exists():
        for file in default_dir.glob("*.pth"):
            models.add(file.name)
    for root in _restore_search_roots():
        if root.exists():
            for file in root.glob("*.pth"):
                models.add(file.name)
    for default_name in _DEFAULT_RESTORE_MODELS:
        models.add(default_name)
    return sorted(models)


def _default_restore_model(options: Tuple[str, ...]) -> str:
    for preferred in _DEFAULT_RESTORE_MODELS:
        if preferred in options:
            return preferred
    return options[0]


def _load_restorer(
    model_name: str,
    device: str,
    upscale: int,
    arch: str,
    channel_multiplier: int,
):
    if GFPGANer is None:
        raise RuntimeError("Face restore requires gfpgan. Please install the gfpgan package.")
    model_dir = str(_default_restore_dir())
    cache_key = (model_name, device, int(upscale), arch, int(channel_multiplier), model_dir)
    cached = _RESTORER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    errors: List[str] = []
    paths_tried: List[str] = []
    retry_download = True

    if model_dir:
        try:
            _ensure_restore_available(model_name, model_dir)
        except Exception as exc:
            errors.append(str(exc))

    device_name = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    for candidate in _candidate_restore_paths(model_name, model_dir):
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        paths_tried.append(str(resolved))
        try:
            try:
                restorer = GFPGANer(
                    model_path=str(resolved),
                    upscale=int(upscale),
                    arch=arch,
                    channel_multiplier=int(channel_multiplier),
                    bg_upsampler=None,
                    device=device_name,
                )
            except TypeError:
                restorer = GFPGANer(
                    model_path=str(resolved),
                    upscale=int(upscale),
                    arch=arch,
                    channel_multiplier=int(channel_multiplier),
                    bg_upsampler=None,
                )
        except Exception as exc:
            errors.append(f"{resolved}: {exc}")
            if retry_download and model_dir and _MODEL_URLS.get(model_name):
                retry_download = False
                try:
                    if resolved.exists():
                        resolved.unlink()
                except Exception:
                    pass
                try:
                    _ensure_restore_available(model_name, model_dir)
                except Exception as download_exc:
                    errors.append(f"download retry failed: {download_exc}")
            continue
        _RESTORER_CACHE[cache_key] = restorer
        return restorer

    available = ", ".join(_list_restore_models())
    detail = "; ".join(errors) if errors else "no additional information"
    raise RuntimeError(
        f"Face restore model '{model_name}' could not be loaded. Paths tried: {paths_tried}. "
        f"Available models: {available}. Details: {detail}"
    )


def _restore_faces(
    image_bgr: np.ndarray,
    model_name: str,
    visibility: float,
    upscale: int,
    arch: str,
    channel_multiplier: int,
    only_center: bool,
    device: str,
) -> np.ndarray:
    if visibility <= 0.0:
        return image_bgr
    if GFPGANer is None:
        global _RESTORE_WARNING_LOGGED
        if not _RESTORE_WARNING_LOGGED:
            print("[comfyui-ess] Face restore skipped: gfpgan is not installed.", flush=True)
            _RESTORE_WARNING_LOGGED = True
        return image_bgr
    restorer = _load_restorer(model_name, device, upscale, arch, channel_multiplier)
    result = restorer.enhance(image_bgr, has_aligned=False, only_center_face=only_center, paste_back=True)
    restored = None
    if isinstance(result, tuple):
        for item in reversed(result):
            if isinstance(item, np.ndarray):
                restored = item
                break
    elif isinstance(result, np.ndarray):
        restored = result
    if restored is None:
        return image_bgr
    if restored.shape[:2] != image_bgr.shape[:2]:
        restored = cv2.resize(restored, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
    if visibility >= 1.0:
        return restored
    return _blend_with_strength(image_bgr, restored, visibility)


def _swapper_get(swapper, target_bgr: np.ndarray, target_face, source_face) -> np.ndarray:
    try:
        result = swapper.get(target_bgr, target_face, source_face, paste_back=True)
    except TypeError:
        result = swapper.get(target_bgr, target_face, source_face)
    if not isinstance(result, np.ndarray):
        raise RuntimeError("Swapper returned unexpected output; selected model may not be compatible.")
    if result.ndim != 3 or result.shape[2] != 3:
        raise RuntimeError("Swapper output is not an image; selected model may not be compatible.")
    return result


def _default_model_dir() -> Path:
    if folder_paths is not None:
        try:
            return Path(folder_paths.models_dir) / _DEFAULT_MODEL_SUBDIR
        except Exception:
            pass
    return _ROPE_DEFAULT_DIR


def _default_restore_dir() -> Path:
    if folder_paths is not None:
        try:
            return Path(folder_paths.models_dir) / _DEFAULT_RESTORE_SUBDIR
        except Exception:
            pass
    return _ROPE_DEFAULT_DIR / "face_restore"


def _ensure_model_dir(model_dir: str) -> Path:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_model_url(url: str) -> str:
    cleaned = url.strip().rstrip("\\'\"")
    cleaned = cleaned.replace("?utm_source=chatgpt.com", "")
    if "huggingface.co" in cleaned and "/blob/" in cleaned:
        cleaned = cleaned.replace("/blob/", "/resolve/")
    if "huggingface.co" in cleaned and "/blame/" in cleaned:
        cleaned = cleaned.replace("/blame/", "/resolve/")
    return cleaned


def _download_model(url: str, dest: Path) -> None:
    import urllib.request
    import sys
    headers = {"User-Agent": "comfyui-ess/face-swapping"}
    req = urllib.request.Request(url, headers=headers)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req) as response:
        total = response.headers.get("Content-Length")
        total_size = int(total) if total and total.isdigit() else 0
        downloaded = 0
        last_pct = -1
        if total_size:
            print(f"[comfyui-ess] Downloading {dest.name} ({total_size / (1024 * 1024):.1f} MB)...", file=sys.stderr)
        else:
            print(f"[comfyui-ess] Downloading {dest.name}...", file=sys.stderr)
        with open(tmp_path, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = int(downloaded * 100 / total_size)
                    if pct != last_pct:
                        last_pct = pct
                        filled = int(pct / 2)
                        bar = "#" * filled + "-" * (50 - filled)
                        msg = f"[comfyui-ess] {dest.name}: [{bar}] {pct}% ({downloaded / (1024 * 1024):.1f} MB)"
                        sys.stderr.write("\r" + msg)
                        sys.stderr.flush()
        if total_size:
            sys.stderr.write("\n")
    tmp_path.replace(dest)


def _ensure_model_available(model_name: str, model_dir: str) -> Optional[Path]:
    urls = _MODEL_URLS.get(model_name)
    if not urls:
        return None
    target_dir = _ensure_model_dir(model_dir)
    target_path = target_dir / model_name
    if target_path.exists() and target_path.stat().st_size >= _MIN_MODEL_BYTES:
        return target_path
    if target_path.exists():
        try:
            target_path.unlink()
        except Exception:
            pass
    errors: List[str] = []
    for url in urls:
        try:
            _download_model(_normalize_model_url(url), target_path)
            if target_path.exists() and target_path.stat().st_size >= _MIN_MODEL_BYTES:
                return target_path
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    if errors:
        raise RuntimeError(f"Failed to download {model_name}. Errors: {'; '.join(errors)}")
    return target_path


def _ensure_restore_available(model_name: str, model_dir: str) -> Optional[Path]:
    urls = _MODEL_URLS.get(model_name)
    if not urls:
        return None
    target_dir = _ensure_model_dir(model_dir)
    target_path = target_dir / model_name
    if target_path.exists() and target_path.stat().st_size >= _MIN_MODEL_BYTES:
        return target_path
    if target_path.exists():
        try:
            target_path.unlink()
        except Exception:
            pass
    errors: List[str] = []
    for url in urls:
        try:
            _download_model(_normalize_model_url(url), target_path)
            if target_path.exists() and target_path.stat().st_size >= _MIN_MODEL_BYTES:
                return target_path
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    if errors:
        raise RuntimeError(f"Failed to download {model_name}. Errors: {'; '.join(errors)}")
    return target_path




def _extract_landmarks(face_obj) -> Optional[np.ndarray]:
    landmarks = getattr(face_obj, "landmark_2d_106", None)
    if landmarks is None:
        landmarks = getattr(face_obj, "kps", None)
    if landmarks is None:
        return None
    landmarks = np.array(landmarks, dtype=np.float32)
    if landmarks.ndim != 2 or landmarks.shape[0] < 5:
        return None
    return landmarks


def _face_mask_from_face(face_obj, shape: Tuple[int, int], expand: int, blur: int) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    landmarks = _extract_landmarks(face_obj)
    if landmarks is None:
        return None
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(landmarks.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)

    return (mask.astype(np.float32) / 255.0)[..., None]


def _ellipse_mask_from_bbox(bbox: np.ndarray, shape: Tuple[int, int], expand: int, blur: int) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    rx = max(int((x2 - x1) / 2), 1)
    ry = max(int((y2 - y1) / 2), 1)
    if expand > 0:
        rx += int(expand)
        ry += int(expand)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]


def _draw_face_debug(image_bgr: np.ndarray, detected: Optional[DetectedFace], mask: Optional[np.ndarray]) -> np.ndarray:
    if cv2 is None or detected is None:
        return image_bgr
    debug = image_bgr.copy()
    x1, y1, x2, y2 = detected.bbox.astype(int)
    h, w = debug.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w - 1, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h - 1, y2)))
    cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    landmarks = _extract_landmarks(detected.face)
    if landmarks is not None:
        for (lx, ly) in landmarks.astype(int):
            if 0 <= lx < w and 0 <= ly < h:
                cv2.circle(debug, (int(lx), int(ly)), 1, (255, 255, 0), -1)

    if mask is not None:
        if mask.ndim == 3:
            mask_alpha = mask[:, :, 0]
        else:
            mask_alpha = mask
        mask_alpha = np.clip(mask_alpha.astype(np.float32), 0.0, 1.0)
        overlay = np.zeros_like(debug)
        overlay[:, :, 2] = 255
        alpha = 0.35
        debug = (debug.astype(np.float32) * (1.0 - alpha * mask_alpha[..., None]) +
                 overlay.astype(np.float32) * (alpha * mask_alpha[..., None]))
        debug = np.clip(debug, 0, 255).astype(np.uint8)
    return debug


def _face_mask_from_detected(
    detected: DetectedFace,
    shape: Tuple[int, int],
    expand: int,
    blur: int,
    mask_type: str,
) -> Optional[np.ndarray]:
    if mask_type == "ellipse":
        return _ellipse_mask_from_bbox(detected.bbox, shape, expand, blur)
    mask = _face_mask_from_face(detected.face, shape, expand, blur)
    if mask is None and mask_type != "landmarks":
        return _ellipse_mask_from_bbox(detected.bbox, shape, expand, blur)
    return mask


def _blend_with_mask(base_bgr: np.ndarray, overlay_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blended = base_bgr.astype(np.float32) * (1.0 - mask) + overlay_bgr.astype(np.float32) * mask
    return np.clip(blended, 0, 255).astype(np.uint8)


def _blend_with_strength(base_bgr: np.ndarray, overlay_bgr: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return base_bgr
    if strength >= 1.0:
        return overlay_bgr
    return cv2.addWeighted(base_bgr, 1.0 - strength, overlay_bgr, strength, 0.0)


def _infer_simswap_size(inputs) -> int:
    for meta in inputs:
        shape = meta.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            size = shape[2]
            if isinstance(size, int) and size in (256, 512):
                return size
    return 256


def _align_face(image_bgr: np.ndarray, face_obj, size: int) -> Tuple[np.ndarray, np.ndarray]:
    kps = getattr(face_obj, "kps", None)
    if kps is None:
        raise RuntimeError("SimSwap alignment requires 5-point keypoints.")
    kps = np.array(kps, dtype=np.float32)
    if face_align is not None and hasattr(face_align, "estimate_norm"):
        try:
            m = face_align.estimate_norm(kps, size)
            if m is not None:
                aligned = cv2.warpAffine(image_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                inv_m = cv2.invertAffineTransform(m)
                return aligned, inv_m
        except Exception:
            pass
    dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    scale = size / 112.0
    dst *= scale
    m, _ = cv2.estimateAffinePartial2D(kps, dst, method=cv2.LMEDS)
    if m is None:
        raise RuntimeError("Failed to compute SimSwap alignment transform.")
    aligned = cv2.warpAffine(image_bgr, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    inv_m = cv2.invertAffineTransform(m)
    return aligned, inv_m


def _build_simswap_feed(inputs, target_bgr: np.ndarray, source_bgr: np.ndarray, source_face) -> dict:
    embed = getattr(source_face, "normed_embedding", None)
    if embed is None:
        embed = getattr(source_face, "embedding", None)
    if embed is not None:
        embed = np.asarray(embed, dtype=np.float32)[None, :]

    def to_tensor(image_bgr: np.ndarray, size: Optional[int]) -> np.ndarray:
        frame = image_bgr
        if size is not None and frame.shape[0] != size:
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        return (frame / 127.5 - 1.0).transpose(2, 0, 1)[None, ...]

    feed = {}
    for meta in inputs:
        name = meta.name
        shape = meta.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            size = None
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                size = shape[2]
            if "source" in name.lower() or "src" in name.lower():
                feed[name] = to_tensor(source_bgr, size)
            elif "target" in name.lower() or "dst" in name.lower():
                feed[name] = to_tensor(target_bgr, size)
            elif name not in feed:
                feed[name] = to_tensor(target_bgr, size) if not feed else to_tensor(source_bgr, size)
        elif isinstance(shape, (list, tuple)) and len(shape) == 2:
            if embed is None:
                raise RuntimeError("SimSwap model expects an embedding input, but none was available.")
            feed[name] = embed
    if not feed:
        raise RuntimeError("SimSwap model inputs could not be resolved.")
    return feed


def _warp_face_ff(temp_frame: np.ndarray, kps: np.ndarray, template: str, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    normed_template = _FACEFUSION_TEMPLATES.get(template, _FACEFUSION_TEMPLATES["ffhq"]) * size[1] / size[0]
    affine_matrix = cv2.estimateAffinePartial2D(kps, normed_template, method=cv2.LMEDS)[0]
    crop_frame = cv2.warpAffine(temp_frame, affine_matrix, (size[1], size[1]), borderMode=cv2.BORDER_REPLICATE)
    return crop_frame, affine_matrix


def _prepare_facefusion_source_frame(source_bgr: np.ndarray, source_face) -> np.ndarray:
    source_frame, _ = _warp_face_ff(source_bgr, source_face.kps, "arcface_v2", (112, 112))
    source_frame = source_frame[:, :, ::-1] / 255.0
    source_frame = source_frame.transpose(2, 0, 1)
    return np.expand_dims(source_frame, axis=0).astype(np.float32)


def _prepare_facefusion_source_embedding(source_face) -> np.ndarray:
    embedding = getattr(source_face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(source_face, "embedding", None)
    if embedding is None:
        raise RuntimeError("FaceFusion model expects an embedding input, but none was available.")
    return np.asarray(embedding, dtype=np.float32).reshape(1, -1)


def _prepare_facefusion_crop(crop_frame: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    crop_frame = crop_frame[:, :, ::-1] / 255.0
    crop_frame = (crop_frame - mean) / std
    crop_frame = crop_frame.transpose(2, 0, 1)
    return np.expand_dims(crop_frame, axis=0).astype(np.float32)


def _normalize_facefusion_crop(crop_frame: np.ndarray) -> np.ndarray:
    crop_frame = crop_frame.transpose(1, 2, 0)
    crop_frame = (crop_frame * 255.0).round()
    crop_frame = crop_frame[:, :, ::-1].astype(np.uint8)
    return crop_frame


def _facefusion_mask_settings(mask_blur: int, mask_expand: int, size: int) -> Tuple[float, Tuple[float, float, float, float]]:
    blur = max(0.0, min(mask_blur / 100.0, 1.0))
    pad_pct = max(0.0, min(mask_expand / max(size, 1) * 100.0, 100.0))
    return blur, (pad_pct, pad_pct, pad_pct, pad_pct)


def _paste_back_ff(temp_frame: np.ndarray, crop_frame: np.ndarray, affine_matrix: np.ndarray, face_mask_blur: float, face_mask_padding: Tuple[float, float, float, float]) -> np.ndarray:
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_frame_size = temp_frame.shape[:2][::-1]
    mask_size = tuple(crop_frame.shape[:2])
    mask_frame = _create_facefusion_mask_frame(mask_size, face_mask_blur, face_mask_padding)
    inverse_mask_frame = cv2.warpAffine(mask_frame, inverse_matrix, temp_frame_size).clip(0, 1)
    inverse_crop_frame = cv2.warpAffine(crop_frame, inverse_matrix, temp_frame_size, borderMode=cv2.BORDER_REPLICATE)
    paste_frame = temp_frame.copy()
    paste_frame[:, :, 0] = inverse_mask_frame * inverse_crop_frame[:, :, 0] + (1 - inverse_mask_frame) * temp_frame[:, :, 0]
    paste_frame[:, :, 1] = inverse_mask_frame * inverse_crop_frame[:, :, 1] + (1 - inverse_mask_frame) * temp_frame[:, :, 1]
    paste_frame[:, :, 2] = inverse_mask_frame * inverse_crop_frame[:, :, 2] + (1 - inverse_mask_frame) * temp_frame[:, :, 2]
    return paste_frame


def _create_facefusion_mask_frame(mask_size: Tuple[int, int], face_mask_blur: float, face_mask_padding: Tuple[float, float, float, float]) -> np.ndarray:
    mask_frame = np.ones(mask_size, np.float32)
    blur_amount = int(mask_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    mask_frame[:max(blur_area, int(mask_size[1] * face_mask_padding[0] / 100)), :] = 0
    mask_frame[-max(blur_area, int(mask_size[1] * face_mask_padding[2] / 100)):, :] = 0
    mask_frame[:, :max(blur_area, int(mask_size[0] * face_mask_padding[3] / 100))] = 0
    mask_frame[:, -max(blur_area, int(mask_size[0] * face_mask_padding[1] / 100)):] = 0
    if blur_amount > 0:
        mask_frame = cv2.GaussianBlur(mask_frame, (0, 0), blur_amount * 0.25)
    return mask_frame


def _tensor_to_bgr(output: np.ndarray) -> np.ndarray:
    if output.ndim == 4:
        output = output[0]
    if output.shape[0] == 3:
        output = output.transpose(1, 2, 0)
    max_val = float(np.max(output)) if output.size else 1.0
    min_val = float(np.min(output)) if output.size else 0.0
    if min_val >= 0.0 and max_val <= 1.5:
        output = output * 255.0
    else:
        output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def _select_swap_output(outputs: List[np.ndarray]) -> np.ndarray:
    best = None
    best_area = -1
    for item in outputs:
        if not isinstance(item, np.ndarray):
            continue
        shape = item.shape
        if item.ndim == 4:
            if shape[1] == 3:
                area = shape[2] * shape[3]
            elif shape[-1] == 3:
                area = shape[1] * shape[2]
            else:
                continue
            if area > best_area:
                best = item
                best_area = area
        elif item.ndim == 3 and (shape[0] == 3 or shape[-1] == 3):
            area = shape[1] * shape[2] if shape[0] == 3 else shape[0] * shape[1]
            if area > best_area:
                best = item
                best_area = area
    if best is not None:
        return best
    if not outputs:
        raise RuntimeError("FaceFusion model returned no outputs.")
    return outputs[0]


def _select_facefusion_output(outputs: List[np.ndarray], metas, target_size: int) -> np.ndarray:
    best = None
    best_score = -1
    best_index = -1
    for idx, item in enumerate(outputs):
        if not isinstance(item, np.ndarray):
            continue
        shape = item.shape
        meta_shape = None
        if metas and idx < len(metas):
            meta_shape = getattr(metas[idx], "shape", None)
        score = 0
        if item.ndim == 4:
            if shape[1] == 3:
                score += 4
                if shape[2] == target_size and shape[3] == target_size:
                    score += 4
            elif shape[-1] == 3:
                score += 3
                if shape[1] == target_size and shape[2] == target_size:
                    score += 3
        elif item.ndim == 3 and (shape[0] == 3 or shape[-1] == 3):
            score += 2
            h = shape[1] if shape[0] == 3 else shape[0]
            w = shape[2] if shape[0] == 3 else shape[1]
            if h == target_size and w == target_size:
                score += 2
        if meta_shape and isinstance(meta_shape, (list, tuple)) and len(meta_shape) == 4:
            if isinstance(meta_shape[2], int) and isinstance(meta_shape[3], int):
                if meta_shape[2] == target_size and meta_shape[3] == target_size:
                    score += 1
        if score > best_score:
            best = item
            best_score = score
            best_index = idx
    if best is not None:
        print(
            f"[comfyui-ess] FaceFusion selected output index {best_index} score {best_score} shape {best.shape}",
            flush=True,
        )
        return best
    return _select_swap_output(outputs)


def _paste_back(
    base_bgr: np.ndarray,
    swapped_aligned: np.ndarray,
    inv_m: np.ndarray,
    expand: int,
    blur: int,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = base_bgr.shape[:2]
    warped = cv2.warpAffine(swapped_aligned, inv_m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros((swapped_aligned.shape[0], swapped_aligned.shape[1]), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (swapped_aligned.shape[1] // 2, int(swapped_aligned.shape[0] * 0.55)),
        (int(swapped_aligned.shape[1] * 0.38), int(swapped_aligned.shape[0] * 0.45)),
        0,
        0,
        360,
        255,
        -1,
    )
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask = cv2.warpAffine(mask, inv_m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = (mask.astype(np.float32) / 255.0)[..., None]
    pasted = np.clip(base_bgr.astype(np.float32) * (1.0 - mask) + warped.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    return pasted, mask
