import base64
import json
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch


def _blank_preview(size: int = 512) -> torch.Tensor:
    return torch.zeros((1, size, size, 3), dtype=torch.float32)


def _decode_preview(preview_png: str) -> torch.Tensor:
    """
    Decode a data URL or raw base64 PNG string into an NHWC float tensor
    in the [0, 1] range. Falls back to a black 512x512 image on failure.
    """
    if not preview_png:
        return _blank_preview()

    try:
        payload = preview_png.split(",", 1)[1] if "base64," in preview_png else preview_png
        img_bytes = base64.b64decode(payload)
        np_buf = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("cv2.imdecode returned None")

        # Convert to RGB, drop alpha if present
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_f = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_f)[None, ...]  # NHWC
        return tensor
    except Exception:
        # Keep the node resilient; return a blank image to avoid breaking graphs.
        return _blank_preview()


class PoseMeshEditor:
    """
    Interactive pose and camera editor for FBX/GLTF characters.

    The heavy UI work lives in the companion JS extension (ESS_POSE_MESH_EDITOR).
    This Python side primarily:
      - Exposes the custom widget to ComfyUI.
      - Parses the widget payload.
      - Returns three IMAGE outputs (preview, depth, edges).
    """

    CATEGORY = "ESS/Pose"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("preview", "depth", "edges")

    @classmethod
    def INPUT_TYPES(cls):
        default_state = ""
        return {
            "required": {
                "state": (
                    "ESS_POSE_MESH_EDITOR",
                    {
                        "multiline": True,
                        "default": default_state,
                        "placeholder": "Use the Pose Mesh Editor UI to build a pose.",
                    },
                ),
                "output_image": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "image",
                        "label_off": "none",
                        "tooltip": "When enabled, decode the preview PNG from the editor into an IMAGE output.",
                    },
                ),
            },
        }

    def render(
        self,
        state: str,
        output_image: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            parsed: Dict[str, Any] = json.loads(state) if state else {}
        except Exception as exc:
            raise ValueError(f"Pose Mesh Editor received invalid JSON: {exc}") from exc

        if output_image:
            preview_png = str(parsed.get("preview_png", "") or "")
            if not preview_png:
                legacy_previews = parsed.get("preview_pngs", None)
                if isinstance(legacy_previews, list) and legacy_previews:
                    preview_png = str(legacy_previews[0] or "")
            image_1 = _decode_preview(preview_png)
            image_2 = _decode_preview(str(parsed.get("depth_png", "") or ""))
            image_3 = _decode_preview(str(parsed.get("edges_png", "") or ""))
        else:
            tiny = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            image_1 = tiny
            image_2 = tiny
            image_3 = tiny

        return (image_1, image_2, image_3)
