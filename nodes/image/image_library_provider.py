from __future__ import annotations

import base64
import io
import json
import random
import time
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from ..assets.workflow_asset_store import resolve_image_library_asset


def _empty_image() -> torch.Tensor:
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def _parse_library_payload(payload: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(payload, dict):
        data = payload
    else:
        text = str(payload or "").strip()
        if not text:
            return {"mode": "manual", "selected_index": 0, "items": []}
        try:
            parsed = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid image library payload: {exc}") from exc
        data = parsed if isinstance(parsed, dict) else {}

    mode = str(data.get("mode") or "manual").strip().lower()
    if mode not in {"manual", "random"}:
        mode = "manual"

    try:
        selected_index = int(data.get("selected_index", data.get("selectedIndex", 0)) or 0)
    except Exception:
        selected_index = 0

    items: list[dict[str, Any]] = []
    for raw in data.get("items") or []:
        if not isinstance(raw, dict):
            continue
        image_data = str(raw.get("image_data") or raw.get("imageData") or raw.get("data") or "").strip()
        asset_id = str(raw.get("asset_id") or raw.get("assetId") or "").strip()
        workflow_relative_path = str(raw.get("workflow_relative_path") or raw.get("workflowRelativePath") or "").strip()
        if not image_data and not asset_id:
            continue
        name = str(raw.get("name") or "").strip()
        prompt = str(raw.get("prompt") or "").strip()
        mime_type = str(raw.get("mime_type") or raw.get("mimeType") or "").strip()
        try:
            weight = float(raw.get("weight", 1) or 0)
        except Exception:
            weight = 1.0
        try:
            width = int(raw.get("width", 0) or 0)
        except Exception:
            width = 0
        try:
            height = int(raw.get("height", 0) or 0)
        except Exception:
            height = 0
        items.append(
            {
                "name": name,
                "prompt": prompt,
                "weight": weight,
                "image_data": image_data,
                "asset_id": asset_id,
                "workflow_relative_path": workflow_relative_path,
                "mime_type": mime_type,
                "width": width,
                "height": height,
            }
        )

    return {
        "mode": mode,
        "selected_index": selected_index,
        "items": items,
    }


def _decode_image_payload(image_data: str) -> tuple[torch.Tensor, int, int]:
    payload = str(image_data or "").strip()
    if not payload:
        raise ValueError("Image payload is empty.")

    if "base64," in payload:
        payload = payload.split("base64,", 1)[1]

    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise ValueError(f"Image payload is not valid base64: {exc}") from exc

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            np_image = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Failed to decode stored image: {exc}") from exc

    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor, int(width), int(height)


def _decode_image_bytes(raw: bytes) -> tuple[torch.Tensor, int, int]:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            np_image = np.asarray(image, dtype=np.float32) / 255.0
    except Exception as exc:
        raise ValueError(f"Failed to read stored asset image: {exc}") from exc
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor, int(width), int(height)


def _choose_item(state: dict[str, Any], seed: int) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    items = list(state.get("items") or [])
    if not items:
        return None, None

    mode = str(state.get("mode") or "manual").strip().lower()
    if mode != "random":
        index = int(state.get("selected_index", 0) or 0)
        index = max(0, min(index, len(items) - 1))
        return index, items[index]

    weighted_items = []
    weights = []
    for index, item in enumerate(items):
        try:
            weight = float(item.get("weight", 1) or 0)
        except Exception:
            weight = 1.0
        if weight > 0:
            weighted_items.append((index, item))
            weights.append(weight)

    if weighted_items:
        rng = random.Random(seed) if int(seed or 0) > 0 else random.SystemRandom()
        return rng.choices(weighted_items, weights=weights, k=1)[0]

    rng = random.Random(seed) if int(seed or 0) > 0 else random.SystemRandom()
    index = rng.randrange(len(items))
    return index, items[index]


class ESSImageLibraryProvider:
    CATEGORY = "ESS/Image"
    FUNCTION = "provide"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "prompt")
    DESCRIPTION = "Stores an image library in the workflow's adjacent .ess asset file, outputs the selected image manually or via weighted random choice."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "library": (
                    "ESS_IMAGE_LIBRARY",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Load one or more images into the workflow's .ess asset library...",
                        "height": 360,
                    },
                ),
            },
            "optional": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Optional seed for random mode. Leave at 0 for a fresh weighted pick each execution.",
                    },
                ),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    @classmethod
    def IS_CHANGED(cls, library: str, seed: int = 0):
        try:
            state = _parse_library_payload(library)
        except Exception:
            return time.time_ns()
        if str(state.get("mode") or "manual").strip().lower() == "random" and int(seed or 0) <= 0:
            return time.time_ns()
        return f"{library}|{int(seed or 0)}"

    def provide(self, library: str, seed: int = 0, extra_pnginfo: Any = None, unique_id: Any = None):
        state = _parse_library_payload(library)
        selected_index, item = _choose_item(state, int(seed or 0))
        if item is None:
            return (_empty_image(), 1, 1, "")

        image_data = str(item.get("image_data") or "")
        if image_data:
            image_tensor, width, height = _decode_image_payload(image_data)
        else:
            asset_bytes = resolve_image_library_asset(item, extra_pnginfo=extra_pnginfo)
            if asset_bytes is None:
                raise ValueError("Stored image asset could not be resolved for this workflow.")
            image_tensor, width, height = _decode_image_bytes(asset_bytes)
        prompt = str(item.get("prompt") or "")
        return {
            "ui": {
                "selected_index": [int(selected_index or 0)],
                "selected_asset_id": [str(item.get("asset_id") or "")],
            },
            "result": (image_tensor, int(width), int(height), prompt),
        }
