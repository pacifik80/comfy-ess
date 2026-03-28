"""Flexible IMAGE batch merge node with dynamic optional inputs."""

from __future__ import annotations

import os
import random
from typing import Any, Iterable

import folder_paths
import torch
from PIL import Image


_MAX_SLOTS = 16


def _iter_image_tensors(value: Any) -> Iterable[torch.Tensor]:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_image_tensors(item)
        return
    raise TypeError(f"Unsupported IMAGE input payload: {type(value)!r}")


def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    if image.ndim == 4:
        return image
    raise ValueError(f"Expected IMAGE tensor with 3 or 4 dims, got shape {tuple(image.shape)}")


def _is_empty_placeholder_batch(image: torch.Tensor) -> bool:
    if tuple(image.shape[1:]) != (1, 1, 3):
        return False
    return bool(torch.count_nonzero(image) == 0)


class ImageBatchMerge:
    CATEGORY = "ESS/Image"
    FUNCTION = "merge"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    def __init__(self):
        suffix = "".join(random.choice("abcdefghijklmnopqrstupvxyz") for _ in range(5))
        self._preview_dir = folder_paths.get_temp_directory()
        self._preview_prefix = f"ess_batch_merge/_temp_{suffix}"
        self._preview_type = "temp"
        self._preview_compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"image_{index}": ("IMAGE", {"tooltip": f"Optional image or batch input {index}."})
            for index in range(1, _MAX_SLOTS + 1)
        }
        return {
            "required": {
                "count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": _MAX_SLOTS,
                        "step": 1,
                        "tooltip": "Number of image inputs to expose (apply to update inputs).",
                    },
                ),
            },
            "optional": optional_inputs,
        }

    def _build_preview_images(self, images: torch.Tensor) -> list[dict[str, str]]:
        if images.ndim != 4 or images.shape[0] == 0:
            return []

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            self._preview_prefix,
            self._preview_dir,
            images[0].shape[1],
            images[0].shape[0],
        )

        results: list[dict[str, str]] = []
        for batch_number, image in enumerate(images):
            image_uint8 = image.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
            image_pil = Image.fromarray(image_uint8)
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            image_pil.save(
                os.path.join(full_output_folder, file),
                compress_level=self._preview_compress_level,
            )
            results.append(
                {
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self._preview_type,
                }
            )
            counter += 1

        return results

    def merge(self, count: int, **kwargs: Any):
        collected: list[torch.Tensor] = []
        target_shape = None
        target_dtype = None
        target_device = None

        for index in range(1, _MAX_SLOTS + 1):
            if index > max(1, int(count or 1)):
                break
            value = kwargs.get(f"image_{index}")
            for tensor in _iter_image_tensors(value):
                normalized = _normalize_image_tensor(tensor)
                if _is_empty_placeholder_batch(normalized):
                    continue
                if target_shape is None:
                    target_shape = tuple(normalized.shape[1:])
                    target_dtype = normalized.dtype
                    target_device = normalized.device
                else:
                    if tuple(normalized.shape[1:]) != target_shape:
                        raise ValueError(
                            "Image Batch Merge received mismatched image sizes/channels: "
                            f"expected {target_shape}, got {tuple(normalized.shape[1:])}."
                        )
                    if normalized.dtype != target_dtype or normalized.device != target_device:
                        normalized = normalized.to(device=target_device, dtype=target_dtype)
                collected.append(normalized)

        if not collected:
            empty = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            return {"ui": {"images": []}, "result": (empty,)}

        merged = torch.cat(collected, dim=0)
        previews = self._build_preview_images(merged)
        return {"ui": {"images": previews}, "result": (merged,)}
