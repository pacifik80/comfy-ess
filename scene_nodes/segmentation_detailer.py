"""Segmentation-driven detailing node."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import comfy.samplers as comfy_samplers
    import comfy.model_management as model_management
    import nodes
except ImportError:
    comfy_samplers = None
    model_management = None
    nodes = None


def _list_files_for_key(key: str) -> list[str]:
    if folder_paths is None:
        return []
    try:
        return folder_paths.get_filename_list(key)
    except Exception:
        return []

def _list_sam_models() -> list[str]:
    if folder_paths is None:
        return []
    candidates = []
    try:
        items = getattr(folder_paths, "folder_names_and_paths", {})
    except Exception:
        items = {}
    for name, paths in items.items():
        if "sam" not in name.lower():
            continue
        for entry in paths:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                folder = entry[1]
            else:
                folder = entry
            try:
                for item in Path(folder).iterdir():
                    if item.is_file():
                        candidates.append(item.name)
            except Exception:
                continue
    return sorted(set(candidates))

def _selector_from_list(options: list[str], *, default: str = "", allow_empty: bool = True):
    if options:
        default_value = default if default and default in options else options[0]
        return (options, {"default": default_value})
    if allow_empty:
        return ("STRING", {"default": default})
    return ("STRING", {"default": default, "tooltip": "Provide a file name."})
def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected image tensor in NHWC format, got shape {tuple(image.shape)}")
    return image


def _to_device(tensor: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
    if tensor.device != reference.device:
        tensor = tensor.to(reference.device)
    return tensor


def _latents_from_image(vae, image: torch.Tensor) -> dict:
    """Encode an image tensor to latent space, preserving scaling."""
    encoded = vae.encode(image[:, :, :, :3])
    if isinstance(encoded, dict):
        samples = encoded.get("samples")
    else:
        samples = encoded
    if samples is None:
        raise RuntimeError("VAE.encode did not return a tensor or samples entry.")

    scaling = getattr(vae, "scaling_factor", None)
    if scaling is None:
        scaling = getattr(getattr(vae, "config", object()), "scaling_factor", None)
    if scaling is None:
        # fall back to Stable Diffusion default
        scaling = 0.18215

    return {"samples": samples * scaling}


def _decode_latent(vae, latent: dict) -> torch.Tensor:
    samples = latent.get("samples")
    if samples is None:
        raise RuntimeError("Latent dictionary is missing 'samples' key.")
    return vae.decode(samples)


def _apply_mask(latent: dict, mask: torch.Tensor) -> dict:
    result = latent.copy()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() != 4:
        raise ValueError("Mask must have shape (batch, 1, height, width).")
    result["noise_mask"] = mask
    return result


def _normalize_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    mask = mask.float()
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.clamp(0.0, 1.0)
    if threshold > 0.0:
        mask = (mask >= threshold).float()
    return mask


def _blur_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    padding = radius
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    blurred = F.conv2d(mask, weight, padding=padding)
    blurred = blurred / weight.numel()
    return blurred.clamp(0.0, 1.0)


def _dilate_mask(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    if iterations <= 0:
        return mask
    kernel_size = iterations * 2 + 1
    padding = iterations
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    dilated = F.conv2d(mask, weight, padding=padding)
    return (dilated > 0).float()


def _erode_mask(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    if iterations <= 0:
        return mask
    kernel_size = iterations * 2 + 1
    padding = iterations
    weight = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device, dtype=mask.dtype)
    eroded = F.conv2d(1.0 - mask, weight, padding=padding)
    return (eroded == 0).float()


def _run_segmentation(
    segmentation_model: Any,
    image: torch.Tensor,
    prompt: Optional[str],
    threshold: float,
) -> list:
    if segmentation_model is None:
        raise ValueError("Segmentation model is required when mask input is not provided.")

    callable_candidates = [
        getattr(segmentation_model, "predict_mask", None),
        getattr(segmentation_model, "segment", None),
        getattr(segmentation_model, "predict", None),
    ]
    callable_candidates = [fn for fn in callable_candidates if callable(fn)]

    if callable(segmentation_model):
        callable_candidates.insert(0, segmentation_model)

    if not callable_candidates:
        raise ValueError(
            "Unable to invoke segmentation model. Expected it to provide a callable interface such as "
            "'predict_mask', 'segment', 'predict', or to be directly callable."
        )

    numpy_image = image.detach().cpu().numpy()
    if numpy_image.ndim == 4 and numpy_image.shape[1] == 3:
        numpy_image = numpy_image.transpose(0, 2, 3, 1)

    kwargs_order = []
    if prompt is not None:
        kwargs_order.append({"prompt": prompt, "text": prompt})
    kwargs_order.append({})

    threshold_kwargs = [
        {"threshold": threshold},
        {"score_threshold": threshold},
        {"mask_threshold": threshold},
        {},
    ]

    for fn in callable_candidates:
        for prompt_kwargs in kwargs_order:
            for threshold_kwargs_entry in threshold_kwargs:
                kwargs = {}
                kwargs.update({k: v for k, v in prompt_kwargs.items() if k is not None})
                kwargs.update({k: v for k, v in threshold_kwargs_entry.items() if k is not None})
                try:
                    result = fn(numpy_image, **kwargs)
                except TypeError:
                    try:
                        result = fn(numpy_image)
                    except Exception:
                        continue
                except Exception:
                    continue

                if result is None:
                    continue
                results = result if isinstance(result, (list, tuple)) else [result]
                processed = []
                for item in results:
                    if isinstance(item, dict):
                        if "mask" in item:
                            item = item["mask"]
                        elif "masks" in item:
                            masks_entry = item["masks"]
                            if isinstance(masks_entry, (list, tuple)):
                                processed.extend(masks_entry)
                                continue
                            item = masks_entry
                    if isinstance(item, (list, tuple)):
                        if item:
                            processed.append(item[0])
                        continue
                    processed.append(item)
                if not processed:
                    continue
                return processed

    raise ValueError("Segmentation model did not return mask data in a recognised format.")

def _ensure_tensor_mask(mask: Any) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask
    if isinstance(mask, dict):
        if "mask" in mask:
            mask = mask["mask"]
        elif "masks" in mask:
            masks_entry = mask["masks"]
            if isinstance(masks_entry, (list, tuple)) and masks_entry:
                mask = masks_entry[0]
            else:
                mask = masks_entry
    return torch.as_tensor(mask)


def _prepare_masks(
    mask_candidates: list,
    *,
    batch_size: int,
    threshold: float,
    blur: int,
    dilate: int,
    erode: int,
    device: torch.device,
) -> list[torch.Tensor]:
    prepared = []
    for candidate in mask_candidates:
        if candidate is None:
            continue
        tensor = _ensure_tensor_mask(candidate).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 3 and tensor.shape[0] != batch_size:
            if tensor.shape[0] == 1:
                tensor = tensor.expand(batch_size, -1, -1)
            else:
                continue
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.dim() != 4:
            continue
        tensor = tensor.to(device)
        tensor = tensor.clamp(0.0, 1.0)
        if threshold > 0.0:
            tensor = (tensor >= threshold).float()
        tensor = _dilate_mask(tensor, dilate)
        tensor = _erode_mask(tensor, erode)
        tensor = _blur_mask(tensor, blur)
        tensor = tensor.clamp(0.0, 1.0)
        prepared.append(tensor)
    return prepared



def _resolve_sam_device(requested: str, fallback_device: torch.device) -> str:
    if requested != "auto":
        return requested
    if fallback_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device_type = fallback_device.type
    except AttributeError:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_type in ("cuda", "cpu", "mps"):
        return device_type
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_checkpoint_bundle(checkpoint_name: str, vae_name: str | None) -> Tuple[Any, Any, Any]:
    if not checkpoint_name:
        raise RuntimeError(
            "SegmentationDetailerNode: checkpoint name is required when model/clip/vae are not supplied."
        )
    loader_cls = getattr(nodes, "CheckpointLoaderSimple", None)
    if loader_cls is None:
        loader_cls = nodes.NODE_CLASS_MAPPINGS.get("CheckpointLoaderSimple")
    if loader_cls is None:
        raise RuntimeError("SegmentationDetailerNode: unable to access CheckpointLoaderSimple node.")
    loader = loader_cls()
    model, clip, checkpoint_vae = loader.load_checkpoint(checkpoint_name)
    vae_obj = checkpoint_vae
    if vae_name and vae_name.strip():
        vae_loader_cls = getattr(nodes, "VAELoader", None) or nodes.NODE_CLASS_MAPPINGS.get("VAELoader")
        if vae_loader_cls is None:
            raise RuntimeError("SegmentationDetailerNode: unable to access VAELoader node for custom VAE selection.")
        vae_loader = vae_loader_cls()
        vae_obj = vae_loader.load_vae(vae_name)[0]
    return model, clip, vae_obj


def _load_sam_model(model_name: str, variant: str, device: str, precision: str) -> Any:
    if not model_name:
        return None
    if nodes is None:
        raise RuntimeError("SegmentationDetailerNode: ComfyUI runtime is required to load SAM models.")
    loader_candidates = [
        "DownloadAndLoadSAM2Model",
        "SAMModelLoader",
        "SAMLoader",
        "SegmentAnythingLoader",
    ]
    node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    for key in loader_candidates:
        loader_cls = getattr(nodes, key, None) or node_mappings.get(key)
        if loader_cls is None:
            continue
        loader = loader_cls()
        try:
            if hasattr(loader, "loadmodel"):
                try:
                    result = loader.loadmodel(model_name, variant, device, precision)
                except TypeError:
                    result = loader.loadmodel(model_name)
            elif hasattr(loader, "load_model"):
                try:
                    result = loader.load_model(model_name, device=device)
                except TypeError:
                    result = loader.load_model(model_name)
            elif hasattr(loader, "load"):
                result = loader.load(model_name)
            else:
                continue
        except Exception:
            continue
        if isinstance(result, (list, tuple)):
            return result[0]
        return result
    raise RuntimeError(
        "SegmentationDetailerNode: unable to load SAM model automatically. Provide a segmentation_model input instead."
    )


class SegmentationDetailerNode:
    CATEGORY = "ESS/Image"
    FUNCTION = "detail"
    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_selector = _selector_from_list(_list_files_for_key("checkpoints"))
        vae_selector = _selector_from_list(_list_files_for_key("vae"), default="", allow_empty=True)
        sam_selector = _selector_from_list(_list_sam_models(), default="", allow_empty=True)
        sampler_selector = comfy_samplers.KSampler.SAMPLERS if comfy_samplers else ("euler",)
        scheduler_selector = comfy_samplers.KSampler.SCHEDULERS if comfy_samplers else ("normal",)
        return {
            "required": {
                "image": ("IMAGE",),
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (sampler_selector,),
                "scheduler": (scheduler_selector,),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "checkpoint": checkpoint_selector,
                "vae_name": vae_selector,
                "sam_model_name": sam_selector,
                "sam_variant": (("automatic_mask_generator", "single_image", "video"), {"default": "automatic_mask_generator"}),
                "sam_device": (("auto", "cuda", "cpu", "mps"), {"default": "auto"}),
                "sam_precision": (("fp16", "bf16", "fp32"), {"default": "fp16"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "segmentation_model": ("SEGMENTATION_MODEL",),
                "mask": ("MASK",),
                "segmentation_prompt": ("STRING", {"multiline": False, "default": ""}),
                "segmentation_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_blur": ("INT", {"default": 4, "min": 0, "max": 64}),
                "mask_dilate": ("INT", {"default": 4, "min": 0, "max": 64}),
                "mask_erode": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
        }

    def detail(
        self,
        image: torch.Tensor,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        seed: int,
        checkpoint: str = "",
        vae_name: str = "",
        sam_model_name: str = "",
        sam_variant: str = "automatic_mask_generator",
        sam_device: str = "auto",
        sam_precision: str = "fp16",
        model=None,
        clip=None,
        vae=None,
        segmentation_model: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None,
        segmentation_prompt: str = "",
        segmentation_threshold: float = 0.5,
        mask_blur: int = 4,
        mask_dilate: int = 4,
        mask_erode: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        if nodes is None or model_management is None or comfy_samplers is None:
            raise RuntimeError("SegmentationDetailerNode requires the ComfyUI runtime environment.")

        image = _ensure_batch(image)
        device = model_management.get_torch_device()
        image = image.to(device)

        if model is None or clip is None or vae is None:
            model, clip, vae = _load_checkpoint_bundle(checkpoint, vae_name)

        segmentation_model_obj = segmentation_model
        if segmentation_model_obj is None and sam_model_name:
            resolved_device = _resolve_sam_device(sam_device, device)
            segmentation_model_obj = _load_sam_model(sam_model_name, sam_variant, resolved_device, sam_precision)

        if mask is None and segmentation_model_obj is None:
            raise RuntimeError("SegmentationDetailerNode: provide either a SAM model or an explicit mask.")

        if mask is None:
            mask_candidates = _run_segmentation(segmentation_model_obj, image, segmentation_prompt.strip() or None, segmentation_threshold)
        else:
            mask_candidates = [_ensure_tensor_mask(mask)]

        masks_prepared = _prepare_masks(
            mask_candidates,
            batch_size=image.shape[0],
            threshold=segmentation_threshold,
            blur=mask_blur,
            dilate=mask_dilate,
            erode=mask_erode,
            device=device,
        )

        if not masks_prepared:
            raise RuntimeError("SegmentationDetailerNode: no valid masks were produced by the segmentation model.")

        latent = _latents_from_image(vae, image)
        latent_samples = latent["samples"]
        latent_height, latent_width = latent_samples.shape[-2], latent_samples.shape[-1]

        current_latent = latent
        current_latent["samples"] = latent_samples.to(device)

        clip_text_encoder = nodes.CLIPTextEncode()
        positive = clip_text_encoder.encode(clip, positive_prompt)[0]
        negative = clip_text_encoder.encode(clip, negative_prompt)[0]

        rng_seed = int(seed)
        for index, prepared_mask in enumerate(masks_prepared):
            mask_resized = F.interpolate(
                prepared_mask,
                size=(latent_height, latent_width),
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            latent_with_mask = _apply_mask(current_latent, mask_resized)

            latent_result = nodes.common_ksampler(
                model,
                rng_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_with_mask,
                denoise=denoise,
                disable_noise=False,
                start_step=0,
                last_step=None,
                force_full_denoise=True,
            )

            current_latent = latent_result[0]
            current_latent["samples"] = current_latent["samples"].to(device)

        decoded_image = _decode_latent(vae, current_latent)
        decoded_image = decoded_image.clamp(0.0, 1.0)

        return decoded_image, current_latent
