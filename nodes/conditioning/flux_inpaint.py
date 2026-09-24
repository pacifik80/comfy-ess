"""All-in-one Flux 2 inpainting helper.

Bundles VAE encoding, latent noise-mask attachment, optional mask grow/blur,
and (optionally) appending the source image as a reference latent so the model
preserves identity / lighting / style of unchanged regions.

Wire after CLIPTextEncode(+FluxGuidance) and feed the resulting CONDITIONING
plus LATENT into ESS Flux Sampler with denoise < 1.0.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

import comfy.utils
import node_helpers

from .flux_reference_images import (
    DEFAULT_MAX_TARGET_PIXELS,
    MIN_TARGET_PIXELS,
    REFERENCE_METHODS,
    _approx_tokens,
    _resize_image,
    _scale_to_pixels,
    _target_dims,
)


logger = logging.getLogger(__name__)


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return mask shape (B, H, W) float in [0, 1]."""
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    elif m.dim() == 4 and m.shape[1] == 1:
        m = m.squeeze(1)
    return m.float().clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    if mask.shape[-2] == target_h and mask.shape[-1] == target_w:
        return mask
    m4 = mask.unsqueeze(1)
    m4 = F.interpolate(m4, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return m4.squeeze(1).clamp(0.0, 1.0)


def _grow_mask(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel = int(pixels) * 2 + 1
    m4 = mask.unsqueeze(1)
    grown = F.max_pool2d(m4, kernel_size=kernel, stride=1, padding=int(pixels))
    return grown.squeeze(1).clamp(0.0, 1.0)


def _gaussian_kernel_1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    k = k / k.sum()
    return k


def _blur_mask(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return mask
    m4 = mask.unsqueeze(1).float()
    k1d = _gaussian_kernel_1d(sigma, m4.device, m4.dtype)
    radius = (k1d.numel() - 1) // 2
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    pad = (radius, radius, radius, radius)
    padded = F.pad(m4, pad, mode="replicate")
    blurred = F.conv2d(padded, kx)
    blurred = F.conv2d(blurred, ky)
    return blurred.squeeze(1).clamp(0.0, 1.0)


class ESSFluxInpaint:
    CATEGORY = "ESS/conditioning"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("conditioning", "latent", "summary")
    FUNCTION = "prepare"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "use_as_reference": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also append the source image as a reference latent. Helps preserve "
                               "identity/lighting/style in the unchanged regions; usually on.",
                }),
                "reference_scale": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "How strongly the source image biases the result as a reference. "
                               "0.7 is balanced; 1.0 can resist prompt edits; <0.5 lets the prompt dominate.",
                }),
                "reference_method": (REFERENCE_METHODS, {
                    "default": "index",
                    "tooltip": "Positional encoding policy for the reference latent. 'index' is the "
                               "best default for Flux 2 inpaint.",
                }),
                "max_target_pixels": ("INT", {
                    "default": DEFAULT_MAX_TARGET_PIXELS,
                    "min": MIN_TARGET_PIXELS, "max": 4 * 1024 * 1024, "step": 65536,
                    "tooltip": "Upper area bound for the reference latent (reference_scale=1.0 maps here).",
                }),
                "mask_grow": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Dilate the mask by N pixels (max-pool). Helps capture edges/halos. "
                               "Try 4–8 for small object replacement.",
                }),
                "mask_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "Gaussian blur sigma applied to the mask. Softens the seam between "
                               "edited and unchanged regions. Try 2–6 for natural blending.",
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the summary string to the console.",
                }),
                # Appended last on purpose: inserting mid-list shifts every later
                # widget's stored value on existing nodes.
                "masked_content": ((("gray", "noise", "original")), {
                    "default": "gray",
                    "tooltip": "What to put in the masked area BEFORE VAE-encoding the inpaint target. "
                               "'gray'/'noise' erase the original so the model actually regenerates the "
                               "region (recommended). 'original' keeps it, which only changes the region "
                               "at denoise=1.0 and otherwise reconstructs what was already there.",
                }),
            },
        }

    def prepare(self, conditioning, vae, image, mask,
                use_as_reference, reference_scale, reference_method, max_target_pixels,
                mask_grow, mask_blur, debug_log, masked_content="gray"):

        # Normalize and align the mask to image dimensions.
        src_h = int(image.shape[1])
        src_w = int(image.shape[2])

        m = _normalize_mask(mask)
        m = _resize_mask(m, src_h, src_w)
        m = _grow_mask(m, int(mask_grow))
        m = _blur_mask(m, float(mask_blur))

        # Optionally erase the masked region before encoding so the model has no
        # original content to reconstruct (mirrors VAEEncodeForInpaint). Done after
        # grow/blur so the same editable region is what gets erased.
        image_for_encode = image
        if masked_content != "original":
            m3 = m.reshape(m.shape[0], m.shape[1], m.shape[2], 1)
            if masked_content == "noise":
                fill = torch.rand_like(image)
            else:  # "gray"
                fill = torch.full_like(image, 0.5)
            image_for_encode = image * (1.0 - m3) + fill * m3

        # Encode the (optionally erased) image for the inpaint latent target.
        full_latent = vae.encode(image_for_encode)

        # Comfy stores noise_mask at image resolution (B, 1, H, W); the sampler
        # downsamples it internally to latent resolution.
        noise_mask = m.reshape((-1, 1, m.shape[-2], m.shape[-1]))

        latent_out = {"samples": full_latent, "noise_mask": noise_mask}

        cond = conditioning
        ref_summary = "ref=off"
        if use_as_reference:
            target_px = _scale_to_pixels(reference_scale, max_target_pixels)
            ref_h, ref_w = _target_dims(src_h, src_w, target_px)
            ref_img = _resize_image(image, ref_h, ref_w)
            ref_latent = vae.encode(ref_img)
            cond = node_helpers.conditioning_set_values(
                cond, {"reference_latents": [ref_latent]}, append=True,
            )
            cond = node_helpers.conditioning_set_values(
                cond, {"reference_latents_method": reference_method},
            )
            tokens = _approx_tokens(ref_h, ref_w)
            ref_summary = (f"ref={ref_w}×{ref_h} (scale={reference_scale:.2f}, "
                           f"~{tokens} tokens, method={reference_method})")

        mask_coverage = float(m.mean().item())
        summary = (
            f"inpaint · src={src_w}×{src_h} · masked={masked_content} · "
            f"mask coverage={mask_coverage * 100:.1f}% "
            f"(grow={mask_grow}, blur={mask_blur:.1f}) · {ref_summary}"
        )

        if mask_coverage < 1e-4:
            summary += "  ⚠ mask is empty — sampling will be a no-op"

        if debug_log:
            logger.info("[ess_flux_inpaint] %s", summary)
            print(f"[ess_flux_inpaint] {summary}")

        return (cond, latent_out, summary)
