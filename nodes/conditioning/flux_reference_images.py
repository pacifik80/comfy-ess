"""Multi-image reference conditioning for Flux / Flux 2.

Appends VAE-encoded images as `reference_latents` to a conditioning, and sets the
`reference_latents_method` the model uses to position-embed them.

Per-image influence is controlled by a 0..1 scale that log-interpolates between a
floor pixel area (low impact, few tokens) and a configurable maximum area (full
impact). Order in slots = order the model sees them.
"""

from __future__ import annotations

import logging
import math

import comfy.utils
import node_helpers


logger = logging.getLogger(__name__)


REFERENCE_METHODS = ["index", "offset", "index_timestep_zero", "uxo"]
MIN_TARGET_PIXELS = 256 * 256  # ~256-token reference; floor of the scale slider.
DEFAULT_MAX_TARGET_PIXELS = 1024 * 1024  # 1 MP, the canonical Flux training area.


def _scale_to_pixels(scale: float, max_pixels: int) -> int:
    s = max(0.0, min(1.0, float(scale)))
    min_px = MIN_TARGET_PIXELS
    max_px = max(min_px + 1, int(max_pixels))
    return int(round(min_px * (max_px / min_px) ** s))


def _target_dims(src_h: int, src_w: int, target_pixels: int) -> tuple[int, int]:
    h = max(1, int(src_h))
    w = max(1, int(src_w))
    aspect = w / h
    th = math.sqrt(max(1, target_pixels) / max(1e-9, aspect))
    tw = aspect * th
    # Snap to multiples of 16 (Flux uses 8x VAE + patch_size 2 = 16).
    th = max(16, int(round(th / 16)) * 16)
    tw = max(16, int(round(tw / 16)) * 16)
    return th, tw


def _resize_image(image, target_h: int, target_w: int):
    if image.shape[1] == target_h and image.shape[2] == target_w:
        return image
    img_chw = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(img_chw, target_w, target_h, "bilinear", "disabled")
    return resized.movedim(1, -1)


def _approx_tokens(target_h: int, target_w: int) -> int:
    # Flux patch token count: (H/8/2) * (W/8/2) = H*W / 256
    return (target_h // 16) * (target_w // 16)


class ESSFluxReferenceImages:
    CATEGORY = "ESS/conditioning"
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "summary")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        def slot_scale():
            return ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "Per-image influence. 0 = small (~256px², few tokens, weak presence). "
                           "1 = max_target_pixels (full influence). Log-scaled between the two.",
            })
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_method": (REFERENCE_METHODS, {
                    "default": "index",
                    "tooltip": "How reference latents are position-encoded into the transformer "
                               "sequence. 'index' gives each image its own positional slot "
                               "(recommended for multi-ref). 'offset' is the Kontext default. "
                               "'index_timestep_zero' pins refs at timestep 0. 'uxo' for UXO/UNO-style models.",
                }),
                "max_target_pixels": ("INT", {
                    "default": DEFAULT_MAX_TARGET_PIXELS,
                    "min": MIN_TARGET_PIXELS, "max": 4 * 1024 * 1024, "step": 65536,
                    "tooltip": "Upper bound mapped to scale=1.0. Default 1024² = 1 MP (canonical Flux). "
                               "Raising this lets a single image dominate token budget.",
                }),
                "image_1_scale": slot_scale(),
                "image_2_scale": slot_scale(),
                "image_3_scale": slot_scale(),
                "image_4_scale": slot_scale(),
                "equalize_token_budget": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Override per-image scales: all wired images get the same target area "
                               "(average of the wired slots' scales). Prevents an oversized image "
                               "from accidentally dominating.",
                }),
                "debug_log": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the same summary string to the console.",
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    def apply(self, conditioning, vae, reference_method,
              max_target_pixels,
              image_1_scale, image_2_scale, image_3_scale, image_4_scale,
              equalize_token_budget, debug_log,
              image_1=None, image_2=None, image_3=None, image_4=None):

        slots = [
            ("image_1", image_1, image_1_scale),
            ("image_2", image_2, image_2_scale),
            ("image_3", image_3, image_3_scale),
            ("image_4", image_4, image_4_scale),
        ]
        wired = [(name, img, sc) for (name, img, sc) in slots if img is not None]

        if not wired:
            summary = "0 references — pass-through (no images wired)"
            if debug_log:
                logger.info("[ess_flux_ref] %s", summary)
                print(f"[ess_flux_ref] {summary}")
            return (conditioning, summary)

        if equalize_token_budget:
            avg = sum(sc for _, _, sc in wired) / len(wired)
            scales_used = [avg] * len(wired)
        else:
            scales_used = [sc for _, _, sc in wired]

        cond = conditioning
        rows = []
        for (name, img, original_sc), eff_sc in zip(wired, scales_used):
            src_h = int(img.shape[1])
            src_w = int(img.shape[2])
            target_pixels = _scale_to_pixels(eff_sc, max_target_pixels)
            th, tw = _target_dims(src_h, src_w, target_pixels)
            img_r = _resize_image(img, th, tw)
            latent = vae.encode(img_r)
            cond = node_helpers.conditioning_set_values(
                cond, {"reference_latents": [latent]}, append=True,
            )
            tokens = _approx_tokens(th, tw)
            scale_note = f"scale={original_sc:.2f}"
            if equalize_token_budget and not math.isclose(original_sc, eff_sc):
                scale_note += f"→{eff_sc:.2f}"
            rows.append(f"  {name}: {src_w}×{src_h} → {tw}×{th}  ({scale_note}, ~{tokens} tokens)")

        cond = node_helpers.conditioning_set_values(
            cond, {"reference_latents_method": reference_method},
        )

        summary = (
            f"{len(wired)} ref{'s' if len(wired) != 1 else ''} · method={reference_method}"
            f"{' · equalized' if equalize_token_budget else ''}\n"
            + "\n".join(rows)
        )
        if debug_log:
            logger.info("[ess_flux_ref] %s", summary)
            print(f"[ess_flux_ref]\n{summary}")

        return (cond, summary)
