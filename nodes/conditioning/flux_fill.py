"""ESS Flux Fill — FLUX.1 Fill inpaint/outpaint in one node.

Wire MODEL (UNETLoader) + CLIP (DualCLIPLoader, clip_l + t5xxl, type 'flux') + VAE,
plus an image + mask (+ optional prompt), and it runs ComfyUI's *official*
"Flux.1 Fill Dev Image Inpainting" pipeline internally and returns the filled image:

    model -> DifferentialDiffusion
    clip  -> CLIPTextEncode(prompt) -> FluxGuidance(30) -> positive
                                  ConditioningZeroOut -> negative
    InpaintModelConditioning(positive, negative, vae, image, mask, noise_mask=True)
    KSampler(model, cfg=1, euler, normal, denoise) -> VAEDecode -> image

It reuses ComfyUI's own stock node classes, so it behaves exactly like that working
template — just collapsed into one node. Models come from standard loader nodes (so
ComfyUI caches them and avoids reloads); the text conditioning is cached so iterating
seed/steps/denoise/image skips the T5 encode.
"""

from __future__ import annotations

import folder_paths
import node_helpers


# Cache the (converted) LoRA state dict by path so the file read isn't repeated each run.
_LORA_SD_CACHE: dict = {}


def _load_lora_sd(path: str):
    sd = _LORA_SD_CACHE.get(path)
    if sd is None:
        import comfy.lora_convert
        import comfy.utils
        sd = comfy.lora_convert.convert_lora(comfy.utils.load_torch_file(path, safe_load=True))
        _LORA_SD_CACHE.clear()  # keep only the current one
        _LORA_SD_CACHE[path] = sd
    return sd


def _apply_loras(model, clip, lora_specs):
    """Apply (name, strength) LoRAs to model+clip, skipping 'None'/zero.

    Drops the input-projection patch (``img_in`` / diffusers ``x_embedder``): a base
    FLUX.1-dev accelerator LoRA (Hyper/Turbo) carries an ``img_in`` delta shaped for the
    64-channel base input, which is invalid on a Fill model's 384-channel ``img_in`` (and
    would be the wrong adaptation anyway). The transformer-block weights — where the
    step-reduction actually lives — still apply cleanly, so the accelerator works without
    the per-step shape error.
    """
    import logging
    import comfy.lora

    for name, strength in lora_specs:
        if not name or name == "None" or abs(float(strength)) < 1e-6:
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            logging.warning("[ess_flux_fill] LoRA not found: %s", name)
            continue
        lora_sd = _load_lora_sd(path)
        key_map = {}
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        key_map = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
        loaded = comfy.lora.load_lora(lora_sd, key_map)

        dropped = [k for k in list(loaded) if "img_in" in k or "x_embedder" in k]
        for k in dropped:
            loaded.pop(k, None)
        if dropped:
            logging.info("[ess_flux_fill] LoRA '%s': skipped %d input-proj patch(es) incompatible "
                         "with Fill (%s)", name, len(dropped), dropped)

        model = model.clone()
        model.add_patches(loaded, float(strength))
        clip = clip.clone()
        clip.add_patches(loaded, float(strength))
    return model, clip


# Text conditioning is image-independent, so cache it: iterating seed/steps/denoise
# (or running on a new image) then reuses it instead of paying the T5 encode again.
_COND_CACHE: dict = {}


def _encode_base(clip, prompt, guidance):
    import nodes
    positive = nodes.CLIPTextEncode().encode(clip, prompt)[0]
    positive = node_helpers.conditioning_set_values(positive, {"guidance": float(guidance)})
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    return (positive, negative)


def _cached_conditioning(key, compute):
    hit = _COND_CACHE.get(key)
    if hit is not None:
        return hit
    _COND_CACHE.clear()  # keep one entry; this is the heavy T5 result
    val = compute()
    _COND_CACHE[key] = val
    return val


def _apply_differential_diffusion(model):
    """Replicate comfy_extras DifferentialDiffusion (strength=1.0) without the v3 node.

    Patches the model's denoise-mask function so the soft mask is thresholded per
    timestep — what makes Fill blend cleanly across feathered/outpaint masks.
    """
    model = model.clone()

    def _denoise_mask_fn(sigma, denoise_mask, extra_options):
        inner = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = inner.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]
        ts_from = inner.inner_model.model_sampling.timestep(sigma_from)
        ts_to = inner.inner_model.model_sampling.timestep(sigma_to)
        current_ts = inner.inner_model.model_sampling.timestep(sigma[0])
        threshold = (current_ts - ts_to) / (ts_from - ts_to)
        return (denoise_mask >= threshold).to(denoise_mask.dtype)

    model.set_model_denoise_mask_function(_denoise_mask_fn)
    return model


def _snap(a0: int, a1: int, limit: int, mult: int = 8):
    """Grow [a0,a1) so its length is a multiple of `mult` and it fits in [0, limit)."""
    length = a1 - a0
    length = ((length + mult - 1) // mult) * mult
    length = min(length, (limit // mult) * mult)  # largest multiple that fits
    length = max(length, mult)
    a0 = max(0, min(a0, limit - length))
    return a0, a0 + length


def _mask_bbox(mask, image_h: int, image_w: int, padding: int, mult: int = 8):
    """Bounding box (x0,y0,x1,y1) of the masked area, padded for context and snapped
    to a multiple of `mult` so the crop needs no further alignment. None if mask empty."""
    import torch

    m = mask
    if m.dim() == 4:
        m = m[:, 0]
    if m.dim() == 3:
        m = m[0]
    if m.shape[-2] != image_h or m.shape[-1] != image_w:
        m = torch.nn.functional.interpolate(
            m.reshape(1, 1, m.shape[-2], m.shape[-1]).float(),
            size=(image_h, image_w), mode="bilinear",
        )[0, 0]

    ys, xs = torch.where(m > 0.5)
    if ys.numel() == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(image_w, x1 + padding), min(image_h, y1 + padding)
    x0, x1 = _snap(x0, x1, image_w, mult)
    y0, y1 = _snap(y0, y1, image_h, mult)
    return (x0, y0, x1, y1)


class ESSFluxFill:
    CATEGORY = "ESS/Image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "fill"

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "FLUX.1 Fill UNET (from a UNETLoader). LoRAs / differential "
                                               "diffusion are applied here."}),
                "clip": ("CLIP", {"tooltip": "One CLIP from a DualCLIPLoader (clip_l + t5xxl, type 'flux'). "
                                             "It already bundles both text encoders."}),
                "vae": ("VAE", {"tooltip": "Flux VAE (ae.safetensors) from a VAELoader."}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Optional. What should fill the masked area / the scene. Empty is fine for "
                               "plain background outpainting. Right-click -> Convert to input to drive it "
                               "from JoyCaption.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "guidance": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Flux Fill guidance. 30 is the template default; drop to ~20-25 if you see "
                               "color drift outside the mask.",
                }),
                "region_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run Flux only on the masked area's bounding box (+ padding) instead of the "
                               "whole image, then composite back. Big speedup when the gap is small relative "
                               "to the frame (e.g. outpaint borders). Turn off to denoise the full image.",
                }),
                "region_padding": ("INT", {
                    "default": 64, "min": 0, "max": 512, "step": 8,
                    "tooltip": "Context margin (px) around the masked area when region_only is on. More "
                               "context = better coherence but slower.",
                }),
                "lora_1": (loras, {
                    "default": "None",
                    "tooltip": "Optional LoRA applied to the model (e.g. an 8-step accelerator). "
                               "Hyper-FLUX.1-dev ~0.125 strength; FLUX.1-Turbo-Alpha ~1.0. With these, set "
                               "steps to ~8. Note: accelerators are tuned for low guidance (~3.5), so try "
                               "lowering 'guidance' if a Fill accelerator looks off.",
                }),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "lora_2": (loras, {"default": "None", "tooltip": "Optional second LoRA (e.g. a style LoRA)."}),
                "lora_2_strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
            },
        }

    def fill(self, model, clip, vae, image, mask, prompt, seed, steps, denoise,
             guidance=30.0, region_only=True, region_padding=64,
             lora_1="None", lora_1_strength=1.0, lora_2="None", lora_2_strength=1.0):
        clip_sig = ("input", id(clip))

        # --- optional LoRAs (e.g. 8-step accelerators) on a patched model/clip copy ---
        lora_specs = [(lora_1, lora_1_strength), (lora_2, lora_2_strength)]
        model, clip = _apply_loras(model, clip, lora_specs)

        # --- model patch: differential diffusion (soft-mask inpaint) ---
        model = _apply_differential_diffusion(model)

        # --- text conditioning (cached: skips T5 when only seed/steps/denoise/image change) ---
        lora_sig = tuple((n, round(float(s), 4)) for n, s in lora_specs)
        cond_key = (clip_sig, lora_sig, str(prompt), round(float(guidance), 4))
        positive, negative = _cached_conditioning(cond_key, lambda: _encode_base(clip, prompt, guidance))

        bbox = _mask_bbox(mask, int(image.shape[1]), int(image.shape[2]), int(region_padding)) if region_only else None

        if bbox is None:
            image_out = self._run_fill(model, vae, image, mask, positive, negative, seed, steps, denoise)
            return (image_out,)

        # --- region-only: crop -> fill the gap -> composite back ---
        import torch
        x0, y0, x1, y1 = bbox
        img_crop = image[:, y0:y1, x0:x1, :]
        m4 = mask
        if m4.dim() == 2:
            m4 = m4.unsqueeze(0)
        if m4.dim() == 4:
            m4 = m4[:, 0]
        if m4.shape[-2] != image.shape[1] or m4.shape[-1] != image.shape[2]:
            m4 = torch.nn.functional.interpolate(
                m4.reshape(-1, 1, m4.shape[-2], m4.shape[-1]).float(),
                size=(image.shape[1], image.shape[2]), mode="bilinear",
            )[:, 0]
        mask_crop = m4[:, y0:y1, x0:x1]

        filled_crop = self._run_fill(model, vae, img_crop, mask_crop, positive, negative, seed, steps, denoise)

        # Composite only inside the mask (alpha = mask) so untouched context stays exact
        # and the feathered edge blends; avoids VAE round-trip drift in context pixels.
        alpha = mask_crop[:1].unsqueeze(-1).to(filled_crop.dtype)  # (1, h, w, 1)
        out = image.clone()
        out[:, y0:y1, x0:x1, :] = filled_crop * alpha + img_crop * (1.0 - alpha)
        return (out,)

    def _run_fill(self, model, vae, image, mask, positive, negative, seed, steps, denoise):
        import nodes

        positive, negative, latent = nodes.InpaintModelConditioning().encode(
            positive, negative, image, vae, mask, noise_mask=True,
        )

        # Diagnostic: model's expected img_in width vs what the conditioning feeds.
        # Flux.1 Fill wants img_in.in_features = 384 (= 96 latent channels x 4).
        try:
            import logging
            dm = model.model.diffusion_model
            img_in = getattr(dm, "img_in", None)
            in_feat = getattr(img_in, "in_features", None)
            cli = positive[0][1].get("concat_latent_image")
            cm = positive[0][1].get("concat_mask")
            samples = latent.get("samples")
            logging.warning(
                "[ess_flux_fill] DIAG img_in.in_features=%s | concat_latent_image=%s | concat_mask=%s | samples=%s",
                in_feat,
                tuple(cli.shape) if cli is not None else None,
                tuple(cm.shape) if cm is not None else None,
                tuple(samples.shape) if samples is not None else None,
            )
        except Exception as _diag_exc:  # never let diagnostics break the run
            import logging
            logging.warning("[ess_flux_fill] DIAG unavailable: %s", _diag_exc)

        out_latent = nodes.KSampler().sample(
            model, int(seed), int(steps), 1.0, "euler", "normal",
            positive, negative, latent, denoise=float(denoise),
        )[0]
        return nodes.VAEDecode().decode(vae, out_latent)[0]
