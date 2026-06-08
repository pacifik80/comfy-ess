"""All-in-one Flux sampler with sigma shift, guidance ramp, true-CFG window,
detail boost, async preview, SDPA backend, precision and torch.compile.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from contextlib import nullcontext

import torch

import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.model_sampling
import comfy.utils
import latent_preview


logger = logging.getLogger(__name__)


PRECISION_OPTIONS = ["auto", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"]
SDPA_OPTIONS = ["auto", "flash", "mem_efficient", "math"]


# ----------------------------- guider -----------------------------------------


class _FluxScheduledGuider(comfy.samplers.CFGGuider):
    """CFGGuider with per-step guidance, true-CFG window and sigma boost."""

    def set_extras(self, sigmas_ref, guidance_schedule, true_cfg, cfg_start, cfg_end, sigma_boost):
        self._sigmas_ref = sigmas_ref.detach().to("cpu") if sigmas_ref is not None else None
        self._guidance_schedule = list(guidance_schedule) if guidance_schedule else None
        self._true_cfg = float(true_cfg)
        self._cfg_start = int(cfg_start)
        self._cfg_end = int(cfg_end)
        self._sigma_boost = float(sigma_boost)

    def _step_from_sigma(self, timestep):
        if self._sigmas_ref is None or self._sigmas_ref.numel() < 2:
            return 0
        s = float(timestep.detach().flatten()[0].cpu().item())
        diffs = (self._sigmas_ref[:-1] - s).abs()
        return int(diffs.argmin().item())

    def _apply_guidance(self, step):
        if not self._guidance_schedule:
            return
        idx = max(0, min(len(self._guidance_schedule) - 1, step))
        value = float(self._guidance_schedule[idx])
        for name in ("positive", "negative"):
            for cond_entry in self.conds.get(name, []) or []:
                model_conds = cond_entry.get("model_conds")
                if not model_conds:
                    continue
                g = model_conds.get("guidance")
                if g is None or not hasattr(g, "cond") or g.cond is None:
                    continue
                g.cond = torch.full_like(g.cond, value)

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        model_options = model_options if model_options is not None else {}
        step = self._step_from_sigma(timestep)

        self._apply_guidance(step)

        if not math.isclose(self._sigma_boost, 1.0):
            timestep = timestep * self._sigma_boost

        in_window = self._cfg_end >= self._cfg_start and self._cfg_start <= step <= self._cfg_end
        if in_window and self._true_cfg > 1.0 and self.conds.get("negative") is not None:
            cond_scale = self._true_cfg
            uncond = self.conds.get("negative")
        else:
            cond_scale = 1.0
            uncond = None

        return comfy.samplers.sampling_function(
            self.inner_model, x, timestep,
            uncond, self.conds.get("positive"), cond_scale,
            model_options=model_options, seed=seed,
        )


# ----------------------------- helpers ----------------------------------------


def _patch_sigma_shift(model, shift_value):
    if shift_value is None or shift_value <= 0.0:
        return model
    m = model.clone()
    sampling_base = comfy.model_sampling.ModelSamplingFlux
    sampling_type = comfy.model_sampling.CONST

    class _Adv(sampling_base, sampling_type):
        pass

    model_sampling = _Adv(model.model.model_config)
    model_sampling.set_parameters(shift=float(shift_value))
    m.add_object_patch("model_sampling", model_sampling)
    return m


def _linear_guidance(start: float, end: float, total_steps: int):
    if total_steps <= 1 or math.isclose(start, end):
        return [float(start)] * max(1, total_steps)
    span = end - start
    return [float(start + span * (i / (total_steps - 1))) for i in range(total_steps)]


def _make_async_callback(model, total_steps, preview_every, x0_output):
    """Sampling-loop callback that offloads preview decode + websocket emit
    to a background thread. The main thread never blocks on JPEG encode or send.

    Returns (callback, shutdown). Call shutdown() after sampling completes.
    """
    pbar = comfy.utils.ProgressBar(total_steps)
    previewer = None
    if preview_every > 0:
        try:
            previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
        except Exception as exc:
            logger.warning("[ess_flux_sampler] preview disabled: %s", exc)
            previewer = None

    if previewer is None or preview_every <= 0:
        def cb(step, x0, x, total):
            if x0_output is not None:
                x0_output["x0"] = x0
            pbar.update_absolute(step + 1, total, None)
        return cb, (lambda: None)

    work_q: "queue.Queue" = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            try:
                item = work_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            x0_slice, step, total = item
            preview_bytes = None
            try:
                preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0_slice)
            except Exception as exc:
                logger.warning("[ess_flux_sampler] preview decode failed: %s", exc)
            try:
                pbar.update_absolute(step + 1, total, preview_bytes)
            except Exception as exc:
                logger.warning("[ess_flux_sampler] pbar update failed: %s", exc)

    thread = threading.Thread(target=worker, name="ess_flux_preview", daemon=True)
    thread.start()

    def cb(step, x0, x, total):
        if x0_output is not None:
            x0_output["x0"] = x0
        do_preview = (((step + 1) % preview_every) == 0) or ((step + 1) >= total)
        if not do_preview:
            pbar.update_absolute(step + 1, total, None)
            return
        # Take a small detached copy so the worker can read it safely while
        # sampling mutates x0 on subsequent steps.
        try:
            x0_for_preview = x0[:1].detach().clone()
        except Exception:
            x0_for_preview = x0
        item = (x0_for_preview, step, total)
        # Drop-on-overrun: if a stale frame is still pending, replace it.
        try:
            work_q.put_nowait(item)
        except queue.Full:
            try:
                work_q.get_nowait()
            except queue.Empty:
                pass
            try:
                work_q.put_nowait(item)
            except queue.Full:
                pass

    def shutdown():
        stop_event.set()
        try:
            work_q.put_nowait(None)
        except queue.Full:
            try:
                work_q.get_nowait()
            except queue.Empty:
                pass
            try:
                work_q.put_nowait(None)
            except queue.Full:
                pass
        thread.join(timeout=3.0)

    return cb, shutdown


def _sdpa_context(backend: str):
    if backend == "auto":
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        return nullcontext()
    mapping = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
    }
    selected = mapping.get(backend)
    if selected is None:
        return nullcontext()
    try:
        return sdpa_kernel(selected)
    except Exception as exc:
        logger.warning("[ess_flux_sampler] sdpa_kernel(%s) unavailable: %s", backend, exc)
        return nullcontext()


def _autocast_context(precision: str):
    if precision in ("auto", "fp8_e4m3fn", "fp8_e5m2"):
        # fp8 is selected at model-load; autocast doesn't accept fp8 dtypes.
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _maybe_compile(model, enabled: bool):
    if not enabled:
        return model
    import importlib.util
    if importlib.util.find_spec("triton") is None:
        logger.warning(
            "[ess_flux_sampler] torch_compile requested but 'triton' is not installed; "
            "skipping compile (Triton is required by inductor and is not available on Windows by default)."
        )
        return model
    try:
        from comfy_api.torch_helpers.torch_compile import set_torch_compile_wrapper
        from comfy_extras.nodes_torch_compile import skip_torch_compile_dict
        m = model.clone(disable_dynamic=True)
        set_torch_compile_wrapper(model=m, backend="inductor",
                                  options={"guard_filter_fn": skip_torch_compile_dict})
        return m
    except Exception as exc:
        logger.warning("[ess_flux_sampler] torch.compile unavailable: %s", exc)
        return model


def _decode_image(vae, samples):
    img = vae.decode(samples)
    if img.dim() == 5:
        img = img.movedim(2, 1).reshape(-1, *img.shape[3:])
    return img


# ------------------------------ node ------------------------------------------


class ESSFluxSampler:
    CATEGORY = "ESS/sampling"
    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                  "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "euler"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sigma_shift": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "tooltip": "Flux time-shift (mu). 0 = leave model as-is. Typical 1.0-1.5 at 1024px, higher for larger.",
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Flux distilled guidance at the FIRST step.",
                }),
                "guidance_end": ("FLOAT", {
                    "default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Flux distilled guidance at the LAST step. Equal to guidance = constant; otherwise linear ramp.",
                }),
                "true_cfg": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Real CFG with a negative cond. 1.0 = distilled only (recommended for Flux dev). Requires the optional 'negative' input to be wired to have any effect.",
                }),
                "true_cfg_until": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of steps from the start where true CFG is active. Has no effect unless true_cfg > 1.0 AND negative is wired.",
                }),
                "detail_boost": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Negative softens, positive sharpens. Internally rescales the sigma seen by the model.",
                }),
                "preview_every": ("INT", {
                    "default": 2, "min": 0, "max": 1000,
                    "tooltip": "Emit preview every N steps (decoded off-thread, never blocks sampling). 0 = no preview.",
                }),
                "precision": (PRECISION_OPTIONS, {"default": "auto"}),
                "sdpa_backend": (SDPA_OPTIONS, {"default": "auto"}),
                "torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Experimental: inductor compile. Requires 'triton' (not available on Windows by default — silently skipped if missing). First step slow, subsequent faster.",
                }),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
        }

    def sample(self, model, positive, latent_image, seed, steps,
               sampler_name, scheduler, denoise,
               sigma_shift, guidance, guidance_end, true_cfg, true_cfg_until,
               detail_boost, preview_every, precision, sdpa_backend, torch_compile,
               negative=None, vae=None):

        m = _patch_sigma_shift(model, sigma_shift) if sigma_shift > 0.0 else model
        m = _maybe_compile(m, torch_compile)

        if denoise <= 0.0:
            out_latent = latent_image.copy()
            image_out = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
            if vae is not None:
                image_out = _decode_image(vae, out_latent["samples"])
            return (out_latent, image_out)

        total_for_schedule = steps if denoise >= 1.0 else max(1, int(round(steps / max(denoise, 1e-3))))
        sigmas = comfy.samplers.calculate_sigmas(
            m.get_model_object("model_sampling"), scheduler, total_for_schedule
        ).cpu()
        sigmas = sigmas[-(steps + 1):]

        guidance_list = _linear_guidance(float(guidance), float(guidance_end), steps)

        # true_cfg requires a negative cond; silently downgrade to distilled-only if missing.
        true_cfg_active = true_cfg > 1.0 and true_cfg_until > 0.0 and negative is not None
        if true_cfg_active:
            cfg_end = max(0, min(steps - 1, int(round(steps * true_cfg_until)) - 1))
            cfg_start = 0
        else:
            cfg_start, cfg_end = 0, -1

        sigma_factor = 1.0 + 0.25 * float(detail_boost)

        sampler = comfy.samplers.sampler_object(sampler_name)

        guider = _FluxScheduledGuider(m)
        if negative is None:
            guider.inner_set_conds({"positive": positive})
        else:
            guider.set_conds(positive, negative)
        guider.set_cfg(float(true_cfg))
        guider.set_extras(
            sigmas_ref=sigmas,
            guidance_schedule=guidance_list,
            true_cfg=float(true_cfg),
            cfg_start=cfg_start,
            cfg_end=cfg_end,
            sigma_boost=sigma_factor,
        )

        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(m, latent_samples)
        latent["samples"] = latent_samples
        noise_mask = latent.get("noise_mask", None)

        from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
        noise_obj = Noise_RandomNoise(seed)
        noise = noise_obj.generate_noise(latent)

        x0_output = {}
        callback, preview_shutdown = _make_async_callback(m, steps, preview_every, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        try:
            with _sdpa_context(sdpa_backend), _autocast_context(precision):
                samples = guider.sample(
                    noise, latent_samples, sampler, sigmas,
                    denoise_mask=noise_mask, callback=callback,
                    disable_pbar=disable_pbar, seed=seed,
                )
        finally:
            preview_shutdown()
        samples = samples.to(comfy.model_management.intermediate_device())

        out_latent = latent.copy()
        out_latent["samples"] = samples

        if vae is not None:
            image_out = _decode_image(vae, samples)
        else:
            image_out = torch.zeros((1, 8, 8, 3), dtype=torch.float32)

        return (out_latent, image_out)
