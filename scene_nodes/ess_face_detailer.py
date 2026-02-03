"""ESS Face Detailer - local copy of Impact Pack FaceDetailer."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import comfy.samplers
import nodes
import numpy as np
import torch
from comfy_extras import nodes_differential_diffusion
from nodes import MAX_RESOLUTION

_custom_nodes_dir = Path(__file__).resolve().parents[2]
_impact_modules_dir = _custom_nodes_dir / "comfyui-impact-pack" / "modules"
if _impact_modules_dir.exists():
    sys.path.append(os.fspath(_impact_modules_dir))

try:
    import impact.core as core
    from impact.core import SEG
    import impact.utils as utils
except Exception as exc:
    raise RuntimeError(
        "ESSFaceDetailer requires ComfyUI-Impact-Pack. Install it or ensure its modules path is available."
    ) from exc

try:
    import insightface  # type: ignore
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:  # pragma: no cover
    insightface = None
    FaceAnalysis = None


_INSIGHTFACE_PROFILES = ("buffalo_l", "buffalo_m", "buffalo_s", "antelopev2")
_DETECTOR_CACHE: dict[tuple[str, str, float], object] = {}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _providers_for_device(device: str) -> list[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_face_detector(profile: str, device: str, det_thresh: float):
    if FaceAnalysis is None:
        raise RuntimeError("ESSFaceDetailer requires insightface for gender-aware detection.")
    key = (profile, device, det_thresh)
    cached = _DETECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    detector = FaceAnalysis(name=profile, providers=_providers_for_device(device))
    ctx_id = 0 if device == "cuda" else -1
    detector.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(640, 640))
    _DETECTOR_CACHE[key] = detector
    return detector


def _gender_matches(face_obj, gender: str) -> bool:
    if gender == "any":
        return True
    face_gender = getattr(face_obj, "gender", None)
    if face_gender is None:
        return False
    return (gender == "male" and face_gender == 1) or (gender == "female" and face_gender == 0)


def _select_largest_face(faces, gender: str):
    if not faces:
        return None
    filtered = [face for face in faces if _gender_matches(face, gender)]
    if not filtered:
        return None
    return max(
        filtered,
        key=lambda f: max(0.0, (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
    )


def _detect_face_segs(
    image,
    detector_profile: str,
    detector_device: str,
    det_thresh: float,
    gender: str,
    bbox_dilation: int,
    bbox_crop_factor: float,
    drop_size: int,
):
    h = image.shape[1]
    w = image.shape[2]
    shape = (h, w)

    frame = image[0].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (frame * 255.0).round().clip(0, 255).astype(np.uint8)
    frame_bgr = frame[:, :, ::-1].copy()

    device = _resolve_device(detector_device)
    detector = _load_face_detector(detector_profile, device, det_thresh)
    faces = detector.get(frame_bgr)
    selected = _select_largest_face(faces, gender)
    if selected is None:
        return (shape, [])

    x1, y1, x2, y2 = [float(v) for v in selected.bbox]
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(x1 + 1, min(w, round(x2))))
    y2 = int(max(y1 + 1, min(h, round(y2))))

    if x2 - x1 <= drop_size or y2 - y1 <= drop_size:
        return (shape, [])

    bbox = [x1, y1, x2, y2]
    crop_region = utils.make_crop_region(w, h, bbox, bbox_crop_factor)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
    cropped_mask = np.zeros((crop_y2 - crop_y1, crop_x2 - crop_x1), dtype=np.float32)
    cropped_mask[y1 - crop_y1:y2 - crop_y1, x1 - crop_x1:x2 - crop_x1] = 1.0
    cropped_mask = utils.dilate_mask(cropped_mask, bbox_dilation)

    segs = (shape, [SEG(None, cropped_mask, 1.0, crop_region, bbox, "face", None)])
    return segs


def _do_detail(
    image,
    segs,
    model,
    clip,
    vae,
    guide_size,
    guide_size_for_bbox,
    max_size,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    denoise,
    feather,
    noise_mask,
    force_inpaint,
    refiner_ratio=None,
    refiner_model=None,
    refiner_clip=None,
    refiner_positive=None,
    refiner_negative=None,
    cycle=1,
    inpaint_model=False,
    noise_mask_feather=0,
    tiled_encode=False,
    tiled_decode=False,
):
    if len(image) > 1:
        raise Exception(
            "[ESS Face Detailer] ERROR: Detailer does not allow image batches.\n"
            "See https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md"
        )

    image = image.clone()
    enhanced_alpha_list = []
    enhanced_list = []
    cropped_list = []
    cnet_pil_list = []

    segs = core.segs_scale_match(segs, image.shape)
    new_segs = []

    ordered_segs = segs[1]

    if not (isinstance(model, str) and model == "DUMMY") and noise_mask_feather > 0 and "denoise_mask_function" not in model.model_options:
        model = nodes_differential_diffusion.DifferentialDiffusion().execute(model)[0]

    for i, seg in enumerate(ordered_segs):
        cropped_image = utils.crop_ndarray4(image.cpu().numpy(), seg.crop_region)
        cropped_image = utils.to_tensor(cropped_image)
        mask = utils.to_tensor(seg.cropped_mask)
        mask = utils.tensor_gaussian_blur_mask(mask, feather)

        is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
        if is_mask_all_zeros:
            logging.info("ESS Face Detailer: segment skip [empty mask]")
            continue

        if noise_mask:
            cropped_mask = seg.cropped_mask
        else:
            cropped_mask = None

        seg_seed = seed + i

        if not isinstance(positive, str):
            cropped_positive = [
                [
                    condition,
                    {
                        k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    },
                ]
                for condition, details in positive
            ]
        else:
            cropped_positive = positive

        if not isinstance(negative, str):
            cropped_negative = [
                [
                    condition,
                    {
                        k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    },
                ]
                for condition, details in negative
            ]
        else:
            cropped_negative = negative

        orig_cropped_image = cropped_image.clone()

        if not (isinstance(model, str) and model == "DUMMY"):
            enhanced_image, cnet_pils = core.enhance_detail(
                cropped_image,
                model,
                clip,
                vae,
                guide_size,
                guide_size_for_bbox,
                max_size,
                seg.bbox,
                seg_seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                cropped_positive,
                cropped_negative,
                denoise,
                cropped_mask,
                force_inpaint,
                refiner_ratio=refiner_ratio,
                refiner_model=refiner_model,
                refiner_clip=refiner_clip,
                refiner_positive=refiner_positive,
                refiner_negative=refiner_negative,
                control_net_wrapper=seg.control_net_wrapper,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                vae_tiled_encode=tiled_encode,
                vae_tiled_decode=tiled_decode,
            )
        else:
            enhanced_image = cropped_image
            cnet_pils = None

        if cnet_pils is not None:
            cnet_pil_list.extend(cnet_pils)

        if enhanced_image is not None:
            image = image.cpu()
            enhanced_image = enhanced_image.cpu()
            utils.tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)
            enhanced_list.append(enhanced_image)

        if enhanced_image is not None:
            enhanced_image_alpha = utils.tensor_convert_rgba(enhanced_image)
            new_seg_image = enhanced_image.numpy()

            mask = utils.tensor_resize(mask, *utils.tensor_get_size(enhanced_image))
            utils.tensor_putalpha(enhanced_image_alpha, mask)
            enhanced_alpha_list.append(enhanced_image_alpha)
        else:
            new_seg_image = None

        cropped_list.append(orig_cropped_image)

        new_seg = SEG(
            new_seg_image,
            seg.cropped_mask,
            seg.confidence,
            seg.crop_region,
            seg.bbox,
            seg.label,
            seg.control_net_wrapper,
        )
        new_segs.append(new_seg)

    image_tensor = utils.tensor_convert_rgb(image)

    cropped_list.sort(key=lambda x: x.shape, reverse=True)
    enhanced_list.sort(key=lambda x: x.shape, reverse=True)
    enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

    return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)


class ESSFaceDetailer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image to detail (single image or batch)."}),
                "model": ("MODEL", {"tooltip": "Diffusion model for the detail pass; ImpactDummyInput skips inference."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the detail pass."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the detail pass."}),
                "vae": ("VAE", {"tooltip": "VAE used for encode/decode in the detail pass."}),
                "clip": ("CLIP", {"tooltip": "CLIP model for conditioning during detail inpaint."}),
                "detection": (
                    "ESS_SEPARATOR",
                    {"label": "detection", "tooltip": "Section label."},
                ),
                "detector_profile": (
                    _INSIGHTFACE_PROFILES,
                    {"default": _INSIGHTFACE_PROFILES[0], "tooltip": "InsightFace detector profile used for face + gender detection."},
                ),
                "detector_device": (
                    ("auto", "cuda", "cpu"),
                    {"default": "auto", "tooltip": "Device used by the InsightFace detector."},
                ),
                "gender": (("any", "male", "female"), {"default": "any", "tooltip": "Filter detected faces by gender metadata."}),
                "render": (
                    "ESS_SEPARATOR",
                    {"label": "render", "tooltip": "Section label."},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Sampling steps for the detail pass."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "CFG scale for the detail pass."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler algorithm for the detail pass."}),
                "scheduler": (core.get_schedulers(), {"tooltip": "Scheduler used for the detail pass."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": "Base seed for the detail pass (offset per face)."}),
                "masking": (
                    "ESS_SEPARATOR",
                    {"label": "masking", "tooltip": "Section label."},
                ),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold for face bbox detection."}),
                "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1, "tooltip": "Expand or shrink detected bbox (pixels)."}),
                "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1, "tooltip": "Scale factor for cropping around the detected bbox."}),
                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10, "tooltip": "Drop detections smaller than this size (pixels)."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength for the detail pass."}),
                "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1, "tooltip": "Feather radius for the paste mask (pixels)."}),
                "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled", "tooltip": "Use the face mask as a noise mask during sampling."}),
                "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1, "tooltip": "Feather radius for the noise mask (pixels)."}),
                "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled", "tooltip": "Force inpaint mode even if the model is not flagged as inpaint."}),
                "misc": (
                    "ESS_SEPARATOR",
                    {"label": "misc", "tooltip": "Section label."},
                ),
                "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "Target guide size for the detail pass (pixels)."}),
                "guide_size_for": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "bbox",
                        "label_off": "crop_region",
                        "tooltip": "Use bbox (on) or crop_region (off) to compute guide size.",
                    },
                ),
                "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8, "tooltip": "Maximum size allowed for the detail crop."}),
                "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "tooltip": "Number of detail refinement cycles per face."}),
                "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", "tooltip": "Treat the model as an inpaint model."}),
                "tiled_encode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", "tooltip": "Use tiled VAE encode to reduce VRAM."}),
                "tiled_decode": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", "tooltip": "Use tiled VAE decode to reduce VRAM."}),
            },
            "optional": {},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"
    CATEGORY = "ESS/Detailer"

    DESCRIPTION = (
        "Local copy of Impact Pack FaceDetailer. Detects a face bbox and refines it via masked inpainting."
    )

    @staticmethod
    def enhance_face(
        image,
        model,
        clip,
        vae,
        guide_size,
        guide_size_for_bbox,
        max_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        denoise,
        feather,
        noise_mask,
        force_inpaint,
        bbox_threshold,
        bbox_dilation,
        bbox_crop_factor,
        drop_size,
        detector_profile,
        detector_device,
        gender,
        refiner_ratio=None,
        refiner_model=None,
        refiner_clip=None,
        refiner_positive=None,
        refiner_negative=None,
        cycle=1,
        inpaint_model=False,
        noise_mask_feather=0,
        tiled_encode=False,
        tiled_decode=False,
    ):
        segs = _detect_face_segs(
            image,
            detector_profile,
            detector_device,
            bbox_threshold,
            gender,
            bbox_dilation,
            bbox_crop_factor,
            drop_size,
        )

        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = _do_detail(
                image,
                segs,
                model,
                clip,
                vae,
                guide_size,
                guide_size_for_bbox,
                max_size,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                denoise,
                feather,
                noise_mask,
                force_inpaint,
                refiner_ratio=refiner_ratio,
                refiner_model=refiner_model,
                refiner_clip=refiner_clip,
                refiner_positive=refiner_positive,
                refiner_negative=refiner_negative,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                tiled_encode=tiled_encode,
                tiled_decode=tiled_decode,
            )
        else:
            enhanced_img = image
            cropped_enhanced = []
            cropped_enhanced_alpha = []
            cnet_pil_list = []
            new_segs = segs

        mask = core.segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [utils.empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [utils.empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [utils.empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list, new_segs

    def doit(
        self,
        image,
        model,
        positive,
        negative,
        clip,
        vae,
        detection,
        detector_profile,
        detector_device,
        gender,
        render,
        steps,
        cfg,
        sampler_name,
        scheduler,
        seed,
        masking,
        bbox_threshold,
        bbox_dilation,
        bbox_crop_factor,
        drop_size,
        denoise,
        feather,
        noise_mask,
        noise_mask_feather,
        force_inpaint,
        misc,
        guide_size,
        guide_size_for,
        max_size,
        cycle=1,
        inpaint_model=False,
        tiled_encode=False,
        tiled_decode=False,
    ):
        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []

        if len(image) > 1:
            logging.warning(
                "[ESS Face Detailer] WARN: FaceDetailer is not designed for video detailing. "
                "Use Detailer For AnimateDiff instead."
            )

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list, _ = ESSFaceDetailer.enhance_face(
                single_image.unsqueeze(0),
                model,
                clip,
                vae,
                guide_size,
                guide_size_for,
                max_size,
                seed + i,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                denoise,
                feather,
                noise_mask,
                force_inpaint,
                bbox_threshold,
                bbox_dilation,
                bbox_crop_factor,
                drop_size,
                detector_profile,
                detector_device,
                gender,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                tiled_encode=tiled_encode,
                tiled_decode=tiled_decode,
            )

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        pipe = (
            model,
            clip,
            vae,
            positive,
            negative,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images
