from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .prompt_parser import PromptParser
from .replacements_utils import apply_replacements

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import comfy.model_management as model_management
    import comfy.sd as comfy_sd
    import comfy.utils as comfy_utils
    from ..comfy_nodes_loader import load_comfy_nodes

    nodes = load_comfy_nodes(required_attrs=("NODE_CLASS_MAPPINGS", "CLIPTextEncode"))
except Exception:
    model_management = None
    comfy_sd = None
    comfy_utils = None
    nodes = None


_LORA_TOKEN_PATTERN = re.compile(r"<lora:([^>\n]+?)>", flags=re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")
_MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")


def _join_prompt_parts(parts: Iterable[str]) -> str:
    return "\n".join([str(part).strip() for part in parts if str(part).strip()])


def _normalize_lora_key(name: str) -> str:
    key = str(name or "").strip().replace("\\", "/").lower()
    if key.endswith(".safetensors"):
        key = key[:-12]
    return key


def _parse_lora_token_body(body: str) -> Tuple[str, float]:
    text = str(body or "").strip()
    if not text:
        raise ValueError("LoRA token has empty name.")

    parts = text.split(":")
    name = text
    weight = 1.0
    if len(parts) >= 2:
        maybe_weight = parts[-1].strip()
        if _NUMERIC_PATTERN.fullmatch(maybe_weight):
            weight = float(maybe_weight)
            name = ":".join(parts[:-1]).strip()

    name = name.replace("\\", "/").strip()
    if not name:
        raise ValueError("LoRA token has empty name.")
    return name, float(weight)


def _extract_loras_and_strip(text: str) -> Tuple[str, List[Tuple[str, float]]]:
    source = str(text or "")
    if not source:
        return "", []

    collected: List[Tuple[str, float]] = []
    out_parts: List[str] = []
    last = 0
    for match in _LORA_TOKEN_PATTERN.finditer(source):
        out_parts.append(source[last:match.start()])
        last = match.end()
        name, weight = _parse_lora_token_body(match.group(1))
        collected.append((name, weight))
    out_parts.append(source[last:])
    return "".join(out_parts), collected


def _clean_prompt_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").splitlines():
        line = _MULTISPACE_PATTERN.sub(" ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _merge_loras_first_weight(loras: Iterable[Tuple[str, float]]) -> List[Tuple[str, float]]:
    merged: List[Tuple[str, float]] = []
    seen: set[str] = set()
    for name, weight in loras:
        key = _normalize_lora_key(name)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append((name, weight))
    return merged


def _resolve_lora_name_for_paths(raw_name: str) -> str:
    candidate = str(raw_name or "").strip().replace("\\", "/")
    if not candidate:
        raise ValueError("LoRA name is empty.")

    if folder_paths is None:
        return candidate

    resolver = getattr(folder_paths, "get_full_path", None)
    resolver_raise = getattr(folder_paths, "get_full_path_or_raise", None)
    candidates = [candidate]
    if not candidate.lower().endswith(".safetensors"):
        candidates.append(f"{candidate}.safetensors")

    for name in candidates:
        if callable(resolver):
            try:
                path = resolver("loras", name)
                if path:
                    return name
            except Exception:
                pass
        if callable(resolver_raise):
            try:
                _ = resolver_raise("loras", name)
                return name
            except Exception:
                pass

    # Case-insensitive / extension-insensitive fallback against known list
    try:
        names = folder_paths.get_filename_list("loras")
    except Exception:
        names = []
    index: Dict[str, str] = {}
    for item in names:
        key = _normalize_lora_key(item)
        if key and key not in index:
            index[key] = item
    mapped = index.get(_normalize_lora_key(candidate))
    if mapped:
        return mapped

    raise ValueError(f"LoRA not found: {candidate}")


def _load_lora_tensor(lora_name: str):
    if folder_paths is None or comfy_utils is None:
        raise RuntimeError("ComfyUI runtime is required for LoRA loading.")

    resolver = getattr(folder_paths, "get_full_path_or_raise", None) or getattr(folder_paths, "get_full_path", None)
    if not callable(resolver):
        raise RuntimeError("Unable to resolve LoRA file path in this ComfyUI runtime.")

    path = resolver("loras", lora_name)
    if not path:
        raise ValueError(f"LoRA not found: {lora_name}")
    try:
        return comfy_utils.load_torch_file(path, safe_load=True)
    except TypeError:
        return comfy_utils.load_torch_file(path)


def _apply_lora_model_only(model, lora_name: str, weight: float):
    if comfy_sd is None:
        raise RuntimeError("ComfyUI runtime is required for LoRA application.")
    lora_tensor = _load_lora_tensor(lora_name)
    applied = comfy_sd.load_lora_for_models(model, None, lora_tensor, float(weight), 0.0)
    if isinstance(applied, (tuple, list)):
        return applied[0]
    return applied


class PromptConditioningBuilder:
    """
    Build conditioning + latent from plain/template prompts:
    - optional replacements
    - optional template parsing
    - LoRA extraction/apply-to-model
    - CLIP encode
    - empty latent generation
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "build"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "clip", "vae", "positive_condition", "negative_condition", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive_prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "negative_prompt": ("STRING", {"forceInput": True, "multiline": True}),
                "parse_template": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "parse",
                        "label_off": "raw",
                        "tooltip": "When enabled, parse ESS template syntax in both prompts.",
                    },
                ),
                "apply_replace_dict": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "replace",
                        "label_off": "ignore",
                        "tooltip": "When enabled, apply %key% replacements from replace_dict.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for template parsing.",
                    },
                ),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384, "step": 8}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
            "optional": {
                "replace_dict": ("DICT", {"forceInput": True}),
            },
        }

    def _process_prompts(
        self,
        positive_prompt: str,
        negative_prompt: str,
        parse_template: bool,
        apply_replace_dict: bool,
        replace_dict: Optional[dict],
        seed: int,
    ) -> Tuple[str, str]:
        pos = str(positive_prompt or "")
        neg = str(negative_prompt or "")

        if apply_replace_dict:
            pos = apply_replacements(pos, replace_dict)
            neg = apply_replacements(neg, replace_dict)

        if not parse_template:
            return pos, neg

        parser = PromptParser(seed=int(seed))
        pos_parsed, pos_inverse = parser.parse(pos)
        neg_parsed, neg_inverse = parser.parse(neg)

        final_pos = _join_prompt_parts([pos_parsed, neg_inverse])
        final_neg = _join_prompt_parts([neg_parsed, pos_inverse])
        return final_pos, final_neg

    def _encode_prompts(self, clip, positive_text: str, negative_text: str):
        if nodes is None:
            raise RuntimeError("ComfyUI runtime is required for prompt encoding.")
        encoder = nodes.CLIPTextEncode()
        positive_cond = encoder.encode(clip, str(positive_text or ""))[0]
        negative_cond = encoder.encode(clip, str(negative_text or ""))[0]
        return positive_cond, negative_cond

    def _build_empty_latent(self, width: int, height: int, batch: int) -> dict:
        latent_w = max(1, int(width) // 8)
        latent_h = max(1, int(height) // 8)
        device = "cpu"
        if model_management is not None and hasattr(model_management, "intermediate_device"):
            try:
                device = model_management.intermediate_device()
            except Exception:
                device = "cpu"
        samples = torch.zeros([int(batch), 4, latent_h, latent_w], device=device)
        return {"samples": samples}

    def build(
        self,
        model,
        clip,
        vae,
        positive_prompt: str,
        negative_prompt: str,
        parse_template: bool,
        apply_replace_dict: bool,
        seed: int,
        width: int,
        height: int,
        batch: int,
        replace_dict: dict | None = None,
    ):
        positive_text, negative_text = self._process_prompts(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            parse_template=bool(parse_template),
            apply_replace_dict=bool(apply_replace_dict),
            replace_dict=replace_dict,
            seed=int(seed),
        )

        positive_clean, positive_loras = _extract_loras_and_strip(positive_text)
        negative_clean, negative_loras = _extract_loras_and_strip(negative_text)
        positive_clean = _clean_prompt_text(positive_clean)
        negative_clean = _clean_prompt_text(negative_clean)
        merged_loras = _merge_loras_first_weight([*positive_loras, *negative_loras])

        model_out = model
        for lora_name_raw, lora_weight in merged_loras:
            resolved_name = _resolve_lora_name_for_paths(lora_name_raw)
            model_out = _apply_lora_model_only(model_out, resolved_name, lora_weight)

        positive_cond, negative_cond = self._encode_prompts(clip, positive_clean, negative_clean)
        latent = self._build_empty_latent(width=width, height=height, batch=batch)

        return model_out, clip, vae, positive_cond, negative_cond, latent
