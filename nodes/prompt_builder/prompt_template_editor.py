from typing import Tuple

from .prompt_parser import PromptParser
from .replacements_utils import apply_replacements


class PromptTemplateEditor:
    """
    Edit a prompt template and optionally parse it into a concrete prompt.
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "render"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": (
                    "ESS_TEMPLATE_EDITOR",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Write template here...",
                        "height": 220,
                    },
                ),
                "parse_template": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "parse",
                        "label_off": "raw",
                        "tooltip": "When enabled, apply template randomization. When disabled, output raw text.",
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
            },
            "optional": {
                "prefix_positive": ("STRING", {"forceInput": True, "multiline": True}),
                "prefix_negative": ("STRING", {"forceInput": True, "multiline": True}),
                "suffix_positive": ("STRING", {"forceInput": True, "multiline": True}),
                "suffix_negative": ("STRING", {"forceInput": True, "multiline": True}),
                "replace_dict": ("DICT", {"forceInput": True}),
            },
        }

    def render(
        self,
        template: str,
        parse_template: bool,
        seed: int,
        prefix_positive: str = "",
        prefix_negative: str = "",
        suffix_positive: str = "",
        suffix_negative: str = "",
        replace_dict: dict = None,
    ) -> Tuple[str, str]:
        template_resolved = apply_replacements(template or "", replace_dict)
        prefix_positive_resolved = apply_replacements(prefix_positive or "", replace_dict)
        prefix_negative_resolved = apply_replacements(prefix_negative or "", replace_dict)
        suffix_positive_resolved = apply_replacements(suffix_positive or "", replace_dict)
        suffix_negative_resolved = apply_replacements(suffix_negative or "", replace_dict)

        if not parse_template:
            positive_parts = [prefix_positive_resolved, template_resolved, suffix_positive_resolved]
            negative_parts = [prefix_negative_resolved, suffix_negative_resolved]
            positive_raw = "\n".join([p for p in positive_parts if p])
            negative_raw = "\n".join([p for p in negative_parts if p])
            return (positive_raw, negative_raw)

        parser = PromptParser(seed=seed)
        try:
            processed_template, inverse_template = parser.parse(template_resolved)
            processed_prefix_pos, inverse_prefix_pos = parser.parse(prefix_positive_resolved)
            processed_suffix_pos, inverse_suffix_pos = parser.parse(suffix_positive_resolved)
            processed_prefix_neg, inverse_prefix_neg = parser.parse(prefix_negative_resolved)
            processed_suffix_neg, inverse_suffix_neg = parser.parse(suffix_negative_resolved)
        except Exception as exc:
            raise ValueError(f"Failed to parse template: {exc}") from exc

        def join_parts(parts):
            return "\n".join([p.strip() for p in parts if p and p.strip()])

        positive = join_parts(
            [
                processed_prefix_pos,
                processed_template,
                processed_suffix_pos,
                inverse_prefix_neg,
                inverse_suffix_neg,
            ]
        )
        negative = join_parts(
            [
                processed_prefix_neg,
                inverse_template,
                processed_suffix_neg,
                inverse_prefix_pos,
                inverse_suffix_pos,
            ]
        )

        return (positive, negative)
