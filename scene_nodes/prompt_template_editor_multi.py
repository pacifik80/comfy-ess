from typing import Tuple, Dict, List

from .prompt_parser import PromptParser


class PromptTemplateEditorMulti:
    """
    Edit a prompt template and output multiple positive/negative prompt pairs.
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "render"

    _MAX_VARIANTS = 10
    _VARIANT_LETTERS = "abcdefghij"

    RETURN_TYPES = ("STRING",) * (_MAX_VARIANTS * 2)
    RETURN_NAMES = tuple(
        name
        for letter in _VARIANT_LETTERS
        for name in (f"positive_{letter}", f"negative_{letter}")
    )

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
                "count": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": cls._MAX_VARIANTS,
                        "step": 1,
                        "tooltip": "Number of output pairs (apply to update outputs).",
                    },
                ),
            },
            "optional": {
                "prefix_positive": ("STRING", {"forceInput": True, "multiline": True}),
                "prefix_negative": ("STRING", {"forceInput": True, "multiline": True}),
                "suffix_positive": ("STRING", {"forceInput": True, "multiline": True}),
                "suffix_negative": ("STRING", {"forceInput": True, "multiline": True}),
            },
        }

    def _join_parts(self, parts: List[str]) -> str:
        return "\n".join([part.strip() for part in parts if part and part.strip()])

    def _variant_list(self, count: int) -> List[str]:
        safe = max(1, min(int(count or 1), self._MAX_VARIANTS))
        return list(self._VARIANT_LETTERS[:safe])

    def _raw_variant_text(self, text: str, parser: PromptParser, variants: List[str]) -> Dict[str, str]:
        if text is None:
            text = ""
        return {
            variant: parser.expand_variant_blocks(text, variant, variants)
            for variant in variants
        }

    def render(
        self,
        template: str,
        parse_template: bool,
        seed: int,
        count: int,
        prefix_positive: str = "",
        prefix_negative: str = "",
        suffix_positive: str = "",
        suffix_negative: str = "",
    ) -> Tuple[str, ...]:
        variants = self._variant_list(count)
        parser = PromptParser(seed=seed)

        if not parse_template:
            positive_text = "\n".join([p for p in [prefix_positive or "", template or "", suffix_positive or ""] if p])
            negative_text = "\n".join([p for p in [prefix_negative or "", suffix_negative or ""] if p])

            positive_by_variant = self._raw_variant_text(positive_text, parser, variants)
            negative_by_variant = self._raw_variant_text(negative_text, parser, variants)

            outputs: List[str] = []
            for letter in self._VARIANT_LETTERS:
                if letter in variants:
                    outputs.append(positive_by_variant.get(letter, ""))
                    outputs.append(negative_by_variant.get(letter, ""))
                else:
                    outputs.append("")
                    outputs.append("")
            return tuple(outputs)

        try:
            template_map = parser.parse_multi(template or "", variants)
            prefix_pos_map = parser.parse_multi(prefix_positive or "", variants)
            suffix_pos_map = parser.parse_multi(suffix_positive or "", variants)
            prefix_neg_map = parser.parse_multi(prefix_negative or "", variants)
            suffix_neg_map = parser.parse_multi(suffix_negative or "", variants)
        except Exception as exc:
            raise ValueError(f"Failed to parse template: {exc}") from exc

        outputs = []
        for letter in self._VARIANT_LETTERS:
            if letter not in variants:
                outputs.extend(["", ""])
                continue

            template_pos, template_neg = template_map.get(letter, ("", ""))
            prefix_pos, prefix_pos_inv = prefix_pos_map.get(letter, ("", ""))
            suffix_pos, suffix_pos_inv = suffix_pos_map.get(letter, ("", ""))
            prefix_neg, prefix_neg_inv = prefix_neg_map.get(letter, ("", ""))
            suffix_neg, suffix_neg_inv = suffix_neg_map.get(letter, ("", ""))

            positive = self._join_parts(
                [prefix_pos, template_pos, suffix_pos, prefix_neg_inv, suffix_neg_inv]
            )
            negative = self._join_parts(
                [prefix_neg, template_neg, suffix_neg, prefix_pos_inv, suffix_pos_inv]
            )

            outputs.extend([positive, negative])

        return tuple(outputs)
