from typing import Tuple

from .prompt_parser import PromptParser


class TextPromptGenerator:
    """
    Generate paired positive/negative prompts from wildcard templates without scene objects.
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_template": ("STRING", {"multiline": True, "default": ""}),
                "negative_template": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    def generate(self, positive_template: str, negative_template: str, seed: int) -> Tuple[str, str]:
        parser = PromptParser(seed=seed)

        try:
            processed_positive, inverse_positive = parser.parse(positive_template or "")
            processed_negative, inverse_negative = parser.parse(negative_template or "")
        except Exception as exc:
            raise ValueError(f"Failed to parse prompt templates: {exc}") from exc

        # Cross-feed inverse parts into the opposite prompt, same as scene logic
        positive_parts = [processed_positive]
        negative_parts = [processed_negative]

        if inverse_positive:
            negative_parts.append(inverse_positive)
        if inverse_negative:
            positive_parts.append(inverse_negative)

        positive = ", ".join(filter(None, [p.strip() for p in positive_parts if p.strip()]))
        negative = ", ".join(filter(None, [n.strip() for n in negative_parts if n.strip()]))

        return positive, negative
