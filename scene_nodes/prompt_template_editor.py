from typing import Tuple

from .prompt_parser import PromptParser


class PromptTemplateEditor:
    """
    Edit a prompt template and optionally parse it into a concrete prompt.
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "render"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")

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
            }
        }

    def render(self, template: str, parse_template: bool, seed: int) -> Tuple[str, str]:
        template = template or ""
        if not parse_template:
            return (template, "")

        parser = PromptParser(seed=seed)
        try:
            processed, inverse = parser.parse(template)
        except Exception as exc:
            raise ValueError(f"Failed to parse template: {exc}") from exc

        return (processed, inverse)
