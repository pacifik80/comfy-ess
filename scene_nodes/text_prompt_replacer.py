from typing import Tuple


class TextPromptReplacer:
    """
    Replace %replace_a%/%replace_b%/%replace_c% placeholders in a prompt block.
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "replace_a": ("STRING", {"forceInput": True, "label": "%replace_a%"}),
                "replace_b": ("STRING", {"forceInput": True, "label": "%replace_b%"}),
                "replace_c": ("STRING", {"forceInput": True, "label": "%replace_c%"}),
            },
        }

    def generate(
        self,
        prompt_text: str,
        replace_a: str = None,
        replace_b: str = None,
        replace_c: str = None,
    ) -> Tuple[str]:
        prompt = prompt_text or ""
        replacements = {
            "%replace_a%": replace_a,
            "%replace_b%": replace_b,
            "%replace_c%": replace_c,
        }

        for token, value in replacements.items():
            prompt = prompt.replace(token, value or "")

        return (prompt,)
