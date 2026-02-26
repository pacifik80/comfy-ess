from __future__ import annotations

import re
from typing import Dict, List, Tuple


_MAX_SLOTS = 16


class StringConcatenate:
    """Concatenate multiple strings with a configurable separator."""

    CATEGORY = "ESS/Utils"
    FUNCTION = "concat"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"input_{index}": (
                "STRING",
                {"forceInput": True, "multiline": True, "tooltip": f"Input string {index}."},
            )
            for index in range(1, _MAX_SLOTS + 1)
        }
        return {
            "required": {
                "count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": _MAX_SLOTS,
                        "step": 1,
                        "tooltip": "Number of input slots to concatenate.",
                    },
                ),
                "separator": ("STRING", {"default": "", "multiline": False}),
                "terminate_string": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Append a newline between each input (except last).",
                    },
                ),
                "clear_excessive_whitespaces": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Collapse repeated spaces/newlines in the result.",
                    },
                ),
            },
            "optional": optional_inputs,
        }

    def concat(
        self,
        count: int,
        separator: str,
        terminate_string: bool,
        clear_excessive_whitespaces: bool,
        **kwargs: Dict[str, str],
    ) -> Tuple[str]:
        inputs = self._collect_inputs(count, kwargs)
        joiner = self._build_joiner(separator or "", terminate_string)
        result = joiner.join(inputs)

        if clear_excessive_whitespaces:
            result = self._clear_excessive_whitespaces(result)

        return (result,)

    @staticmethod
    def _collect_inputs(count: int, kwargs: Dict[str, str]) -> List[str]:
        parts = []
        max_count = max(1, min(_MAX_SLOTS, int(count)))
        for index in range(1, max_count + 1):
            value = kwargs.get(f"input_{index}")
            if value is None:
                continue
            parts.append(value)
        return parts

    @staticmethod
    def _build_joiner(separator: str, terminate_string: bool) -> str:
        if not terminate_string:
            return separator
        if separator.endswith("\n"):
            return separator
        return f"{separator}\n"

    @staticmethod
    def _clear_excessive_whitespaces(text: str) -> str:
        normalized = text.replace("\r\n", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{2,}", "\n", normalized)
        return normalized
