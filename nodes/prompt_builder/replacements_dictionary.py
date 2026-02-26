from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Tuple

from .replacements_utils import normalize_replacements, normalize_replacement_key


_MAX_PAIRS = 24


class ReplaceDict:
    """
    Build a placeholder replacement dictionary.

    Placeholders are expected in form: %key%
    - key must not contain spaces.
    - value can be any text/template.
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "create_replacements"
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("replace_dict",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": _MAX_PAIRS,
                        "step": 1,
                        "tooltip": "Number of key/value pairs to keep (apply to update editor rows).",
                    },
                ),
                "replacements_editor": (
                    "ESS_REPLACEMENTS_EDITOR",
                    {
                        "multiline": True,
                        "default": "[]",
                        "placeholder": "Open and edit replacement pairs...",
                        "height": 220,
                    },
                ),
            },
            "optional": {
                "replace_dict": ("DICT", {"forceInput": True}),
            },
        }

    def _parse_editor_pairs(self, raw: Any) -> list[dict]:
        if raw is None:
            return []

        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except Exception:
                return []
        else:
            parsed = raw

        if isinstance(parsed, Mapping):
            # Accept {key:value} shape as fallback
            return [{"key": str(k), "value": "" if v is None else str(v)} for k, v in parsed.items()]
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, Mapping):
                    out.append(
                        {
                            "key": str(item.get("key", "") or ""),
                            "value": str(item.get("value", "") or ""),
                        }
                    )
            return out
        return []

    def create_replacements(
        self,
        count: int,
        replacements_editor: str,
        replace_dict: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str]]:
        merged = normalize_replacements(replace_dict)

        safe_count = max(1, min(int(count or 1), _MAX_PAIRS))
        pairs = self._parse_editor_pairs(replacements_editor)[:safe_count]

        for index, item in enumerate(pairs, start=1):
            raw_key = str(item.get("key", "") or "")
            key_candidate = raw_key.strip()
            if not key_candidate:
                continue
            if "%" in key_candidate or any(ch.isspace() for ch in key_candidate):
                raise ValueError(
                    f"Invalid replacement key at row {index}: '{raw_key}'. "
                    "Use key without spaces and without '%'."
                )
            key = normalize_replacement_key(key_candidate)
            if not key:
                raise ValueError(
                    f"Invalid replacement key at row {index}: '{raw_key}'."
                )
            merged[key] = str(item.get("value", "") or "")

        return (merged,)
