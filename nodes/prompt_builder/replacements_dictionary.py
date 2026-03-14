from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional, Tuple

from .prompt_parser import PromptParser
from .replacements_utils import apply_replacements, normalize_replacements, normalize_replacement_key


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
                "replace_values": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "replace values",
                        "label_off": "keep raw",
                        "tooltip": "Replace %key% placeholders in replacement values.",
                    },
                ),
                "recursive_replacements": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "recursive",
                        "label_off": "single pass",
                        "tooltip": "When enabled with replace_values, keep replacing until value stabilizes.",
                    },
                ),
                "parse_values": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "parse values",
                        "label_off": "raw values",
                        "tooltip": "Parse replacement values as template syntax using this node seed.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seed for parsing replacement values when parse_values is enabled.",
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

    @staticmethod
    def _replace_value(
        value: str,
        context: Mapping[str, str],
        *,
        recursive: bool,
        max_passes: int = 32,
    ) -> str:
        current = str(value or "")
        if not recursive:
            return apply_replacements(current, context)

        seen = {current}
        for _ in range(max_passes):
            updated = apply_replacements(current, context)
            if updated == current:
                break
            if updated in seen:
                # Cycle/oscillation protection (e.g. a -> %b%, b -> %a%).
                break
            seen.add(updated)
            current = updated
        return current

    def create_replacements(
        self,
        count: int,
        replace_values: bool,
        recursive_replacements: bool,
        parse_values: bool,
        seed: int,
        replacements_editor: str,
        replace_dict: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str]]:
        merged = normalize_replacements(replace_dict)

        # `count` is treated as UI/layout intent; applied rows are serialized by the editor.
        pairs = self._parse_editor_pairs(replacements_editor)[:_MAX_PAIRS]

        ordered_pairs: list[tuple[str, str]] = []
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
            ordered_pairs.append((key, str(item.get("value", "") or "")))

        # Stage raw values first so local keys can reference each other.
        for key, value in ordered_pairs:
            merged[key] = value

        if replace_values:
            context = dict(merged)
            for key, _value in ordered_pairs:
                merged[key] = self._replace_value(
                    merged.get(key, ""),
                    context,
                    recursive=bool(recursive_replacements),
                )

        if parse_values:
            parser = PromptParser(seed=seed)
            for row_index, (key, _value) in enumerate(ordered_pairs, start=1):
                try:
                    parsed_value, _inverse = parser.parse(merged.get(key, ""))
                    merged[key] = parsed_value
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse replacement value at row {row_index} ('{key}'): {exc}"
                    ) from exc

        return (merged,)
