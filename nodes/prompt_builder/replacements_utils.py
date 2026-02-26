from __future__ import annotations

import re
from typing import Any, Dict, Mapping


_PLACEHOLDER_PATTERN = re.compile(r"%([^\s%]+)%")


def normalize_replacement_key(raw_key: Any) -> str:
    key_text = str(raw_key).strip()
    if not key_text:
        return ""
    if "%" in key_text:
        return ""
    if any(ch.isspace() for ch in key_text):
        return ""
    return key_text


def normalize_replacements(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}

    out: Dict[str, str] = {}
    for key, value in raw.items():
        key_text = normalize_replacement_key(key)
        if not key_text:
            continue
        out[key_text] = "" if value is None else str(value)
    return out


def apply_replacements(text: str, replace_dict: Mapping[str, Any] | None) -> str:
    source = str(text or "")
    if not source:
        return source

    normalized = normalize_replacements(replace_dict)
    if not normalized:
        return source

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in normalized:
            return normalized[key]
        return match.group(0)

    return _PLACEHOLDER_PATTERN.sub(_replace, source)
