from __future__ import annotations

class LabelNote:
    """Canvas-only label node for annotating the graph."""

    CATEGORY = "ESS/Utils"
    FUNCTION = "noop"
    RETURN_TYPES = ()
    RETURN_NAMES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def noop(self) -> tuple:
        return tuple()
