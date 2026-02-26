"""Group reroute node with a dynamic number of passthrough slots."""

from __future__ import annotations

from typing import Any, Dict, Tuple


_MAX_SLOTS = 16


class _AnyType(str):
    def __ne__(self, value: object) -> bool:
        if value == "*" or self == "*":
            return False
        return super().__ne__(value)


_ANY = _AnyType("*")


class GroupReroute:
    @classmethod
    def INPUT_TYPES(cls):
        optional_inputs = {
            f"input_{index}": (_ANY, {"tooltip": f"Passthrough input {index}."})
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
                        "tooltip": "Number of reroute slots (apply to update inputs/outputs).",
                    },
                ),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = tuple(_ANY for _ in range(_MAX_SLOTS))
    RETURN_NAMES = tuple(f"output_{index}" for index in range(1, _MAX_SLOTS + 1))
    FUNCTION = "passthrough"
    CATEGORY = "ESS/Utils"
    OUTPUT_TOOLTIPS = tuple(f"Passthrough output {index}." for index in range(1, _MAX_SLOTS + 1))

    def passthrough(self, count: int, **kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
        outputs = []
        for index in range(1, _MAX_SLOTS + 1):
            outputs.append(kwargs.get(f"input_{index}"))
        return tuple(outputs)

    @classmethod
    def VALIDATE_INPUTS(cls, input_types, **kwargs):
        # Accept any input types; this node is a typed passthrough.
        return True
