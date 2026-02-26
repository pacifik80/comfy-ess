from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType


def load_comfy_nodes(*, required_attrs: tuple[str, ...] = ()) -> ModuleType:
    module = importlib.import_module("nodes")
    if all(hasattr(module, attr) for attr in required_attrs):
        return module

    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "nodes.py"
        if not candidate.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("comfyui_core_nodes", candidate)
            if spec is None or spec.loader is None:
                continue
            loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded)
        except Exception:
            continue
        if all(hasattr(loaded, attr) for attr in required_attrs):
            return loaded

    missing = ", ".join(required_attrs) if required_attrs else "<none>"
    raise ImportError(f"Unable to resolve ComfyUI nodes module. Required attributes: {missing}")
