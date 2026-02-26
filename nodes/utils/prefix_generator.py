import random
import time
from typing import Tuple


class PrefixGenerator:
    """
    Utility node that creates path prefixes for organizing generated assets.
    """

    CATEGORY = "ESS/Utils"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("project_prefix", "version_prefix", "run_prefix", "seed")
    FUNCTION = "generate"
    _MAX_SEED = 0xffffffffffffffff

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "project"}),
                "project_version": ("STRING", {"default": "v1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": cls._MAX_SEED}),
                "seed_mode": (["fixed", "randomize", "increment", "decrement"], {"default": "fixed"}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, seed_mode: str = "fixed", **_kwargs):
        if seed_mode in ("randomize", "increment", "decrement"):
            return time.time()
        return None

    @classmethod
    def _resolve_seed(cls, seed: int, seed_mode: str) -> int:
        seed_value = 0
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            seed_value = 0
        if seed_value < 0:
            seed_value = 0

        if seed_mode == "randomize":
            return random.SystemRandom().randint(0, cls._MAX_SEED)
        if seed_mode == "increment":
            return (seed_value + 1) & cls._MAX_SEED
        if seed_mode == "decrement":
            return (seed_value - 1) & cls._MAX_SEED
        return seed_value

    def generate(self, project_name: str, project_version: str, seed: int, seed_mode: str) -> Tuple[str, str, str, int]:
        project = (project_name or "").strip()
        version = (project_version or "").strip()
        resolved_seed = self._resolve_seed(seed, seed_mode)

        # Keep names simple to reduce risk of invalid paths
        project = project or "project"
        version = version or "v1"

        project_prefix = project
        version_prefix = f"{project}/{version}"
        timestamp = time.strftime("%Y%m%d%H%M%S")
        run_prefix = f"{version_prefix}/{timestamp}_{resolved_seed}"

        return project_prefix, version_prefix, run_prefix, resolved_seed
