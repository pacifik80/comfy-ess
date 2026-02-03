from .scene import Scene
import random
import time
import copy
from .global_state import GlobalState

class SceneSelector:
    CATEGORY = "ESS/Scene"

    @staticmethod
    def _resolve_scene_candidate(candidate):
        """
        Unpack a scene candidate that might carry UI metadata and report if it is hidden.
        Scenes can arrive wrapped in a list/tuple from ComfyUI, and some inputs may
        include a metadata dict flagging them as hidden when the UI collapses inputs.
        """
        if candidate is None:
            return None, False

        hidden = False
        resolved_scene = candidate

        if isinstance(candidate, (list, tuple)) and candidate:
            resolved_scene = candidate[0]
            if len(candidate) > 1 and isinstance(candidate[1], dict):
                meta = candidate[1]
                hidden = bool(meta.get("hidden") or meta.get("is_hidden"))

        if isinstance(resolved_scene, dict):
            hidden = hidden or bool(resolved_scene.get("hidden") or resolved_scene.get("is_hidden"))
            resolved_scene = resolved_scene.get("scene", resolved_scene)

        if not isinstance(resolved_scene, Scene):
            return None, hidden

        if isinstance(resolved_scene, Scene):
            metadata = getattr(resolved_scene, "metadata", None)
            if isinstance(metadata, dict):
                hidden = hidden or bool(metadata.get("hidden"))

        return resolved_scene, hidden

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "scene_1": ("SCENE",),
                "name_1": ("STRING", {"default": "Scene 1"}),
                "weight_1": ("FLOAT", {"default": 1.0}),

                "scene_2": ("SCENE",),
                "name_2": ("STRING", {"default": "Scene 2"}),
                "weight_2": ("FLOAT", {"default": 0.0}),

                "scene_3": ("SCENE",),
                "name_3": ("STRING", {"default": "Scene 3"}),
                "weight_3": ("FLOAT", {"default": 0.0}),

                "scene_4": ("SCENE",),
                "name_4": ("STRING", {"default": "Scene 4"}),
                "weight_4": ("FLOAT", {"default": 0.0}),

                "scene_5": ("SCENE",),
                "name_5": ("STRING", {"default": "Scene 5"}),
                "weight_5": ("FLOAT", {"default": 0.0}),

                "scene_6": ("SCENE",),
                "name_6": ("STRING", {"default": "Scene 6"}),
                "weight_6": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("SCENE",)
    RETURN_NAMES = ("selected_scene",)
    FUNCTION = "select"

    def select(self, scene_1=None, name_1="Scene 1", weight_1=1.0,
                     scene_2=None, name_2="Scene 2", weight_2=0.0,
                     scene_3=None, name_3="Scene 3", weight_3=0.0,
                     scene_4=None, name_4="Scene 4", weight_4=0.0,
                     scene_5=None, name_5="Scene 5", weight_5=0.0,
                     scene_6=None, name_6="Scene 6", weight_6=0.0):

        scenes_with_weights = [
            (scene_1, weight_1),
            (scene_2, weight_2),
            (scene_3, weight_3),
            (scene_4, weight_4),
            (scene_5, weight_5),
            (scene_6, weight_6),
        ]

        available = []
        for candidate, weight in scenes_with_weights:
            resolved, hidden = self._resolve_scene_candidate(candidate)
            if resolved is None or hidden:
                continue
            available.append((resolved, 0.0 if weight is None else float(weight)))

        if not available:
            return (None,)

        seed = GlobalState.get_seed()
        if seed is None:
            for scene_obj, _ in available:
                if isinstance(scene_obj, Scene):
                    seed = scene_obj.get_seed()
                    if seed is not None:
                        break

        if seed is None:
            seed = int(time.time() * 1000)

        rng = random.Random(seed)

        weights = [max(0.0, weight) for _, weight in available]
        if all(weight <= 0 for weight in weights):
            weights = [1.0 for _ in weights]

        scenes = [scene for scene, _ in available]
        selected = rng.choices(scenes, weights=weights, k=1)[0]
        return (copy.deepcopy(selected),)

