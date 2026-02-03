from .scene import Scene
import random
import time
import copy
from .global_state import GlobalState
from .scene_selector import SceneSelector


class SceneSelectorTest:
    CATEGORY = "ESS/Scene"
    MAX_SCENES = 10

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "num_scenes": (
                    "INT",
                    {"default": 3, "min": 1, "max": cls.MAX_SCENES},
                ),
            },
            "optional": {},
        }

        for idx in range(1, cls.MAX_SCENES + 1):
            inputs["optional"][f"scene_{idx}"] = ("SCENE",)
            inputs["optional"][f"name_{idx}"] = ("STRING", {"default": f"Scene {idx}"})
            inputs["optional"][f"weight_{idx}"] = (
                "FLOAT",
                {"default": 1.0 if idx == 1 else 0.0},
            )

        return inputs

    RETURN_TYPES = ("SCENE",)
    RETURN_NAMES = ("selected_scene",)
    FUNCTION = "select"

    def select(
        self,
        num_scenes=3,
        scene_1=None,
        name_1="Scene 1",
        weight_1=1.0,
        scene_2=None,
        name_2="Scene 2",
        weight_2=0.0,
        scene_3=None,
        name_3="Scene 3",
        weight_3=0.0,
        scene_4=None,
        name_4="Scene 4",
        weight_4=0.0,
        scene_5=None,
        name_5="Scene 5",
        weight_5=0.0,
        scene_6=None,
        name_6="Scene 6",
        weight_6=0.0,
        scene_7=None,
        name_7="Scene 7",
        weight_7=0.0,
        scene_8=None,
        name_8="Scene 8",
        weight_8=0.0,
        scene_9=None,
        name_9="Scene 9",
        weight_9=0.0,
        scene_10=None,
        name_10="Scene 10",
        weight_10=0.0,
    ):
        target_count = int(num_scenes) if num_scenes is not None else 1
        target_count = max(1, min(self.MAX_SCENES, target_count))

        scene_inputs = [
            scene_1,
            scene_2,
            scene_3,
            scene_4,
            scene_5,
            scene_6,
            scene_7,
            scene_8,
            scene_9,
            scene_10,
        ]
        weight_inputs = [
            weight_1,
            weight_2,
            weight_3,
            weight_4,
            weight_5,
            weight_6,
            weight_7,
            weight_8,
            weight_9,
            weight_10,
        ]

        scenes_with_weights = []
        for idx in range(target_count):
            candidate = scene_inputs[idx]
            if candidate is None:
                continue

            resolved, hidden = SceneSelector._resolve_scene_candidate(candidate)
            if resolved is None or hidden:
                continue
            weight_value = 0.0 if weight_inputs[idx] is None else float(weight_inputs[idx])
            scenes_with_weights.append((resolved, weight_value))

        if not scenes_with_weights:
            return (None,)

        # Prefer the global seed for reproducibility, fall back to the first scene seed, then time.
        seed = GlobalState.get_seed()
        if seed is None:
            for scene_obj, _ in scenes_with_weights:
                if isinstance(scene_obj, Scene):
                    seed = scene_obj.get_seed()
                    if seed is not None:
                        break

        if seed is None:
            seed = int(time.time() * 1000)

        rng = random.Random(seed)

        weights = [max(0.0, weight) for _, weight in scenes_with_weights]
        if all(weight <= 0 for weight in weights):
            weights = [1.0 for _ in weights]

        scenes = [scene for scene, _ in scenes_with_weights]
        selected = rng.choices(scenes, weights=weights, k=1)[0]
        return (copy.deepcopy(selected),)
