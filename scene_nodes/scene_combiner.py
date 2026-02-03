import random
import time
from typing import List, Tuple, Optional, Any, Dict
from .scene import Scene
from .global_state import GlobalState


class SceneCombiner:
    """
    A node for combining multiple scenes into a single scene with merged prompts and LoRA stacks.
    """
    CATEGORY = "ESS/Scene"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_1": ("SCENE",),
            },
            "optional": {
                "scene_2": ("SCENE",),
                "scene_3": ("SCENE",),
                "scene_4": ("SCENE",),
                "scene_5": ("SCENE",),
                "scene_6": ("SCENE",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK", "INT", "INT", "INT")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "lora_stack", "width", "height", "seed")
    FUNCTION = "combine"

    def combine(
        self,
        scene_1: Scene,
        scene_2: Optional[Scene] = None,
        scene_3: Optional[Scene] = None,
        scene_4: Optional[Scene] = None,
        scene_5: Optional[Scene] = None,
        scene_6: Optional[Scene] = None,
    ) -> Tuple[str, str, List[Tuple[str, float, float]], int, int, int]:
        """
        Combine multiple scenes into a single scene.

        Args:
            scene_1: First scene (required)
            scene_2-6: Optional additional scenes

        Returns:
            Tuple containing:
            - Combined positive prompt
            - Combined negative prompt
            - Combined LoRA stack
            - Scene width
            - Scene height
            - Seed used for random generation
        """
        try:
            all_scenes = [scene_1, scene_2, scene_3, scene_4, scene_5, scene_6]
            combined = Scene()
            lora_stack_raw: List[Dict[str, Any]] = []

            # Use seed from global state if available
            global_seed = GlobalState.get_seed()
            if global_seed is not None:
                random.seed(global_seed)
            else:
                global_seed = int(time.time() * 1000)
                random.seed(global_seed)

            # Get resolution from global state if available
            width, height = GlobalState.get_resolution()
            if width is None or height is None:
                width = 512  # Default width
                height = 512  # Default height

            for s in all_scenes:
                if isinstance(s, list):
                    s = s[0] if s else None
                if s is None:
                    continue
                if not isinstance(s, Scene):
                    raise ValueError(f"Expected Scene object, got {type(s)}")

                combined.merge_from(s)
                lora_stack_raw.extend(s.get_lora_stack())

            # Remove duplicates by LoRA name, keeping the maximum weight
            lora_dict: Dict[str, float] = {}
            for lora in lora_stack_raw:
                if not isinstance(lora, dict):
                    continue
                name = lora.get("lora_name")
                if not name:
                    continue
                strength = lora.get("strength", 1.0)
                lora_dict[name] = max(lora_dict.get(name, strength), strength)

            combined_lora_stack = [
                (name, lora_dict[name], lora_dict[name])  # strength and clip_strength are the same
                for name in sorted(lora_dict, key=lambda n: -lora_dict[n])
            ]

            def join_by_type(elements: List[Dict[str, Any]], attr: str) -> str:
                def group_lines(group_name: str) -> str:
                    lines: List[str] = []
                    for element in elements:
                        if isinstance(element, dict):
                            if element.get("type") != group_name:
                                continue
                            value = element.get(attr, "")
                        else:
                            if getattr(element, "type", None) != group_name:
                                continue
                            value = getattr(element, attr, "")
                        if value:
                            lines.append(value)
                    return "\n".join(lines) if lines else ""

                parts: List[str] = []

                for group in ["quality", "action"]:
                    grouped = group_lines(group)
                    if grouped:
                        parts.append(grouped)

                scene_group = group_lines("scene")
                if scene_group:
                    parts.append("BREAK")
                    parts.append(scene_group)

                girl_group = group_lines("girl")
                if girl_group:
                    parts.append("BREAK")
                    parts.append(girl_group)

                boy_group = group_lines("boy")
                if boy_group:
                    parts.append("BREAK")
                    parts.append(boy_group)

                return "\n".join(filter(None, parts))

            pos_text = join_by_type(combined.elements, "positive")
            neg_text = join_by_type(combined.elements, "negative")

            # Append embeddings to the appropriate prompt so they actually get used downstream
            pos_embeddings = combined.get_embeddings(False)
            if pos_embeddings:
                embeds_block = "\n".join(pos_embeddings)
                pos_text = "\n".join(filter(None, [pos_text, embeds_block]))

            neg_embeddings = combined.get_embeddings(True)
            if neg_embeddings:
                embeds_block = "\n".join(neg_embeddings)
                neg_text = "\n".join(filter(None, [neg_text, embeds_block]))

            # Append LoRA tokens to the positive prompt so CLIPTextEncode can pick them up
            if combined_lora_stack:
                lora_tokens = [f"<lora:{name}:{strength}>" for name, strength, _ in combined_lora_stack]
                pos_text = "\n".join(filter(None, [pos_text, *lora_tokens]))

            # Print the resulting prompts to console
            print("\n=== Scene Combiner Output ===")
            print("Positive Prompt:")
            print(pos_text.strip())
            print("\nNegative Prompt:")
            print(neg_text.strip())
            print("===========================\n")

            return (pos_text.strip(), neg_text.strip(), combined_lora_stack, width, height, global_seed)

        except Exception as e:
            raise RuntimeError(f"Error combining scenes: {str(e)}")
