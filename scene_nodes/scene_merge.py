"""
SceneAppend - Appends multiple scene objects into a single scene.
Takes up to 6 optional scene inputs and combines their elements.
"""

from __future__ import annotations
from typing import Optional
from .scene import Scene


class SceneMerge:
    """
    A node for merging multiple scenes into a single scene.
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

    RETURN_TYPES = ("SCENE",)
    RETURN_NAMES = ("scene",)
    FUNCTION = "merge"

    def merge(
        self,
        scene_1: Scene,
        scene_2: Optional[Scene] = None,
        scene_3: Optional[Scene] = None,
        scene_4: Optional[Scene] = None,
        scene_5: Optional[Scene] = None,
        scene_6: Optional[Scene] = None,
    ) -> tuple:
        """
        Merge multiple scenes into a single scene.

        Args:
            scene_1: First scene (required)
            scene_2-6: Optional additional scenes

        Returns:
            Tuple containing the merged scene
        """
        try:
            all_scenes = [scene_1, scene_2, scene_3, scene_4, scene_5, scene_6]
            merged = Scene()

            for s in all_scenes:
                if isinstance(s, list):
                    s = s[0] if s else None
                if s is None:
                    continue
                if not isinstance(s, Scene):
                    raise ValueError(f"Expected Scene object, got {type(s)}")
                merged.merge_from(s)

            return (merged,)

        except Exception as e:
            raise RuntimeError(f"Error merging scenes: {str(e)}")
