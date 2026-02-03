import torch
import random
import numpy as np
from typing import Dict, Any, Optional
from .scene import Scene
from .global_state import GlobalState

class InitScene:
    """
    A node for initializing a new scene with global parameters.
    """
    CATEGORY = "ESS/Scene"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            },
            "optional": {
                "replacements": ("DICT", {"default": {}}),
            }
        }

    RETURN_TYPES = ("SCENE",)
    RETURN_NAMES = ("scene",)
    FUNCTION = "initialize_scene"

    def initialize_scene(self, seed: int, width: int, height: int, replacements: dict = None) -> tuple:
        """
        Initialize a new scene with the given parameters.
        
        Args:
            seed: Random seed for the scene
            width: Width of the scene
            height: Height of the scene
            replacements: Optional dictionary of text replacements
            
        Returns:
            Tuple containing the initialized scene
        """
        # Update global state
        GlobalState.set_seed(seed)
        GlobalState.set_resolution(width, height)
        if replacements:
            GlobalState.set_replacements(replacements)

        # Create and return scene
        scene = Scene()
        scene.set_seed(seed)
        return (scene,) 
