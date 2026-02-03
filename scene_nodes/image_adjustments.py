"""
ImageAdjustmentsNode - A node for adjusting image properties like saturation, brightness, and RGB components.
"""

import torch
import torch.nn.functional as F
from typing import Optional

class ImageAdjustmentsNode:
    """
    A node for adjusting image properties.
    """
    CATEGORY = "ESS/Image Processing"
    FUNCTION = "adjust_image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "brightness": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "red": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "green": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blue": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }

    def adjust_image(
        self,
        image: torch.Tensor,
        saturation: float,
        brightness: float,
        red: float,
        green: float,
        blue: float,
    ) -> tuple[torch.Tensor]:
        """
        Adjust image properties.
        
        Args:
            image: Input image tensor (NHWC or NCHW format)
            saturation: Saturation adjustment (-1 to 1)
            brightness: Brightness adjustment (-1 to 1)
            red: Red channel adjustment (-1 to 1)
            green: Green channel adjustment (-1 to 1)
            blue: Blue channel adjustment (-1 to 1)
            
        Returns:
            Adjusted image tensor in the same format as input
        """
        # Check if input is NHWC format (channels last)
        is_nhwc = image.shape[-1] == 3
        
        # Convert to NCHW if needed
        if is_nhwc:
            adjusted = image.permute(0, 3, 1, 2)
        else:
            # Ensure input is in NCHW format
            if image.shape[1] != 3:
                raise ValueError(f"Expected 3 channels in NCHW format, got shape {tuple(image.shape)}")
            adjusted = image.clone()

        # Apply RGB adjustments
        if red != 0:
            adjusted[:, 0] = torch.clamp(adjusted[:, 0] + red, 0, 1)
        if green != 0:
            adjusted[:, 1] = torch.clamp(adjusted[:, 1] + green, 0, 1)
        if blue != 0:
            adjusted[:, 2] = torch.clamp(adjusted[:, 2] + blue, 0, 1)

        # Apply brightness adjustment
        if brightness != 0:
            adjusted = torch.clamp(adjusted + brightness, 0, 1)

        # Apply saturation adjustment
        if saturation != 0:
            # Convert to grayscale
            grayscale = adjusted.mean(dim=1, keepdim=True)
            # Blend between grayscale and color based on saturation
            adjusted = torch.lerp(grayscale.expand_as(adjusted), adjusted, 1.0 + saturation)

        # Convert back to original format if needed
        if is_nhwc:
            adjusted = adjusted.permute(0, 2, 3, 1)

        return (adjusted,) 