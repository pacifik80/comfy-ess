import torch
import torch.nn.functional as F


class PoseResizeWithPadding:
    """
    Resize an image to the target size while keeping aspect ratio and adding padding.
    Intended for pose/OpenPose style images where black padding is acceptable.
    """

    CATEGORY = "ESS/Pose"
    FUNCTION = "resize_with_padding"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "padding_top": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "padding_bottom": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "padding_left": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "padding_right": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
            }
        }

    def resize_with_padding(
        self,
        image: torch.Tensor,
        width: int,
        height: int,
        padding_top: int,
        padding_bottom: int,
        padding_left: int,
        padding_right: int,
    ) -> tuple[torch.Tensor]:
        """
        Resize while keeping aspect ratio, then place on a black canvas with padding.
        """
        if image.ndim != 4:
            raise ValueError(f"Expected 4D tensor (batch,image), got shape {tuple(image.shape)}")

        is_nhwc = image.shape[-1] == 3
        if is_nhwc:
            img = image.permute(0, 3, 1, 2)
        else:
            if image.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got shape {tuple(image.shape)}")
            img = image

        _, _, src_h, src_w = img.shape

        avail_w = width - padding_left - padding_right
        avail_h = height - padding_top - padding_bottom
        if avail_w <= 0 or avail_h <= 0:
            raise ValueError("Available area after padding must be positive in both dimensions.")

        scale = min(avail_w / src_w, avail_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)

        canvas = torch.zeros(
            (img.shape[0], 3, height, width), device=img.device, dtype=img.dtype
        )

        offset_x = padding_left + max((avail_w - new_w) // 2, 0)
        offset_y = padding_top + max((avail_h - new_h) // 2, 0)

        canvas[:, :, offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        if is_nhwc:
            canvas = canvas.permute(0, 2, 3, 1)

        return (canvas,)
