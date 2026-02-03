import sys
from pathlib import Path

FaceSwapInSwapperNode = None
FaceSwapSimSwapNode = None
FaceSwapFaceFusionNode = None
_face_swapper_error = None
PersonCropToSize = None
_person_crop_error = None

try:
    from .scene_nodes.scene_selector import SceneSelector
    from .scene_nodes.scene_selector_test import SceneSelectorTest
    from .scene_nodes.define_scene_part import DefineScenePart, SceneDebug
    from .scene_nodes.scene_combiner import SceneCombiner
    from .scene_nodes.scene import Scene
    from .scene_nodes.scene_merge import SceneMerge
    from .scene_nodes.scene_initializer import InitScene
    from .scene_nodes.replacements_dictionary import ReplaceDict
    from .scene_nodes.image_adjustments import ImageAdjustmentsNode
    from .scene_nodes.segmentation_detailer import SegmentationDetailerNode
    from .scene_nodes.ess_face_detailer import ESSFaceDetailer
    from .scene_nodes.group_reroute import GroupReroute
    from .scene_nodes.string_concat import StringConcatenate
    from .scene_nodes.label_note import LabelNote
    try:
        from .scene_nodes.face_swapping import FaceSwapInSwapperNode, FaceSwapSimSwapNode, FaceSwapFaceFusionNode
    except Exception as exc:
        _face_swapper_error = exc
    from .scene_nodes.pose_figure_editor import PoseFigureEditor
    from .scene_nodes.pose_resize_with_padding import PoseResizeWithPadding
    from .scene_nodes.prefix_generator import PrefixGenerator
    from .scene_nodes.text_prompt_generator import TextPromptGenerator
    from .scene_nodes.text_prompt_replacer import TextPromptReplacer
    from .scene_nodes.prompt_template_editor import PromptTemplateEditor
    try:
        from .scene_nodes.person_crop_to_size import PersonCropToSize
    except Exception as exc:
        _person_crop_error = exc
except ImportError as exc:
    if "attempted relative import" not in str(exc):
        raise
    package_dir = Path(__file__).resolve().parent
    if str(package_dir) not in sys.path:
        sys.path.insert(0, str(package_dir))
    from scene_nodes.scene_selector import SceneSelector
    from scene_nodes.scene_selector_test import SceneSelectorTest
    from scene_nodes.define_scene_part import DefineScenePart, SceneDebug
    from scene_nodes.scene_combiner import SceneCombiner
    from scene_nodes.scene import Scene
    from scene_nodes.scene_merge import SceneMerge
    from scene_nodes.scene_initializer import InitScene
    from scene_nodes.replacements_dictionary import ReplaceDict
    from scene_nodes.image_adjustments import ImageAdjustmentsNode
    from scene_nodes.segmentation_detailer import SegmentationDetailerNode
    from scene_nodes.ess_face_detailer import ESSFaceDetailer
    from scene_nodes.group_reroute import GroupReroute
    from scene_nodes.string_concat import StringConcatenate
    from scene_nodes.label_note import LabelNote
    try:
        from scene_nodes.face_swapping import FaceSwapInSwapperNode, FaceSwapSimSwapNode, FaceSwapFaceFusionNode
    except Exception as exc:
        _face_swapper_error = exc
    from scene_nodes.pose_figure_editor import PoseFigureEditor
    from scene_nodes.pose_resize_with_padding import PoseResizeWithPadding
    from scene_nodes.prefix_generator import PrefixGenerator
    from scene_nodes.text_prompt_generator import TextPromptGenerator
    from scene_nodes.text_prompt_replacer import TextPromptReplacer
    from scene_nodes.prompt_template_editor import PromptTemplateEditor
    try:
        from scene_nodes.person_crop_to_size import PersonCropToSize
    except Exception as exc:
        _person_crop_error = exc


_NODE_PREFIX = "ESS/"

_BASE_NODE_CLASS_MAPPINGS = {
    "SceneSelector": SceneSelector,
    "SceneSelectorTest": SceneSelectorTest,
    "DefineScenePart": DefineScenePart,
    "SceneCombiner": SceneCombiner,
    "SceneDebug": SceneDebug,
    "SceneMerge": SceneMerge,
    "InitScene": InitScene,
    "ReplaceDict": ReplaceDict,
    "ImageAdjustments": ImageAdjustmentsNode,
    "SegmentationDetailer": SegmentationDetailerNode,
    "ESSFaceDetailer": ESSFaceDetailer,
    "GroupReroute": GroupReroute,
    "StringConcatenate": StringConcatenate,
    "LabelNote": LabelNote,
    "PoseFigureEditor": PoseFigureEditor,
    "PoseResizeWithPadding": PoseResizeWithPadding,
    "PrefixGenerator": PrefixGenerator,
    "TextPromptGenerator": TextPromptGenerator,
    "TextPromptReplacer": TextPromptReplacer,
    "PromptTemplateEditor": PromptTemplateEditor,
}

_BASE_NODE_DISPLAY_NAME_MAPPINGS = {
    "SceneSelector": "Scene Selector",
    "SceneSelectorTest": "Scene Selector Test",
    "DefineScenePart": "Define Scene Part",
    "SceneCombiner": "Scene Combiner",
    "SceneDebug": "Scene Debug",
    "SceneMerge": "Scene Merge",
    "InitScene": "Init Scene",
    "ReplaceDict": "Replace Dict",
    "ImageAdjustments": "Image Adjustments",
    "SegmentationDetailer": "Segmentation Detailer",
    "ESSFaceDetailer": "Face Detailer (ESS)",
    "GroupReroute": "Group Reroute",
    "StringConcatenate": "String Concatenate",
    "LabelNote": "Label Note",
    "PoseFigureEditor": "Pose Figure Editor",
    "PoseResizeWithPadding": "Pose Resize With Padding",
    "PrefixGenerator": "Prefix Generator",
    "TextPromptGenerator": "Text Prompt Generator",
    "TextPromptReplacer": "Text Prompt Replacer",
    "PromptTemplateEditor": "Prompt Template Editor",
}

NODE_CLASS_MAPPINGS = {
    f"{_NODE_PREFIX}{key}": value for key, value in _BASE_NODE_CLASS_MAPPINGS.items()
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"{_NODE_PREFIX}{key}": f"ESS - {value}"
    for key, value in _BASE_NODE_DISPLAY_NAME_MAPPINGS.items()
}

if FaceSwapInSwapperNode is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}FaceSwapInSwapper"] = FaceSwapInSwapperNode
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}FaceSwapInSwapper"] = "ESS - Face Swap (InSwapper)"
if FaceSwapSimSwapNode is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}FaceSwapSimSwap"] = FaceSwapSimSwapNode
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}FaceSwapSimSwap"] = "ESS - Face Swap (SimSwap)"
if FaceSwapFaceFusionNode is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}FaceSwapFaceFusion"] = FaceSwapFaceFusionNode
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}FaceSwapFaceFusion"] = "ESS - Face Swap (FaceFusion)"
elif _face_swapper_error:
    print(f"[comfyui-ess] FaceSwap node disabled: {_face_swapper_error}", file=sys.stderr)

if PersonCropToSize is not None:
    NODE_CLASS_MAPPINGS[f"{_NODE_PREFIX}PersonCropToSize"] = PersonCropToSize
    NODE_DISPLAY_NAME_MAPPINGS[f"{_NODE_PREFIX}PersonCropToSize"] = "ESS - Person Crop To Size"
elif _person_crop_error:
    print(f"[comfyui-ess] PersonCropToSize node disabled: {_person_crop_error}", file=sys.stderr)

# Define categories
# Processing: ColorField
# Scene: SceneSelector, DefineScenePart, SceneCombiner, SceneDebug, SceneAppend, InitScene, ReplaceDict
# Image Processing: ImageAdjustments


def get_custom_types():
    return {
        "SCENE": {
            "input": Scene,
            "output": lambda x: isinstance(x, Scene)
        },
        "SEGMENTATION_MODEL": {
            "input": object,
            "output": lambda x: callable(x) or any(callable(getattr(x, attr, None)) for attr in ("predict_mask", "segment", "predict"))
        }
    }


WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
