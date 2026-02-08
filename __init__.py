import sys
from urllib.parse import unquote
from pathlib import Path

try:
    from aiohttp import web
except Exception:  # pragma: no cover - optional in some test contexts
    web = None

FaceSwapInSwapperNode = None
FaceSwapSimSwapNode = None
FaceSwapFaceFusionNode = None
_face_swapper_error = None
PersonCropToSize = None
_person_crop_error = None

try:
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
    from .scene_nodes.pose_mesh_editor import PoseMeshEditor
    from .scene_nodes.prefix_generator import PrefixGenerator
    from .scene_nodes.text_prompt_generator import TextPromptGenerator
    from .scene_nodes.text_prompt_replacer import TextPromptReplacer
    from .scene_nodes.prompt_template_editor import PromptTemplateEditor
    from .scene_nodes.prompt_template_editor_multi import PromptTemplateEditorMulti
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
    from scene_nodes.pose_mesh_editor import PoseMeshEditor
    from scene_nodes.prefix_generator import PrefixGenerator
    from scene_nodes.text_prompt_generator import TextPromptGenerator
    from scene_nodes.text_prompt_replacer import TextPromptReplacer
    from scene_nodes.prompt_template_editor import PromptTemplateEditor
    from scene_nodes.prompt_template_editor_multi import PromptTemplateEditorMulti
    try:
        from scene_nodes.person_crop_to_size import PersonCropToSize
    except Exception as exc:
        _person_crop_error = exc


_NODE_PREFIX = "ESS/"

_BASE_NODE_CLASS_MAPPINGS = {
    "ReplaceDict": ReplaceDict,
    "ImageAdjustments": ImageAdjustmentsNode,
    "SegmentationDetailer": SegmentationDetailerNode,
    "ESSFaceDetailer": ESSFaceDetailer,
    "GroupReroute": GroupReroute,
    "StringConcatenate": StringConcatenate,
    "LabelNote": LabelNote,
    "PoseMeshEditor": PoseMeshEditor,
    "PrefixGenerator": PrefixGenerator,
    "TextPromptGenerator": TextPromptGenerator,
    "TextPromptReplacer": TextPromptReplacer,
    "PromptTemplateEditor": PromptTemplateEditor,
    "PromptTemplateEditorMulti": PromptTemplateEditorMulti,
}

_BASE_NODE_DISPLAY_NAME_MAPPINGS = {
    "ReplaceDict": "Replace Dict",
    "ImageAdjustments": "Image Adjustments",
    "SegmentationDetailer": "Segmentation Detailer",
    "ESSFaceDetailer": "Face Detailer (ESS)",
    "GroupReroute": "Group Reroute",
    "StringConcatenate": "String Concatenate",
    "LabelNote": "Label Note",
    "PoseMeshEditor": "Pose Mesh Editor",
    "PrefixGenerator": "Prefix Generator",
    "TextPromptGenerator": "Text Prompt Generator",
    "TextPromptReplacer": "Text Prompt Replacer",
    "PromptTemplateEditor": "Prompt Template Editor",
    "PromptTemplateEditorMulti": "Prompt Template Editor - Multioutput",
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
# Image Processing: ImageAdjustments


def get_custom_types():
    return {
        "SEGMENTATION_MODEL": {
            "input": object,
            "output": lambda x: callable(x) or any(callable(getattr(x, attr, None)) for attr in ("predict_mask", "segment", "predict"))
        }
    }


WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _get_rigged_meshes_dir() -> Path:
    return Path(__file__).resolve().parent / "meshes" / "rigged"


def _list_rigged_meshes() -> list[str]:
    meshes_dir = _get_rigged_meshes_dir()
    if not meshes_dir.exists():
        return []
    allowed = {".fbx", ".obj", ".glb", ".gltf"}
    return sorted(
        [
            p.name
            for p in meshes_dir.iterdir()
            if p.is_file() and p.suffix.lower() in allowed
        ]
    )


if web is not None:
    try:
        from server import PromptServer

        @PromptServer.instance.routes.get("/ess/rigged/list")
        async def ess_rigged_list(_request):
            return web.json_response({"items": _list_rigged_meshes()})

        @PromptServer.instance.routes.get("/ess/rigged/get")
        async def ess_rigged_get(request):
            raw_name = request.rel_url.query.get("name", "")
            name = unquote(raw_name).strip()
            if not name:
                return web.json_response({"error": "missing name"}, status=400)

            meshes_dir = _get_rigged_meshes_dir().resolve()
            target = (meshes_dir / name).resolve()
            if not str(target).startswith(str(meshes_dir)):
                return web.json_response({"error": "invalid name"}, status=400)
            if not target.exists() or not target.is_file():
                return web.json_response({"error": "not found"}, status=404)

            return web.Response(
                body=target.read_bytes(),
                headers={"X-Mesh-Name": target.name},
                content_type="application/octet-stream",
            )
    except Exception as exc:  # pragma: no cover - route registration is optional
        print(f"[comfyui-ess] Mesh routes unavailable: {exc}", file=sys.stderr)
