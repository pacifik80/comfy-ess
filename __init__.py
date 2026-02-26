import sys
import json
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
    from .nodes.prompt_builder.replacements_dictionary import ReplaceDict
    from .nodes.image_processing.image_adjustments import ImageAdjustmentsNode
    from .nodes.image.segmentation_detailer import SegmentationDetailerNode
    from .nodes.detailer.ess_face_detailer import ESSFaceDetailer
    from .nodes.utils.group_reroute import GroupReroute
    from .nodes.utils.string_concat import StringConcatenate
    from .nodes.utils.label_note import LabelNote
    try:
        from .nodes.face_swapping.face_swapping import FaceSwapInSwapperNode, FaceSwapSimSwapNode, FaceSwapFaceFusionNode
    except Exception as exc:
        _face_swapper_error = exc
    from .nodes.pose.pose_mesh_editor import PoseMeshEditor
    from .nodes.utils.prefix_generator import PrefixGenerator
    from .nodes.prompt_builder.text_prompt_generator import TextPromptGenerator
    from .nodes.prompt_builder.text_prompt_replacer import TextPromptReplacer
    from .nodes.prompt_builder.prompt_template_editor import PromptTemplateEditor
    from .nodes.prompt_builder.prompt_template_editor_multi import PromptTemplateEditorMulti
    from .nodes.prompt_builder.scene_flow_editor import SceneFlowEditor
    try:
        from .nodes.image.person_crop_to_size import PersonCropToSize
    except Exception as exc:
        _person_crop_error = exc
except ImportError as exc:
    if "attempted relative import" not in str(exc):
        raise
    import importlib
    import importlib.util

    package_dir = Path(__file__).resolve().parent
    nodes_dir = package_dir / "nodes"
    local_nodes_pkg = "comfyui_ess_local_nodes"
    if local_nodes_pkg not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            local_nodes_pkg,
            nodes_dir / "__init__.py",
            submodule_search_locations=[str(nodes_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load local nodes package from: {nodes_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[local_nodes_pkg] = module
        spec.loader.exec_module(module)

    def _import_node_attr(module_name: str, attr_name: str):
        module = importlib.import_module(f"{local_nodes_pkg}.{module_name}")
        return getattr(module, attr_name)

    ReplaceDict = _import_node_attr("prompt_builder.replacements_dictionary", "ReplaceDict")
    ImageAdjustmentsNode = _import_node_attr("image_processing.image_adjustments", "ImageAdjustmentsNode")
    SegmentationDetailerNode = _import_node_attr("image.segmentation_detailer", "SegmentationDetailerNode")
    ESSFaceDetailer = _import_node_attr("detailer.ess_face_detailer", "ESSFaceDetailer")
    GroupReroute = _import_node_attr("utils.group_reroute", "GroupReroute")
    StringConcatenate = _import_node_attr("utils.string_concat", "StringConcatenate")
    LabelNote = _import_node_attr("utils.label_note", "LabelNote")
    try:
        FaceSwapInSwapperNode = _import_node_attr("face_swapping.face_swapping", "FaceSwapInSwapperNode")
        FaceSwapSimSwapNode = _import_node_attr("face_swapping.face_swapping", "FaceSwapSimSwapNode")
        FaceSwapFaceFusionNode = _import_node_attr("face_swapping.face_swapping", "FaceSwapFaceFusionNode")
    except Exception as exc:
        _face_swapper_error = exc
    PoseMeshEditor = _import_node_attr("pose.pose_mesh_editor", "PoseMeshEditor")
    PrefixGenerator = _import_node_attr("utils.prefix_generator", "PrefixGenerator")
    TextPromptGenerator = _import_node_attr("prompt_builder.text_prompt_generator", "TextPromptGenerator")
    TextPromptReplacer = _import_node_attr("prompt_builder.text_prompt_replacer", "TextPromptReplacer")
    PromptTemplateEditor = _import_node_attr("prompt_builder.prompt_template_editor", "PromptTemplateEditor")
    PromptTemplateEditorMulti = _import_node_attr("prompt_builder.prompt_template_editor_multi", "PromptTemplateEditorMulti")
    SceneFlowEditor = _import_node_attr("prompt_builder.scene_flow_editor", "SceneFlowEditor")
    try:
        PersonCropToSize = _import_node_attr("image.person_crop_to_size", "PersonCropToSize")
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
    "SceneFlowEditor": SceneFlowEditor,
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
    "SceneFlowEditor": "Scene Flow Editor",
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

        def _run_scene_flow_test_payload(payload):
            if not isinstance(payload, dict):
                payload = {}

            flow_script = str(payload.get("flow_script", "") or "")

            raw_seed = payload.get("seed", 0)
            try:
                seed = int(raw_seed)
            except Exception:
                seed = 0
            if seed < 0:
                seed = 0

            raw_parse = payload.get("parse_templates", True)
            if isinstance(raw_parse, str):
                parse_templates = raw_parse.strip().lower() in {"1", "true", "yes", "on", "parse"}
            else:
                parse_templates = bool(raw_parse)

            generator = SceneFlowEditor()
            positive, negative, debug_json = generator.generate(
                flow_script=flow_script,
                seed=seed,
                parse_templates=parse_templates,
            )
            debug_obj = {}
            try:
                parsed_debug = json.loads(debug_json) if isinstance(debug_json, str) else {}
                if isinstance(parsed_debug, dict):
                    debug_obj = parsed_debug
            except Exception:
                debug_obj = {"raw": str(debug_json)}

            return {
                "ok": True,
                "positive": positive,
                "negative": negative,
                "debug": debug_obj,
            }

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

        @PromptServer.instance.routes.post("/ess/scene_flow/test")
        async def ess_scene_flow_test(request):
            try:
                payload = await request.json()
            except Exception:
                payload = {}

            try:
                return web.json_response(_run_scene_flow_test_payload(payload))
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=500)

        @PromptServer.instance.routes.get("/ess/scene_flow/test")
        async def ess_scene_flow_test_get(request):
            payload = {
                "flow_script": request.rel_url.query.get("flow_script", ""),
                "seed": request.rel_url.query.get("seed", 0),
                "parse_templates": request.rel_url.query.get("parse_templates", True),
            }
            try:
                return web.json_response(_run_scene_flow_test_payload(payload))
            except Exception as exc:
                return web.json_response({"ok": False, "error": str(exc)}, status=500)
    except Exception as exc:  # pragma: no cover - route registration is optional
        print(f"[comfyui-ess] Mesh routes unavailable: {exc}", file=sys.stderr)
