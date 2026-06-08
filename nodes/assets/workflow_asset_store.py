from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from PIL import Image, ImageOps


_ARCHIVE_SUFFIX = ".ess"
_IMAGE_LIBRARY_KIND = "image_library"
_WORKFLOW_EXTRA_KEY = "ess_workflow_path"
_MANIFEST_NAME = "manifest.json"
_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def _sanitize_segment(value: Any, fallback: str = "asset") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = _SAFE_SEGMENT_RE.sub("_", text).strip(" .")
    return text or fallback


def _candidate_workflow_roots() -> list[Path]:
    current = Path(__file__).resolve()
    roots: list[Path] = []
    seen: set[str] = set()
    for parent in current.parents:
        candidate = parent / "user" / "default" / "workflows"
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            roots.append(candidate.resolve())
    return roots


def _default_workflow_root() -> Path:
    roots = _candidate_workflow_roots()
    if roots:
        return roots[0]
    fallback = Path(__file__).resolve().parents[3] / "user" / "default" / "workflows"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


def _extract_workflow_reference(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, dict):
        for key in (
            "path",
            "file",
            "filename",
            "relative_path",
            "relativePath",
            "name",
            "id",
        ):
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _normalize_reference_text(reference: str) -> str:
    text = unquote(str(reference or "")).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    while text.startswith("/"):
        text = text[1:]
    lower = text.lower()
    if lower == "workflows":
        return ""
    if lower.startswith("workflows/"):
        text = text[len("workflows/") :]
    return text.strip()


def _resolve_workflow_path(raw: Any) -> tuple[Path, str]:
    reference = _extract_workflow_reference(raw)
    if not reference:
        raise ValueError("Workflow reference is required.")

    reference = _normalize_reference_text(reference)
    workflow_root = _default_workflow_root()

    candidate = Path(reference)
    if candidate.is_absolute():
        workflow_path = candidate.resolve()
        try:
            relative = workflow_path.relative_to(workflow_root).as_posix()
        except Exception:
            relative = str(workflow_path)
    else:
        relative = reference
        workflow_path = (workflow_root / relative).resolve()
        if workflow_root not in {workflow_path, *workflow_path.parents}:
            raise ValueError("Workflow path is outside the managed workflows directory.")

    if workflow_path.suffix.lower() != ".json":
        workflow_path = workflow_path.with_suffix(".json")
        relative = _workflow_reference_for_path(workflow_path)
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    return workflow_path, relative


def _workflow_reference_for_path(workflow_path: Path) -> str:
    workflow_path = workflow_path.resolve()
    workflow_root = _default_workflow_root()
    try:
        return workflow_path.relative_to(workflow_root).as_posix()
    except Exception:
        return str(workflow_path)


def _archive_path_for_workflow(workflow_path: Path) -> Path:
    return workflow_path.with_name(workflow_path.name + _ARCHIVE_SUFFIX)


def _empty_manifest() -> dict[str, Any]:
    return {"version": 1, "nodes": {}}


def _load_archive(workflow_path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    archive_path = _archive_path_for_workflow(workflow_path)
    if not archive_path.exists():
        return _empty_manifest(), {}

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            try:
                manifest_raw = archive.read(_MANIFEST_NAME)
                manifest = json.loads(manifest_raw.decode("utf-8"))
            except Exception:
                manifest = _empty_manifest()
            if not isinstance(manifest, dict):
                manifest = _empty_manifest()
            manifest.setdefault("version", 1)
            manifest.setdefault("nodes", {})

            assets: dict[str, bytes] = {}
            for name in archive.namelist():
                if not name.startswith("assets/") or name.endswith("/"):
                    continue
                assets[name] = archive.read(name)
            return manifest, assets
    except Exception:
        return _empty_manifest(), {}


def _find_existing_workflow_refs(payload: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("workflow_relative_path") or item.get("workflowRelativePath") or "").strip()
        if not ref:
            continue
        if ref in seen:
            continue
        seen.add(ref)
        refs.append(ref)
    return refs


def _adopt_previous_archive_if_needed(workflow_path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    manifest, assets = _load_archive(workflow_path)
    if manifest.get("nodes") or assets:
        return manifest, assets

    target_archive = _archive_path_for_workflow(workflow_path)
    if target_archive.exists():
        return manifest, assets

    for ref in _find_existing_workflow_refs(payload):
        try:
            source_workflow_path, _ = _resolve_workflow_path(ref)
        except Exception:
            continue
        if source_workflow_path.resolve() == workflow_path.resolve():
            continue
        source_archive = _archive_path_for_workflow(source_workflow_path)
        if not source_archive.exists():
            continue
        try:
            target_archive.write_bytes(source_archive.read_bytes())
            try:
                source_archive.unlink()
            except Exception:
                pass
            return _load_archive(workflow_path)
        except Exception:
            continue

    return manifest, assets


def _save_archive(workflow_path: Path, manifest: dict[str, Any], assets: dict[str, bytes]) -> None:
    archive_path = _archive_path_for_workflow(workflow_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for asset_name in sorted(assets):
            archive.writestr(asset_name, assets[asset_name])


def _decode_data_url(item: dict[str, Any]) -> tuple[bytes, str, int, int]:
    raw_value = str(item.get("image_data") or item.get("imageData") or "").strip()
    if not raw_value:
        raise ValueError("image_data is required when saving an asset.")
    header = ""
    payload = raw_value
    if "base64," in raw_value:
        header, payload = raw_value.split("base64,", 1)
    mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip()
    if not mime_type and header.startswith("data:"):
        mime_type = header[5:].rstrip(";")
    raw_bytes = base64.b64decode(payload)
    with Image.open(BytesIO(raw_bytes)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), "image/png", int(width), int(height)


def _prune_missing_nodes(manifest: dict[str, Any], known_node_ids: list[Any] | None) -> None:
    if not isinstance(known_node_ids, list):
        return
    known = {str(node_id).strip() for node_id in known_node_ids if str(node_id).strip()}
    nodes = manifest.get("nodes")
    if not isinstance(nodes, dict):
        return

    remove_ids: list[str] = []
    for node_id, entry in nodes.items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("kind") or "").strip() != _IMAGE_LIBRARY_KIND:
            continue
        if str(node_id).strip() not in known:
            remove_ids.append(str(node_id))

    for node_id in remove_ids:
        nodes.pop(node_id, None)


def _collect_used_assets(manifest: dict[str, Any]) -> set[str]:
    used_assets: set[str] = set()
    nodes = manifest.get("nodes") or {}
    if not isinstance(nodes, dict):
        return used_assets
    for entry in nodes.values():
        if not isinstance(entry, dict):
            continue
        for item in entry.get("items") or []:
            if not isinstance(item, dict):
                continue
            asset_name = str(item.get("asset_name") or "").strip()
            if asset_name:
                used_assets.add(asset_name)
    return used_assets


def _collect_manifest_items_by_asset_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    nodes = manifest.get("nodes") or {}
    if not isinstance(nodes, dict):
        return indexed
    for entry in nodes.values():
        if not isinstance(entry, dict):
            continue
        for item in entry.get("items") or []:
            if not isinstance(item, dict):
                continue
            asset_id = str(item.get("asset_id") or "").strip()
            if asset_id and asset_id not in indexed:
                indexed[asset_id] = item
    return indexed


def _find_asset_in_archive_by_id(workflow_ref: Any, asset_id: str) -> tuple[dict[str, Any] | None, bytes | None]:
    try:
        workflow_path, _ = _resolve_workflow_path(workflow_ref)
    except Exception:
        return None, None
    manifest, assets = _load_archive(workflow_path)
    entry = _find_asset_entry(manifest, asset_id)
    if not entry:
        return None, None
    asset_name = str(entry.get("asset_name") or "").strip()
    if not asset_name:
        return None, None
    raw_bytes = assets.get(asset_name)
    if raw_bytes is None:
        return None, None
    return entry, raw_bytes


def _iter_archive_paths() -> list[Path]:
    workflow_root = _default_workflow_root()
    try:
        return list(workflow_root.rglob(f"*{_ARCHIVE_SUFFIX}"))
    except Exception:
        return []


def _find_asset_globally_by_id(asset_id: str, preferred_refs: list[Any] | None = None) -> tuple[dict[str, Any] | None, bytes | None, str]:
    checked: set[str] = set()

    for workflow_ref in preferred_refs or []:
        ref_text = _extract_workflow_reference(workflow_ref)
        if not ref_text:
            continue
        try:
            workflow_path, workflow_norm = _resolve_workflow_path(ref_text)
        except Exception:
            continue
        key = str(workflow_path).lower()
        if key in checked:
            continue
        checked.add(key)
        entry, raw_bytes = _find_asset_in_archive_by_id(workflow_norm, asset_id)
        if entry is not None and raw_bytes is not None:
            return entry, raw_bytes, _workflow_reference_for_path(workflow_path)

    for archive_path in _iter_archive_paths():
        try:
            workflow_path = archive_path.with_suffix("")
            workflow_path = workflow_path if workflow_path.suffix.lower() == ".json" else Path(str(archive_path)[: -len(_ARCHIVE_SUFFIX)])
            key = str(workflow_path.resolve()).lower()
            if key in checked:
                continue
            checked.add(key)
            manifest, assets = _load_archive(workflow_path)
            entry = _find_asset_entry(manifest, asset_id)
            if not entry:
                continue
            asset_name = str(entry.get("asset_name") or "").strip()
            if not asset_name:
                continue
            raw_bytes = assets.get(asset_name)
            if raw_bytes is None:
                continue
            return entry, raw_bytes, _workflow_reference_for_path(workflow_path)
        except Exception:
            continue

    return None, None, ""


def _persist_image_library_node(
    workflow_path: Path,
    manifest: dict[str, Any],
    assets: dict[str, bytes],
    payload: dict[str, Any],
) -> dict[str, Any]:
    node_id = str(payload.get("node_id") or payload.get("nodeId") or "").strip()
    if not node_id:
        raise ValueError("node_id is required.")

    nodes = manifest.setdefault("nodes", {})
    existing_node = nodes.get(node_id) if isinstance(nodes.get(node_id), dict) else {}
    existing_items = {
        str(item.get("asset_id") or ""): item
        for item in existing_node.get("items", [])
        if isinstance(item, dict) and str(item.get("asset_id") or "").strip()
    }
    all_manifest_items = _collect_manifest_items_by_asset_id(manifest)

    saved_items: list[dict[str, Any]] = []
    unresolved_refs: list[str] = []
    for index, raw_item in enumerate(payload.get("items") or []):
        if not isinstance(raw_item, dict):
            continue
        asset_id = str(raw_item.get("asset_id") or raw_item.get("assetId") or "").strip()
        name = str(raw_item.get("name") or "").strip() or f"Image {index + 1}"
        prompt = str(raw_item.get("prompt") or "")
        try:
            weight = float(raw_item.get("weight", 1) or 0)
        except Exception:
            weight = 1.0

        image_data = str(raw_item.get("image_data") or raw_item.get("imageData") or "").strip()
        if image_data:
            raw_bytes, mime_type, width, height = _decode_data_url(raw_item)
            asset_id = asset_id or uuid.uuid4().hex
            content_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            asset_name = f"assets/{asset_id}_{content_hash}.png"
            assets[asset_name] = raw_bytes
        else:
            if not asset_id:
                continue
            existing_item = existing_items.get(asset_id) or all_manifest_items.get(asset_id) or {}
            asset_name = str(existing_item.get("asset_name") or "").strip()
            if not asset_name or asset_name not in assets:
                source_ref = str(raw_item.get("workflow_relative_path") or raw_item.get("workflowRelativePath") or "").strip()
                source_entry, source_bytes = _find_asset_in_archive_by_id(source_ref, asset_id)
                if source_entry is None or source_bytes is None:
                    source_entry, source_bytes, _resolved_ref = _find_asset_globally_by_id(
                        asset_id,
                        preferred_refs=[source_ref, payload.get("workflow")],
                    )
                if source_entry is None or source_bytes is None:
                    unresolved_refs.append(asset_id or f"item_{index + 1}")
                    continue
                source_asset_name = str(source_entry.get("asset_name") or "").strip()
                source_suffix = Path(source_asset_name).suffix or ".bin"
                asset_name = f"assets/{asset_id}_{uuid.uuid4().hex[:8]}{source_suffix}"
                assets[asset_name] = source_bytes
                existing_item = source_entry
            width = int(existing_item.get("width", raw_item.get("width", 0)) or 0)
            height = int(existing_item.get("height", raw_item.get("height", 0)) or 0)
            mime_type = str(existing_item.get("mime_type") or raw_item.get("mime_type") or "image/png")

        saved_items.append(
            {
                "asset_id": asset_id,
                "asset_name": asset_name,
                "name": name,
                "prompt": prompt,
                "weight": weight,
                "width": width,
                "height": height,
                "mime_type": mime_type,
            }
        )

    incoming_count = sum(1 for item in (payload.get("items") or []) if isinstance(item, dict))
    if incoming_count > 0 and not saved_items:
        unresolved_text = ", ".join(unresolved_refs[:5]) if unresolved_refs else "unknown assets"
        raise ValueError(
            f"Could not persist image library assets. Missing asset data for: {unresolved_text}."
        )

    nodes[node_id] = {
        "kind": _IMAGE_LIBRARY_KIND,
        "items": saved_items,
    }

    used_assets = _collect_used_assets(manifest)
    for asset_name in list(assets):
        if asset_name not in used_assets:
            assets.pop(asset_name, None)

    _save_archive(workflow_path, manifest, assets)
    workflow_relative = _workflow_reference_for_path(workflow_path)
    return {
        "mode": str(payload.get("mode") or "manual"),
        "selected_index": int(payload.get("selected_index", payload.get("selectedIndex", 0)) or 0),
        "items": [
            {
                "asset_id": item["asset_id"],
                "name": item["name"],
                "prompt": item["prompt"],
                "weight": item["weight"],
                "width": item["width"],
                "height": item["height"],
                "mime_type": item["mime_type"],
                "workflow_relative_path": workflow_relative,
            }
            for item in saved_items
        ],
    }


def _find_asset_entry(manifest: dict[str, Any], asset_id: str) -> dict[str, Any] | None:
    for node_entry in (manifest.get("nodes") or {}).values():
        if not isinstance(node_entry, dict):
            continue
        for item in node_entry.get("items") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("asset_id") or "").strip() == asset_id:
                return item
    return None


def resolve_image_library_asset(item: dict[str, Any], extra_pnginfo: Any = None) -> bytes | None:
    asset_id = str(item.get("asset_id") or item.get("assetId") or "").strip()
    if not asset_id:
        return None

    workflow_refs: list[str] = []
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if isinstance(workflow, dict):
            extra = workflow.get("extra") if isinstance(workflow.get("extra"), dict) else {}
            ref = str(extra.get(_WORKFLOW_EXTRA_KEY) or "").strip()
            if ref:
                workflow_refs.append(ref)
    item_ref = str(item.get("workflow_relative_path") or "").strip()
    if item_ref and item_ref not in workflow_refs:
        workflow_refs.append(item_ref)

    for workflow_ref in workflow_refs:
        try:
            workflow_path, _ = _resolve_workflow_path(workflow_ref)
        except Exception:
            continue
        manifest, assets = _load_archive(workflow_path)
        entry = _find_asset_entry(manifest, asset_id)
        if not entry:
            continue
        asset_name = str(entry.get("asset_name") or "").strip()
        if not asset_name:
            continue
        raw_bytes = assets.get(asset_name)
        if raw_bytes is not None:
            return raw_bytes

    _entry, raw_bytes, _workflow_ref = _find_asset_globally_by_id(asset_id, preferred_refs=workflow_refs)
    if raw_bytes is not None:
        return raw_bytes
    return None


def register_workflow_asset_routes(prompt_server, web_module):
    if prompt_server is None or web_module is None:
        return

    @prompt_server.instance.routes.post("/ess/workflow_assets/save")
    async def ess_workflow_assets_save(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        try:
            workflow_path, relative = _resolve_workflow_path(payload.get("workflow"))
            manifest, assets = _adopt_previous_archive_if_needed(workflow_path, payload)
            _prune_missing_nodes(manifest, payload.get("known_node_ids"))
            node_state = _persist_image_library_node(workflow_path, manifest, assets, payload)
            return web_module.json_response(
                {
                    "ok": True,
                    "workflow_relative_path": relative,
                    "state": node_state,
                }
            )
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    @prompt_server.instance.routes.get("/ess/workflow_assets/file")
    async def ess_workflow_assets_file(request):
        try:
            workflow_path, _ = _resolve_workflow_path(request.rel_url.query.get("workflow", ""))
            asset_id = str(request.rel_url.query.get("asset_id", "") or "").strip()
            if not asset_id:
                return web_module.json_response({"ok": False, "error": "asset_id is required"}, status=400)

            manifest, assets = _load_archive(workflow_path)
            entry = _find_asset_entry(manifest, asset_id)
            if not entry:
                return web_module.json_response({"ok": False, "error": "asset not found"}, status=404)

            asset_name = str(entry.get("asset_name") or "").strip()
            raw_bytes = assets.get(asset_name)
            if raw_bytes is None:
                return web_module.json_response({"ok": False, "error": "asset not found"}, status=404)

            content_type = str(entry.get("mime_type") or "application/octet-stream")
            return web_module.Response(body=raw_bytes, content_type=content_type)
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)
