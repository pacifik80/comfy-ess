from __future__ import annotations

import asyncio
import hashlib
import html
import importlib
import importlib.util
import inspect
import json
import mimetypes
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import comfy.sd
import folder_paths


_SUPPORTED_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
_SIDECAR_SUFFIX = ".ess.json"
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_RETRY_LIMIT = 5
_DEFAULT_CIVITAI_LIMIT = 20
_CIVITAI_MODELS_ENDPOINT = "https://civitai.com/api/v1/models"
_CIVITAI_MODEL_VERSIONS_ENDPOINT = "https://civitai.com/api/v1/model-versions"
_HTTP_USER_AGENT = "comfyui-ess/1.0"
_DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
_DOWNLOAD_JOBS_LOCK = threading.Lock()
_GGUF_MODULE_CACHE = None
_GGUF_MODULE_LOCK = threading.Lock()
_COMPONENT_UNKNOWN = None


def _sanitize_segment(value: Any, fallback: str = "untitled") -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    text = re.sub(r"[<>:\"/\\\\|?*\x00-\x1f]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or fallback


def _strip_html(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_format(format_name: Any = None, filename: Any = None) -> str:
    raw = str(format_name or "").strip().lower()
    suffix = Path(str(filename or "")).suffix.lower()
    if raw in {"safetensor", "safetensors", "safe_tensor"} or suffix == ".safetensors":
        return "SafeTensor"
    if raw in {"gguf"} or suffix == ".gguf":
        return "GGUF"
    if raw in {"ckpt", "checkpoint"} or suffix == ".ckpt":
        return "CKPT"
    if suffix in {".pt", ".pth", ".bin"}:
        return suffix.lstrip(".").upper()
    return "Unknown"


def _normalize_preview_entry(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": preview.get("id"),
        "url": preview.get("url") or "",
        "width": preview.get("width"),
        "height": preview.get("height"),
        "nsfw": preview.get("nsfw"),
        "nsfwLevel": preview.get("nsfwLevel"),
        "type": preview.get("type"),
        "hash": preview.get("hash"),
        "mimeType": preview.get("mimeType") or preview.get("mime_type"),
        "postId": preview.get("postId"),
        "createdAt": preview.get("createdAt"),
        "username": preview.get("username"),
        "stats": preview.get("stats") if isinstance(preview.get("stats"), dict) else None,
        "meta": preview.get("meta") if isinstance(preview.get("meta"), dict) else None,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _infer_base_model(*texts: Any) -> str:
    haystack = " ".join(str(text or "") for text in texts).lower()
    checks = [
        ("Illustrious", ["illustrious"]),
        ("WAN", ["wan2.2", "wan2.1", "wan"]),
        ("SDXL", ["sdxl", "xl", "ponyxl"]),
        ("Pony", ["pony"]),
        ("FLUX", ["flux"]),
        ("SD 1.5", ["sd1.5", "sd15", "1.5", "sd 1.5"]),
        ("Qwen", ["qwen"]),
    ]
    for label, tokens in checks:
        if any(token in haystack for token in tokens):
            return label
    return ""


def _relative_to_root(path: Path, root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return None


def _folder_from_relative_path(relative_path: str | None) -> str:
    if not relative_path:
        return ""
    parent = Path(relative_path).parent.as_posix().replace("\\", "/")
    return "" if parent == "." else parent


def _local_source_label(source_kind: str) -> str:
    if source_kind == "ess":
        return "ESS"
    if source_kind == "unet":
        return "UNet"
    return "Checkpoints"


def _item_base_models(item: dict[str, Any]) -> list[str]:
    labels: set[str] = set()
    for label in item.get("baseModels") or []:
        text = str(label or "").strip()
        if text:
            labels.add(text)
    for version in item.get("versions") or []:
        text = str(version.get("baseModel") or "").strip()
        if text:
            labels.add(text)
    return sorted(labels, key=str.lower)


def _collect_available_base_models(items: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for item in items:
        labels.update(_item_base_models(item))
    return sorted(labels, key=str.lower)


def _attach_local_browser_metadata(
    item: dict[str, Any],
    path: Path,
    search_root: Path,
    *,
    source_kind: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    browser_relative_path = _relative_to_root(resolved, search_root) or resolved.name
    browser_folder = _folder_from_relative_path(browser_relative_path)
    source_root_name = search_root.name or str(search_root)
    source_label = _local_source_label(source_kind)

    base_models: set[str] = set()
    for version in item.get("versions") or []:
        trained_words = list(version.get("trainedWords") or [])
        version_file_names = [str(file.get("name") or "") for file in version.get("files") or []]
        base_model = str(version.get("baseModel") or "").strip()
        if not base_model:
            base_model = _infer_base_model(
                item.get("name"),
                version.get("name"),
                item.get("description"),
                browser_relative_path,
                browser_folder,
                source_root_name,
                " ".join(trained_words),
                " ".join(version_file_names),
                " ".join(item.get("tags") or []),
            )
            version["baseModel"] = base_model
        if base_model:
            base_models.add(base_model)

        for file in version.get("files") or []:
            file["browserRelativePath"] = browser_relative_path
            file["browserFolder"] = browser_folder
            file["sourceLabel"] = source_label
            file["sourceRootName"] = source_root_name

    item["browserRelativePath"] = browser_relative_path
    item["browserFolder"] = browser_folder
    item["sourceKind"] = source_kind
    item["sourceLabel"] = source_label
    item["sourceRootName"] = source_root_name
    item["sourceRootPath"] = str(search_root)
    item["baseModels"] = sorted(base_models, key=str.lower)
    return item


def _safe_json_read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return None


def _safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_ess_root(raw_root: Any = None) -> Path:
    root_text = str(raw_root or "").strip()
    if root_text:
        root = Path(root_text).expanduser()
    else:
        root = Path(folder_paths.models_dir) / "ess"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _normalize_subfolder(raw_name: Any, default: str = "checkpoints") -> str:
    name = str(raw_name or default).replace("\\", "/").strip().strip("/")
    parts = [_sanitize_segment(part) for part in name.split("/") if part and part not in {".", ".."}]
    return "/".join(parts) if parts else default


def _resolve_checkpoint_dir(raw_root: Any = None, subfolder: Any = None, raw_directory: Any = None) -> tuple[Path, Path, str]:
    directory_text = str(raw_directory or "").replace("\\", "/").strip().strip("/")
    if directory_text:
        root = _resolve_ess_root(None)
        normalized_directory = _normalize_subfolder(directory_text, default="checkpoints")
        checkpoint_dir = (root / normalized_directory).resolve()
        if root not in {checkpoint_dir, *checkpoint_dir.parents}:
            raise ValueError("Invalid ESS checkpoint directory configuration.")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return root, checkpoint_dir, normalized_directory

    root = _resolve_ess_root(raw_root)
    normalized = _normalize_subfolder(subfolder, default="checkpoints")
    checkpoint_dir = (root / normalized).resolve()
    if root not in {checkpoint_dir, *checkpoint_dir.parents}:
        raise ValueError("Invalid ESS checkpoint folder configuration.")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return root, checkpoint_dir, normalized


def _file_is_supported(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _SUPPORTED_MODEL_EXTENSIONS


def _sidecar_path_for_model(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + _SIDECAR_SUFFIX)


def _preview_dir_for_model(model_path: Path) -> Path:
    return model_path.parent / f".{model_path.name}.ess_previews"


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_existing_file(path: Path, expected_size: int | None, expected_sha256: str | None) -> tuple[bool, str | None]:
    if not path.exists() or not path.is_file():
        return False, None
    stat = path.stat()
    if expected_size and stat.st_size != expected_size:
        return False, None
    if expected_sha256:
        actual = _compute_sha256(path)
        if actual.lower() != expected_sha256.lower():
            return False, actual
        return True, actual
    return True, None


def _guess_component_presence_for_file(path: Path) -> tuple[bool | None, bool | None]:
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        return False, False
    if suffix != ".safetensors":
        return _COMPONENT_UNKNOWN, _COMPONENT_UNKNOWN

    try:
        from safetensors import safe_open
    except Exception:
        return _COMPONENT_UNKNOWN, _COMPONENT_UNKNOWN

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception:
        return _COMPONENT_UNKNOWN, _COMPONENT_UNKNOWN

    has_clip = any(
        key.startswith(prefix)
        for key in keys
        for prefix in (
            "cond_stage_model.",
            "conditioner.embedders.",
            "text_encoders.",
            "clip_l.",
            "clip_g.",
        )
    )
    has_vae = any(
        key.startswith(prefix)
        for key in keys
        for prefix in (
            "first_stage_model.",
            "vae.",
            "decoder.",
            "encoder.",
        )
    )
    return has_clip, has_vae


def _bool_state_label(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _http_headers(api_key: str | None = None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, */*;q=0.8",
        "User-Agent": _HTTP_USER_AGENT,
    }
    api_key_text = str(api_key or "").strip()
    if api_key_text:
        headers["Authorization"] = api_key_text if api_key_text.lower().startswith("bearer ") else f"Bearer {api_key_text}"
    if extra:
        headers.update(extra)
    return headers


def _http_json(url: str, *, headers: dict[str, str] | None = None, timeout: int = 30) -> dict[str, Any]:
    request = Request(url, headers=headers or {}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_download(
    url: str,
    destination: Path,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    progress_cb=None,
) -> None:
    request = Request(url, headers=headers or {}, method="GET")
    with urlopen(request, timeout=timeout) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None
        downloaded = 0
        while True:
            chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if progress_cb:
                progress_cb(downloaded, total_bytes)


def _download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    headers: dict[str, str],
    job_id: str,
) -> str | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".part")
    if expected_size and temp_path.exists() and temp_path.stat().st_size > expected_size:
        temp_path.unlink(missing_ok=True)

    for attempt in range(1, _DOWNLOAD_RETRY_LIMIT + 1):
        existing = temp_path.stat().st_size if temp_path.exists() else 0
        request_headers = dict(headers)
        if existing > 0:
            request_headers["Range"] = f"bytes={existing}-"
        _update_download_job(
            job_id,
            status="running",
            stage="downloading",
            message="Downloading model...",
            attempt=attempt,
            bytes_downloaded=existing,
            total_bytes=expected_size,
        )

        try:
            request = Request(url, headers=request_headers, method="GET")
            with urlopen(request, timeout=30) as response:
                status = getattr(response, "status", response.getcode())
                content_length = response.headers.get("Content-Length")
                reported = int(content_length) if content_length and content_length.isdigit() else None

                if status == 200 and existing > 0:
                    temp_path.unlink(missing_ok=True)
                    existing = 0
                mode = "ab" if existing > 0 else "wb"
                total_bytes = expected_size
                if reported is not None and status == 206:
                    total_bytes = existing + reported
                elif reported is not None and total_bytes is None:
                    total_bytes = reported

                downloaded = existing
                with temp_path.open(mode) as handle:
                    while True:
                        chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        _update_download_job(
                            job_id,
                            status="running",
                            stage="downloading",
                            message="Downloading model...",
                            bytes_downloaded=downloaded,
                            total_bytes=total_bytes,
                        )

                if total_bytes and temp_path.stat().st_size < total_bytes:
                    raise RuntimeError("Connection closed before the download completed.")
                break
        except HTTPError as exc:
            if exc.code == 416 and expected_size and temp_path.exists() and temp_path.stat().st_size == expected_size:
                break
            if attempt >= _DOWNLOAD_RETRY_LIMIT:
                raise RuntimeError(f"Download failed with HTTP {exc.code}.") from exc
        except (URLError, TimeoutError, RuntimeError) as exc:
            if attempt >= _DOWNLOAD_RETRY_LIMIT:
                raise RuntimeError(f"Download failed after {attempt} attempts: {exc}") from exc
        time.sleep(min(2 ** (attempt - 1), 8))

    valid, existing_sha = _validate_existing_file(temp_path, expected_size, expected_sha256)
    if not valid:
        raise RuntimeError("Downloaded file did not pass validation.")

    temp_path.replace(destination)
    if expected_sha256:
        return expected_sha256.lower()
    return existing_sha


def _download_preview_images(
    model_path: Path,
    root: Path,
    preview_images: list[dict[str, Any]],
    *,
    headers: dict[str, str],
    job_id: str,
) -> list[dict[str, Any]]:
    preview_dir = _preview_dir_for_model(model_path)
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for index, preview in enumerate(preview_images[:3], start=1):
        url = str(preview.get("url") or "").strip()
        if not url:
            continue
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        preview_type = str(preview.get("type") or "").strip().lower()
        if preview_type == "video" and suffix not in {".mp4", ".webm", ".mov"}:
            suffix = ".mp4"
        elif preview_type != "video" and suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        preview_path = preview_dir / f"{index:02d}{suffix}"
        try:
            _update_download_job(job_id, stage="preview", message=f"Saving preview {index}...")
            _http_download(url, preview_path, headers=headers)
        except Exception:
            continue
        relative = _relative_to_root(preview_path, root)
        saved.append(
            {
                "relative_path": relative,
                "absolute_path": str(preview_path.resolve()),
                "url": url,
                "width": preview.get("width"),
                "height": preview.get("height"),
                "nsfw": preview.get("nsfw"),
                "nsfwLevel": preview.get("nsfwLevel"),
                "type": preview.get("type"),
                "hash": preview.get("hash"),
                "mimeType": preview.get("mimeType") or preview.get("mime_type"),
                "postId": preview.get("postId"),
                "createdAt": preview.get("createdAt"),
                "username": preview.get("username"),
                "stats": preview.get("stats") if isinstance(preview.get("stats"), dict) else None,
                "meta": preview.get("meta") if isinstance(preview.get("meta"), dict) else None,
            }
        )
    return saved


def _selection_payload_from_local_file(
    path: Path,
    *,
    root: Path | None = None,
    sidecar: dict[str, Any] | None = None,
    managed: bool = False,
) -> dict[str, Any]:
    resolved = path.resolve()
    sidecar = sidecar or _safe_json_read(_sidecar_path_for_model(path)) or {}
    local_info = sidecar.get("local") if isinstance(sidecar, dict) else {}
    capabilities = sidecar.get("capabilities") if isinstance(sidecar, dict) else {}
    format_name = _normalize_format(local_info.get("format") if isinstance(local_info, dict) else None, resolved.name)

    relative_path = None
    if root:
        try:
            relative_path = resolved.relative_to(root).as_posix()
        except Exception:
            relative_path = None

    return {
        "selection_type": "local",
        "model_name": (sidecar.get("model") or {}).get("name") if isinstance(sidecar, dict) else resolved.stem,
        "version_name": (sidecar.get("version") or {}).get("name") if isinstance(sidecar, dict) else "",
        "file_name": resolved.name,
        "format": format_name,
        "local_path": str(resolved),
        "relative_path": relative_path,
        "managed": managed,
        "has_clip": capabilities.get("has_clip", _COMPONENT_UNKNOWN) if isinstance(capabilities, dict) else _COMPONENT_UNKNOWN,
        "has_vae": capabilities.get("has_vae", _COMPONENT_UNKNOWN) if isinstance(capabilities, dict) else _COMPONENT_UNKNOWN,
        "description": (sidecar.get("model") or {}).get("description_text", "") if isinstance(sidecar, dict) else "",
        "tags": (sidecar.get("model") or {}).get("tags", []) if isinstance(sidecar, dict) else [],
        "trained_words": (sidecar.get("version") or {}).get("trainedWords", []) if isinstance(sidecar, dict) else [],
    }


def _build_preview_url(root: Path | None, relative_path: str | None = None, absolute_path: str | None = None) -> str:
    params_payload: dict[str, str] = {}
    if root is not None:
        params_payload["root"] = str(root)
    if relative_path:
        params_payload["path"] = relative_path
    if absolute_path:
        params_payload["absolute"] = absolute_path
    params = urlencode(params_payload)
    return f"/ess/checkpoints/preview?{params}"


def _transform_sidecar_to_group(path: Path, root: Path, sidecar: dict[str, Any], managed: bool) -> dict[str, Any]:
    model_info = sidecar.get("model") or {}
    version_info = sidecar.get("version") or {}
    file_info = sidecar.get("file") or {}
    capabilities = sidecar.get("capabilities") or {}
    previews = sidecar.get("previews") or []
    local_path = str(path.resolve())
    relative_path = _relative_to_root(path, root)
    modified_at = int(path.stat().st_mtime) if path.exists() else 0
    model_stats = model_info.get("stats") or {}
    version_stats = version_info.get("stats") or {}

    return {
        "id": f"local-file-{hashlib.sha1(local_path.encode('utf-8')).hexdigest()}",
        "source": "local",
        "remoteModelId": model_info.get("id"),
        "name": model_info.get("name") or path.stem,
        "creator": (model_info.get("creator") or {}).get("username") or (model_info.get("creator") or {}).get("name") or "",
        "description": model_info.get("description_text") or "",
        "tags": list(model_info.get("tags") or []),
        "modifiedAt": modified_at,
        "createdAt": _safe_int(version_info.get("createdAt") or model_info.get("createdAt")),
        "updatedAt": _safe_int(version_info.get("updatedAt") or model_info.get("updatedAt") or modified_at),
        "downloadCount": _safe_int(version_stats.get("downloadCount") or model_stats.get("downloadCount")),
        "ratingCount": _safe_int(version_stats.get("ratingCount") or model_stats.get("ratingCount")),
        "rating": float(version_stats.get("rating") or model_stats.get("rating") or 0.0),
        "nsfw": model_info.get("nsfw"),
        "versions": [
            {
                "id": str(version_info.get("id") or hashlib.sha1(local_path.encode("utf-8")).hexdigest()),
                "name": version_info.get("name") or path.stem,
                "baseModel": version_info.get("baseModel") or "",
                "trainedWords": list(version_info.get("trainedWords") or []),
                "createdAt": _safe_int(version_info.get("createdAt")),
                "updatedAt": _safe_int(version_info.get("updatedAt")),
                "downloadCount": _safe_int(version_stats.get("downloadCount")),
                "ratingCount": _safe_int(version_stats.get("ratingCount")),
                "rating": float(version_stats.get("rating") or 0.0),
                "images": [
                    {
                        **preview,
                        "url": (
                            preview.get("url")
                            if str(preview.get("type") or "").strip().lower() == "video" and preview.get("url")
                            else (
                                _build_preview_url(root, preview.get("relative_path"), preview.get("absolute_path"))
                                if preview.get("relative_path") or preview.get("absolute_path")
                                else preview.get("url")
                            )
                        ),
                        "meta": preview.get("meta") if isinstance(preview.get("meta"), dict) else None,
                    }
                    for preview in previews
                ],
                "files": [
                    {
                        "id": str(file_info.get("id") or hashlib.sha1(local_path.encode("utf-8")).hexdigest()),
                        "name": file_info.get("name") or path.name,
                        "format": _normalize_format(file_info.get("format"), path.name),
                        "sizeBytes": int((path.stat().st_size if path.exists() else 0)),
                        "hashes": file_info.get("hashes") or {},
                        "primary": True,
                        "downloaded": True,
                        "managed": managed,
                        "localPath": local_path,
                        "relativePath": relative_path,
                        "hasClip": capabilities.get("has_clip", _COMPONENT_UNKNOWN),
                        "hasVae": capabilities.get("has_vae", _COMPONENT_UNKNOWN),
                        "modifiedAt": modified_at,
                    }
                ],
            }
        ],
    }


def _group_local_files(root: Path, checkpoint_dir: Path, query: str = "") -> dict[str, Any]:
    files_by_group: dict[str, dict[str, Any]] = {}
    scanned_paths: set[str] = set()

    search_roots: list[tuple[Path, bool, set[str] | None, str]] = [(checkpoint_dir, True, None, "ess")]
    for path_str in folder_paths.get_folder_paths("checkpoints"):
        search_roots.append((Path(path_str).resolve(), False, None, "checkpoints"))
    for path_str in folder_paths.get_folder_paths("unet"):
        search_roots.append((Path(path_str).resolve(), False, {".gguf"}, "unet"))

    for search_root, managed, allowed_suffixes, source_kind in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not _file_is_supported(path):
                continue
            if allowed_suffixes is not None and path.suffix.lower() not in allowed_suffixes:
                continue
            resolved_key = str(path.resolve())
            if resolved_key in scanned_paths:
                continue
            scanned_paths.add(resolved_key)

            sidecar = _safe_json_read(_sidecar_path_for_model(path))
            if sidecar:
                group = _transform_sidecar_to_group(path, root, sidecar, managed)
                group = _attach_local_browser_metadata(group, path, search_root, source_kind=source_kind)
                existing_group = files_by_group.get(group["id"])
                if existing_group is None:
                    files_by_group[group["id"]] = group
                else:
                    existing_versions = existing_group.setdefault("versions", [])
                    for incoming_version in group.get("versions") or []:
                        match = next(
                            (
                                version
                                for version in existing_versions
                                if str(version.get("id")) == str(incoming_version.get("id"))
                            ),
                            None,
                        )
                        if match is None:
                            existing_versions.append(incoming_version)
                            continue
                        existing_files = match.setdefault("files", [])
                        for incoming_file in incoming_version.get("files") or []:
                            if not any(str(file.get("id")) == str(incoming_file.get("id")) for file in existing_files):
                                existing_files.append(incoming_file)
                        if not match.get("images") and incoming_version.get("images"):
                            match["images"] = incoming_version.get("images")
                continue

            group_id = f"local-file-{hashlib.sha1(resolved_key.encode('utf-8')).hexdigest()}"
            format_name = _normalize_format(filename=path.name)
            relative_path = _relative_to_root(path, root)
            item = {
                "id": group_id,
                "source": "local",
                "name": path.stem,
                "creator": "",
                "description": "",
                "tags": [],
                "modifiedAt": int(path.stat().st_mtime),
                "createdAt": 0,
                "updatedAt": int(path.stat().st_mtime),
                "downloadCount": 0,
                "ratingCount": 0,
                "rating": 0.0,
                "nsfw": None,
                "versions": [
                    {
                        "id": group_id + "-v1",
                        "name": "Local file",
                        "baseModel": "",
                        "trainedWords": [],
                        "createdAt": 0,
                        "updatedAt": int(path.stat().st_mtime),
                        "downloadCount": 0,
                        "ratingCount": 0,
                        "rating": 0.0,
                        "images": [],
                        "files": [
                            {
                                "id": group_id + "-f1",
                                "name": path.name,
                                "format": format_name,
                                "sizeBytes": int(path.stat().st_size),
                                "hashes": {},
                                "primary": True,
                                "downloaded": True,
                                "managed": managed,
                                "localPath": resolved_key,
                                "relativePath": relative_path,
                                "hasClip": False if format_name == "GGUF" else _COMPONENT_UNKNOWN,
                                "hasVae": False if format_name == "GGUF" else _COMPONENT_UNKNOWN,
                                "modifiedAt": int(path.stat().st_mtime),
                            }
                        ],
                    }
                ],
            }
            files_by_group[group_id] = _attach_local_browser_metadata(item, path, search_root, source_kind=source_kind)

    items = list(files_by_group.values())
    query_terms = [term for term in query.lower().split() if term]
    if query_terms:
        filtered: list[dict[str, Any]] = []
        for item in items:
            haystack = " ".join(
                [
                    str(item.get("name") or ""),
                    str(item.get("description") or ""),
                    str(item.get("browserRelativePath") or ""),
                    str(item.get("browserFolder") or ""),
                    " ".join(item.get("baseModels") or []),
                    " ".join(item.get("tags") or []),
                    " ".join(
                        str(version.get("name") or "") + " " + str(version.get("baseModel") or "") + " " + " ".join(version.get("trainedWords") or [])
                        for version in item.get("versions") or []
                    ),
                    " ".join(
                        str(file.get("name") or "")
                        for version in item.get("versions") or []
                        for file in version.get("files") or []
                    ),
                ]
            ).lower()
            if all(term in haystack for term in query_terms):
                filtered.append(item)
        items = filtered

    items.sort(key=lambda item: str(item.get("name") or "").lower())
    return {
        "items": items,
        "availableBaseModels": _collect_available_base_models(items),
    }


def _transform_civitai_model(model: dict[str, Any]) -> dict[str, Any]:
    versions = []
    item_name = model.get("name") or "Untitled model"
    item_description = _strip_html(model.get("description") or "")
    item_tags = list(model.get("tags") or [])
    model_stats = model.get("stats") or {}
    item_created_at = _safe_int(model.get("createdAt"))
    item_updated_at = _safe_int(model.get("updatedAt"))
    for version in model.get("modelVersions") or []:
        files = []
        version_stats = version.get("stats") or {}
        for file in version.get("files") or []:
            format_name = _normalize_format((file.get("metadata") or {}).get("format"), file.get("name"))
            if format_name == "Unknown":
                continue
            hashes = file.get("hashes") or {}
            files.append(
                {
                    "id": str(file.get("id") or file.get("name") or uuid.uuid4().hex),
                    "name": file.get("name") or "",
                    "format": format_name,
                    "sizeBytes": int(float(file.get("sizeKB") or 0) * 1024),
                    "downloadUrl": file.get("downloadUrl") or version.get("downloadUrl") or "",
                    "hashes": hashes,
                    "primary": bool(file.get("primary")),
                    "downloaded": False,
                    "managed": True,
                    "hasClip": False if format_name == "GGUF" else _COMPONENT_UNKNOWN,
                    "hasVae": False if format_name == "GGUF" else _COMPONENT_UNKNOWN,
                    "modifiedAt": _safe_int(version.get("updatedAt") or model.get("updatedAt")),
                }
            )

        if not files:
            continue
        files.sort(key=lambda entry: (not bool(entry.get("primary")), str(entry.get("name") or "").lower()))
        base_model = str(version.get("baseModel") or "").strip()
        if not base_model:
            base_model = _infer_base_model(
                item_name,
                version.get("name"),
                item_description,
                " ".join(item_tags),
                " ".join(version.get("trainedWords") or []),
                " ".join(file.get("name") or "" for file in files),
            )
        versions.append(
            {
                "id": str(version.get("id") or uuid.uuid4().hex),
                "name": version.get("name") or "Version",
                "baseModel": base_model,
                "trainedWords": list(version.get("trainedWords") or []),
                "createdAt": _safe_int(version.get("createdAt")),
                "updatedAt": _safe_int(version.get("updatedAt")),
                "downloadCount": _safe_int(version_stats.get("downloadCount")),
                "ratingCount": _safe_int(version_stats.get("ratingCount")),
                "rating": float(version_stats.get("rating") or 0.0),
                "images": [
                    _normalize_preview_entry(image)
                    for image in (version.get("images") or [])
                    if image.get("url")
                ],
                "files": files,
            }
        )

    return {
        "id": f"civitai-{model.get('id')}",
        "source": "civitai",
        "remoteModelId": model.get("id"),
        "name": item_name,
        "creator": (model.get("creator") or {}).get("username") or "",
        "description": item_description,
        "tags": item_tags,
        "createdAt": item_created_at,
        "updatedAt": item_updated_at,
        "downloadCount": _safe_int(model_stats.get("downloadCount")),
        "ratingCount": _safe_int(model_stats.get("ratingCount")),
        "rating": float(model_stats.get("rating") or 0.0),
        "nsfw": model.get("nsfw"),
        "versions": versions,
        "baseModels": sorted({str(version.get("baseModel") or "").strip() for version in versions if str(version.get("baseModel") or "").strip()}, key=str.lower),
    }


def _fetch_civitai_catalog(
    query: str,
    cursor: str | None,
    api_key: str | None,
    limit: int,
    *,
    username: str = "",
    tag: str = "",
    base_model: str = "",
    sort: str = "",
    period: str = "",
    allow_nsfw: str = "",
) -> dict[str, Any]:
    params = {
        "types": "Checkpoint",
        "limit": max(1, min(limit, 50)),
    }
    if query.strip():
        params["query"] = query.strip()
    if cursor:
        params["cursor"] = cursor
    if username.strip():
        params["username"] = username.strip()
    if tag.strip():
        params["tag"] = tag.strip()
    if base_model.strip():
        params["baseModels"] = base_model.strip()
    if sort.strip():
        params["sort"] = sort.strip()
    if period.strip():
        params["period"] = period.strip()
    allow_nsfw_text = str(allow_nsfw or "").strip().lower()
    if allow_nsfw_text in {"true", "false"}:
        params["nsfw"] = allow_nsfw_text
    payload = _http_json(
        f"{_CIVITAI_MODELS_ENDPOINT}?{urlencode(params)}",
        headers=_http_headers(api_key),
        timeout=30,
    )
    items = [_transform_civitai_model(model) for model in payload.get("items") or []]
    items = [item for item in items if item.get("versions")]
    metadata = payload.get("metadata") or {}
    return {
        "items": items,
        "nextCursor": metadata.get("nextCursor"),
        "availableBaseModels": _collect_available_base_models(items),
    }


def _fetch_civitai_tags(query: str, api_key: str | None, limit: int = 20) -> dict[str, Any]:
    params = {
        "limit": max(1, min(limit, 100)),
    }
    if query.strip():
        params["query"] = query.strip()
    payload = _http_json(
        f"https://civitai.com/api/v1/tags?{urlencode(params)}",
        headers=_http_headers(api_key),
        timeout=30,
    )
    items = []
    for item in payload.get("items") or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        items.append(
            {
                "name": name,
                "modelCount": _safe_int(item.get("modelCount")),
                "link": item.get("link") or "",
            }
        )
    return {
        "items": items,
    }


def _fetch_civitai_version_images(model_version_id: Any, api_key: str | None, limit: int = 100) -> dict[str, Any]:
    model_version_id_text = str(model_version_id or "").strip()
    if not model_version_id_text:
        raise ValueError("Model version id is required.")
    params = {
        "modelVersionId": model_version_id_text,
        "limit": max(1, min(limit, 200)),
    }
    payload = _http_json(
        f"https://civitai.com/api/v1/images?{urlencode(params)}",
        headers=_http_headers(api_key),
        timeout=30,
    )
    items = [_normalize_preview_entry(item) for item in payload.get("items") or [] if item.get("url")]
    metadata = payload.get("metadata") or {}
    return {
        "items": items,
        "nextCursor": metadata.get("nextCursor"),
    }


def _fetch_civitai_model_detail(model_id: Any, api_key: str | None) -> dict[str, Any]:
    model_id_text = str(model_id or "").strip()
    if not model_id_text:
        raise ValueError("CivitAI model id is missing.")
    return _http_json(
        f"{_CIVITAI_MODELS_ENDPOINT}/{model_id_text}",
        headers=_http_headers(api_key),
        timeout=30,
    )


def _fetch_civitai_model_version_by_hash(hash_value: str, api_key: str | None) -> dict[str, Any]:
    normalized_hash = str(hash_value or "").strip()
    if not normalized_hash:
        raise ValueError("Hash value is required for CivitAI lookup.")
    return _http_json(
        f"{_CIVITAI_MODEL_VERSIONS_ENDPOINT}/by-hash/{normalized_hash}",
        headers=_http_headers(api_key),
        timeout=30,
    )


def _extract_lookup_hash(path: Path, sidecar: dict[str, Any] | None) -> tuple[str, str]:
    sidecar = sidecar or {}
    file_hashes = ((sidecar.get("file") or {}).get("hashes") or {}) if isinstance(sidecar, dict) else {}
    for key in ("SHA256", "sha256", "AutoV2", "autoV2", "BLAKE3", "blake3", "CRC32", "crc32"):
        value = str(file_hashes.get(key) or "").strip()
        if value:
            return key.upper(), value
    return "SHA256", _compute_sha256(path)


def _resolve_local_model_path(raw_path: Any, root: Path, checkpoint_dir: Path) -> Path:
    candidate = Path(str(raw_path or "")).expanduser()
    if not candidate.exists() or not candidate.is_file():
        raise ValueError("Local model file not found.")
    resolved = candidate.resolve()
    allowed_roots = [root.resolve(), checkpoint_dir.resolve()]
    for folder_name in ("checkpoints", "unet"):
        for path_str in folder_paths.get_folder_paths(folder_name):
            allowed_roots.append(Path(path_str).resolve())
    if not any(allowed_root in {resolved, *resolved.parents} for allowed_root in allowed_roots):
        raise ValueError("Selected file is outside known checkpoint folders.")
    if not _file_is_supported(resolved):
        raise ValueError("Selected file is not a supported checkpoint.")
    return resolved


def _classify_local_source(path: Path, root: Path, checkpoint_dir: Path) -> tuple[Path, str, bool]:
    resolved = path.resolve()
    candidates: list[tuple[Path, str, bool]] = [(checkpoint_dir.resolve(), "ess", True)]
    for path_str in folder_paths.get_folder_paths("checkpoints"):
        candidates.append((Path(path_str).resolve(), "checkpoints", False))
    for path_str in folder_paths.get_folder_paths("unet"):
        candidates.append((Path(path_str).resolve(), "unet", False))

    matches: list[tuple[int, Path, str, bool]] = []
    for candidate_root, source_kind, managed in candidates:
        if candidate_root in {resolved, *resolved.parents}:
            matches.append((len(candidate_root.parts), candidate_root, source_kind, managed))
    if matches:
        _, match_root, source_kind, managed = max(matches, key=lambda entry: entry[0])
        return match_root, source_kind, managed
    return root.resolve(), "checkpoints", False


def _merge_enriched_sidecar(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    existing = existing or {}
    merged = json.loads(json.dumps(existing)) if existing else {}

    for key in ("schema", "source", "saved_at", "local", "capabilities"):
        if key in incoming:
            merged[key] = incoming[key]

    existing_model = merged.get("model") if isinstance(merged.get("model"), dict) else {}
    incoming_model = incoming.get("model") if isinstance(incoming.get("model"), dict) else {}
    merged["model"] = {
        **existing_model,
        **incoming_model,
        "name": existing_model.get("name") or incoming_model.get("name") or "",
        "creator": existing_model.get("creator") or incoming_model.get("creator") or {},
        "description_text": existing_model.get("description_text") or incoming_model.get("description_text") or "",
        "tags": list(existing_model.get("tags") or incoming_model.get("tags") or []),
    }

    existing_version = merged.get("version") if isinstance(merged.get("version"), dict) else {}
    incoming_version = incoming.get("version") if isinstance(incoming.get("version"), dict) else {}
    merged["version"] = {
        **existing_version,
        **incoming_version,
        "name": existing_version.get("name") or incoming_version.get("name") or "",
        "baseModel": existing_version.get("baseModel") or incoming_version.get("baseModel") or "",
        "trainedWords": list(existing_version.get("trainedWords") or incoming_version.get("trainedWords") or []),
    }

    existing_file = merged.get("file") if isinstance(merged.get("file"), dict) else {}
    incoming_file = incoming.get("file") if isinstance(incoming.get("file"), dict) else {}
    merged["file"] = {
        **existing_file,
        **incoming_file,
        "name": existing_file.get("name") or incoming_file.get("name") or "",
        "format": existing_file.get("format") or incoming_file.get("format") or "",
        "hashes": {
            **(incoming_file.get("hashes") or {}),
            **(existing_file.get("hashes") or {}),
        },
    }

    existing_previews = list(merged.get("previews") or [])
    incoming_previews = list(incoming.get("previews") or [])
    if not existing_previews:
        merged["previews"] = incoming_previews
    elif incoming_previews and not any(isinstance(preview.get("meta"), dict) and preview.get("meta") for preview in existing_previews):
        merged["previews"] = incoming_previews
    else:
        merged["previews"] = existing_previews
    return merged


def _download_target_paths(root: Path, checkpoint_dir: Path, model: dict[str, Any], version: dict[str, Any], file: dict[str, Any]) -> tuple[Path, str]:
    family_name = _sanitize_segment(
        version.get("baseModel") or _infer_base_model(model.get("name"), version.get("name"), " ".join(model.get("tags") or [])),
        fallback="Unknown Family",
    )
    author_name = _sanitize_segment(model.get("creator"), fallback="Unknown Author")
    model_dir_name = _sanitize_segment(model.get("name"), fallback="model")
    version_dir_name = _sanitize_segment(version.get("name"), fallback="version")
    file_name = _sanitize_segment(file.get("name"), fallback="model")
    destination = (checkpoint_dir / family_name / author_name / model_dir_name / version_dir_name / file_name).resolve()
    if root not in {destination, *destination.parents}:
        raise ValueError("Invalid destination path.")
    relative = destination.relative_to(root).as_posix()
    return destination, relative


def _build_sidecar_payload(
    root: Path,
    destination: Path,
    *,
    model: dict[str, Any],
    version: dict[str, Any],
    file: dict[str, Any],
    format_name: str,
    preview_entries: list[dict[str, Any]],
    sha256_value: str | None,
    has_clip: bool | None,
    has_vae: bool | None,
) -> dict[str, Any]:
    relative_path = _relative_to_root(destination, root)
    return {
        "schema": 1,
        "source": "civitai",
        "saved_at": int(time.time()),
        "model": {
            "id": model.get("remoteModelId") or model.get("id"),
            "name": model.get("name") or destination.stem,
            "creator": {"username": model.get("creator") or ""},
            "description_text": model.get("description") or "",
            "tags": list(model.get("tags") or []),
            "createdAt": _safe_int(model.get("createdAt")),
            "updatedAt": _safe_int(model.get("updatedAt")),
            "stats": {
                "downloadCount": _safe_int(model.get("downloadCount")),
                "ratingCount": _safe_int(model.get("ratingCount")),
                "rating": float(model.get("rating") or 0.0),
            },
            "nsfw": model.get("nsfw"),
        },
        "version": {
            "id": version.get("id"),
            "name": version.get("name") or "",
            "baseModel": version.get("baseModel") or "",
            "trainedWords": list(version.get("trainedWords") or []),
            "createdAt": _safe_int(version.get("createdAt")),
            "updatedAt": _safe_int(version.get("updatedAt")),
            "stats": {
                "downloadCount": _safe_int(version.get("downloadCount")),
                "ratingCount": _safe_int(version.get("ratingCount")),
                "rating": float(version.get("rating") or 0.0),
            },
        },
        "file": {
            "id": file.get("id"),
            "name": destination.name,
            "format": format_name,
            "hashes": {
                **(file.get("hashes") or {}),
                **({"SHA256": sha256_value} if sha256_value else {}),
            },
            "sizeBytes": int(destination.stat().st_size),
        },
        "local": {
            "path": str(destination.resolve()),
            "relative_path": relative_path,
            "format": format_name,
            "sizeBytes": int(destination.stat().st_size),
        },
        "capabilities": {
            "has_clip": has_clip,
            "has_vae": has_vae,
        },
        "previews": preview_entries,
    }


def _update_download_job(job_id: str, **changes: Any) -> None:
    with _DOWNLOAD_JOBS_LOCK:
        job = dict(_DOWNLOAD_JOBS.get(job_id) or {})
        if "started_at" not in job:
            job["started_at"] = time.time()
        job.update(changes)
        bytes_downloaded = job.get("bytes_downloaded")
        total_bytes = job.get("total_bytes")
        started_at = job.get("started_at") or time.time()
        elapsed = max(time.time() - started_at, 0.001)
        if isinstance(bytes_downloaded, (int, float)):
            job["speedBytesPerSecond"] = int(bytes_downloaded / elapsed)
        if isinstance(bytes_downloaded, (int, float)) and isinstance(total_bytes, (int, float)) and total_bytes > 0:
            job["progress"] = max(0.0, min(1.0, float(bytes_downloaded) / float(total_bytes)))
        _DOWNLOAD_JOBS[job_id] = job


def _get_download_job(job_id: str) -> dict[str, Any]:
    with _DOWNLOAD_JOBS_LOCK:
        return dict(_DOWNLOAD_JOBS.get(job_id) or {})


def _load_gguf_nodes_module():
    global _GGUF_MODULE_CACHE
    if _GGUF_MODULE_CACHE is not None:
        return _GGUF_MODULE_CACHE

    with _GGUF_MODULE_LOCK:
        if _GGUF_MODULE_CACHE is not None:
            return _GGUF_MODULE_CACHE

        current = Path(__file__).resolve()
        candidate_dirs: list[Path] = []
        for parent in current.parents:
            candidate_dirs.append(parent / "ComfyUI-GGUF")
            candidate_dirs.append(parent / "custom_nodes" / "ComfyUI-GGUF")

        for package_dir in candidate_dirs:
            init_file = package_dir / "__init__.py"
            nodes_file = package_dir / "nodes.py"
            if not init_file.exists() or not nodes_file.exists():
                continue
            package_name = "comfyui_ess_external_gguf"
            if package_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    package_name,
                    init_file,
                    submodule_search_locations=[str(package_dir)],
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = module
                spec.loader.exec_module(module)
            nodes_module = importlib.import_module(f"{package_name}.nodes")
            _GGUF_MODULE_CACHE = nodes_module
            return nodes_module

    raise RuntimeError("ComfyUI-GGUF is not installed, so GGUF checkpoints cannot be loaded.")


def _load_checkpoint_from_selection(path: Path, format_name: str):
    if not path.exists() or not path.is_file():
        raise ValueError(f"Checkpoint file not found: {path}")

    if format_name == "GGUF" or path.suffix.lower() == ".gguf":
        gguf_nodes = _load_gguf_nodes_module()
        ops = gguf_nodes.GGMLOps()
        sd, extra = gguf_nodes.gguf_sd_loader(str(path))
        kwargs = {}
        valid_params = inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters
        if "metadata" in valid_params:
            kwargs["metadata"] = extra.get("metadata", {})
        model = comfy.sd.load_diffusion_model_state_dict(
            sd,
            model_options={"custom_operations": ops},
            **kwargs,
        )
        if model is None:
            raise RuntimeError(f"Could not detect GGUF model type for: {path}")
        model = gguf_nodes.GGUFModelPatcher.clone(model)
        return (model, None, None)

    out = comfy.sd.load_checkpoint_guess_config(
        str(path),
        output_vae=True,
        output_clip=True,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    return out[:3]


def _parse_selection_payload(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        data = payload
    else:
        text = str(payload or "").strip()
        if not text:
            raise ValueError("No model selection saved. Open the browser and choose a checkpoint.")
        try:
            data = json.loads(text)
        except Exception as exc:
            raise ValueError(f"Invalid checkpoint selection payload: {exc}") from exc

    local_path = str(data.get("local_path") or data.get("localPath") or "").strip()
    if not local_path:
        raise ValueError("Selected entry does not have a local file path yet. Download or choose a local checkpoint first.")
    format_name = _normalize_format(data.get("format"), local_path)
    return {
        "local_path": local_path,
        "format": format_name,
    }


class ESSCheckpointBrowserLoader:
    CATEGORY = "ESS/Loaders"
    FUNCTION = "load_checkpoint"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    DESCRIPTION = "Browse local checkpoints or CivitAI models, download them into the ESS library, and load MODEL/CLIP/VAE."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint": (
                    "ESS_CHECKPOINT_BROWSER",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Open browser and select a checkpoint...",
                        "height": 140,
                    },
                )
            }
        }

    def load_checkpoint(self, checkpoint: str):
        selection = _parse_selection_payload(checkpoint)
        return _load_checkpoint_from_selection(Path(selection["local_path"]), selection["format"])


def _run_download_job(job_id: str, payload: dict[str, Any]) -> None:
    try:
        root, checkpoint_dir, _ = _resolve_checkpoint_dir(
            payload.get("ess_root"),
            payload.get("checkpoints_subdir"),
            payload.get("checkpoint_directory"),
        )
        model = payload.get("model") or {}
        version = payload.get("version") or {}
        file = payload.get("file") or {}
        format_name = _normalize_format(file.get("format"), file.get("name"))
        if format_name == "Unknown":
            raise ValueError("Unsupported checkpoint format for download.")

        destination, _relative = _download_target_paths(root, checkpoint_dir, model, version, file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(file.get("sizeBytes") or 0) or None
        expected_sha256 = ((file.get("hashes") or {}).get("SHA256") or "").strip() or None

        if destination.exists():
            valid, actual_sha = _validate_existing_file(destination, expected_size, expected_sha256)
            if not valid:
                destination.unlink(missing_ok=True)
            else:
                preview_entries = []
                sidecar_existing = _safe_json_read(_sidecar_path_for_model(destination))
                if sidecar_existing:
                    preview_entries = sidecar_existing.get("previews") or []
                else:
                    has_clip, has_vae = _guess_component_presence_for_file(destination)
                    preview_entries = _download_preview_images(
                        destination,
                        root,
                        version.get("images") or [],
                        headers=_http_headers(payload.get("api_key")),
                        job_id=job_id,
                    )
                    sidecar_payload = _build_sidecar_payload(
                        root,
                        destination,
                        model=model,
                        version=version,
                        file=file,
                        format_name=format_name,
                        preview_entries=preview_entries,
                        sha256_value=actual_sha,
                        has_clip=has_clip,
                        has_vae=has_vae,
                    )
                    _safe_json_write(_sidecar_path_for_model(destination), sidecar_payload)

                selection = _selection_payload_from_local_file(
                    destination,
                    root=root,
                    managed=True,
                )
                _update_download_job(
                    job_id,
                    status="completed",
                    stage="done",
                    message="Model already exists locally.",
                    completed_at=time.time(),
                    bytes_downloaded=expected_size or destination.stat().st_size,
                    total_bytes=expected_size or destination.stat().st_size,
                    result=selection,
                )
                return

        if not str(file.get("downloadUrl") or "").strip():
            raise ValueError("Selected CivitAI file does not include a download URL.")

        _update_download_job(job_id, status="running", stage="starting", message="Preparing download...")
        sha256_value = _download_with_resume(
            str(file.get("downloadUrl")),
            destination,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
            headers=_http_headers(payload.get("api_key")),
            job_id=job_id,
        )
        has_clip, has_vae = _guess_component_presence_for_file(destination)
        preview_entries = _download_preview_images(
            destination,
            root,
            version.get("images") or [],
            headers=_http_headers(payload.get("api_key")),
            job_id=job_id,
        )
        sidecar_payload = _build_sidecar_payload(
            root,
            destination,
            model=model,
            version=version,
            file=file,
            format_name=format_name,
            preview_entries=preview_entries,
            sha256_value=sha256_value,
            has_clip=has_clip,
            has_vae=has_vae,
        )
        _safe_json_write(_sidecar_path_for_model(destination), sidecar_payload)
        selection = _selection_payload_from_local_file(destination, root=root, sidecar=sidecar_payload, managed=True)
        _update_download_job(
            job_id,
            status="completed",
            stage="done",
            message="Download completed.",
            completed_at=time.time(),
            bytes_downloaded=expected_size or destination.stat().st_size,
            total_bytes=expected_size or destination.stat().st_size,
            result=selection,
        )
    except Exception as exc:
        _update_download_job(
            job_id,
            status="failed",
            stage="error",
            message=str(exc),
            error=str(exc),
            completed_at=time.time(),
        )


def _enrich_local_model_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    root, checkpoint_dir, _ = _resolve_checkpoint_dir(
        payload.get("ess_root"),
        payload.get("checkpoints_subdir"),
        payload.get("checkpoint_directory"),
    )
    model_path = _resolve_local_model_path(payload.get("local_path"), root, checkpoint_dir)
    search_root, source_kind, managed = _classify_local_source(model_path, root, checkpoint_dir)
    sidecar_path = _sidecar_path_for_model(model_path)
    existing_sidecar = _safe_json_read(sidecar_path) or {}
    _, lookup_hash = _extract_lookup_hash(model_path, existing_sidecar)
    version_payload = _fetch_civitai_model_version_by_hash(lookup_hash, payload.get("api_key"))

    model_id = version_payload.get("modelId") or ((version_payload.get("model") or {}).get("id"))
    if not model_id:
        raise ValueError("CivitAI did not return a model id for this checkpoint.")
    model_payload = _fetch_civitai_model_detail(model_id, payload.get("api_key"))

    transformed_model = _transform_civitai_model(model_payload)
    if not transformed_model.get("versions"):
        raise ValueError("CivitAI model does not expose any compatible checkpoint files.")

    version_id = str(version_payload.get("id") or "")
    transformed_version = next(
        (entry for entry in transformed_model.get("versions") or [] if str(entry.get("id")) == version_id),
        None,
    )
    if transformed_version is None:
        transformed_version = transformed_model.get("versions")[0]

    transformed_file = None
    version_hashes = (version_payload.get("files") or [])
    candidate_hashes = {lookup_hash.lower()}
    if isinstance(existing_sidecar, dict):
        for value in (((existing_sidecar.get("file") or {}).get("hashes") or {}).values()):
            text = str(value or "").strip().lower()
            if text:
                candidate_hashes.add(text)

    for file_entry in transformed_version.get("files") or []:
        file_hashes = {str(value or "").strip().lower() for value in (file_entry.get("hashes") or {}).values() if str(value or "").strip()}
        if candidate_hashes & file_hashes:
            transformed_file = file_entry
            break

    if transformed_file is None:
        for api_file in version_hashes:
            api_hashes = {str(value or "").strip().lower() for value in (api_file.get("hashes") or {}).values() if str(value or "").strip()}
            if candidate_hashes & api_hashes:
                transformed_file = {
                    "id": str(api_file.get("id") or model_path.name),
                    "name": model_path.name,
                    "format": _normalize_format((api_file.get("metadata") or {}).get("format"), model_path.name),
                    "sizeBytes": int(model_path.stat().st_size),
                    "hashes": api_file.get("hashes") or {},
                    "primary": bool(api_file.get("primary")),
                    "downloaded": True,
                    "managed": managed,
                    "localPath": str(model_path.resolve()),
                    "relativePath": _relative_to_root(model_path, root),
                    "hasClip": _COMPONENT_UNKNOWN,
                    "hasVae": _COMPONENT_UNKNOWN,
                }
                break

    format_name = _normalize_format(
        (transformed_file or {}).get("format"),
        model_path.name,
    )
    has_clip, has_vae = _guess_component_presence_for_file(model_path)
    preview_entries = list(existing_sidecar.get("previews") or [])
    if not preview_entries:
        preview_entries = _download_preview_images(
            model_path,
            root,
            transformed_version.get("images") or [],
            headers=_http_headers(payload.get("api_key")),
            job_id=f"enrich-{uuid.uuid4().hex}",
        )

    sidecar_payload = _build_sidecar_payload(
        root,
        model_path,
        model=transformed_model,
        version=transformed_version,
        file=transformed_file or {
            "id": model_path.name,
            "name": model_path.name,
            "format": format_name,
            "hashes": {"SHA256": lookup_hash},
        },
        format_name=format_name,
        preview_entries=preview_entries,
        sha256_value=lookup_hash if len(lookup_hash) == 64 else None,
        has_clip=has_clip,
        has_vae=has_vae,
    )
    merged_sidecar = _merge_enriched_sidecar(existing_sidecar, sidecar_payload)
    _safe_json_write(sidecar_path, merged_sidecar)

    selection = _selection_payload_from_local_file(
        model_path,
        root=root,
        sidecar=merged_sidecar,
        managed=managed,
    )
    return {
        "selection": selection,
        "item": _attach_local_browser_metadata(
            _transform_sidecar_to_group(model_path, root, merged_sidecar, managed),
            model_path,
            search_root,
            source_kind=source_kind,
        ),
    }


def register_checkpoint_routes(prompt_server, web_module):
    if prompt_server is None or web_module is None:
        return

    @prompt_server.instance.routes.get("/ess/checkpoints/local")
    async def ess_checkpoints_local(request):
        root_raw = request.rel_url.query.get("root", "")
        subdir_raw = request.rel_url.query.get("subdir", "checkpoints")
        query = request.rel_url.query.get("query", "")
        try:
            checkpoint_directory = request.rel_url.query.get("checkpoint_directory", "")
            root, checkpoint_dir, normalized = _resolve_checkpoint_dir(root_raw, subdir_raw, checkpoint_directory)
            payload = await asyncio.to_thread(_group_local_files, root, checkpoint_dir, query)
            return web_module.json_response(
                {
                    "ok": True,
                    **payload,
                    "ess_root": str(root),
                "checkpoints_subdir": normalized,
                "checkpoint_directory": normalized,
            }
            )
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    @prompt_server.instance.routes.get("/ess/checkpoints/civitai/search")
    async def ess_checkpoints_civitai_search(request):
        query = request.rel_url.query.get("query", "")
        cursor = request.rel_url.query.get("cursor")
        api_key = request.rel_url.query.get("api_key", "")
        username = request.rel_url.query.get("username", "")
        tag = request.rel_url.query.get("tag", "")
        base_model = request.rel_url.query.get("base_model", "")
        sort = request.rel_url.query.get("sort", "")
        period = request.rel_url.query.get("period", "")
        allow_nsfw = request.rel_url.query.get("nsfw", "")
        try:
            limit = int(request.rel_url.query.get("limit", _DEFAULT_CIVITAI_LIMIT))
        except Exception:
            limit = _DEFAULT_CIVITAI_LIMIT
        try:
            payload = await asyncio.to_thread(
                _fetch_civitai_catalog,
                query,
                cursor,
                api_key,
                limit,
                username=username,
                tag=tag,
                base_model=base_model,
                sort=sort,
                period=period,
                allow_nsfw=allow_nsfw,
            )
            return web_module.json_response({"ok": True, **payload})
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    @prompt_server.instance.routes.get("/ess/checkpoints/civitai/tags")
    async def ess_checkpoints_civitai_tags(request):
        query = request.rel_url.query.get("query", "")
        api_key = request.rel_url.query.get("api_key", "")
        try:
            limit = int(request.rel_url.query.get("limit", 20))
        except Exception:
            limit = 20
        try:
            payload = await asyncio.to_thread(_fetch_civitai_tags, query, api_key, limit)
            return web_module.json_response({"ok": True, **payload})
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    @prompt_server.instance.routes.get("/ess/checkpoints/civitai/images")
    async def ess_checkpoints_civitai_images(request):
        model_version_id = request.rel_url.query.get("model_version_id", "")
        api_key = request.rel_url.query.get("api_key", "")
        try:
            limit = int(request.rel_url.query.get("limit", 100))
        except Exception:
            limit = 100
        try:
            payload = await asyncio.to_thread(_fetch_civitai_version_images, model_version_id, api_key, limit)
            return web_module.json_response({"ok": True, **payload})
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    @prompt_server.instance.routes.post("/ess/checkpoints/download")
    async def ess_checkpoints_download(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        job_id = str(payload.get("job_id") or uuid.uuid4().hex)
        _update_download_job(
            job_id,
            id=job_id,
            status="queued",
            stage="queued",
            message="Queued for download.",
            created_at=time.time(),
            progress=0.0,
        )
        thread = threading.Thread(target=_run_download_job, args=(job_id, payload), daemon=True)
        thread.start()
        return web_module.json_response({"ok": True, "job_id": job_id})

    @prompt_server.instance.routes.get("/ess/checkpoints/download_status")
    async def ess_checkpoints_download_status(request):
        job_id = str(request.rel_url.query.get("job_id", "") or "").strip()
        if not job_id:
            return web_module.json_response({"ok": False, "error": "job_id is required"}, status=400)
        job = _get_download_job(job_id)
        if not job:
            return web_module.json_response({"ok": False, "error": "job not found"}, status=404)
        return web_module.json_response({"ok": True, **job})

    async def _handle_enrich_local(request):
        payload = {}
        if request.method == "GET":
            payload = {
                "ess_root": request.rel_url.query.get("ess_root", ""),
                "checkpoints_subdir": request.rel_url.query.get("checkpoints_subdir", "checkpoints"),
                "checkpoint_directory": request.rel_url.query.get("checkpoint_directory", ""),
                "api_key": request.rel_url.query.get("api_key", ""),
                "local_path": request.rel_url.query.get("local_path", ""),
            }
        else:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        try:
            result = await asyncio.to_thread(_enrich_local_model_metadata, payload)
            return web_module.json_response({"ok": True, **result})
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    prompt_server.instance.routes.get("/ess/checkpoints/enrich_local")(_handle_enrich_local)
    prompt_server.instance.routes.post("/ess/checkpoints/enrich_local")(_handle_enrich_local)

    @prompt_server.instance.routes.get("/ess/checkpoints/preview")
    async def ess_checkpoints_preview(request):
        try:
            absolute = str(request.rel_url.query.get("absolute", "") or "").strip()
            if absolute:
                target = Path(absolute).expanduser().resolve()
                allowed_roots = [Path(folder_paths.models_dir).resolve()]
                for folder_name in ("checkpoints", "unet"):
                    for path_str in folder_paths.get_folder_paths(folder_name):
                        allowed_roots.append(Path(path_str).resolve())
                if not any(allowed_root in {target, *target.parents} for allowed_root in allowed_roots):
                    return web_module.json_response({"ok": False, "error": "invalid absolute preview path"}, status=400)
            else:
                root = _resolve_ess_root(request.rel_url.query.get("root", ""))
                relative = str(request.rel_url.query.get("path", "") or "").replace("\\", "/").strip().strip("/")
                if not relative:
                    return web_module.json_response({"ok": False, "error": "path is required"}, status=400)
                target = (root / relative).resolve()
                if root not in {target, *target.parents}:
                    return web_module.json_response({"ok": False, "error": "invalid path"}, status=400)
            if not target.exists() or not target.is_file():
                return web_module.json_response({"ok": False, "error": "preview not found"}, status=404)
            mime_type, _ = mimetypes.guess_type(str(target))
            return web_module.Response(
                body=target.read_bytes(),
                content_type=mime_type or "application/octet-stream",
            )
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)

    @prompt_server.instance.routes.get("/ess/checkpoints/remote_preview")
    async def ess_checkpoints_remote_preview(request):
        try:
            raw_url = str(request.rel_url.query.get("url", "") or "").strip()
            if not raw_url:
                return web_module.json_response({"ok": False, "error": "url is required"}, status=400)

            parsed = urlparse(raw_url)
            if parsed.scheme not in {"http", "https"}:
                return web_module.json_response({"ok": False, "error": "unsupported preview url"}, status=400)

            hostname = (parsed.hostname or "").lower()
            if not hostname or (hostname != "civitai.com" and not hostname.endswith(".civitai.com")):
                return web_module.json_response({"ok": False, "error": "unsupported preview host"}, status=400)

            request_headers = _http_headers(extra={"Accept": "image/*,video/*,*/*;q=0.5"})
            upstream = Request(raw_url, headers=request_headers, method="GET")
            with urlopen(upstream, timeout=30) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type") or mimetypes.guess_type(raw_url)[0] or "application/octet-stream"
                return web_module.Response(body=body, content_type=content_type)
        except HTTPError as exc:
            return web_module.json_response({"ok": False, "error": f"upstream preview http {exc.code}"}, status=exc.code)
        except URLError as exc:
            return web_module.json_response({"ok": False, "error": f"preview fetch failed: {exc.reason}"}, status=502)
        except Exception as exc:
            return web_module.json_response({"ok": False, "error": str(exc)}, status=500)
