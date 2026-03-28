from __future__ import annotations

import base64
import importlib
import os
import subprocess
import sys
import threading
import types
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


_ROOT_DIR = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT_DIR / "models"
_EXTERNAL_REPOS_DIR = _MODELS_DIR / "external_repos"

_WILDCAMERA_REPO = _EXTERNAL_REPOS_DIR / "WildCamera"
_MULTIHMR_REPO = _EXTERNAL_REPOS_DIR / "multi-hmr"

_WILDCAMERA_CKPT = _WILDCAMERA_REPO / "model_zoo" / "Release" / "wild_camera_all.pth"
_MULTIHMR_MODELS_DIR = _MULTIHMR_REPO / "models"
_MULTIHMR_CKPT = _MULTIHMR_MODELS_DIR / "multiHMR" / "multiHMR_896_L.pt"
_MULTIHMR_MEAN_PARAMS = _MULTIHMR_MODELS_DIR / "smpl_mean_params.npz"
_MULTIHMR_SMPLX = _MULTIHMR_MODELS_DIR / "smplx" / "SMPLX_NEUTRAL.npz"

_PROJECT_SMPLX = _MODELS_DIR / "smplx" / "SMPLX_NEUTRAL.npz"

_WILDCAMERA_REPO_URL = "https://github.com/ShngJZ/WildCamera"
_MULTIHMR_REPO_URL = "https://github.com/naver/multi-hmr"

_WILDCAMERA_URLS = [
    "https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/checkpoint/wild_camera_all.pth?download=true",
]
_MULTIHMR_URLS = [
    "https://huggingface.co/naver/multiHMR_896_L/resolve/main/multiHMR_896_L.pt?download=true",
    "https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multiHMR_896_L.pt",
]
_MULTIHMR_MEAN_PARAMS_URLS = [
    "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz?versionId=CAEQHhiBgICN6M3V6xciIDU1MzUzNjZjZGNiOTQ3OWJiZTJmNThiZmY4NmMxMTM4",
    "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmhuman3d/models/smpl_mean_params.npz",
]

_MIN_LARGE_MODEL_BYTES = 1024 * 1024
_MIN_SMALL_ASSET_BYTES = 256
_MODEL_LOCK = threading.Lock()
_WILDCAMERA_MODEL = None
_MULTIHMR_MODEL = None
_MULTIHMR_DEVICE = None
_FACE_ANALYZER = None

# First SMPL-X joints match the standard body skeleton ordering.
_SMPLX_BODY_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _report(progress_cb, message: str, **extra) -> None:
    if callable(progress_cb):
        try:
            progress_cb(str(message or ""), **extra)
        except Exception:
            pass


def _normalize_url(url: str) -> str:
    cleaned = str(url or "").strip().rstrip("\\'\"")
    if "huggingface.co" in cleaned and "/blob/" in cleaned:
        cleaned = cleaned.replace("/blob/", "/resolve/")
    if "huggingface.co" in cleaned and "/blame/" in cleaned:
        cleaned = cleaned.replace("/blame/", "/resolve/")
    return cleaned


def _download_file(urls: list[str], dest: Path, user_agent: str, progress_cb=None, min_bytes: int = _MIN_LARGE_MODEL_BYTES) -> Path:
    _ensure_dir(dest.parent)
    min_bytes = max(1, int(min_bytes or 1))
    if dest.exists() and dest.stat().st_size >= min_bytes:
        _report(progress_cb, f"Using cached file: {dest.name}")
        return dest
    if dest.exists():
        try:
            dest.unlink()
        except Exception:
            pass

    errors: list[str] = []
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    for raw_url in urls:
        url = _normalize_url(raw_url)
        try:
            _report(progress_cb, f"Downloading: {dest.name}")
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req) as response, open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(dest)
            if dest.exists() and dest.stat().st_size >= min_bytes:
                _report(progress_cb, f"Downloaded: {dest.name}")
                return dest
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
    raise RuntimeError(f"Failed to download {dest.name}. Errors: {'; '.join(errors)}")


def _ensure_git_repo(repo_url: str, dest: Path, progress_cb=None) -> Path:
    if (dest / ".git").exists():
        _report(progress_cb, f"Using cached repo: {dest.name}")
        return dest
    if dest.exists() and any(dest.iterdir()):
        _report(progress_cb, f"Using existing folder: {dest.name}")
        return dest
    _ensure_dir(dest.parent)
    _report(progress_cb, f"Cloning repo: {dest.name}")
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _report(progress_cb, f"Repo ready: {dest.name}")
    return dest


@contextmanager
def _sys_path(path: Path):
    path_s = str(path)
    sys.path.insert(0, path_s)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_s)
        except ValueError:
            pass


def _purge_modules(prefixes: tuple[str, ...]) -> None:
    for key in list(sys.modules.keys()):
        if key in prefixes or any(key.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(key, None)


def _install_mmcv_convmodule_shim() -> None:
    if "mmcv.cnn" in sys.modules:
        return
    try:
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError(f"torch.nn is required for WildCamera shim: {exc}") from exc

    class ConvModule(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias="auto",
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            inplace=True,
            **_kwargs,
        ):
            super().__init__()
            use_bias = bool(bias) if bias != "auto" else norm_cfg is None
            self.conv = nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=use_bias,
            )
            self.with_norm = False
            self.with_activation = False
            self.norm_name = None
            if isinstance(norm_cfg, dict):
                norm_type = str(norm_cfg.get("type", "") or "").upper()
                if norm_type == "BN":
                    self.bn = nn.BatchNorm2d(int(out_channels))
                    self.norm_name = "bn"
                    self.with_norm = True
                elif norm_type == "GN":
                    num_groups = int(norm_cfg.get("num_groups", 32) or 32)
                    num_groups = max(1, min(num_groups, int(out_channels)))
                    while int(out_channels) % num_groups != 0 and num_groups > 1:
                        num_groups -= 1
                    self.gn = nn.GroupNorm(num_groups, int(out_channels))
                    self.norm_name = "gn"
                    self.with_norm = True
            if act_cfg is not None:
                act_type = "RELU"
                if isinstance(act_cfg, dict):
                    act_type = str(act_cfg.get("type", "ReLU") or "ReLU").upper()
                if act_type == "RELU":
                    self.activate = nn.ReLU(inplace=bool(inplace))
                    self.with_activation = True
                elif act_type == "LEAKYRELU":
                    negative_slope = float(act_cfg.get("negative_slope", 0.01)) if isinstance(act_cfg, dict) else 0.01
                    self.activate = nn.LeakyReLU(negative_slope=negative_slope, inplace=bool(inplace))
                    self.with_activation = True
                elif act_type == "GELU":
                    self.activate = nn.GELU()
                    self.with_activation = True

        @property
        def norm(self):
            if self.norm_name:
                return getattr(self, self.norm_name)
            return None

        def forward(self, x):
            x = self.conv(x)
            if self.with_norm and self.norm is not None:
                x = self.norm(x)
            if self.with_activation and hasattr(self, "activate"):
                x = self.activate(x)
            return x

    mmcv_mod = types.ModuleType("mmcv")
    cnn_mod = types.ModuleType("mmcv.cnn")
    cnn_mod.ConvModule = ConvModule
    mmcv_mod.cnn = cnn_mod
    sys.modules["mmcv"] = mmcv_mod
    sys.modules["mmcv.cnn"] = cnn_mod


def _decode_image_from_b64(image_b64: str):
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"opencv/numpy are required: {exc}") from exc

    payload = str(image_b64 or "").strip()
    if "base64," in payload:
        payload = payload.split("base64,", 1)[1]
    if not payload:
        raise ValueError("empty image payload")
    raw = base64.b64decode(payload, validate=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image")
    return image


def _encode_preview_image_b64(image_bgr) -> str | None:
    try:
        import cv2
        ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok or encoded is None:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    except Exception:
        return None


def _draw_labeled_box(image, box, label: str, color) -> None:
    try:
        import cv2
    except Exception:
        return
    if not box or len(box) < 4:
        return
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = str(label or "").strip()
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(th + 6, y1 - 6)
    cv2.rectangle(image, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
    cv2.putText(image, text, (x1 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _build_preview_image(image_bgr, faces=None, persons=None):
    try:
        import cv2
    except Exception:
        return None
    preview = image_bgr.copy()
    for idx, face in enumerate(faces or []):
        gender = str(face.get("gender", "unknown"))
        age = face.get("age")
        age_text = f", {int(age)}" if age is not None else ""
        _draw_labeled_box(
            preview,
            face.get("bbox"),
            f"face {idx + 1}: {gender}{age_text}",
            (64, 196, 255),
        )
    for idx, person in enumerate(persons or []):
        bbox = person.get("bbox")
        if isinstance(bbox, dict):
            bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
        gender = str(person.get("gender", "unknown"))
        stage = str(person.get("life_stage", "adult"))
        _draw_labeled_box(
            preview,
            bbox,
            f"person {idx + 1}: {gender}, {stage}",
            (86, 255, 123),
        )
        for joint in (person.get("joints2d") or [])[:22]:
            try:
                x, y = int(round(float(joint[0]))), int(round(float(joint[1])))
                cv2.circle(preview, (x, y), 2, (86, 255, 123), -1, lineType=cv2.LINE_AA)
            except Exception:
                continue
    return _encode_preview_image_b64(preview)


def _bgr_to_pil(image_bgr):
    try:
        import cv2
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f"PIL/OpenCV are required: {exc}") from exc
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _get_device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_face_analyzer():
    global _FACE_ANALYZER
    if _FACE_ANALYZER is not None:
        return _FACE_ANALYZER

    try:
        import torch
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception:
        return None

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
    analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_thresh=0.30, det_size=(640, 640))
    _FACE_ANALYZER = analyzer
    return _FACE_ANALYZER


def _ensure_multihmr_assets(progress_cb=None) -> None:
    _report(progress_cb, "Preparing Multi-HMR assets")
    _ensure_git_repo(_MULTIHMR_REPO_URL, _MULTIHMR_REPO, progress_cb=progress_cb)
    _download_file(
        _MULTIHMR_URLS,
        _MULTIHMR_CKPT,
        "comfyui-ess/multi-hmr",
        progress_cb=progress_cb,
        min_bytes=_MIN_LARGE_MODEL_BYTES,
    )
    _download_file(
        _MULTIHMR_MEAN_PARAMS_URLS,
        _MULTIHMR_MEAN_PARAMS,
        "comfyui-ess/multi-hmr",
        progress_cb=progress_cb,
        min_bytes=_MIN_SMALL_ASSET_BYTES,
    )
    if not _MULTIHMR_SMPLX.exists() and _PROJECT_SMPLX.exists():
        _ensure_dir(_MULTIHMR_SMPLX.parent)
        _MULTIHMR_SMPLX.write_bytes(_PROJECT_SMPLX.read_bytes())
        _report(progress_cb, "Copied SMPL-X neutral model into Multi-HMR cache")
    if not _MULTIHMR_SMPLX.exists():
        raise RuntimeError(
            f"SMPL-X model is required at {_MULTIHMR_SMPLX}. "
            f"You can also place it at {_PROJECT_SMPLX} and retry."
        )
    _report(progress_cb, "Multi-HMR assets ready")


def _ensure_wildcamera_assets(progress_cb=None) -> None:
    _report(progress_cb, "Preparing WildCamera assets")
    _ensure_git_repo(_WILDCAMERA_REPO_URL, _WILDCAMERA_REPO, progress_cb=progress_cb)
    _download_file(
        _WILDCAMERA_URLS,
        _WILDCAMERA_CKPT,
        "comfyui-ess/wildcamera",
        progress_cb=progress_cb,
        min_bytes=_MIN_LARGE_MODEL_BYTES,
    )
    _report(progress_cb, "WildCamera assets ready")


def _load_wildcamera_model(progress_cb=None):
    global _WILDCAMERA_MODEL
    if _WILDCAMERA_MODEL is not None:
        _report(progress_cb, "WildCamera model already loaded")
        return _WILDCAMERA_MODEL

    import torch

    _ensure_wildcamera_assets(progress_cb=progress_cb)
    _purge_modules(("tools",))
    try:
        import mmcv.cnn  # type: ignore  # pragma: no cover - optional dependency
    except Exception:
        _report(progress_cb, "mmcv not found, using local ConvModule shim")
        _install_mmcv_convmodule_shim()
    _report(progress_cb, "Loading WildCamera model")
    with _sys_path(_WILDCAMERA_REPO):
        wild_module = importlib.import_module("WildCamera.newcrfs.newcrf_incidencefield")
        model = wild_module.NEWCRFIF(version="large07", pretrained=None)
        state = torch.load(_WILDCAMERA_CKPT, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to(_get_device())
        model.eval()
    _WILDCAMERA_MODEL = model
    _report(progress_cb, "WildCamera model loaded")
    return _WILDCAMERA_MODEL


def _load_multihmr_model(progress_cb=None):
    global _MULTIHMR_MODEL, _MULTIHMR_DEVICE
    if _MULTIHMR_MODEL is not None:
        _report(progress_cb, "Multi-HMR model already loaded")
        return _MULTIHMR_MODEL

    import torch

    _ensure_multihmr_assets(progress_cb=progress_cb)
    repo_root = _MULTIHMR_REPO
    device = _get_device()
    _purge_modules(("utils", "model", "blocks", "multi_hmr_anny"))
    _report(progress_cb, "Loading Multi-HMR model")
    with _sys_path(repo_root):
        constants = importlib.import_module("utils.constants")
        constants.SMPLX_DIR = str(repo_root / "models")
        constants.MEAN_PARAMS = str(_MULTIHMR_MEAN_PARAMS)
        constants.CACHE_DIR_MULTIHMR = str(_MULTIHMR_MODELS_DIR / "multiHMR")

        model_mod = importlib.import_module("model")
        ckpt = torch.load(_MULTIHMR_CKPT, map_location=device)
        kwargs = {k: v for k, v in vars(ckpt["args"]).items()}
        kwargs["type"] = ckpt["args"].train_return_type
        kwargs["img_size"] = ckpt["args"].img_size[0]
        model = model_mod.Model(**kwargs).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

    _MULTIHMR_MODEL = model
    _MULTIHMR_DEVICE = device
    _report(progress_cb, "Multi-HMR model loaded")
    return _MULTIHMR_MODEL


def _estimate_camera_intrinsics(pil_image, progress_cb=None):
    _report(progress_cb, "Estimating camera intrinsics")
    model = _load_wildcamera_model(progress_cb=progress_cb)
    intrinsic, _ = model.inference(pil_image, wtassumption=False)
    _report(progress_cb, "Camera intrinsics estimated")
    return intrinsic


def _normalize_rgb_uint8(img):
    import numpy as np

    arr = img.astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return (arr - mean) / std


def _prepare_multihmr_input(pil_image, img_size: int):
    from PIL import ImageOps
    import numpy as np
    import torch

    src_w, src_h = pil_image.size
    contained = ImageOps.contain(pil_image, (img_size, img_size))
    cont_w, cont_h = contained.size
    pad_x = int((img_size - cont_w) // 2)
    pad_y = int((img_size - cont_h) // 2)
    padded = ImageOps.pad(contained, size=(img_size, img_size), color=(0, 0, 0))

    arr = np.asarray(padded)
    norm = _normalize_rgb_uint8(arr)
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(_MULTIHMR_DEVICE)

    scale = min(img_size / max(1, src_w), img_size / max(1, src_h))
    transform = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "src_w": src_w,
        "src_h": src_h,
        "img_size": img_size,
    }
    return tensor, transform


def _adapt_intrinsics_for_square(K_original, transform):
    import numpy as np
    import torch

    scale = float(transform["scale"])
    pad_x = float(transform["pad_x"])
    pad_y = float(transform["pad_y"])
    K = np.asarray(K_original, dtype=np.float32).copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = K[0, 2] * scale + pad_x
    K[1, 2] = K[1, 2] * scale + pad_y
    return torch.from_numpy(K).unsqueeze(0).to(_MULTIHMR_DEVICE)


def _run_multihmr(pil_image, K_original, det_thresh: float = 0.25, nms_kernel_size: int = 3, progress_cb=None):
    import torch

    _report(progress_cb, "Preparing Multi-HMR inference input")
    model = _load_multihmr_model(progress_cb=progress_cb)
    img_size = int(getattr(model, "img_size", 896) or 896)
    input_tensor, transform = _prepare_multihmr_input(pil_image, img_size)
    K_padded = _adapt_intrinsics_for_square(K_original, transform)

    _report(progress_cb, "Running Multi-HMR inference")
    with torch.no_grad():
        humans = model(
            input_tensor,
            is_training=False,
            nms_kernel_size=int(nms_kernel_size),
            det_thresh=float(det_thresh),
            K=K_padded,
        )

    _report(progress_cb, f"Multi-HMR detected {len(humans or [])} person(s)")
    return humans, transform


def _extract_faces(image_bgr, progress_cb=None):
    import numpy as np

    analyzer = _get_face_analyzer()
    if analyzer is None:
        _report(progress_cb, "InsightFace unavailable, skipping age/gender estimation")
        return []
    try:
        _report(progress_cb, "Running face analysis for age/gender hints")
        face_objs = analyzer.get(image_bgr)
    except Exception:
        return []

    out = []
    for face in face_objs or []:
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        gender_raw = getattr(face, "gender", None)
        gender = "unknown"
        if gender_raw == 0:
            gender = "female"
        elif gender_raw == 1:
            gender = "male"
        age = getattr(face, "age", None)
        age_val = int(age) if age is not None and str(age).strip() != "" else None
        kps = getattr(face, "kps", None)
        landmark_106 = getattr(face, "landmark_2d_106", None)

        def _as_points(value):
            if value is None:
                return None
            try:
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < 2:
                    return None
                return [[float(p[0]), float(p[1])] for p in arr]
            except Exception:
                return None

        out.append(
            {
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                "gender": gender,
                "age": age_val,
                "kps": _as_points(kps),
                "landmark_106": _as_points(landmark_106),
            }
        )
    _report(progress_cb, f"Face analysis found {len(out)} face(s)")
    return out


def _age_to_stage(age_val: int | None) -> str:
    if age_val is None:
        return "adult"
    if age_val < 4:
        return "baby"
    if age_val < 13:
        return "child"
    if age_val < 18:
        return "teen"
    return "adult"


def _distance_named(named_joints: dict[str, Any], a: str, b: str) -> float | None:
    ja = named_joints.get(a)
    jb = named_joints.get(b)
    if not ja or not jb:
        return None
    try:
        dx = float(ja["x"]) - float(jb["x"])
        dy = float(ja["y"]) - float(jb["y"])
        dz = float(ja["z"]) - float(jb["z"])
        return (dx * dx + dy * dy + dz * dz) ** 0.5
    except Exception:
        return None


def _average_named_point(named_joints: dict[str, Any], names: list[str]) -> tuple[float, float, float] | None:
    pts = []
    for name in names:
        joint = named_joints.get(name)
        if not joint:
            continue
        try:
            pts.append((float(joint["x"]), float(joint["y"]), float(joint["z"])))
        except Exception:
            continue
    if not pts:
        return None
    inv = 1.0 / len(pts)
    return (
        sum(p[0] for p in pts) * inv,
        sum(p[1] for p in pts) * inv,
        sum(p[2] for p in pts) * inv,
    )


def _compute_body_metrics(named_joints: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    shoulder_width = _distance_named(named_joints, "left_shoulder", "right_shoulder")
    hip_width = _distance_named(named_joints, "left_hip", "right_hip")
    head = _average_named_point(named_joints, ["head", "neck"])
    ankles = _average_named_point(named_joints, ["left_ankle", "right_ankle"])
    pelvis = _average_named_point(named_joints, ["pelvis", "left_hip", "right_hip"])
    if shoulder_width is not None:
        metrics["shoulder_width"] = shoulder_width
    if hip_width is not None:
        metrics["hip_width"] = hip_width
    if head and ankles:
        dx = head[0] - ankles[0]
        dy = head[1] - ankles[1]
        dz = head[2] - ankles[2]
        metrics["body_height"] = (dx * dx + dy * dy + dz * dz) ** 0.5
    if head and pelvis:
        dx = head[0] - pelvis[0]
        dy = head[1] - pelvis[1]
        dz = head[2] - pelvis[2]
        metrics["head_to_pelvis"] = (dx * dx + dy * dy + dz * dz) ** 0.5
    if metrics.get("hip_width", 0.0) > 1e-6 and metrics.get("shoulder_width", 0.0) > 1e-6:
        metrics["shoulder_hip_ratio"] = metrics["shoulder_width"] / metrics["hip_width"]
    if metrics.get("body_height", 0.0) > 1e-6 and metrics.get("head_to_pelvis", 0.0) > 1e-6:
        metrics["head_body_ratio"] = metrics["head_to_pelvis"] / metrics["body_height"]
    return metrics


def _infer_stage_from_body(metrics: dict[str, float]) -> str:
    head_body_ratio = float(metrics.get("head_body_ratio", 0.0) or 0.0)
    if head_body_ratio >= 0.46:
        return "baby"
    if head_body_ratio >= 0.36:
        return "child"
    if head_body_ratio >= 0.30:
        return "teen"
    return "adult"


def _infer_gender_from_body(metrics: dict[str, float]) -> str:
    ratio = float(metrics.get("shoulder_hip_ratio", 0.0) or 0.0)
    if ratio >= 1.18:
        return "male"
    if ratio > 0.0 and ratio <= 1.02:
        return "female"
    return "unknown"


def _suggest_mesh(gender: str, stage: str) -> str:
    if stage in {"baby", "child"}:
        return "MQ chil male.fbx"
    if stage == "teen":
        if gender == "male":
            return "MQ chil male.fbx"
        return "MQ teen female.fbx"
    if gender == "male":
        return "MQ adult male.fbx"
    return "MQ adult female.fbx"


def _match_face_to_person(person_bbox: list[float] | None, faces: list[dict[str, Any]]):
    if not person_bbox or len(person_bbox) != 4 or not faces:
        return None
    x1, y1, x2, y2 = [float(v) for v in person_bbox]
    pcx = (x1 + x2) * 0.5
    pcy = (y1 + y2) * 0.5
    best = None
    best_rank = float("inf")
    for face in faces:
        fx1, fy1, fx2, fy2 = face["bbox"]
        fcx, fcy = face["center"]
        inside = (x1 <= fcx <= x2) and (y1 <= fcy <= y2)
        dx = fcx - pcx
        dy = fcy - pcy
        rank = (dx * dx + dy * dy) ** 0.5
        if not inside:
            rank += 1e6
        if rank < best_rank:
            best_rank = rank
            best = face
    return best


def _to_float_list(values) -> list[float]:
    return [float(v) for v in values]


def _restore_points_to_source_image(points2d, transform: dict[str, Any], image_w: int, image_h: int):
    import numpy as np

    scale = max(1e-8, float(transform.get("scale", 1.0) or 1.0))
    pad_x = float(transform.get("pad_x", 0.0) or 0.0)
    pad_y = float(transform.get("pad_y", 0.0) or 0.0)
    restored = []
    for point in points2d:
        try:
            x = (float(point[0]) - pad_x) / scale
            y = (float(point[1]) - pad_y) / scale
            x = max(0.0, min(float(image_w) - 1.0, x))
            y = max(0.0, min(float(image_h) - 1.0, y))
            restored.append([x, y])
        except Exception:
            restored.append([float("nan"), float("nan")])
    return np.asarray(restored, dtype=np.float32)


def _bbox_iou(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if not box_a or not box_b or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-8:
        return 0.0
    return inter / union


def _deduplicate_persons(persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(persons) <= 1:
        return persons
    ordered = sorted(persons, key=lambda p: float(p.get("score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for person in ordered:
        bbox = person.get("bbox")
        bbox_list = None
        if isinstance(bbox, dict):
            bbox_list = [bbox.get("x1", 0.0), bbox.get("y1", 0.0), bbox.get("x2", 0.0), bbox.get("y2", 0.0)]
        duplicate = False
        for existing in kept:
            eb = existing.get("bbox")
            eb_list = None
            if isinstance(eb, dict):
                eb_list = [eb.get("x1", 0.0), eb.get("y1", 0.0), eb.get("x2", 0.0), eb.get("y2", 0.0)]
            iou = _bbox_iou(bbox_list, eb_list)
            if not bbox_list or not eb_list:
                continue
            ax1, ay1, ax2, ay2 = bbox_list
            bx1, by1, bx2, by2 = eb_list
            a_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            area_ratio = a_area / b_area
            acx = (ax1 + ax2) * 0.5
            acy = (ay1 + ay2) * 0.5
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5
            center_dx = acx - bcx
            center_dy = acy - bcy
            center_dist = (center_dx * center_dx + center_dy * center_dy) ** 0.5
            max_dim = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
            center_dist_norm = center_dist / max_dim
            depth_a = float((person.get("transl_pelvis") or [0.0, 0.0, 0.0])[-1])
            depth_b = float((existing.get("transl_pelvis") or [0.0, 0.0, 0.0])[-1])
            depth_delta = abs(depth_a - depth_b)
            if (
                iou >= 0.72
                and 0.75 <= area_ratio <= 1.3333333333
                and center_dist_norm <= 0.12
                and depth_delta <= 0.45
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(person)
    kept.sort(key=lambda p: float((p.get("center_hint") or {}).get("x", 0.5)))
    return kept


def _package_person(human: dict[str, Any], image_w: int, image_h: int, faces: list[dict[str, Any]], transform: dict[str, Any]):
    import numpy as np

    joints3d = human["j3d"].detach().cpu().numpy()
    joints2d = _restore_points_to_source_image(human["j2d"].detach().cpu().numpy(), transform, image_w, image_h)
    transl_pelvis = human["transl_pelvis"].detach().cpu().numpy().reshape(-1)
    transl = human["transl"].detach().cpu().numpy().reshape(-1)
    rotvec = human["rotvec"].detach().cpu().numpy()
    score = float(human["scores"].detach().cpu().numpy().reshape(-1)[0])

    named_joints = {}
    for idx, name in enumerate(_SMPLX_BODY_JOINT_NAMES):
        if idx >= len(joints3d):
            break
        j3 = joints3d[idx]
        j2 = joints2d[idx] if idx < len(joints2d) else None
        named_joints[name] = {
            "x": float(j3[0]),
            "y": float(j3[1]),
            "z": float(j3[2]),
            "u": float(j2[0]) if j2 is not None else None,
            "v": float(j2[1]) if j2 is not None else None,
            "un": float(j2[0]) / max(1.0, float(image_w)) if j2 is not None else None,
            "vn": float(j2[1]) / max(1.0, float(image_h)) if j2 is not None else None,
        }

    xs = [float(j[0]) for j in joints2d if j is not None]
    ys = [float(j[1]) for j in joints2d if j is not None]
    bbox = None
    if xs and ys:
        bbox = [min(xs), min(ys), max(xs), max(ys)]

    body_metrics = _compute_body_metrics(named_joints)
    face = _match_face_to_person(bbox, faces)
    face_gender = face["gender"] if face else "unknown"
    age = face["age"] if face else None
    face_stage = _age_to_stage(age) if age is not None else None
    body_stage = _infer_stage_from_body(body_metrics)
    body_gender = _infer_gender_from_body(body_metrics)
    gender = face_gender if face_gender != "unknown" else body_gender
    stage = face_stage or body_stage

    center_hint = None
    scale_hint = None
    if bbox is not None:
        center_hint = {
            "x": ((bbox[0] + bbox[2]) * 0.5) / max(1.0, float(image_w)),
            "y": ((bbox[1] + bbox[3]) * 0.5) / max(1.0, float(image_h)),
        }
        scale_hint = {
            "width": (bbox[2] - bbox[0]) / max(1.0, float(image_w)),
            "height": (bbox[3] - bbox[1]) / max(1.0, float(image_h)),
        }

    return {
        "score": score,
        "bbox": {
            "x1": float(bbox[0]),
            "y1": float(bbox[1]),
            "x2": float(bbox[2]),
            "y2": float(bbox[3]),
            "xn1": float(bbox[0]) / max(1.0, float(image_w)),
            "yn1": float(bbox[1]) / max(1.0, float(image_h)),
            "xn2": float(bbox[2]) / max(1.0, float(image_w)),
            "yn2": float(bbox[3]) / max(1.0, float(image_h)),
        } if bbox is not None else None,
        "center_hint": center_hint,
        "scale_hint": scale_hint,
        "gender": gender,
        "gender_source": "face" if face_gender != "unknown" else ("body" if body_gender != "unknown" else "unknown"),
        "age": age,
        "life_stage": stage,
        "life_stage_source": "face" if face_stage is not None else "body",
        "mesh_suggestion": _suggest_mesh(gender, stage),
        "body_metrics": body_metrics,
        "face": face,
        "transl": _to_float_list(transl),
        "transl_pelvis": _to_float_list(transl_pelvis),
        "rotvec": np.asarray(rotvec).astype(float).tolist(),
        "joints3d_named": named_joints,
        "joints2d": np.asarray(joints2d).astype(float).tolist(),
    }


def infer_scene_from_image(image_b64: str, det_thresh: float = 0.25, progress_cb=None) -> dict[str, Any]:
    with _MODEL_LOCK:
        _report(progress_cb, "Decoding input image")
        image_bgr = _decode_image_from_b64(image_b64)
        _report(progress_cb, "Source image loaded", preview_image_base64=_encode_preview_image_b64(image_bgr))
        pil_image = _bgr_to_pil(image_bgr)
        image_w, image_h = pil_image.size

        K = _estimate_camera_intrinsics(pil_image, progress_cb=progress_cb)
        humans, _transform = _run_multihmr(pil_image, K, det_thresh=det_thresh, nms_kernel_size=3, progress_cb=progress_cb)
        faces = _extract_faces(image_bgr, progress_cb=progress_cb)
        _report(
            progress_cb,
            f"Face analysis found {len(faces)} face(s)",
            preview_image_base64=_build_preview_image(image_bgr, faces=faces, persons=None),
        )

        _report(progress_cb, "Packaging detected persons")
        persons = []
        for human in humans or []:
            person = _package_person(human, image_w, image_h, faces, _transform)
            persons.append(person)

        if not persons:
            raise RuntimeError("No people reconstructed by Multi-HMR.")

        deduped = _deduplicate_persons(persons)
        if len(deduped) != len(persons):
            _report(progress_cb, f"Removed {len(persons) - len(deduped)} duplicate detection(s)")
        persons = deduped
        persons.sort(key=lambda p: float((p.get("center_hint") or {}).get("x", 0.5)))
        primary = persons[0]
        _report(
            progress_cb,
            f"Scene initialization ready with {len(persons)} person(s)",
            preview_image_base64=_build_preview_image(image_bgr, faces=faces, persons=persons),
        )

        return {
            "ok": True,
            "engine": "advanced_multihmr",
            "camera_engine": "wildcamera",
            "image_width": int(image_w),
            "image_height": int(image_h),
            "camera_intrinsics": [[float(v) for v in row] for row in K.tolist()],
            "persons": persons,
            "person_count": len(persons),
            # Backward-compatible summary fields.
            "center_hint": primary.get("center_hint"),
            "scale_hint": primary.get("scale_hint"),
            "bbox": primary.get("bbox"),
            "gender": primary.get("gender"),
            "age": primary.get("age"),
            "life_stage": primary.get("life_stage"),
            "mesh_suggestion": primary.get("mesh_suggestion"),
        }
