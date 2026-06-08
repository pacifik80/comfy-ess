# CLAUDE.md

ComfyUI custom-node pack (`comfyui-ess`, node prefix `ESS/`). Pure plugin: no build
step — install by placing in `ComfyUI/custom_nodes/` and restarting ComfyUI.

## Layout
- `__init__.py` — single entry point. Registers nodes + aiohttp routes. See below.
- `nodes/<category>/*.py` — node implementations. Categories: `prompt_builder`, `image`,
  `pose`, `detailer`, `face_swapping`, `sampling`, `conditioning`, `checkpoints`,
  `assets`, `image_processing`, `utils`.
- `js/*.js` — one front-end widget per interactive node. `js/vendor/three/` is bundled
  Three.js (pose mesh editor). Exposed via `WEB_DIRECTORY = "./js"`.
- `meshes/fbx/` — rig meshes served by `/ess/rigged/*`.
- `models/`, `.cache_src/` — gitignored, multi-GB local ML assets / vendored repos
  (multi-hmr, WildCamera). Not part of the package; never commit.

## How `__init__.py` works
1. **Dual-mode import**: tries relative imports; on "attempted relative import" falls back
   to loading `nodes/` as standalone package `comfyui_ess_local_nodes`. Keep both paths in
   sync when adding nodes.
2. **Defensive loading**: each optional node is wrapped in `try/except` saving `_<x>_error`;
   a failed import prints a `[comfyui-ess] ... disabled:` warning but never breaks the pack.
3. **Registration**: nodes added to `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`
   only `if X is not None`.
4. **Routes**: registered on `PromptServer.instance.routes` (guarded by `if web is not None`):
   `/ess/rigged/{list,get}`, `/ess/scene_flow/test`, `/ess/pose/init_from_image`,
   `/ess/pose/init_progress`, plus checkpoint/workflow-asset/recrop routes.

## Adding a node
1. Implement in `nodes/<category>/your_node.py`.
2. In `__init__.py`: add the import in BOTH the relative block and the `_import_node_attr`
   fallback block, each in its own `try/except`.
3. Register in the mappings (relevant `if ... is not None` block).
4. If it has UI, add `js/your_node.js` (auto-served from `WEB_DIRECTORY`).

## Conventions / gotchas
- Heavy ML deps (insightface, ultralytics, smplx, GFPGAN, pyrender, torch, cv2) are
  imported lazily inside functions so the pack loads without them. Keep it that way.
- `numpy<2` and `pyglet<2` are pinned in `requirements.txt` — respect the upper bounds.
- No test suite yet despite `pytest`/`mypy` in requirements.
- `git status` may list large untracked `models/`/`.cache_src/` on older clones — now
  gitignored.
