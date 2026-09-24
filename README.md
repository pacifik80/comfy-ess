# comfyui-ess

Clean starting point for a complete rewrite of the ESS ComfyUI extension.
The `rewrite` branch currently contains only an empty ComfyUI entry point and
repository housekeeping. No nodes or frontend extensions are registered yet.

## Previous implementation

The complete implementation immediately before the rewrite is preserved at:

- Snapshot commit: `e7111aa4aa0d6c66d3e2afca4a92235062b44198`
- Tag: [`pre-rewrite-2026-09-25`](https://github.com/pacifik80/comfy-ess/tree/pre-rewrite-2026-09-25)
- Archive branch: `archive/pre-rewrite-2026-09-25`
- `master` also points to this snapshot at the start of the rewrite.

The snapshot includes the previously uncommitted Flux Fill, Flux Inpaint, and
Recrop changes. Earlier implementations, including removed crop nodes, remain
available in Git history. The rewrite retains that history.

Read an archived file without switching branches:

```sh
git show pre-rewrite-2026-09-25:nodes/image/recrop.py
```

Find older versions of a removed file:

```sh
git log --all -- nodes/image/composition_crop.py
```

## Local reference checkout

On the original development machine, a linked Git worktree is available at:

```text
C:\AIApps\Data\ess-archive\comfyui-ess-pre-rewrite-2026-09-25
```

It contains the old source together with the local `models/`, `.cache_src/`, and
other ignored caches moved out of the rewrite workspace. Those local assets are
not uploaded to GitHub. The archive is outside `custom_nodes`, so ComfyUI does
not automatically load a second copy of the extension.

On another machine, create a reference checkout outside `custom_nodes`:

```sh
git worktree add /path/outside/custom_nodes/comfyui-ess-reference archive/pre-rewrite-2026-09-25
```
