# comfy-ess

Custom ComfyUI nodes for ESS prompt tooling, including a template editor with syntax highlighting and utility image/pose nodes.

## Features
- Prompt template editor with scope-aware highlighting
- Optional template parsing with deterministic seed
- Utility nodes for image adjustment, pose generation, and optional face workflows

**Nodes**
All nodes appear under `ESS/*` categories in the ComfyUI add menu.

**Prompt Builder**
- `ESS - Prompt Template Editor`: rich template editor; optionally parses with seed and supports prefix/suffix positive/negative inputs.
- `ESS - Prompt Template Editor - Multioutput`: outputs up to 10 positive/negative pairs using `<< >>` variant blocks (a-j).
- `ESS - Text Prompt Generator`: parses positive/negative templates into final prompts (inverse parts cross-fed).
- `ESS - Text Prompt Replacer`: replaces `%replace_a%`, `%replace_b%`, `%replace_c%` placeholders in a prompt.
- `ESS - Replace Dict`: creates or extends a replacement dictionary (key/value pairs).

**Image**
- `ESS - Image Adjustments`: per-channel RGB, brightness, and saturation adjustments on `IMAGE`.
- `ESS - Segmentation Detailer`: segmentation-guided detail pass using a mask or SAM model; outputs `IMAGE` and `LATENT`.
- `ESS - Person Crop To Size`: detects a person/head with Ultralytics YOLO and crops to the target size; outputs a debug overlay.

**Detailer**
- `ESS - Face Detailer (ESS)`: Impact Pack face detailer clone; detects a face and refines via masked inpainting.

**Pose**
- `ESS - Pose Figure Editor`: parametric pose generator; outputs a pose image plus OpenPose JSON.
- `ESS - Pose Resize With Padding`: aspect-preserving resize to target size with custom padding.

**Face Swapping**
- `ESS - Face Swap (InSwapper)`: InsightFace InSwapper ONNX; optional mask blending and GFPGAN restoration.
- `ESS - Face Swap (SimSwap)`: SimSwap ONNX with optional GFPGAN restoration and mask blending.
- `ESS - Face Swap (FaceFusion)`: HyperSwap/Ghost/BlendSwap/HiFiFace/UniFace ONNX models; outputs swapped image.

**Utils**
- `ESS - Prefix Generator`: creates project/version/run prefixes and resolves seeds (fixed/random/increment/decrement).
- `ESS - Group Reroute`: dynamic passthrough (up to 16 inputs/outputs).
- `ESS - String Concatenate`: concatenates multiple strings with separator/newline options and whitespace cleanup.
- `ESS - Label Note`: canvas-only label node (no outputs).

**Optional Dependencies**
- Face swapping nodes require `insightface`, `onnxruntime`, and `opencv-python` (plus optional `gfpgan` for restoration).
- `ESS - Face Detailer (ESS)` requires ComfyUI-Impact-Pack and InsightFace.
- `ESS - Segmentation Detailer` requires the ComfyUI runtime and optionally SAM models.
- `ESS - Person Crop To Size` requires `ultralytics` and YOLO model weights.

## Install
Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.
