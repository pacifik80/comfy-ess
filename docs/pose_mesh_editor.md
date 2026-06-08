# Pose Mesh Editor Requirements

## Document Purpose
This document captures the actual current requirements, scope, behavior, outputs, and known gaps for the `ESS - Pose Mesh Editor` node as it exists in this repository today.

The goal is to provide a reviewable baseline before further redesign or bug-fixing work.

## Node Identity
- ComfyUI display name: `ESS - Pose Mesh Editor`
- Internal node key: `ESS/PoseMeshEditor`
- Python class: `PoseMeshEditor`
- Category: `ESS/Pose`
- Companion frontend/editor widget type: `ESS_POSE_MESH_EDITOR`

## Primary Purpose
The node provides an interactive scene editor for rigged humanoid meshes, with:
- manual pose editing
- OpenPose-style overlay visualization
- image-based scene initialization
- preview output generation
- auxiliary depth and edge output generation

The editor is primarily implemented in `js/ess_pose_mesh_editor.js`. The Python node is a thin state carrier and output decoder in `nodes/pose/pose_mesh_editor.py`.

## High-Level Functional Requirements

### 1. Interactive Scene Editing
The node must let the user build and edit a scene interactively inside a custom editor overlay.

Current editor scope includes:
- loading rigged meshes
- creating and removing characters
- selecting an active character
- selecting and transforming bones
- selecting and transforming the whole placed figure
- framing the model in view
- visualizing an OpenPose-style skeleton on top of the rendered mesh

### 2. Image-Based Scene Initialization
The editor must support initializing a scene from a source image.

Current intended flow:
- user chooses an image via `Init From Image`
- backend estimates scene/person structure from the image
- frontend rebuilds the scene from detected people
- frontend applies pose retargeting to the loaded rigged meshes
- frontend shows a progress/review dialog before the user closes it

### 3. Output Generation
The node must return three image outputs derived from the current editor state:
- `preview`
- `depth`
- `edges`

These outputs are encoded into the serialized state by the frontend and decoded by Python.

## Current Python Node Contract

### Required Inputs
- `state`
  - custom widget payload
  - JSON string produced by the frontend editor
- `output_image`
  - boolean
  - when disabled, the node returns tiny blank images
- `draw_fingers`
  - boolean
  - controls whether OpenPose-style hand/finger data is drawn into the final preview render
- `draw_face_mask`
  - boolean
  - controls whether face mask geometry is drawn into the final preview render

### Outputs
- `preview`
  - OpenPose-style rendered preview image from the editor
- `depth`
  - depth render of the scene
- `edges`
  - contour/edge render derived from the scene

### Current Output Behavior
If `output_image` is enabled, Python decodes:
- `preview_png`
- `depth_png`
- `edges_png`

from the serialized `state` payload.

Legacy compatibility currently remains for:
- `preview_pngs`

## Mesh Source Requirements

### Active Mesh Repository
The active rigged-mesh source is now:
- `meshes/fbx`

Current backend routes and frontend mesh loading logic use this directory.

### Current Selection Behavior
Scene-init mesh selection currently prefers filenames matching the following FBX mannequins when available:
- `MQ adult female.fbx`
- `MQ adult male.fbx`
- `MQ teen female.fbx`
- `MQ chil male.fbx`

If a backend `mesh_suggestion` exists and matches an available file, that suggestion wins.

If no preferred match is found, the first available mesh is used.

### Import Scale Normalization
Meshes loaded from `meshes/fbx` are normalized on import to reduce extreme authoring-scale differences.

Current normalization targets:
- baby-like filenames (`baby`) -> target height about `0.96`
- child-like filenames (`kid`, `child`, `chil`) -> target height about `1.28`
- teen-like filenames (`teen`) -> target height about `1.56`
- all others -> target height about `1.72`

This is implemented at import time in the editor, before the character becomes active.

### Important Current Status
- `meshes/fbx` is active: implemented
- import-scale normalization: implemented
- exact restriction to only two fixed models: not fully enforced

The current system still selects from the available files in `meshes/fbx`, with strong preference rules, rather than hard-locking to exactly two filenames.

## Camera Requirements

### Current Intended Camera Model
The node now works as a single-camera node for output generation.

Current output generation uses the active camera payload and produces:
- one preview render
- one depth render
- one edge render

### Backward Compatibility
Older serialized camera/state structures may still be read for restore compatibility, but the active output path is single-camera oriented.

### Current Status
- single preview/depth/edges output path: implemented
- old multi-camera legacy compatibility retained in restore paths: partially retained for compatibility

## OpenPose Overlay Requirements

### Visible During Editing
An OpenPose-style skeleton must be drawn in the editor while posing.

Current overlay includes:
- body skeleton
- OpenPose body joints
- optional hands/fingers
- optional face mask

### Final Preview Output
The final `preview` output is not just a mesh render. It is currently an OpenPose-style render generated from the editor scene.

### Hands/Fingers Toggle
Main node boolean input:
- `draw_fingers`

Controls whether OpenPose-style hand/finger lines are drawn in the preview render and editor overlay.

### Face Mask Toggle
Main node boolean input:
- `draw_face_mask`

Controls whether face-mask geometry is drawn in the preview render and editor overlay.

### Current Status
- OpenPose body skeleton visible during editing: implemented
- optional fingers/hands drawing: implemented
- optional face mask drawing: implemented

## Pose Editing Requirements

### Bone Selection and Manipulation
The editor must support selecting bones and applying transforms.

Current behavior includes:
- selected bone highlighting
- transform gizmo interaction
- state serialization after edits

### Whole-Figure Placement
The editor also supports whole-figure placement controls.

Current behavior:
- `Pos X`, `Pos Y`, `Pos Z` act on a dedicated per-character scene root, not directly on a selected bone
- imported meshes are grounded on load so the scene root origin represents floor-level placement more reliably
- whole-figure transform is serialized and restored with the character state

Current status:
- whole-figure placement controls: implemented
- whole-figure transform persistence across save/load/history: implemented

### Paired Limb / Mirrored Adjustment
The editor currently contains paired-bone mirroring support.

Current implemented behavior:
- when a mirrored counterpart is found, the UI can expose mirror behavior
- rotation/position/scale deltas can be mirrored to the paired bone

This is intended for paired limbs and symmetrical rig edits.

### Current Status
- mirror-paired-bone logic: implemented in editor code
- end-to-end validation against the current `meshes/fbx` rig set: not yet fully validated

## Scene Initialization From Image

### Intended Behavior
Image-based initialization should:
- detect people in an input image
- estimate their pose/camera structure
- choose suitable rigged meshes
- rebuild the scene
- retarget pose to the loaded character rigs
- present a review dialog

### Current Review Dialog Requirements
The progress/review dialog currently includes:
- `Source` pane
  - original image with detection overlay
- `Model` pane
  - simplified visual structure of the recognized model output
- `Result` pane
  - rendered scene result
- log section
- explicit `Finish` button to close the dialog

### Current Progress Behavior
The dialog shows:
- status text
- elapsed time
- live logs from backend stages
- error display when initialization fails

### Current Backend/Model Pipeline
The scene-init feature currently depends on:
- image-to-scene backend route(s) defined in `__init__.py`
- modern monocular scene/person estimation pipeline added earlier
- face analysis hints for age/gender where available

This document intentionally describes the requirements/behavior, not the exact external-model inventory.

### Current Retarget Goal
The `Model` pane should represent the recognized pose structure.
The `Result` pane should make the rigged mesh visually match that recognized structure as closely as practical.

Current implementation note:
- after the main 3D retarget pass, the editor now performs an additional visible-pose refinement pass against the projected named-joint structure used by the `Model` pane, rather than against the raw source OpenPose payload
- this is intended to keep the final visible rig closer to what the review dialog actually shows in `Model`

### Current Status
- Source/Model/Result review dialog: implemented
- scene rebuild from detected persons: implemented
- scene-init retarget quality matching the recognized pose: not yet fully satisfactory

This remains one of the main active problem areas.

## Slider Interaction Requirements

### Relative Adjustment Sliders
Transform sliders are currently designed as relative adjustment controls rather than persistent absolute handles.

Current behavior:
- while dragging, the slider updates the target transform live
- when released, the slider springs back to center
- the adjusted transform value remains applied
- the user can drag again in the same direction to continue the change

This currently applies to:
- bone rotation sliders
- whole-figure position sliders
- scale sliders

Current status:
- relative spring-back transform sliders: implemented
- serialization/restoration of resulting transform values: implemented

## Scene Visibility / Lighting Requirements

The editor scene must remain readable while posing and reviewing imported scenes.

Current lighting behavior:
- ACES filmic tone mapping enabled
- elevated exposure
- hemisphere light
- ambient light
- two directional lights
- dark blue-gray background instead of near-black

Current status:
- brighter editor scene lighting: implemented
- exact artistic/look-dev tuning: not finalized

## State Serialization Requirements

The custom editor must serialize enough information to fully reconstruct the editor session.

Current serialized payload includes, at minimum:
- pose/editor state
- character list
- active character id
- active camera
- per-character scene-root transform
- preview/depth/edges PNG data for node outputs
- metadata timestamp/model label

### Compatibility Requirement
Older saved payload formats should degrade gracefully where practical.

Currently present compatibility behavior includes:
- old preview list fallback
- older camera/state restore paths where possible

## UI/UX Requirements

### Mesh Selection UX
The editor must let the user add characters from the active mesh repository.

Current messaging and picker text explicitly reference:
- `meshes/fbx`

### Init Review UX
The image-init dialog must stay open after a run, so the user can compare the outputs before leaving.

Current behavior:
- review dialog remains visible after success/failure
- user must press `Finish` to dismiss

### Preview UX
The init dialog must use three square preview panes:
- Source
- Model
- Result

The preview image inside each pane must preserve aspect ratio.

## Current Known Gaps / Open Issues

The following are not fully solved at the time of this document:

### 1. Scene-init Retarget Accuracy
Current result mesh pose can still diverge from the recognized model pose, especially in:
- torso lean
- head direction
- limb orientation
- lower-leg / foot behavior

This is the primary active quality issue.

### 2. New Rig Validation
The code has been switched to `meshes/fbx`, but full validation of all previously implemented pose-editing features against the new rigs is not complete.

That includes:
- paired-limb/mirror behavior
- scene-init retarget behavior
- bone aliasing assumptions

### 3. Exact Fixed Mesh Set
The system now uses `meshes/fbx`, but it is not yet strictly limited to exactly two hardcoded model files.

### 4. Scene-Init Matching Requirement
Current user expectation:
- the final scene result should closely match the pose shown in the `Model` pane

Current implementation:
- attempts this
- does not yet consistently achieve it

## Requirements Status Summary

### Implemented
- node exists as `ESS - Pose Mesh Editor`
- custom editor UI exists
- mesh source uses `meshes/fbx`
- import scale normalization exists
- OpenPose body overlay is visible during editing
- final outputs are `preview`, `depth`, `edges`
- fingers/hands toggle exists
- face mask toggle exists
- image-init dialog shows `Source`, `Model`, `Result`
- review dialog stays open until `Finish`

### Implemented But Not Fully Validated
- paired/mirrored limb editing against the new rig set
- scene-init mesh choice against the active `meshes/fbx` models
- scene-init retarget matching against the recognized pose

### Not Fully Enforced Yet
- exact restriction to only two fixed mesh files

## Acceptance Criteria For “Working As Intended”
For future work, the node should be considered aligned with its intended behavior when all of the following are true:

1. Mesh loading
- characters load from `meshes/fbx`
- loaded rigs appear at sane human scale

2. Editing
- OpenPose body overlay is always visible during posing
- optional fingers and face mask behave according to the node toggles
- mirrored paired-bone editing behaves predictably on the active rigs

3. Outputs
- `preview` is a stable OpenPose-style render
- `depth` is a stable scene depth render
- `edges` is a stable contour render

4. Scene init
- Source / Model / Result panes are readable
- the result mesh visually follows the recognized pose with acceptable fidelity
- head direction and torso lean match the source/model output
- feet and lower legs do not deform unnaturally

5. Persistence
- editor state reloads correctly from saved workflow data

## Files Governing This Node
- Python node: `nodes/pose/pose_mesh_editor.py`
- Frontend/editor: `js/ess_pose_mesh_editor.js`
- Node registration and routes: `__init__.py`
