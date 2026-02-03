from typing import Optional

import random

from scene_nodes.scene import Scene
from scene_nodes.scene_selector import SceneSelector
from scene_nodes.global_state import GlobalState


def _make_scene(label: str, seed: Optional[int] = None) -> Scene:
    scene = Scene()
    scene.add_element('tag', label, '')
    if seed is not None:
        scene.set_seed(seed)
    return scene


def test_scene_selector_prefers_scene_seed_over_global():
    GlobalState.reset()
    GlobalState.set_seed(999)

    scene_a = _make_scene('A', seed=42)
    scene_b = _make_scene('B')

    selector = SceneSelector()
    selected_scene, = selector.select(
        scene_1=scene_a,
        weight_1=1.0,
        scene_2=scene_b,
        weight_2=3.0,
    )

    expected_scene = random.Random(42).choices(
        [scene_a, scene_b],
        weights=[1.0, 3.0],
        k=1,
    )[0]

    assert selected_scene.elements[0]['positive'] == expected_scene.elements[0]['positive']


def test_scene_selector_falls_back_to_global_seed_when_scene_seed_missing():
    GlobalState.reset()
    GlobalState.set_seed(7)

    scene_a = _make_scene('A')
    scene_b = _make_scene('B')

    selector = SceneSelector()
    selected_scene, = selector.select(
        scene_1=scene_a,
        weight_1=2.0,
        scene_2=scene_b,
        weight_2=5.0,
    )

    expected_scene = random.Random(7).choices(
        [scene_a, scene_b],
        weights=[2.0, 5.0],
        k=1,
    )[0]

    assert selected_scene.elements[0]['positive'] == expected_scene.elements[0]['positive']


def test_scene_selector_defaults_when_all_weights_zero():
    GlobalState.reset()

    scene_a = _make_scene('A', seed=101)
    scene_b = _make_scene('B', seed=101)

    selector = SceneSelector()
    selected_scene, = selector.select(
        scene_1=scene_a,
        weight_1=0.0,
        scene_2=scene_b,
        weight_2=0.0,
    )

    expected_scene = random.Random(101).choices(
        [scene_a, scene_b],
        weights=[1.0, 1.0],
        k=1,
    )[0]

    assert selected_scene.elements[0]['positive'] == expected_scene.elements[0]['positive']


def test_scene_selector_skips_hidden_scenes_by_metadata():
    GlobalState.reset()

    scene_visible = _make_scene('Visible', seed=321)
    scene_hidden = _make_scene('Hidden', seed=321)
    scene_hidden.metadata["hidden"] = True

    selector = SceneSelector()
    selected_scene, = selector.select(
        scene_1=scene_visible,
        weight_1=1.0,
        scene_2=scene_hidden,
        weight_2=50.0,
    )

    assert selected_scene.elements[0]['positive'] == 'Visible'


def test_scene_selector_skips_hidden_wrapped_inputs():
    GlobalState.reset()

    scene_visible = _make_scene('Visible', seed=222)
    hidden_wrapper = [_make_scene('Hidden', seed=222), {"hidden": True}]

    selector = SceneSelector()
    selected_scene, = selector.select(
        scene_1=scene_visible,
        weight_1=0.5,
        scene_2=hidden_wrapper,
        weight_2=10.0,
    )

    assert selected_scene.elements[0]['positive'] == 'Visible'


def test_scene_selector_ignores_non_scene_candidates():
    GlobalState.reset()

    valid_scene = _make_scene('Valid', seed=55)
    selector = SceneSelector()

    selected_scene, = selector.select(
        scene_1=[],
        weight_1=100.0,
        scene_2={"hidden": False},
        weight_2=100.0,
        scene_3=valid_scene,
        weight_3=0.1,
    )

    assert isinstance(selected_scene, Scene)
    assert selected_scene.elements[0]['positive'] == 'Valid'


def test_scene_selector_returns_none_when_all_candidates_invalid():
    GlobalState.reset()
    selector = SceneSelector()

    selected_scene, = selector.select(
        scene_1=[],
        weight_1=1.0,
        scene_2={"hidden": False},
        weight_2=2.0,
    )

    assert selected_scene is None
