import sys
from types import ModuleType

if 'folder_paths' not in sys.modules:
    folder_paths_stub = ModuleType('folder_paths')
    folder_paths_stub.get_folder_paths = lambda *_, **__: []
    sys.modules['folder_paths'] = folder_paths_stub

from scene_nodes.scene import Scene
from scene_nodes.scene_combiner import SceneCombiner
from scene_nodes.global_state import GlobalState


def test_scene_initialization():
    scene = Scene()
    assert scene.elements == []
    assert scene.metadata == {}
    assert scene.get_embeddings() == []
    assert scene.get_embeddings(is_negative=True) == []
    assert scene.get_lora_stack() == []

    elements = [{'type': 'test', 'positive': 'pos', 'negative': 'neg'}]
    metadata = {'seed': 123}
    embeddings = {'positive': ['embed'], 'negative': ['neg_embed']}
    lora_stack = [{'lora_name': 'test_lora', 'strength': 0.5, 'clip_strength': 0.4}]
    scene = Scene(elements, metadata=metadata, embeddings=embeddings, lora_stack=lora_stack)

    assert scene.elements == elements
    assert scene.metadata == metadata
    assert scene.get_embeddings() == ['embed']
    assert scene.get_embeddings(is_negative=True) == ['neg_embed']
    assert scene.get_lora_stack() == [{'lora_name': 'test_lora', 'strength': 0.5, 'clip_strength': 0.4}]


def test_add_element():
    scene = Scene()
    scene.add_element('test', 'pos', 'neg')
    assert len(scene.elements) == 1
    assert scene.elements[0] == {
        'type': 'test',
        'positive': 'pos',
        'negative': 'neg'
    }


def test_add_embedding_and_deduplication():
    scene = Scene()
    scene.add_embedding('embed')
    scene.add_embedding('embed')
    scene.add_embedding('neg_embed', is_negative=True)
    scene.add_embedding('neg_embed', is_negative=True)

    assert scene.get_embeddings() == ['embed']
    assert scene.get_embeddings(is_negative=True) == ['neg_embed']


def test_add_lora_updates_existing_entry():
    scene = Scene()
    scene.add_lora('test_lora', 0.5)
    scene.add_lora('test_lora', 0.8, 0.7)

    assert scene.get_lora_stack() == [{'lora_name': 'test_lora', 'strength': 0.8, 'clip_strength': 0.7}]


def test_merge_from_creates_independent_copy():
    source = Scene()
    source.add_element('type1', 'pos1', 'neg1')
    source.add_embedding('embed_pos')
    source.add_embedding('embed_neg', is_negative=True)
    source.add_lora('lora1', 0.5)
    source.metadata['key'] = 'value'

    target = Scene()
    target.merge_from(source)

    assert target.elements[0] == {'type': 'type1', 'positive': 'pos1', 'negative': 'neg1'}
    assert target.get_embeddings() == ['embed_pos']
    assert target.get_embeddings(is_negative=True) == ['embed_neg']
    assert target.get_lora_stack() == [{'lora_name': 'lora1', 'strength': 0.5, 'clip_strength': 0.5}]
    assert target.metadata == {'key': 'value'}

    source.elements[0]['positive'] = 'changed'
    source.metadata['key'] = 'other'
    assert target.elements[0]['positive'] == 'pos1'
    assert target.metadata['key'] == 'value'


def test_to_json():
    scene = Scene()
    scene.add_element('test', 'pos', 'neg')
    scene.add_embedding('embed')
    scene.add_embedding('neg_embed', is_negative=True)
    scene.add_lora('test_lora', 0.6, 0.4)
    scene.metadata['foo'] = 'bar'

    json_data = scene.to_json()
    assert json_data == {
        'elements': [{'type': 'test', 'positive': 'pos', 'negative': 'neg'}],
        'metadata': {'foo': 'bar'},
        'embeddings': {
            'positive': ['embed'],
            'negative': ['neg_embed'],
        },
        'lora_stack': [{'lora_name': 'test_lora', 'strength': 0.6, 'clip_strength': 0.4}],
    }


def test_from_json():
    json_data = {
        'elements': [{'type': 'test', 'positive': 'pos', 'negative': 'neg'}],
        'metadata': {'foo': 'bar'},
        'embeddings': {
            'positive': ['embed'],
            'negative': ['neg_embed'],
        },
        'lora_stack': [{'lora_name': 'test_lora', 'strength': 0.6, 'clip_strength': 0.4}],
    }

    scene = Scene.from_json(json_data)
    assert scene.elements == json_data['elements']
    assert scene.metadata == {'foo': 'bar'}
    assert scene.get_embeddings() == ['embed']
    assert scene.get_embeddings(is_negative=True) == ['neg_embed']
    assert scene.get_lora_stack() == [{'lora_name': 'test_lora', 'strength': 0.6, 'clip_strength': 0.4}]


from scene_nodes.define_scene_part import SceneDebug


def test_scene_debug_updates_ui_widget():
    scene = Scene()
    scene.add_element('type', 'pos', 'neg')
    node = SceneDebug()
    extra_pnginfo = [{"workflow": {"nodes": [{"id": 1, "widgets_values": ["default"]}]}}]

    result = node.debug_scene(scene, "", unique_id=[1], extra_pnginfo=extra_pnginfo)

    assert 'Scene Elements:' in result['ui']['text'][0]
    stored_value = extra_pnginfo[0]['workflow']['nodes'][0]['widgets_values'][0]
    assert stored_value.startswith('Scene Elements:')


def test_scene_seed_roundtrip():
    scene = Scene()
    assert scene.get_seed() is None
    scene.set_seed(123)
    assert scene.get_seed() == 123

    scene.set_seed(None)
    assert scene.get_seed() is None

    scene.set_seed(456)
    json_data = scene.to_json()
    assert json_data['metadata']['seed'] == 456

    restored = Scene.from_json(json_data)
    assert restored.get_seed() == 456


def test_scene_combiner_carries_embeddings_and_loras_into_prompts():
    GlobalState.reset()
    GlobalState.set_resolution(640, 480)
    GlobalState.set_seed(11)

    scene = Scene()
    scene.add_element("scene", "river", "storm")
    scene.add_embedding("pos_embed")
    scene.add_embedding("neg_embed", is_negative=True)
    scene.add_lora("my_lora", 0.7)

    combiner = SceneCombiner()
    pos, neg, lora_stack, width, height, seed = combiner.combine(scene)

    assert "pos_embed" in pos
    assert "neg_embed" in neg
    assert "<lora:my_lora:0.7>" in pos
    # Ensure metadata still reflected
    assert (width, height) == (640, 480)
    assert seed == 11
    assert lora_stack == [("my_lora", 0.7, 0.7)]
