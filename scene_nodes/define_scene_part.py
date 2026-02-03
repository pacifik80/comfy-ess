import os
import json
import copy
import random
import re
import folder_paths
import time
from .scene import Scene
from .global_state import GlobalState
from .prompt_parser import PromptParser

def recursive_list_files(base_path):
    files_list = []
    for root, _, files in os.walk(base_path):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), base_path)
            files_list.append(relative_path.replace('\\', '/'))
    return files_list

def get_embeddings_list():
    embeddings = ["None"]
    embeddings_folders = folder_paths.get_folder_paths("embeddings")
    for embeddings_path in embeddings_folders:
        if os.path.exists(embeddings_path):
            embeddings += recursive_list_files(embeddings_path)
    return embeddings

def get_loras_list():
    loras = ["None"]
    loras_folders = folder_paths.get_folder_paths("loras")
    for loras_path in loras_folders:
        if os.path.exists(loras_path):
            loras += recursive_list_files(loras_path)
    return loras

def weighted_choice(options):
    filtered_options = []
    weights = []
    for option in options:
        opt = option.strip()
        if not opt:
            continue
        # Match format -[name:weight]> or -[weight]> or ->
        match = re.match(r'^-\s*\[([^:]+)(?::(\d+(?:\.\d+)?))?\]\s*>\s*(.*)$', opt)
        if match:
            name = match.group(1)
            weight = float(match.group(2)) if match.group(2) else 1.0
            content = match.group(3).strip()
            weights.append(weight)
            filtered_options.append(content)
        else:
            # Handle simple -> format
            match = re.match(r'^-\s*>\s*(.*)$', opt)
            if match:
                weights.append(1.0)
                filtered_options.append(match.group(1).strip())
            else:
                # Handle old format [weight]option
                match = re.match(r'^\[(\d+)\](.*)$', opt)
                if match:
                    weights.append(int(match.group(1)))
                    filtered_options.append(match.group(2).strip())
                else:
                    weights.append(1.0)
                    filtered_options.append(opt)
    if not filtered_options:
        return ""
    return random.choices(filtered_options, weights=weights, k=1)[0]

def extract_block_options(text):
    lines = text.splitlines()
    result_lines = []
    inside_block = False
    block_lines = []
    current_indent = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{{{'):
            inside_block = True
            block_lines = []
            current_indent = len(line) - len(line.lstrip())
            continue
        elif stripped.startswith('}}}'):
            inside_block = False
            chosen = weighted_choice(block_lines)
            result_lines.append(chosen)
            continue
        if inside_block:
            # Preserve indentation for nested options
            indent = len(line) - len(line.lstrip())
            if indent > current_indent:
                # This is a nested option, add it to the last option
                if block_lines:
                    block_lines[-1] += "\n" + line
                else:
                    block_lines.append(line)
            else:
                block_lines.append(line)
        else:
            result_lines.append(line)
    return '\n'.join(result_lines)

def extract_prompts(text):
    text = extract_block_options(text)
    positive_parts = []
    negative_parts = []
    def replace_nested(match):
        choice = weighted_choice(match.group(1).split('|'))
        pos, neg = split_prompt(choice)
        positive_parts.append(pos)
        if neg:
            negative_parts.append(neg)
        return pos
    pattern = re.compile(r'\{([^{}]*)\}')
    while re.search(pattern, text):
        text = re.sub(pattern, replace_nested, text)
    return text, ", ".join(negative_parts)

def process_text(text):
    result, extra_neg = extract_prompts(text)
    return result, extra_neg

def remove_comments(text):
    lines = text.splitlines()
    clean_lines = [line for line in lines if not line.strip().startswith("#")]
    return "\n".join(clean_lines)

def split_prompt(text):
    parts = text.split("/", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    else:
        return text.strip(), ""

class DefineScenePart:
    CATEGORY = "ESS/Scene"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "type": (["quality", "scene", "action", "girl", "boy"], {"default": "scene"}),
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "positive_embedding": (get_embeddings_list(), {"default": "None"}),
                "negative_embedding": (get_embeddings_list(), {"default": "None"}),
                "lora": (get_loras_list(), {"default": "None"}),
                "lora_weight": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                "scene": ("SCENE",),
            }
        }

    RETURN_TYPES = ("SCENE",)
    FUNCTION = "modify_scene"

    def modify_scene(self, type, positive_prompt, negative_prompt,
                     positive_embedding, negative_embedding, lora, lora_weight=1.0, scene=None):
        # Use seed from global state if available
        global_seed = GlobalState.get_seed()
        if global_seed is not None:
            random.seed(global_seed)

        # Create new scene if none provided
        if scene is None:
            scene = Scene()
            if global_seed is not None:
                scene.set_seed(global_seed)
        else:
            # Create a deep copy of the input scene to avoid modifying the original
            scene = copy.deepcopy(scene)

        # Parse positive and negative prompts
        parser = PromptParser()
        try:
            processed_positive, inverse_positive = parser.parse(positive_prompt)
            processed_negative, inverse_negative = parser.parse(negative_prompt)
            
            # Add inverse content to opposite prompts
            if inverse_positive:
                negative_prompt = (negative_prompt + ", " + inverse_positive) if negative_prompt else inverse_positive
            if inverse_negative:
                positive_prompt = (positive_prompt + ", " + inverse_negative) if positive_prompt else inverse_negative
                
            # Update scene with processed prompts
            scene.add_element(type, processed_positive, processed_negative)
            
            # Add embeddings if specified
            if positive_embedding != "None":
                scene.add_embedding(positive_embedding)
            if negative_embedding != "None":
                scene.add_embedding(negative_embedding, is_negative=True)
                
            # Add LoRA if specified
            if lora != "None":
                scene.add_lora(lora, lora_weight)
                
            return (scene,)
            
        except Exception as e:
            raise ValueError(f"Error processing prompts: {str(e)}")

class SceneDebug:
    CATEGORY = "ESS/Scene"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene": ("SCENE",),
                "_debug_text": ("STRING", {
                    "multiline": True,
                    "default": "Debug information will appear here...",
                    "readonly": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "debug_scene"

    def debug_scene(self, scene, _debug_text, unique_id=None, extra_pnginfo=None):
        if scene is None:
            message = "Scene is empty."
            if unique_id is not None and extra_pnginfo is not None:
                try:
                    workflow_data = extra_pnginfo[0].get("workflow") if isinstance(extra_pnginfo, list) and extra_pnginfo else None
                    if isinstance(workflow_data, dict):
                        nodes = workflow_data.get("nodes", [])
                        node = next((item for item in nodes if str(item.get("id")) == str(unique_id[0])), None)
                        if node is not None:
                            node["widgets_values"] = [message]
                except Exception as exc:
                    print(f"SceneDebug: failed to update widget value ({exc})")
            payload = {"text": (message,), "_debug_text": (message,), "debug_text": (message,)}
            return {"ui": payload, "result": (message,)}

        debug_info = ["Scene Elements:"]
        elements = getattr(scene, "elements", [])
        if not elements:
            debug_info.append("  (none)")
        else:
            for element in elements:
                if isinstance(element, dict):
                    element_type = element.get("type", "<unknown>")
                    positive = element.get("positive", "")
                    negative = element.get("negative", "")
                else:
                    element_type = getattr(element, "type", type(element).__name__)
                    positive = getattr(element, "positive", getattr(element, "positive_prompt", ""))
                    negative = getattr(element, "negative", getattr(element, "negative_prompt", ""))

                debug_info.append(f"  Type: {element_type}")
                if positive:
                    debug_info.append(f"    Positive: {positive}")
                if negative:
                    debug_info.append(f"    Negative: {negative}")

        embeddings_callable = getattr(scene, "get_embeddings", None)
        if callable(embeddings_callable):
            pos_embeddings = scene.get_embeddings(False)
            neg_embeddings = scene.get_embeddings(True)
        else:
            embeddings_attr = getattr(scene, "embeddings", None)
            if isinstance(embeddings_attr, dict):
                pos_embeddings = list(embeddings_attr.get("positive", []))
                neg_embeddings = list(embeddings_attr.get("negative", []))
            elif isinstance(embeddings_attr, (list, tuple)):
                pos_embeddings = list(embeddings_attr)
                neg_embeddings = []
            else:
                pos_embeddings = []
                neg_embeddings = []

        if pos_embeddings or neg_embeddings:
            debug_info.append("")
            debug_info.append("Embeddings:")
            if pos_embeddings:
                debug_info.append(f"  Positive: {', '.join(pos_embeddings)}")
            if neg_embeddings:
                debug_info.append(f"  Negative: {', '.join(neg_embeddings)}")

        lora_callable = getattr(scene, "get_lora_stack", None)
        lora_stack = scene.get_lora_stack() if callable(lora_callable) else list(getattr(scene, "lora_stack", []))
        if lora_stack:
            debug_info.append("")
            debug_info.append("LoRA Stack:")
            for lora in lora_stack:
                if isinstance(lora, dict):
                    name = lora.get("lora_name", "<unknown>")
                    strength = lora.get("strength", 1.0)
                    clip_strength = lora.get("clip_strength", strength)
                else:
                    name = getattr(lora, "name", str(lora))
                    strength = getattr(lora, "weight", getattr(lora, "strength", 1.0))
                    clip_strength = getattr(lora, "clip_strength", strength)
                debug_info.append(f"  {name} (strength: {strength}, clip_strength: {clip_strength})")

        metadata = getattr(scene, "metadata", None)
        if metadata:
            debug_info.append("")
            debug_info.append("Metadata:")
            for key, value in metadata.items():
                debug_info.append(f"  {key}: {value}")

        debug_output = "\n".join(debug_info)

        if unique_id is not None and extra_pnginfo is not None:
            try:
                workflow_data = extra_pnginfo[0].get("workflow") if isinstance(extra_pnginfo, list) and extra_pnginfo else None
                if isinstance(workflow_data, dict):
                    nodes = workflow_data.get("nodes", [])
                    node = next((item for item in nodes if str(item.get("id")) == str(unique_id[0])), None)
                    if node is not None:
                        node["widgets_values"] = [debug_output]
            except Exception as exc:
                print(f"SceneDebug: failed to update widget value ({exc})")

        ui_payload = {"text": (debug_output,), "_debug_text": (debug_output,), "debug_text": (debug_output,)}
        return {"ui": ui_payload, "result": (debug_output,)}
