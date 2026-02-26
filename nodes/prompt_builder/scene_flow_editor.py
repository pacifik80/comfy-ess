from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from .prompt_parser import PromptParser


class SceneFlowEditor:
    """
    Generate one scene prompt from a graph script authored in the Scene Flow Editor UI.

    Node types:
      - element: prompt template leaf
      - sequential: concatenates all inbound branches
      - random: selects one inbound branch by edge weight
      - output: concatenates all inbound branches (final scene output)
    """

    CATEGORY = "ESS/PromptBuilder"
    FUNCTION = "generate"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "debug_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_script": (
                    "ESS_SCENE_FLOW_EDITOR",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Open Scene Flow Editor and design a flow graph...",
                        "height": 260,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "parse_templates": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "parse",
                        "label_off": "raw",
                        "tooltip": "When enabled, parse element templates with ESS parser.",
                    },
                ),
            }
        }

    def generate(self, flow_script: str, seed: int, parse_templates: bool) -> Tuple[str, str, str]:
        payload = self._normalize_payload(flow_script or "")
        parser = PromptParser(seed=seed)
        nodes = payload["nodes"]
        edges = payload["edges"]
        sections = payload["sections"]

        node_by_id = {str(node["id"]): node for node in nodes}
        inbound: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for index, edge in enumerate(edges):
            source = str(edge.get("from", ""))
            target = str(edge.get("to", ""))
            if source not in node_by_id or target not in node_by_id:
                continue
            if edge.get("enabled", True) is False:
                continue
            edge_copy = dict(edge)
            edge_copy["_order"] = int(edge.get("order", index))
            inbound[target].append(edge_copy)

        for target_id in inbound:
            inbound[target_id].sort(key=lambda item: (item.get("_order", 0), str(item.get("id", ""))))

        scene_section_id = self._pick_scene_section_id(payload, sections)
        output_id = self._pick_output_node_id(payload, nodes, scene_section_id)

        trace: List[Dict[str, Any]] = []
        eval_cache: Dict[str, Tuple[str, str]] = {}
        eval_stack: set[str] = set()

        def stable_seed(token: str) -> int:
            digest = hashlib.sha256(f"{seed}|{token}".encode("utf-8")).digest()
            return int.from_bytes(digest[:8], "big", signed=False)

        def join_text(parts: List[str]) -> str:
            return "\n".join([part.strip() for part in parts if part and part.strip()])

        def evaluate_node(node_id: str) -> Tuple[str, str]:
            node_id = str(node_id)
            cached = eval_cache.get(node_id)
            if cached is not None:
                return cached

            if node_id in eval_stack:
                raise ValueError(f"Cycle detected in flow graph at node '{node_id}'.")

            node = node_by_id.get(node_id)
            if node is None:
                return "", ""

            eval_stack.add(node_id)
            node_type = str(node.get("type", "element")).lower()
            title = str(node.get("title", node_type or "node"))

            try:
                if node_type == "element":
                    template = str(node.get("template", "") or "")
                    if parse_templates:
                        pos, neg = parser.parse(template)
                    else:
                        pos, neg = template.strip(), ""
                    trace.append(
                        {
                            "node_id": node_id,
                            "node_type": node_type,
                            "title": title,
                            "action": "emit",
                            "positive_preview": pos[:120],
                            "negative_preview": neg[:120],
                        }
                    )
                elif node_type == "random":
                    candidates = []
                    for edge in inbound.get(node_id, []):
                        try:
                            weight = float(edge.get("weight", 1.0))
                        except (TypeError, ValueError):
                            weight = 1.0
                        if weight <= 0:
                            continue
                        candidates.append((edge, weight))

                    if not candidates:
                        pos, neg = "", ""
                        trace.append(
                            {
                                "node_id": node_id,
                                "node_type": node_type,
                                "title": title,
                                "action": "empty",
                                "reason": "no eligible inbound edges",
                            }
                        )
                    else:
                        rng = random.Random(stable_seed(f"random-node:{node_id}"))
                        selected_index = rng.choices(
                            list(range(len(candidates))),
                            weights=[weight for _, weight in candidates],
                            k=1,
                        )[0]
                        selected_edge, _selected_weight = candidates[selected_index]
                        selected_source = str(selected_edge.get("from", ""))
                        pos, neg = evaluate_node(selected_source)

                        trace.append(
                            {
                                "node_id": node_id,
                                "node_type": node_type,
                                "title": title,
                                "action": "random_pick",
                                "selected_source": selected_source,
                                "candidates": [
                                    {
                                        "edge_id": str(edge.get("id", "")),
                                        "source": str(edge.get("from", "")),
                                        "weight": weight,
                                    }
                                    for edge, weight in candidates
                                ],
                            }
                        )
                else:
                    child_positive: List[str] = []
                    child_negative: List[str] = []
                    for edge in inbound.get(node_id, []):
                        source_id = str(edge.get("from", ""))
                        pos_part, neg_part = evaluate_node(source_id)
                        if pos_part.strip():
                            child_positive.append(pos_part)
                        if neg_part.strip():
                            child_negative.append(neg_part)

                    pos = join_text(child_positive)
                    neg = join_text(child_negative)
                    trace.append(
                        {
                            "node_id": node_id,
                            "node_type": node_type,
                            "title": title,
                            "action": "concat",
                            "inputs": [str(edge.get("from", "")) for edge in inbound.get(node_id, [])],
                        }
                    )

                eval_cache[node_id] = (pos, neg)
                return pos, neg
            finally:
                eval_stack.remove(node_id)

        if not output_id:
            debug = {
                "error": "no output node found",
                "scene_section_id": scene_section_id,
                "sections_count": len(sections),
                "nodes_count": len(nodes),
                "edges_count": len(edges),
            }
            return "", "", json.dumps(debug, ensure_ascii=False, indent=2)

        try:
            positive, negative = evaluate_node(output_id)
            debug = {
                "scene_section_id": scene_section_id,
                "scene_output_id": output_id,
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "trace": trace,
            }
            return positive.strip(), negative.strip(), json.dumps(debug, ensure_ascii=False, indent=2)
        except Exception as exc:
            debug = {
                "error": str(exc),
                "scene_section_id": scene_section_id,
                "scene_output_id": output_id,
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "trace": trace,
            }
            return "", "", json.dumps(debug, ensure_ascii=False, indent=2)

    def _pick_scene_section_id(self, payload: Dict[str, Any], sections: List[Dict[str, Any]]) -> Optional[str]:
        candidate = payload.get("scene_section_id")
        if candidate is not None:
            return str(candidate)
        for section in sections:
            name = str(section.get("name", "")).strip().lower()
            section_id = str(section.get("id", "")).strip()
            if name == "scene" and section_id:
                return section_id
            if section_id == "scene":
                return section_id
        if sections:
            fallback = str(sections[0].get("id", "")).strip()
            return fallback or None
        return None

    def _pick_output_node_id(
        self,
        payload: Dict[str, Any],
        nodes: List[Dict[str, Any]],
        scene_section_id: Optional[str],
    ) -> Optional[str]:
        explicit_output = payload.get("scene_output_id")
        if explicit_output is not None:
            return str(explicit_output)

        for node in nodes:
            if str(node.get("type", "")).lower() != "output":
                continue
            if scene_section_id is None:
                return str(node.get("id"))
            if str(node.get("section_id", "")) == scene_section_id:
                return str(node.get("id"))

        for node in nodes:
            if str(node.get("type", "")).lower() == "output":
                return str(node.get("id"))
        return None

    def _normalize_payload(self, raw: str) -> Dict[str, Any]:
        payload = {}
        if raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = {}

        sections = payload.get("sections", [])
        if not isinstance(sections, list):
            sections = []
        normalized_sections: List[Dict[str, Any]] = []
        seen_section_ids: set[str] = set()
        for index, section in enumerate(sections):
            if not isinstance(section, dict):
                continue
            section_id = str(section.get("id", f"section_{index + 1}")).strip()
            if not section_id or section_id in seen_section_ids:
                continue
            seen_section_ids.add(section_id)
            normalized_sections.append(
                {
                    "id": section_id,
                    "name": str(section.get("name", section_id)).strip() or section_id,
                }
            )

        if not normalized_sections:
            normalized_sections = [{"id": "scene", "name": "scene"}]
            seen_section_ids = {"scene"}

        nodes = payload.get("nodes", [])
        if not isinstance(nodes, list):
            nodes = []
        normalized_nodes: List[Dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", f"node_{index + 1}")).strip()
            if not node_id or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            node_type = str(node.get("type", "element")).strip().lower()
            if node_type not in {"element", "sequential", "random", "output"}:
                node_type = "element"
            section_id = str(node.get("section_id", "")).strip()
            if section_id not in seen_section_ids:
                section_id = normalized_sections[0]["id"]
            normalized_nodes.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "section_id": section_id,
                    "title": str(node.get("title", node_type)).strip() or node_type,
                    "template": str(node.get("template", "")),
                }
            )

        if not normalized_nodes:
            normalized_nodes = [
                {
                    "id": "scene_out",
                    "type": "output",
                    "section_id": normalized_sections[0]["id"],
                    "title": "Scene Output",
                    "template": "",
                }
            ]

        edges = payload.get("edges", [])
        if not isinstance(edges, list):
            edges = []
        normalized_edges: List[Dict[str, Any]] = []
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("from", "")).strip()
            target = str(edge.get("to", "")).strip()
            if not source or not target:
                continue
            normalized_edges.append(
                {
                    "id": str(edge.get("id", f"edge_{index + 1}")),
                    "from": source,
                    "to": target,
                    "weight": edge.get("weight", 1.0),
                    "order": edge.get("order", index),
                    "enabled": bool(edge.get("enabled", True)),
                }
            )

        return {
            "version": int(payload.get("version", 1)),
            "sections": normalized_sections,
            "nodes": normalized_nodes,
            "edges": normalized_edges,
            "scene_section_id": payload.get("scene_section_id"),
            "scene_output_id": payload.get("scene_output_id"),
        }
