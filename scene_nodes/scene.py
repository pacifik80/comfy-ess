from typing import List, Dict, Any, Optional
import copy


class Scene:
    """
    A scene aggregates prompt fragments, embeddings, and LoRA strengths that can be
    composed into a final prompt or merged with other scenes.
    """

    def __init__(
        self,
        elements: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embeddings: Optional[Dict[str, List[str]]] = None,
        lora_stack: Optional[List[Any]] = None,
    ) -> None:
        """Initialize a Scene instance."""
        self.elements: List[Dict[str, Any]] = [dict(element) for element in (elements or [])]
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        normalized_seed = self._normalize_seed_value(self.metadata.get("seed"))
        if normalized_seed is not None:
            self.metadata["seed"] = normalized_seed
        elif "seed" in self.metadata:
            self.metadata.pop("seed", None)
        self.embeddings: Dict[str, List[str]] = {"positive": [], "negative": []}
        self.lora_stack: List[Dict[str, Any]] = []

        if embeddings:
            if isinstance(embeddings, dict):
                for embedding in embeddings.get("positive", []):
                    self.add_embedding(embedding)
                for embedding in embeddings.get("negative", []):
                    self.add_embedding(embedding, is_negative=True)
            else:
                for embedding in embeddings:
                    self.add_embedding(embedding)

        if lora_stack:
            for entry in lora_stack:
                normalized = self._normalize_lora_entry(entry)
                if normalized["lora_name"]:
                    self.add_lora(
                        normalized["lora_name"],
                        normalized["strength"],
                        normalized["clip_strength"],
                    )

    @staticmethod
    def _normalize_lora_entry(entry: Any) -> Dict[str, Any]:
        """Convert various LoRA representations into the canonical dict form."""
        if isinstance(entry, dict):
            name = entry.get("lora_name")
            strength = entry.get("strength", 1.0)
            clip_strength = entry.get("clip_strength", strength)
            return {"lora_name": name, "strength": strength, "clip_strength": clip_strength}

        if isinstance(entry, (list, tuple)) and entry:
            name = entry[0]
            strength = entry[1] if len(entry) > 1 else 1.0
            clip_strength = entry[2] if len(entry) > 2 else strength
            return {"lora_name": name, "strength": strength, "clip_strength": clip_strength}

        return {"lora_name": str(entry) if entry else None, "strength": 1.0, "clip_strength": 1.0}

    @staticmethod
    def _normalize_seed_value(seed: Any) -> Optional[int]:
        """Normalize seed inputs to non-negative ints or None."""
        if seed is None:
            return None
        try:
            value = int(seed)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value

    def set_seed(self, seed: Optional[int]) -> None:
        """Store a seed value in the scene metadata."""
        normalized = self._normalize_seed_value(seed)
        if normalized is None:
            self.metadata.pop("seed", None)
        else:
            self.metadata["seed"] = normalized

    def get_seed(self) -> Optional[int]:
        """Return the seed stored in the scene metadata, if any."""
        return self._normalize_seed_value(self.metadata.get("seed"))

    def add_element(self, type_: str, positive: str, negative: str) -> None:
        """Add a new element to the scene."""
        self.elements.append({
            "type": type_,
            "positive": positive,
            "negative": negative,
        })

    def get_elements_by_type(self, type_: str) -> List[Dict[str, Any]]:
        """Get all elements of a specific type."""
        return [el for el in self.elements if el.get("type") == type_]

    def add_embedding(self, embedding: str, is_negative: bool = False) -> None:
        """Register an embedding for the scene."""
        if not embedding:
            return
        key = "negative" if is_negative else "positive"
        target = self.embeddings[key]
        if embedding not in target:
            target.append(embedding)

    def get_embeddings(self, is_negative: bool = False) -> List[str]:
        """Return a copy of the stored embeddings."""
        key = "negative" if is_negative else "positive"
        return list(self.embeddings[key])

    def add_lora(self, lora_name: str, strength: float = 1.0, clip_strength: Optional[float] = None) -> None:
        """Register a LoRA weight for the scene, updating duplicates."""
        if not lora_name:
            return
        if clip_strength is None:
            clip_strength = strength

        for entry in self.lora_stack:
            if entry.get("lora_name") == lora_name:
                entry["strength"] = strength
                entry["clip_strength"] = clip_strength
                return

        self.lora_stack.append({
            "lora_name": lora_name,
            "strength": strength,
            "clip_strength": clip_strength,
        })

    def get_lora_stack(self) -> List[Dict[str, Any]]:
        """Return a copy of the LoRA stack."""
        return [dict(entry) for entry in self.lora_stack]

    def merge_from(self, other: "Scene") -> None:
        """Merge another scene into this one without mutating the source."""
        if not isinstance(other, Scene):
            return

        self.elements.extend(copy.deepcopy(other.elements))

        for embedding in other.get_embeddings(False):
            self.add_embedding(embedding)
        for embedding in other.get_embeddings(True):
            self.add_embedding(embedding, is_negative=True)

        for lora in other.get_lora_stack():
            self.add_lora(
                lora.get("lora_name"),
                lora.get("strength", 1.0),
                lora.get("clip_strength", lora.get("strength", 1.0)),
            )

        if other.metadata:
            other_metadata = copy.deepcopy(other.metadata)
            other_seed = self._normalize_seed_value(other_metadata.pop("seed", None))
            self.metadata.update(other_metadata)
            if self.get_seed() is None and other_seed is not None:
                self.set_seed(other_seed)

    def to_json(self) -> Dict[str, Any]:
        """Convert the scene to a JSON-serializable dictionary."""
        return {
            "elements": copy.deepcopy(self.elements),
            "metadata": copy.deepcopy(self.metadata),
            "embeddings": {
                "positive": self.get_embeddings(False),
                "negative": self.get_embeddings(True),
            },
            "lora_stack": self.get_lora_stack(),
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Scene":
        """Create a Scene instance from a JSON-serialized dictionary."""
        return cls(
            elements=data.get("elements", []),
            metadata=data.get("metadata", {}),
            embeddings=data.get("embeddings", {}),
            lora_stack=data.get("lora_stack", []),
        )
