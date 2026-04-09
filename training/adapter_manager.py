"""
LeanAI — LoRA Adapter Manager
Manages fine-tuned LoRA adapters that personalize the model to YOUR coding style.

Each adapter is a small (~50MB) file that modifies the base model's behavior
without changing the original weights. You can have multiple adapters:
  - "work" adapter trained on your work projects
  - "personal" adapter for side projects
  - "devops" adapter for infrastructure code
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class AdapterInfo:
    """Information about a LoRA adapter."""
    name: str
    path: str
    created: float
    training_examples: int = 0
    base_model: str = ""
    quality_score: float = 0.0
    description: str = ""
    tags: List[str] = field(default_factory=list)
    training_hours: float = 0.0
    version: int = 1

    @property
    def size_mb(self) -> float:
        if os.path.exists(self.path):
            return os.path.getsize(self.path) / (1024 * 1024)
        return 0.0

    @property
    def age_days(self) -> float:
        return (time.time() - self.created) / 86400

    def to_dict(self) -> dict:
        return {
            "name": self.name, "path": self.path, "created": self.created,
            "training_examples": self.training_examples,
            "base_model": self.base_model,
            "quality_score": self.quality_score,
            "description": self.description,
            "tags": self.tags,
            "training_hours": self.training_hours,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdapterInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        size = f"{self.size_mb:.1f}MB" if self.size_mb > 0 else "not built"
        return (
            f"{self.name} v{self.version} | {self.training_examples} examples | "
            f"{size} | quality: {self.quality_score:.0%} | {self.age_days:.0f} days old"
        )


class AdapterManager:
    """
    Manages LoRA adapters for personalized fine-tuning.
    
    Usage:
        mgr = AdapterManager()
        
        # Create a new adapter config
        mgr.create("work", base_model="qwen-7b", description="Work projects")
        
        # After training produces an adapter file
        mgr.register_trained("work", adapter_path, num_examples=200)
        
        # List adapters
        print(mgr.list_adapters())
        
        # Get active adapter path for model loading
        path = mgr.get_active_adapter_path()
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or str(Path.home() / ".leanai" / "finetune" / "adapters")
        os.makedirs(self.data_dir, exist_ok=True)
        self._adapters: Dict[str, AdapterInfo] = {}
        self._active_adapter: Optional[str] = None
        self._load_registry()

    def create(self, name: str, base_model: str = "qwen-7b",
               description: str = "", tags: Optional[List[str]] = None) -> AdapterInfo:
        """Create a new adapter configuration (doesn't train yet)."""
        adapter_dir = os.path.join(self.data_dir, name)
        os.makedirs(adapter_dir, exist_ok=True)

        adapter = AdapterInfo(
            name=name,
            path=adapter_dir,
            created=time.time(),
            base_model=base_model,
            description=description or f"LoRA adapter: {name}",
            tags=tags or [],
        )
        self._adapters[name] = adapter
        self._save_registry()
        return adapter

    def register_trained(self, name: str, adapter_path: str,
                         num_examples: int = 0, quality_score: float = 0.0,
                         training_hours: float = 0.0):
        """Register a completed training run for an adapter."""
        if name not in self._adapters:
            self.create(name)

        adapter = self._adapters[name]
        adapter.path = adapter_path
        adapter.training_examples = num_examples
        adapter.quality_score = quality_score
        adapter.training_hours = training_hours
        adapter.version += 1
        self._save_registry()

    def set_active(self, name: str) -> bool:
        """Set the active adapter."""
        if name not in self._adapters:
            return False
        self._active_adapter = name
        self._save_registry()
        return True

    def deactivate(self):
        """Deactivate all adapters (use base model only)."""
        self._active_adapter = None
        self._save_registry()

    def get_active(self) -> Optional[AdapterInfo]:
        """Get the currently active adapter."""
        if self._active_adapter and self._active_adapter in self._adapters:
            return self._adapters[self._active_adapter]
        return None

    def get_active_adapter_path(self) -> Optional[str]:
        """Get the path to the active adapter weights."""
        adapter = self.get_active()
        if adapter and os.path.exists(adapter.path):
            # Look for adapter files in the directory
            for f in os.listdir(adapter.path):
                if f.endswith((".bin", ".safetensors", ".gguf")):
                    return os.path.join(adapter.path, f)
        return None

    def get_adapter(self, name: str) -> Optional[AdapterInfo]:
        return self._adapters.get(name)

    def list_adapters(self) -> str:
        """Human-readable list of adapters."""
        if not self._adapters:
            return "No adapters created yet. Use /finetune create <name> to start."

        lines = ["LoRA Adapters:"]
        for name, adapter in self._adapters.items():
            active = " ← ACTIVE" if name == self._active_adapter else ""
            lines.append(f"  {adapter.summary()}{active}")
        return "\n".join(lines)

    def delete(self, name: str) -> bool:
        """Delete an adapter."""
        if name not in self._adapters:
            return False
        if name == self._active_adapter:
            self._active_adapter = None
        del self._adapters[name]
        self._save_registry()
        return True

    # ── Persistence ───────────────────────────────────────────────

    def _registry_path(self) -> str:
        return os.path.join(self.data_dir, "adapter_registry.json")

    def _save_registry(self):
        data = {
            "active": self._active_adapter,
            "adapters": {k: v.to_dict() for k, v in self._adapters.items()},
        }
        try:
            with open(self._registry_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_registry(self):
        path = self._registry_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._active_adapter = data.get("active")
            for k, v in data.get("adapters", {}).items():
                self._adapters[k] = AdapterInfo.from_dict(v)
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "total_adapters": len(self._adapters),
            "active": self._active_adapter,
            "adapters": {k: v.to_dict() for k, v in self._adapters.items()},
        }
