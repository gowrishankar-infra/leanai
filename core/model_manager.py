"""
LeanAI — Model Manager
Downloads, manages, and auto-switches between local models.

Tier System:
  - FAST (7B):  ~25 sec response, good for simple questions
  - QUALITY (32B): ~90 sec response, near-GPT-4 for coding
  - AUTO: Liquid router decides based on query complexity

The key insight: you don't need the 32B model for "what is my name?"
and you don't want the 7B model for "design a microservice architecture."
Auto mode gives you the best of both worlds.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


# ── Model Registry ────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    name: str                  # display name
    filename: str              # GGUF filename
    repo_id: str               # HuggingFace repo
    size_gb: float             # approximate file size
    ram_needed_gb: float       # RAM needed when loaded
    quality_score: int         # relative quality (1-100)
    speed_label: str           # "fast", "medium", "slow"
    prompt_format: str         # "chatml", "phi3", etc.
    description: str = ""

    @property
    def full_path(self) -> str:
        return str(Path.home() / ".leanai" / "models" / self.filename)

    @property
    def is_downloaded(self) -> bool:
        if os.path.exists(self.full_path):
            return True
        # Check for alternate filenames (user might have different quantization)
        models_dir = Path.home() / ".leanai" / "models"
        if not models_dir.exists():
            return False
        # Match by model size key (e.g. "7b", "14b", "32b")
        size_key = ""
        for part in self.filename.lower().replace("-", "").replace("_", "").split("."):
            for s in ["7b", "14b", "32b"]:
                if s in part:
                    size_key = s
                    break
        if size_key:
            for f in models_dir.glob("*.gguf"):
                fname = f.name.lower().replace("-", "").replace("_", "")
                if size_key in fname and "qwen" in fname:
                    return True
        return False

    @property
    def resolved_path(self) -> str:
        """Find the actual file path, handling alternate filenames."""
        if os.path.exists(self.full_path):
            return self.full_path
        models_dir = Path.home() / ".leanai" / "models"
        if not models_dir.exists():
            return self.full_path
        size_key = ""
        for part in self.filename.lower().replace("-", "").replace("_", "").split("."):
            for s in ["7b", "14b", "32b"]:
                if s in part:
                    size_key = s
                    break
        if size_key:
            for f in models_dir.glob("*.gguf"):
                fname = f.name.lower().replace("-", "").replace("_", "")
                if size_key in fname and "qwen" in fname:
                    return str(f)
        return self.full_path


# Models that work well for coding on consumer hardware
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "qwen-7b": ModelInfo(
        name="Qwen2.5 Coder 7B",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        size_gb=4.5,
        ram_needed_gb=6,
        quality_score=70,
        speed_label="fast",
        prompt_format="chatml",
        description="Fast, good for simple tasks. ~25s on CPU.",
    ),
    "qwen-14b": ModelInfo(
        name="Qwen2.5 Coder 14B",
        filename="qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        repo_id="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        size_gb=8.5,
        ram_needed_gb=10,
        quality_score=82,
        speed_label="medium",
        prompt_format="chatml",
        description="Balanced quality/speed. ~45s on CPU.",
    ),
    "qwen-32b": ModelInfo(
        name="Qwen2.5 Coder 32B",
        filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        size_gb=18.0,
        ram_needed_gb=20,
        quality_score=92,
        speed_label="slow",
        prompt_format="chatml",
        description="Near-GPT-4 for coding. ~90s on CPU. 92% HumanEval.",
    ),
}


# ── Complexity classifier ─────────────────────────────────────────

COMPLEX_INDICATORS = {
    # Architecture & design
    "architect", "design", "microservice", "distributed", "scalable",
    "system design", "trade-off", "pattern",
    # Multi-step reasoning
    "step by step", "explain in detail", "comprehensive", "thorough",
    "analyze", "compare", "evaluate", "optimize", "explain this",
    "explain the", "break down", "walk through", "how does this work",
    # DevOps & CI/CD (needs deep knowledge)
    "pipeline", "ci/cd", "cicd", "devops", "azure devops", "github actions",
    "jenkins", "gitlab ci", "kubernetes", "k8s", "docker compose",
    "terraform", "ansible", "helm", "deployment", "infrastructure",
    "yaml", "dockerfile", "nginx", "load balancer",
    # Complex code tasks
    "refactor", "redesign", "implement from scratch", "build a complete",
    "multi-file", "full application", "production-ready",
    "algorithm", "data structure", "dynamic programming",
    "concurrent", "async", "parallel", "thread",
    "knapsack", "graph traversal", "binary tree",
    # Complex data structures
    "lru", "cache", "ttl", "expiration", "thread-safe", "thread safe",
    "lock-free", "lockfree", "atomic",
    "rate limiter", "rate limit", "circuit breaker",
    "load balancer", "connection pool",
    # Debugging complex issues
    "race condition", "memory leak", "deadlock", "performance issue",
    "security vulnerability",
    # Architecture patterns
    "decorator", "singleton", "factory", "observer", "strategy",
    "middleware", "event driven", "pub sub",
    "message queue", "websocket",
    # Review & explain
    "review this", "what's wrong", "improve this", "best practice",
}

SIMPLE_INDICATORS = {
    # Simple lookups
    "what is", "who is", "define", "meaning of",
    # Simple code
    "hello world", "print", "for loop", "if else",
    "simple function", "basic script",
    # Memory/personal
    "my name", "remember", "you know",
    # Quick tasks
    "fix this", "syntax", "typo", "rename",
}


def classify_complexity(query: str) -> str:
    """
    Classify query complexity.
    Returns: "simple", "medium", or "complex"
    """
    lower = query.lower()
    word_count = len(lower.split())

    # Check for complex indicators
    complex_matches = sum(1 for ind in COMPLEX_INDICATORS if ind in lower)
    simple_matches = sum(1 for ind in SIMPLE_INDICATORS if ind in lower)

    # Short queries are usually simple
    if word_count <= 5 and complex_matches == 0:
        return "simple"

    # Lots of code in the query (pasted code) = medium to complex
    if lower.count("\n") > 5 or word_count > 50:
        return "complex"

    if complex_matches >= 2:
        return "complex"
    elif complex_matches == 1 and word_count > 15:
        return "complex"
    elif simple_matches >= 1:
        return "simple"
    elif word_count > 20:
        return "medium"
    else:
        return "medium"


# ── Model Manager ─────────────────────────────────────────────────

class ModelManager:
    """
    Manages multiple local models and routes queries to the best one.
    
    Usage:
        manager = ModelManager()
        manager.list_models()          # see what's available
        manager.download("qwen-32b")   # download the 32B model
        manager.set_mode("auto")       # auto-switch by complexity
        
        model_key = manager.select_model("design a microservice")  # -> "qwen-32b"
        model_key = manager.select_model("what is my name")        # -> "qwen-7b"
    """

    def __init__(self):
        self.models = dict(MODEL_REGISTRY)
        self._mode = "auto"  # "fast", "quality", "auto", or a specific model key
        self._state_path = str(Path.home() / ".leanai" / "model_manager.json")
        self._stats = {
            "queries_routed": 0,
            "fast_count": 0,
            "quality_count": 0,
        }
        self._load_state()

    # ── Model listing ─────────────────────────────────────────────

    def list_models(self) -> str:
        """List all available models with download status."""
        lines = ["Available Models:", ""]
        for key, model in self.models.items():
            status = "DOWNLOADED" if model.is_downloaded else f"not downloaded ({model.size_gb:.1f} GB)"
            lines.append(
                f"  {key:12s}  {model.name:30s}  {model.speed_label:6s}  "
                f"quality:{model.quality_score}%  {status}"
            )
        lines.append(f"\nCurrent mode: {self._mode}")
        downloaded = [k for k, m in self.models.items() if m.is_downloaded]
        lines.append(f"Downloaded: {', '.join(downloaded) or 'none'}")
        return "\n".join(lines)

    def get_downloaded_models(self) -> List[str]:
        """Get keys of all downloaded models."""
        return [k for k, m in self.models.items() if m.is_downloaded]

    def get_model_info(self, key: str) -> Optional[ModelInfo]:
        """Get info about a specific model."""
        return self.models.get(key)

    # ── Downloading ───────────────────────────────────────────────

    def download(self, model_key: str) -> Tuple[bool, str]:
        """
        Download a model from HuggingFace.
        Returns (success, message).
        """
        model = self.models.get(model_key)
        if not model:
            return False, f"Unknown model: {model_key}. Available: {list(self.models.keys())}"

        if model.is_downloaded:
            return True, f"{model.name} already downloaded at {model.full_path}"

        models_dir = str(Path.home() / ".leanai" / "models")
        os.makedirs(models_dir, exist_ok=True)

        print(f"Downloading {model.name} ({model.size_gb:.1f} GB)...")
        print(f"From: {model.repo_id}")
        print(f"To: {models_dir}")
        print(f"This may take a while on slow connections.\n")

        try:
            # Write a download script to avoid path escaping issues
            import tempfile
            script = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            script.write(
                "from huggingface_hub import hf_hub_download\n"
                "import os\n"
                f"repo = '{model.repo_id}'\n"
                f"filename = '{model.filename}'\n"
                f"dest = os.path.join(os.path.expanduser('~'), '.leanai', 'models')\n"
                "os.makedirs(dest, exist_ok=True)\n"
                "print(f'Downloading to {{dest}}')\n"
                "hf_hub_download(repo, filename, local_dir=dest)\n"
                "print('Download complete!')\n"
            )
            script.close()

            result = subprocess.run(
                [sys.executable, script.name],
                capture_output=False, text=True, timeout=7200,
            )
            os.unlink(script.name)

            if model.is_downloaded:
                return True, f"Downloaded {model.name} successfully!"
            else:
                # Check alternate locations
                alt_path = os.path.join(models_dir, model.filename)
                if os.path.exists(alt_path):
                    return True, f"Downloaded {model.name} successfully!"
                return False, f"Download may have failed. Check {models_dir} manually."

        except subprocess.TimeoutExpired:
            return False, "Download timed out. Try downloading manually."
        except Exception as e:
            return False, f"Download error: {e}"

    def download_command(self, model_key: str) -> str:
        """Get the manual download command for a model."""
        model = self.models.get(model_key)
        if not model:
            return f"Unknown model: {model_key}"
        return (
            f"# Download {model.name} ({model.size_gb:.1f} GB)\n"
            f"pip install huggingface_hub\n"
            f'python -c "from huggingface_hub import hf_hub_download; '
            f"hf_hub_download('{model.repo_id}', '{model.filename}', "
            f"local_dir='{Path.home() / '.leanai' / 'models'}')\""
        )

    # ── Mode & Selection ──────────────────────────────────────────

    def set_mode(self, mode: str):
        """Set routing mode: 'fast', 'quality', 'auto', or a model key."""
        valid = {"fast", "quality", "auto"} | set(self.models.keys())
        if mode not in valid:
            raise ValueError(f"Invalid mode: {mode}. Valid: {valid}")
        self._mode = mode
        self._save_state()

    @property
    def mode(self) -> str:
        return self._mode

    def select_model(self, query: str) -> str:
        """
        Select the best model for a query based on current mode.
        Returns the model key to use.
        """
        downloaded = self.get_downloaded_models()
        self._stats["queries_routed"] += 1
        if not downloaded:
            self._stats["fast_count"] += 1
            return "qwen-7b"  # default even if not downloaded

        # Explicit mode
        if self._mode == "fast":
            self._stats["fast_count"] += 1
            return self._pick_fastest(downloaded)

        elif self._mode == "quality":
            self._stats["quality_count"] += 1
            return self._pick_best_quality(downloaded)

        elif self._mode in self.models:
            return self._mode if self._mode in downloaded else self._pick_fastest(downloaded)

        # Auto mode — use complexity classifier
        complexity = classify_complexity(query)

        if complexity == "simple":
            self._stats["fast_count"] += 1
            return self._pick_fastest(downloaded)
        elif complexity == "complex":
            self._stats["quality_count"] += 1
            return self._pick_best_quality(downloaded)
        else:
            # Medium — use mid-tier if available, else fastest
            if "qwen-14b" in downloaded:
                return "qwen-14b"
            elif "qwen-32b" in downloaded:
                self._stats["quality_count"] += 1
                return "qwen-32b"
            else:
                self._stats["fast_count"] += 1
                return self._pick_fastest(downloaded)

    def _pick_fastest(self, downloaded: List[str]) -> str:
        """Pick the fastest (smallest) downloaded model."""
        by_speed = sorted(downloaded, key=lambda k: self.models[k].size_gb)
        return by_speed[0]

    def _pick_best_quality(self, downloaded: List[str]) -> str:
        """Pick the highest quality downloaded model."""
        by_quality = sorted(downloaded, key=lambda k: self.models[k].quality_score, reverse=True)
        return by_quality[0]

    # ── Model path resolution ─────────────────────────────────────

    def get_model_path(self, model_key: str) -> Optional[str]:
        """Get the filesystem path for a model."""
        model = self.models.get(model_key)
        if not model:
            return None
        if model.is_downloaded:
            return model.resolved_path
        return None

    def get_prompt_format(self, model_key: str) -> str:
        """Get the prompt format for a model."""
        model = self.models.get(model_key)
        return model.prompt_format if model else "chatml"

    # ── Persistence ───────────────────────────────────────────────

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w") as f:
                json.dump({"mode": self._mode, "stats": self._stats}, f)
        except Exception:
            pass

    def _load_state(self):
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                self._mode = data.get("mode", "auto")
                self._stats = data.get("stats", self._stats)
            except Exception:
                pass

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        downloaded = self.get_downloaded_models()
        return {
            "mode": self._mode,
            "downloaded_models": downloaded,
            "available_models": list(self.models.keys()),
            **self._stats,
        }
