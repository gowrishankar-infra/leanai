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


def get_leanai_home():
    """Get LeanAI home directory. Supports LEANAI_HOME env var."""
    return Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai")))


def get_models_dir():
    """Get models directory. Supports LEANAI_MODELS env var for reusing LM Studio models."""
    custom = os.environ.get("LEANAI_MODELS")
    if custom:
        return Path(custom)
    return get_leanai_home() / "models"


def _guess_format(filename: str) -> str:
    """Best-guess prompt format from a filename (mirrors the engine's
    auto-detection). The engine re-detects on load, so this is for display."""
    name = filename.lower()
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "chatml"
    if "phi" in name:
        return "phi3"
    if "llama" in name:
        return "llama3"
    if "mistral" in name or "nemo" in name:
        return "chatml"
    return "chatml"


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
    is_local: bool = False     # auto-discovered local .gguf (not curated registry)
    is_remote: bool = False    # served over HTTP (OpenAI-compatible / Ollama)
    model_uri: str = ""        # for remote: "remote:<alias>" identifier

    @property
    def full_path(self) -> str:
        if self.is_remote:
            return self.model_uri
        return str(get_models_dir() / self.filename)

    @property
    def is_downloaded(self) -> bool:
        if self.is_remote:
            return True  # remote endpoints are always "available" (no download)
        if os.path.exists(self.full_path):
            return True
        # Check for alternate filenames (user might have different quantization)
        models_dir = get_models_dir()
        if not models_dir.exists():
            return False
        # Match by model family + size key
        fname_lower = self.filename.lower()
        # Gemma 4 26B detection
        if "gemma" in fname_lower and "26b" in fname_lower:
            for f in models_dir.rglob("*.gguf"):
                fl = f.name.lower()
                if "gemma" in fl and "26b" in fl:
                    return True
            return False
        # Qwen3.5-27B detection
        if "qwen3.5" in fname_lower and "27b" in fname_lower:
            for f in models_dir.rglob("*.gguf"):
                fl = f.name.lower()
                if "qwen3.5" in fl and "27b" in fl:
                    return True
            return False
        # Qwen3-Coder-Next detection
        if "qwen3" in fname_lower and "coder" in fname_lower:
            for f in models_dir.rglob("*.gguf"):
                fl = f.name.lower()
                if "qwen3" in fl and "coder" in fl:
                    return True
            return False
        # Standard size key matching (7b, 14b, 32b)
        size_key = ""
        for part in fname_lower.replace("-", "").replace("_", "").split("."):
            for s in ["7b", "14b", "32b"]:
                if s in part:
                    size_key = s
                    break
        if size_key:
            for f in models_dir.rglob("*.gguf"):
                fname = f.name.lower().replace("-", "").replace("_", "")
                if size_key in fname and "qwen" in fname:
                    return True
        return False

    @property
    def resolved_path(self) -> str:
        """Find the actual file path, handling alternate filenames."""
        if self.is_remote:
            return self.model_uri
        if os.path.exists(self.full_path):
            return self.full_path
        models_dir = get_models_dir()
        if not models_dir.exists():
            return self.full_path
        # Gemma 4 26B detection
        fname_lower = self.filename.lower()
        if "gemma" in fname_lower and "26b" in fname_lower:
            for f in models_dir.rglob("*.gguf"):
                fl = f.name.lower()
                if "gemma" in fl and "26b" in fl:
                    return str(f)
            return self.full_path
        # Qwen3.5-27B detection
        if "qwen3.5" in fname_lower and "27b" in fname_lower:
            for f in models_dir.rglob("*.gguf"):
                fl = f.name.lower()
                if "qwen3.5" in fl and "27b" in fl:
                    return str(f)
            return self.full_path
        # Qwen3-Coder-Next detection
        if "qwen3" in fname_lower and "coder" in fname_lower:
            for f in models_dir.rglob("*.gguf"):
                fl = f.name.lower()
                if "qwen3" in fl and "coder" in fl:
                    return str(f)
            return self.full_path
        # Standard size key matching
        size_key = ""
        for part in fname_lower.replace("-", "").replace("_", "").split("."):
            for s in ["7b", "14b", "32b"]:
                if s in part:
                    size_key = s
                    break
        if size_key:
            for f in models_dir.rglob("*.gguf"):
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
    "qwen3-coder": ModelInfo(
        name="Qwen3 Coder Next (80B MoE, 3B active)",
        filename="Qwen3-Coder-Next-UD-Q3_K_XL.gguf",
        repo_id="unsloth/Qwen3-Coder-Next-GGUF",
        size_gb=30.0,
        ram_needed_gb=32,
        quality_score=97,
        speed_label="fast",
        prompt_format="chatml",
        description="Near-Sonnet quality. 80B MoE but only 3B active = fast. Needs 32GB RAM.",
    ),
    "qwen3-coder-q4": ModelInfo(
        name="Qwen3 Coder Next Q4 (80B MoE, best quality)",
        filename="Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
        repo_id="unsloth/Qwen3-Coder-Next-GGUF",
        size_gb=46.0,
        ram_needed_gb=48,
        quality_score=98,
        speed_label="fast",
        prompt_format="chatml",
        description="Best local coding model. Needs 48GB RAM. Near-Sonnet 4.5 quality.",
    ),
    "qwen35-27b": ModelInfo(
        name="Qwen3.5 27B (newest, best dense model)",
        filename="Qwen3.5-27B-Q4_K_M.gguf",
        repo_id="unsloth/Qwen3.5-27B-GGUF",
        size_gb=16.7,
        ram_needed_gb=20,
        quality_score=96,
        speed_label="medium",
        prompt_format="chatml",
        description="Latest Qwen3.5. Best dense model for coding. Thinking mode. 16.7GB Q4.",
    ),
    "gemma4-26b": ModelInfo(
        name="Gemma 4 26B-A4B (fastest quality, best for UI/frontend)",
        filename="gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        repo_id="unsloth/gemma-4-26B-A4B-it-GGUF",
        size_gb=16.9,
        ram_needed_gb=20,
        quality_score=95,
        speed_label="fast",
        prompt_format="gemma",
        description="Google's best local model. MoE 4B active = very fast. Rock solid under quantization.",
    ),
    # ── User-added local models (exact filenames present in ~/.leanai/models) ──
    "qwen-coder-josie": ModelInfo(
        name="Qwen2.5 Coder 7B (Josiefied, uncensored)",
        filename="Josiefied-Qwen2.5-Coder-7B-Instruct-v1.Q6_K.gguf",
        repo_id="Josiefied/Qwen2.5-Coder-7B-Instruct-v1-GGUF",
        size_gb=6.3,
        ram_needed_gb=8,
        quality_score=72,
        speed_label="fast",
        prompt_format="chatml",
        description="Code-specialized 7B, uncensored community build (Q6). Good for C/Python.",
    ),
    "llama-8b": ModelInfo(
        name="Meta Llama 3.1 8B Instruct",
        filename="Meta-Llama-3.1-8B-Instruct.Q6_K.gguf",
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        size_gb=6.6,
        ram_needed_gb=8,
        quality_score=74,
        speed_label="fast",
        prompt_format="llama3",
        description="General-purpose 8B (Q6). Broader knowledge than the coder models.",
    ),
    "mistral-nemo": ModelInfo(
        name="Mistral Nemo 12B Instruct",
        filename="mistralai-Mistral-Nemo-Instruct-2407-extensive-BP-12B.Q6_K.gguf",
        repo_id="bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        size_gb=9.6,
        ram_needed_gb=12,
        quality_score=80,
        speed_label="medium",
        prompt_format="chatml",
        description="12B general model (Q6). Higher quality, heavier/slower on 4GB VRAM.",
    ),
}


# ── Complexity classifier ─────────────────────────────────────────

COMPLEX_INDICATORS = {
    # Architecture & design
    "architect", "design", "microservice", "distributed", "scalable",
    "system design", "trade-off", "pattern",
    # Multi-step reasoning & explanation
    "step by step", "explain in detail", "comprehensive", "thorough",
    "analyze", "compare", "evaluate", "optimize", "explain this",
    "explain the", "break down", "walk through", "how does this work",
    "explain", "review", "audit", "critique",
    # DevOps & CI/CD
    "pipeline", "ci/cd", "cicd", "devops", "azure devops", "github actions",
    "jenkins", "gitlab ci", "kubernetes", "k8s", "docker compose",
    "terraform", "ansible", "helm", "deployment", "infrastructure",
    "yaml", "dockerfile", "nginx", "load balancer",
    # API & Backend
    "rest api", "graphql", "endpoint", "middleware", "authentication",
    "authorization", "oauth", "jwt", "cors", "rate limit",
    "api gateway", "reverse proxy", "grpc",
    # Database
    "sql injection", "orm", "migration", "transaction", "index",
    "connection pool", "query optimization", "normalization",
    "nosql", "redis", "mongodb", "postgres", "mysql",
    # Complex code tasks
    "refactor", "redesign", "implement from scratch", "build a complete",
    "multi-file", "full application", "production-ready",
    "algorithm", "data structure", "dynamic programming",
    "concurrent", "async", "parallel", "thread",
    "knapsack", "graph traversal", "binary tree",
    # Complex data structures
    "lru", "cache", "ttl", "expiration", "thread-safe", "thread safe",
    "lock-free", "lockfree", "atomic",
    "rate limiter", "circuit breaker",
    # Debugging complex issues
    "race condition", "memory leak", "deadlock", "performance issue",
    "security vulnerability", "buffer overflow", "stack overflow",
    # Architecture patterns
    "decorator", "singleton", "factory", "observer", "strategy",
    "event driven", "pub sub", "message queue", "websocket",
    # Review & improve (any language)
    "review this", "what's wrong", "what is wrong", "improve this", "best practice",
    "what's missing", "security review", "code review",
    "debug this", "why is this", "why does this",
    # Language-specific complex topics
    "pointer", "reference", "ownership", "borrow", "lifetime",  # Rust/C++
    "goroutine", "channel", "defer", "interface",                # Go
    "promise", "callback", "closure", "prototype",               # JavaScript
    "generic", "trait", "enum", "macro",                         # Rust/Java
    "spring boot", "django", "flask", "express", "fastapi",      # Frameworks
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
    "syntax", "typo", "rename",
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

    # Complex always wins over simple (e.g. "what is wrong" has both "what is" and "what is wrong")
    if complex_matches >= 1:
        return "complex"

    # Short queries are usually simple
    if word_count <= 5 and complex_matches == 0:
        return "simple"

    # Lots of code in the query (pasted code) = medium to complex
    if lower.count("\n") > 5 or word_count > 50:
        return "complex"

    if simple_matches >= 1:
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
        self._state_path = str(get_leanai_home() / "model_manager.json")
        self._stats = {
            "queries_routed": 0,
            "fast_count": 0,
            "quality_count": 0,
        }
        self._load_state()
        # Surface any .gguf the user dropped in that no registry entry covers,
        # so EVERY local model is listable + switchable, not just curated ones.
        self._discover_local_models()
        # Surface remote endpoints (OpenAI-compatible / Ollama) from
        # ~/.leanai/endpoints.yaml so they're listable + switchable too.
        self._remote_specs = {}
        self._load_remote_endpoints()

    def _load_remote_endpoints(self) -> None:
        """Add any remote model aliases from endpoints.yaml as switchable
        entries. They report as 'available' (no download) and carry a
        'remote:<alias>' URI the engine knows how to load. Never fatal."""
        try:
            from core.endpoints import load_endpoints
            specs = load_endpoints()
        except Exception as e:
            print(f"[LeanAI] Could not load remote endpoints: {e}")
            return
        self._remote_specs = specs
        for alias, spec in specs.items():
            if alias in self.models:
                # Don't let a remote alias shadow a curated/local key.
                print(f"[LeanAI] Remote alias '{alias}' clashes with an existing "
                      f"model key — skipped.")
                continue
            self.models[alias] = ModelInfo(
                name=f"{spec.model_id} (remote · {spec.endpoint_name})",
                filename="",
                repo_id="(remote endpoint)",
                size_gb=0.0,
                ram_needed_gb=0.0,
                quality_score=spec.quality_score,
                speed_label=spec.speed_label,
                prompt_format=spec.prompt_format,
                description=spec.description,
                is_local=False,
                is_remote=True,
                model_uri=f"remote:{alias}",
            )

    def _discover_local_models(self) -> None:
        """Scan the models folder; add any unregistered .gguf as a switchable
        entry (filename-derived key, auto-detected prompt format). Curated
        registry entries keep their nice names; loose files still show up."""
        models_dir = get_models_dir()
        if not models_dir.exists():
            return
        # Filenames already covered by a registry entry (so we don't double-list).
        covered = set()
        for m in MODEL_REGISTRY.values():
            covered.add(m.filename.lower())
            if m.is_downloaded:
                try:
                    covered.add(os.path.basename(self.get_model_path_for(m)).lower())
                except Exception:
                    pass
        for f in sorted(models_dir.rglob("*.gguf")):
            if f.name.lower() in covered:
                continue
            key = self._local_key(f.name)
            if key in self.models:
                continue
            try:
                size_gb = f.stat().st_size / (1024 ** 3)
            except Exception:
                size_gb = 0.0
            self.models[key] = ModelInfo(
                name=f.name,
                filename=f.name,
                repo_id="(local file)",
                size_gb=round(size_gb, 1),
                ram_needed_gb=round(size_gb * 1.3, 1),
                quality_score=0,
                speed_label="local",
                prompt_format=_guess_format(f.name),
                description="Auto-detected local file.",
                is_local=True,
            )

    @staticmethod
    def _local_key(filename: str) -> str:
        """Short, typeable key from a filename, e.g.
        'Meta-Llama-3.1-8B-Instruct.Q6_K.gguf' -> 'local-meta-llama-3.1-8b'."""
        stem = filename.lower()
        for suf in (".gguf", ".q6_k", ".q4_k_m", ".q5_k_m", ".q8_0", "-instruct"):
            stem = stem.replace(suf, "")
        stem = stem.replace("_", "-").replace(" ", "-").strip("-.")
        parts = [p for p in stem.split("-") if p][:4]
        return "local-" + "-".join(parts) if parts else "local-model"

    def get_model_path_for(self, model: "ModelInfo") -> str:
        """Resolve a ModelInfo to an on-disk path (exact, else its full_path)."""
        return model.full_path

    # ── Model listing ─────────────────────────────────────────────

    def list_models(self) -> str:
        """List all available models with download status."""
        lines = ["Available Models:", ""]
        local = []
        remote = []
        for key, model in self.models.items():
            if getattr(model, "is_remote", False):
                remote.append((key, model))
                continue
            if getattr(model, "is_local", False):
                local.append((key, model))
                continue
            status = "DOWNLOADED" if model.is_downloaded else f"not downloaded ({model.size_gb:.1f} GB)"
            lines.append(
                f"  {key:12s}  {model.name:30s}  {model.speed_label:6s}  "
                f"quality:{model.quality_score}%  {status}"
            )
        if remote:
            lines.append("")
            lines.append("Remote endpoints (OpenAI-compatible / Ollama — no GPU needed):")
            for key, model in remote:
                lines.append(f"  {key:20s}  {model.name}  ({model.prompt_format})")
            lines.append("  (test reachability with: /model test)")
        else:
            lines.append("")
            lines.append("Remote endpoints: none yet.")
            lines.append("  Run /model connect to add an Ollama or OpenAI-compatible "
                         "server (no GPU needed here).")
        if local:
            lines.append("")
            lines.append("Local files (auto-detected in your models folder):")
            for key, model in local:
                lines.append(f"  {key:28s}  {model.name}  ({model.prompt_format})")
        lines.append(f"\nCurrent mode: {self._mode}")
        downloaded = [k for k, m in self.models.items() if m.is_downloaded]
        lines.append(f"Available: {', '.join(downloaded) or 'none'}")
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

        models_dir = str(get_models_dir())
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
                # Save as active model
                self._save_active(model.full_path)
                return True, f"Downloaded {model.name} successfully!"
            else:
                # Check alternate locations
                alt_path = os.path.join(models_dir, model.filename)
                if os.path.exists(alt_path):
                    self._save_active(alt_path)
                    return True, f"Downloaded {model.name} successfully!"
                return False, f"Download may have failed. Check {models_dir} manually."

        except subprocess.TimeoutExpired:
            return False, "Download timed out. Try downloading manually."
        except Exception as e:
            return False, f"Download error: {e}"

    def _save_active(self, path):
        """Save active model path so it persists across restarts."""
        try:
            config = get_leanai_home() / "active_model.txt"
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(str(path))
        except Exception:
            pass

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
            f"local_dir='{get_models_dir()}')\""
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
        lower_query = query.lower()

        # Frontend tasks → prefer Gemma 4 (better UI code, faster)
        frontend_keywords = {"react", "vue", "angular", "svelte", "next.js", "css",
                            "tailwind", "html", "frontend", "ui", "component", "form",
                            "button", "modal", "navbar", "sidebar", "dashboard", "layout"}
        is_frontend = any(kw in lower_query for kw in frontend_keywords)

        if is_frontend and "gemma4-26b" in downloaded:
            return "gemma4-26b"

        if complexity == "simple":
            self._stats["fast_count"] += 1
            return self._pick_fastest(downloaded)
        elif complexity == "complex":
            self._stats["quality_count"] += 1
            # Prefer Qwen3.5 27B for complex reasoning, then best available
            if "qwen35-27b" in downloaded:
                return "qwen35-27b"
            return self._pick_best_quality(downloaded)
        else:
            # Medium — prefer Gemma 4 (fast + reliable), then other options
            if "gemma4-26b" in downloaded:
                return "gemma4-26b"
            elif "qwen3-coder" in downloaded:
                return "qwen3-coder"
            elif "qwen-14b" in downloaded:
                return "qwen-14b"
            elif "qwen-32b" in downloaded:
                self._stats["quality_count"] += 1
                return "qwen-32b"
            else:
                self._stats["fast_count"] += 1
                return self._pick_fastest(downloaded)

    def _pick_fastest(self, downloaded: List[str]) -> str:
        """Pick the fastest (smallest) available model. Preference order:
        curated local registry → remote endpoints → auto-detected local files.
        This keeps a dropped-in .gguf from hijacking auto-routing, while still
        letting a no-GPU machine (only remote endpoints) route automatically."""
        curated = [k for k in downloaded
                   if not getattr(self.models[k], "is_local", False)
                   and not getattr(self.models[k], "is_remote", False)]
        remote = [k for k in downloaded if getattr(self.models[k], "is_remote", False)]
        pool = curated or remote or downloaded
        by_speed = sorted(pool, key=lambda k: self.models[k].size_gb)
        return by_speed[0]

    def _pick_best_quality(self, downloaded: List[str]) -> str:
        """Pick the highest quality available model (curated → remote → local)."""
        curated = [k for k in downloaded
                   if not getattr(self.models[k], "is_local", False)
                   and not getattr(self.models[k], "is_remote", False)]
        remote = [k for k in downloaded if getattr(self.models[k], "is_remote", False)]
        pool = curated or remote or downloaded
        by_quality = sorted(pool, key=lambda k: self.models[k].quality_score, reverse=True)
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
