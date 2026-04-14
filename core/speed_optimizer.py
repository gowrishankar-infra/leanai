"""
LeanAI — Speed Optimizer
Maximizes inference speed without compromising quality.

Techniques:
  1. Optimized llama.cpp parameters (n_batch, flash_attn, mmap, mlock)
  2. Response caching — identical/similar questions return instantly
  3. Prompt KV caching — reuse system prompt processing across calls
  4. Smart early stopping — detect complete answers, stop generating
  5. Hybrid passes — use 7B for critique/verification, 32B for generation
  6. GPU layer offloading — use any available GPU (even Intel iGPU)
"""

import os
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from pathlib import Path


@dataclass
class SpeedConfig:
    """Optimized configuration for maximum inference speed."""
    # llama.cpp optimization parameters
    n_batch: int = 512            # process 512 tokens at once (default is often 8)
    n_ubatch: int = 512           # micro-batch size
    flash_attn: bool = True       # use flash attention (faster, less memory)
    use_mmap: bool = True         # memory-map model file (faster loading)
    use_mlock: bool = False       # lock model in RAM (prevents swapping)
    n_gpu_layers: int = 0         # layers to offload to GPU (0 = CPU only)
    numa: bool = False            # NUMA optimization for multi-socket
    type_k: int = 1               # KV cache key quantization (1=f16, reduces memory)
    type_v: int = 1               # KV cache value quantization

    # Response caching
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.9  # how similar a query must be to use cache
    max_cache_entries: int = 500

    # Generation optimization
    max_tokens_simple: int = 256    # short responses for simple queries
    max_tokens_medium: int = 512    # medium responses
    max_tokens_complex: int = 1024  # full responses for complex queries

    # Hybrid mode
    hybrid_enabled: bool = True     # use 7B for critique passes


def detect_gpu() -> Dict[str, any]:
    """Detect available GPU for layer offloading."""
    gpu_info = {"available": False, "type": "none", "recommended_layers": 0}

    # Check for NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            name, mem = result.stdout.strip().split(",")
            mem_gb = int(mem.strip().split()[0]) / 1024
            gpu_info["available"] = True
            gpu_info["type"] = "nvidia"
            gpu_info["name"] = name.strip()
            gpu_info["memory_gb"] = mem_gb
            # Recommend layers based on VRAM
            if mem_gb >= 16:
                gpu_info["recommended_layers"] = 40  # full offload for 32B
            elif mem_gb >= 8:
                gpu_info["recommended_layers"] = 20  # partial
            elif mem_gb >= 4:
                gpu_info["recommended_layers"] = 10
            else:
                gpu_info["recommended_layers"] = 5
            return gpu_info
    except (FileNotFoundError, Exception):
        pass

    # Check for Vulkan (Intel/AMD iGPU)
    try:
        import subprocess
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            gpu_info["available"] = True
            gpu_info["type"] = "vulkan"
            gpu_info["name"] = "Vulkan-compatible GPU"
            gpu_info["recommended_layers"] = 5  # conservative for iGPU
            return gpu_info
    except (FileNotFoundError, Exception):
        pass

    return gpu_info


class ResponseCache:
    """
    Cache for LLM responses. Similar queries return instantly.
    Uses content hashing for exact matches and keyword overlap for similar matches.
    """

    def __init__(self, cache_dir: Optional[str] = None, max_entries: int = 500):
        self.cache_dir = cache_dir or str(Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai"))) / "response_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_entries = max_entries
        self._memory_cache: Dict[str, Tuple[str, float, float]] = {}  # hash -> (response, confidence, timestamp)
        self._hits = 0
        self._misses = 0
        self._load()

    def _hash(self, query: str) -> str:
        """Create a normalized hash of a query."""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _keyword_hash(self, query: str) -> str:
        """Create a keyword-based hash for fuzzy matching."""
        words = sorted(set(query.lower().split()))
        # Remove common stop words
        stop = {"a", "an", "the", "is", "are", "was", "were", "do", "does",
                "what", "how", "why", "when", "where", "who", "in", "on",
                "at", "to", "for", "of", "with", "and", "or", "but", "not",
                "can", "could", "would", "should", "will", "shall", "may",
                "i", "me", "my", "you", "your", "it", "its", "this", "that"}
        content_words = [w for w in words if w not in stop and len(w) > 2]
        return hashlib.md5(" ".join(content_words).encode()).hexdigest()

    def get(self, query: str) -> Optional[Tuple[str, float]]:
        """Look up a cached response. Returns (response, confidence) or None."""
        exact_hash = self._hash(query)
        if exact_hash in self._memory_cache:
            response, confidence, _ = self._memory_cache[exact_hash]
            self._hits += 1
            return response, confidence
        kw_hash = self._keyword_hash(query)
        if kw_hash in self._memory_cache:
            response, confidence, _ = self._memory_cache[kw_hash]
            self._hits += 1
            return response, confidence * 0.9
        self._misses += 1
        return None

    def get_semantic_draft(self, query: str, embedder=None, threshold_low: float = 0.6, threshold_high: float = 0.85) -> Optional[Tuple[str, str, float]]:
        """
        NOVEL: Semantic Speculative Caching with Draft Adaptation.
        Find a cached response for a SIMILAR (not identical) query.
        Returns (cached_query, cached_response, similarity) or None.
        The caller should ask the model to ADAPT the cached response instead of generating from scratch.
        """
        if not embedder or not self._query_texts:
            return None
        try:
            query_emb = embedder.encode([query])[0]
            best_sim = 0.0
            best_key = None
            for cached_query, cache_hash in self._query_texts.items():
                cached_emb = embedder.encode([cached_query])[0]
                # Cosine similarity
                import numpy as np
                sim = float(np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb) + 1e-8))
                if sim > best_sim:
                    best_sim = sim
                    best_key = cached_query
            if best_key and threshold_low <= best_sim < threshold_high:
                if cache_hash in self._memory_cache:
                    resp, conf, _ = self._memory_cache[self._hash(best_key)]
                    return best_key, resp, best_sim
        except Exception:
            pass
        return None

    def put(self, query: str, response: str, confidence: float = 0.8):
        """Cache a response."""
        exact_hash = self._hash(query)
        kw_hash = self._keyword_hash(query)
        timestamp = time.time()
        self._memory_cache[exact_hash] = (response, confidence, timestamp)
        self._memory_cache[kw_hash] = (response, confidence, timestamp)
        # Track query texts for semantic draft matching
        if not hasattr(self, '_query_texts'):
            self._query_texts = {}
        self._query_texts[query] = exact_hash
        if len(self._memory_cache) > self.max_entries * 2:
            self._evict()
        if (self._hits + self._misses) % 20 == 0:
            self._save()

    def _evict(self):
        """Remove oldest entries to stay under limit."""
        entries = sorted(self._memory_cache.items(), key=lambda x: x[1][2])
        to_remove = len(entries) - self.max_entries
        for key, _ in entries[:to_remove]:
            del self._memory_cache[key]

    def _save(self):
        """Persist cache to disk."""
        path = os.path.join(self.cache_dir, "response_cache.json")
        try:
            data = {
                k: {"response": v[0][:2000], "confidence": v[1], "timestamp": v[2]}
                for k, v in list(self._memory_cache.items())[:self.max_entries]
            }
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load(self):
        """Load cache from disk."""
        path = os.path.join(self.cache_dir, "response_cache.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                self._memory_cache[k] = (v["response"], v["confidence"], v["timestamp"])
        except Exception:
            pass

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "entries": len(self._memory_cache) // 2,  # each entry has 2 hashes
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.0%}",
        }


def get_optimal_model_params(model_size_gb: float, ram_gb: float = 32) -> dict:
    """
    Calculate optimal llama.cpp parameters based on model size and available RAM.
    """
    gpu = detect_gpu()

    params = {
        "n_batch": 512,
        "n_ubatch": 512,
        "flash_attn": True,
        "use_mmap": True,
        "use_mlock": False,
        "n_gpu_layers": 0,
        "verbose": False,
    }

    # GPU offloading
    if gpu["available"]:
        params["n_gpu_layers"] = gpu["recommended_layers"]

    # RAM-based tuning
    free_ram = ram_gb - model_size_gb - 4  # 4GB for OS/apps
    if free_ram > 8:
        params["use_mlock"] = True  # lock in RAM if plenty available
        params["n_batch"] = 1024
    elif free_ram > 4:
        params["n_batch"] = 512
    else:
        params["n_batch"] = 256  # conservative

    # Large batch for prompt processing
    params["n_ubatch"] = min(params["n_batch"], 512)

    return params


def get_max_tokens_for_query(query: str, model_size: str = "32b") -> int:
    """
    Determine optimal max_tokens based on query type.
    Short responses for simple queries = faster generation.
    Generous for complex tasks to avoid truncation.
    """
    lower = query.lower()
    word_count = len(lower.split())

    # Code generation (needs lots of tokens for complete implementations)
    code_words = {"implement", "write", "code", "function", "class", "create", "build",
                  "design", "develop", "refactor", "complete", "full"}
    if any(w in lower for w in code_words):
        return 2048

    # Reasoning/planning/analysis (needs room for multi-step thought + improved code)
    reason_words = {"explain", "detail", "comprehensive", "thorough", "compare",
                    "analyze", "reason", "plan", "decompose", "evaluate", "why",
                    "how does", "describe", "architecture", "trade-off"}
    if any(w in lower for w in reason_words):
        return 2048

    # Questions about files/code (need room to describe + show improved version)
    project_words = {"file", "module", "engine", "brain", "router", "what does"}
    if any(w in lower for w in project_words):
        return 1536

    # Very short queries → short responses
    if word_count <= 5:
        return 384

    # Yes/no questions
    if lower.startswith(("is ", "are ", "does ", "do ", "can ", "will ", "should ")):
        return 384

    # Definition questions
    if lower.startswith(("what is ", "what are ", "define ", "meaning of ")):
        return 512

    # Default — generous enough to avoid truncation
    return 768


class SpeedOptimizer:
    """
    Central speed optimization manager.
    
    Usage:
        optimizer = SpeedOptimizer()
        
        # Check for cached response before calling model
        cached = optimizer.cache.get(query)
        if cached:
            response, confidence = cached
            # Use cached response — instant!
        
        # Get optimal parameters for model loading
        params = optimizer.get_model_params(model_size_gb=18.0)
        
        # Get appropriate max_tokens for this query
        max_tokens = optimizer.get_max_tokens(query)
    """

    def __init__(self, config: Optional[SpeedConfig] = None):
        self.config = config or SpeedConfig()
        self.cache = ResponseCache(max_entries=self.config.max_cache_entries)
        self._gpu_info = None

    def get_gpu_info(self) -> dict:
        """Detect and return GPU information."""
        if self._gpu_info is None:
            self._gpu_info = detect_gpu()
        return self._gpu_info

    def get_model_params(self, model_size_gb: float = 18.0, ram_gb: float = 32) -> dict:
        """Get optimized parameters for model loading."""
        params = get_optimal_model_params(model_size_gb, ram_gb)
        params["flash_attn"] = self.config.flash_attn
        params["n_gpu_layers"] = max(params["n_gpu_layers"], self.config.n_gpu_layers)
        return params

    def get_max_tokens(self, query: str) -> int:
        """Get optimal max_tokens for a query."""
        return get_max_tokens_for_query(query)

    def should_use_cache(self, query: str) -> Optional[Tuple[str, float]]:
        """Check if we have a cached response."""
        if not self.config.cache_enabled:
            return None
        return self.cache.get(query)

    def cache_response(self, query: str, response: str, confidence: float = 0.8):
        """Cache a response for future use."""
        if self.config.cache_enabled and len(response) > 10:
            self.cache.put(query, response, confidence)

    def stats(self) -> dict:
        gpu = self.get_gpu_info()
        return {
            "gpu": gpu,
            "cache": self.cache.stats(),
            "config": {
                "n_batch": self.config.n_batch,
                "flash_attn": self.config.flash_attn,
                "hybrid": self.config.hybrid_enabled,
                "n_gpu_layers": self.config.n_gpu_layers,
            },
        }

    def optimization_report(self) -> str:
        """Generate a human-readable optimization report."""
        gpu = self.get_gpu_info()
        cache = self.cache.stats()

        lines = ["═══ Speed Optimization Report ═══"]

        # GPU
        if gpu["available"]:
            lines.append(f"GPU: {gpu.get('name', 'detected')} ({gpu['type']})")
            lines.append(f"  Recommended layers to offload: {gpu['recommended_layers']}")
            lines.append(f"  Currently offloading: {self.config.n_gpu_layers} layers")
            if self.config.n_gpu_layers == 0:
                lines.append("  ⚡ TIP: Set n_gpu_layers > 0 for 1.5-3x speedup!")
        else:
            lines.append("GPU: none detected (CPU only)")
            lines.append("  TIP: A GPU (even iGPU via Vulkan) would help significantly")

        # Batch size
        lines.append(f"Batch size: {self.config.n_batch}")
        if self.config.n_batch < 512:
            lines.append("  ⚡ TIP: Increase n_batch to 512+ for faster prompt processing")

        # Flash attention
        lines.append(f"Flash attention: {'enabled' if self.config.flash_attn else 'disabled'}")
        if not self.config.flash_attn:
            lines.append("  ⚡ TIP: Enable flash_attn for faster attention computation")

        # Cache
        lines.append(f"Response cache: {cache['entries']} entries, {cache['hit_rate']} hit rate")

        # Hybrid
        lines.append(f"Hybrid mode: {'enabled' if self.config.hybrid_enabled else 'disabled'}")
        if self.config.hybrid_enabled:
            lines.append("  Using 7B for critique passes, 32B for generation")

        return "\n".join(lines)
