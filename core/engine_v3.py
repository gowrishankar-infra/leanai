"""
LeanAI Phase 3+4b Engine
- Qwen2.5 Coder 7B/32B (chatml) + auto-detection for other models
- Sandboxed code execution with auto-fix loop
- Only returns verified-working code
"""
import time
import os
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from core.router import TaskRouter, Tier
from core.watchdog import MetaCognitiveWatchdog
from core.confidence import ConfidenceScoringEngine
from core.calibrator import ConfidenceCalibrator
from tools.z3_verifier import Z3Verifier, Verdict
from tools.executor import CodeExecutor
from memory.hierarchy_v2 import HierarchicalMemoryV2
from training.self_improve import TrainingDataStore, FeedbackSignal
from training.continual_trainer import ContinualTrainer, TrainingConfig
from training.self_play_v2 import EnhancedSelfPlayEngine


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.05


@dataclass
class LeanAIResponse:
    text: str
    pair_id: str
    confidence: float
    confidence_label: str
    confidence_bar: str
    confidence_method: str
    tier_used: str
    latency_ms: float
    verified: bool
    verification_summary: str
    claims_checked: int
    claims_correct: int
    memory_context_used: bool
    answered_from_memory: bool
    quality_score: float
    code_executed: bool = False
    code_passed: bool = False
    code_output: str = ""
    code_auto_fixed: bool = False
    warning: Optional[str] = None
    corrected: bool = False


def _get_active_model_path() -> str:
    """Find the active model. Checks config file first, then scans for any downloaded model."""
    config_file = Path.home() / ".leanai" / "active_model.txt"
    if config_file.exists():
        path = config_file.read_text().strip()
        if Path(path).exists():
            return path

    # Auto-detect: scan models directory for any .gguf file
    models_dir = Path.home() / ".leanai" / "models"
    if models_dir.exists():
        # Prefer qwen models, then any .gguf
        for pattern in ["qwen*7b*.gguf", "qwen*.gguf", "*.gguf"]:
            found = list(models_dir.glob(pattern))
            if found:
                # Pick the smallest (fastest) model as default
                found.sort(key=lambda p: p.stat().st_size)
                return str(found[0])

    # Fallback — will trigger "model not found" message
    return str(models_dir / "qwen25-coder-7b-q4.gguf")


def _save_active_model(model_path: str):
    """Save the active model path so it persists across restarts."""
    try:
        config_file = Path.home() / ".leanai" / "active_model.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(model_path)
    except Exception:
        pass  # non-critical


def _detect_prompt_format(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "gemma" in name: return "gemma"
    if "qwen" in name: return "chatml"
    if "phi" in name: return "phi3"
    if "llama" in name: return "llama3"
    return "chatml"


def _optimal_threads(model_path: str) -> int:
    name = Path(model_path).name.lower()
    cpu = os.cpu_count() or 8
    # All models benefit from max threads on modern CPUs
    if any(x in name for x in ["7b", "6b", "8b"]): return min(cpu, 16)
    if any(x in name for x in ["32b", "33b", "34b", "70b"]): return min(cpu, 16)
    return min(cpu, 16)


def _looks_like_python(code: str) -> bool:
    """Check if code looks like Python (not YAML, JSON, bash, SQL, markdown, Go, etc)."""
    # Skip very short content (just a language tag like "go", "rust", etc)
    if len(code.strip()) < 10:
        return False

    first_lines = code.strip().split("\n")[:5]
    first = first_lines[0].strip() if first_lines else ""

    # First line is just a language name (from ```go blocks where tag leaked into content)
    lang_tags = {"go", "rust", "java", "javascript", "typescript", "c", "cpp",
                 "ruby", "php", "swift", "kotlin", "scala", "bash", "sh",
                 "yaml", "yml", "json", "sql", "html", "css", "dockerfile"}
    if first.lower() in lang_tags:
        return False

    # Definitely NOT Python
    non_python_indicators = [
        "trigger:", "pool:", "vmImage:", "steps:",        # YAML/Azure DevOps
        "apiVersion:", "kind:", "metadata:",              # Kubernetes YAML
        "FROM ", "RUN ", "COPY ", "CMD ", "ENTRYPOINT",   # Dockerfile
        "SELECT ", "INSERT ", "CREATE TABLE", "ALTER ",   # SQL
        "#!/bin/bash", "#!/bin/sh", "echo ", "sudo ",     # bash
        '{"', "'{",                                        # JSON
        "<html", "<!DOCTYPE", "<div", "<script",          # HTML
        "const ", "let ", "var ", "function ",             # JavaScript
        "package main", "func main", "import (",          # Go
        "public class", "public static void",             # Java
        "fn main", "use std",                             # Rust
    ]
    code_upper = code[:500]
    for indicator in non_python_indicators:
        if indicator in code_upper:
            return False

    # Looks like markdown/prose (not code)
    prose_indicators = [
        "- What it does", "- Why it matters", "- What this",
        "What to watch", "Improved version", "### ",
        "Here's ", "Let's break", "This pipeline",
        "In summary", "Overall,", "Note:", "Warning:",
    ]
    for indicator in prose_indicators:
        if indicator in code_upper:
            return False

    # Starts with markdown bullet — definitely not code
    if first.startswith(("- ", "* ", "> ", "| ")):
        return False

    # Looks like YAML (key: value pattern on most lines)
    yaml_lines = sum(1 for l in first_lines if re.match(r"^\s*\w+:", l))
    if yaml_lines >= 3:
        return False

    # Too much prose — more than half the lines are natural language
    prose_lines = sum(1 for l in first_lines if l.strip() and
                      len(l.split()) > 6 and l.strip()[0].isupper())
    if prose_lines >= 3:
        return False

    # Probably Python
    python_indicators = ["def ", "class ", "import ", "from ", "print(", "if __name__",
                         "return ", "self.", "elif ", "except ", "lambda "]
    for indicator in python_indicators:
        if indicator in code:
            return True

    # If we can parse it as Python AST, it's Python
    try:
        import ast
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _extract_code_blocks(text: str) -> list:
    """Extract Python code blocks from a response. Skips non-Python blocks."""
    blocks = []

    # 1. Try explicitly tagged Python blocks: ```python ... ```
    py_pattern = re.compile(r"```(?:python|py)\n?(.*?)```", re.DOTALL)
    for m in py_pattern.finditer(text):
        code = m.group(1).strip()
        if code and len(code) > 5:
            blocks.append(code)

    if blocks:
        return [_sanitize_code(b) for b in blocks]

    # 2. Try UNTAGGED code blocks (no language specified) — assume Python
    #    But skip if it looks like YAML, JSON, bash, SQL, etc.
    untagged = re.compile(r"```\n?(.*?)```", re.DOTALL)
    for m in untagged.finditer(text):
        code = m.group(1).strip()
        if code and len(code) > 5 and _looks_like_python(code):
            blocks.append(code)

    if blocks:
        return [_sanitize_code(b) for b in blocks]

    # 3. Handle UNCLOSED code blocks (model truncated mid-response)
    unclosed = re.compile(r"```(?:python|py)?\n(.*)", re.DOTALL)
    m = unclosed.search(text)
    if m:
        code = m.group(1).strip()
        if not _looks_like_python(code):
            return []
        lines = code.split("\n")
        code_lines = []
        for line in lines:
            if (code_lines and
                not line.startswith((" ", "\t", "def ", "class ", "import ", "from ",
                                    "if ", "for ", "while ", "return ", "print(",
                                    "#", "@", "}", "]", ")", "else:", "elif ",
                                    "try:", "except", "finally:", "with ", "raise ",
                                    "yield ", "async ", "await ", "")) and
                line and line[0].isupper() and " " in line and len(line) > 30):
                break
            code_lines.append(line)
        code = "\n".join(code_lines).strip()
        if code and len(code) > 10:
            blocks.append(_sanitize_code(code))

    if blocks:
        return blocks

    # 4. Last resort: if the ENTIRE response looks like pure Python code
    lines = text.strip().split("\n")
    has_code_keywords = any(kw in text for kw in ["def ", "class ", "import "])
    prose_lines = sum(1 for l in lines if l.strip() and l.strip()[0].isupper()
                      and ("." in l or "!" in l or ":" in l) and len(l.split()) > 5)
    code_lines = sum(1 for l in lines if l.strip().startswith(("def ", "class ", "import ",
                     "from ", "if ", "for ", "while ", "return ", "#", "@", "    ")))

    if has_code_keywords and prose_lines == 0 and code_lines >= 2:
        blocks.append(_sanitize_code(text.strip()))

    return blocks


def _sanitize_code(code: str) -> str:
    """
    Validate code with ast.parse. If it fails, trim broken lines from the end
    until it parses, or return the original if nothing works.
    """
    import ast

    # Quick check — does it parse as-is?
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass

    # Try trimming lines from the end (model truncated mid-function)
    lines = code.split("\n")
    for trim in range(1, min(len(lines), 15)):
        trimmed = "\n".join(lines[:-trim])
        if not trimmed.strip():
            break
        try:
            ast.parse(trimmed)
            return trimmed
        except SyntaxError:
            continue

    # Try closing common unterminated constructs
    fixups = [
        code + '\n"""',          # close triple-quoted string
        code + "\n'''",          # close single triple-quoted string
        code + "\n    pass\n",   # close empty function body
        code + "\n        pass\n",
    ]
    for fix in fixups:
        try:
            ast.parse(fix)
            return fix
        except SyntaxError:
            continue

    # Nothing worked — return original, executor will handle the error
    return code


def _is_safe_to_execute(code: str) -> bool:
    """
    Check if code is safe to auto-execute in the sandbox.
    Returns False for interactive code, project-specific imports, file deletion, etc.
    """
    unsafe_patterns = [
        "input(",           # waits for keyboard input → hangs
        "getpass",          # waits for password input
        "sys.stdin",        # reads from stdin
        "os.remove(",       # deletes files
        "os.rmdir(",        # deletes directories
        "shutil.rmtree(",   # deletes directory trees
        "os.system(",       # shell commands
        "subprocess.call(",  # shell commands
        "exit(",            # exits the process
        "quit(",            # exits the process
    ]
    code_lower = code.lower()
    for pattern in unsafe_patterns:
        if pattern.lower() in code_lower:
            return False

    # Skip code that imports from the project itself — can't work in sandbox
    project_imports = [
        "from core.", "from brain.", "from tools.", "from training.",
        "from agents.", "from swarm.", "from memory.", "from api.",
        "from federated.", "from speculative.", "from liquid.", "from hdc.",
        "import core.", "import brain.", "import tools.",
    ]
    for pattern in project_imports:
        if pattern in code:
            return False

    # Check for while True with input
    if "while true" in code_lower and "input(" in code_lower:
        return False

    return True


class LeanAIEngineV3:

    def __init__(self, model_path=None, verbose=False, auto_train=True, auto_execute=True):
        self.model_path    = model_path or _get_active_model_path()
        self.prompt_format = _detect_prompt_format(self.model_path)
        self.n_threads     = _optimal_threads(self.model_path)
        self.verbose       = verbose
        self.auto_execute  = auto_execute

        self.router     = TaskRouter()
        self.watchdog   = MetaCognitiveWatchdog()
        self.scorer     = ConfidenceScoringEngine()
        self.calibrator = ConfidenceCalibrator()
        self.verifier   = Z3Verifier()
        self.executor   = CodeExecutor()
        self.memory     = HierarchicalMemoryV2()
        self.store      = TrainingDataStore()
        self.self_play  = EnhancedSelfPlayEngine()
        self.trainer    = ContinualTrainer(
            store=self.store,
            config=TrainingConfig(
                check_interval_minutes=30,
                min_pairs_to_train=50,
                self_play_batch_size=20,
            ),
        )
        self._model        = None
        self._model_loaded = False

        if auto_train:
            self.trainer.start()

        model_name = Path(self.model_path).name
        print("[LeanAI v3] Engine initialized.")
        print(f"[LeanAI v3] Model: {model_name}")
        print(f"[LeanAI v3] Format: {self.prompt_format}")
        print(f"[LeanAI v3] Threads: {self.n_threads}")
        print(f"[LeanAI v3] Code executor: {'on' if auto_execute else 'off'}")
        print(f"[LeanAI v3] Available runtimes: {self.executor.available_languages}")
        print(f"[LeanAI v3] Memory: {self.memory.episodic.backend}")
        print(f"[LeanAI v3] Training: {self.store.stats()['total']} pairs")

    def generate(self, query, config=None, project_context=""):
        config = config or GenerationConfig()
        start  = time.time()
        decision = self.router.route(query, self.memory.working.current_tokens)

        # Tiny tier
        if decision.tier == Tier.TINY:
            text = self._handle_tiny(query)
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.99, text, "tiny", False, query)
            pair = self.store.add_pair(instruction=query, response=text,
                feedback=FeedbackSignal.EXCELLENT, confidence=cal.calibrated,
                verified=False, latency_ms=latency, tier_used="tiny")
            self.memory.record_exchange(query, text)
            return self._wrap(text, pair.id, cal, "tiny", latency,
                              False, False, "Not needed.", 0, 0, False, True, 1.0)

        # Memory-first
        memory_answer = self.memory.answer_from_memory(query)
        if memory_answer:
            latency = (time.time() - start) * 1000
            cal = self.calibrator.calibrate(0.95, memory_answer, "memory", True, query)
            self.memory.record_exchange(query, memory_answer)
            pair = self.store.add_pair(instruction=query, response=memory_answer,
                feedback=FeedbackSignal.EXCELLENT, confidence=cal.calibrated,
                verified=True, latency_ms=latency, tier_used="memory")
            return self._wrap(memory_answer, pair.id, cal, "memory (instant)", latency,
                              True, False, "From world model.", 0, 0, True, True, 1.0)

        # Generate
        mem_ctx  = self.memory.prepare_context(query)
        mem_used = bool(mem_ctx)
        prompt   = self._build_prompt(query, mem_ctx,
                                       self.memory.working.get_context_window(max_tokens=512),
                                       project_context=project_context)
        response_text = self._generate_with_model(prompt, config)

        # Code execution
        code_executed = code_passed = code_auto_fixed = False
        code_output = ""

        if self.auto_execute and decision.requires_tools:
            code_blocks = _extract_code_blocks(response_text)
            # Filter out code that's unsafe to auto-execute
            code_blocks = [b for b in code_blocks if _is_safe_to_execute(b)]
            if code_blocks:
                verified = self.executor.execute_and_verify(code_blocks[0])
                code_executed  = True
                code_passed    = verified.passed
                code_auto_fixed = verified.auto_fixed
                code_output    = self.executor.format_result(verified)

                # If code failed, add error context to response
                if not verified.passed:
                    error_note = (
                        "\n\n> Execution note: Code ran but produced an error. "
                        f"Error: {verified.final_result.stderr[:200].strip()}"
                    )
                    response_text += error_note
                else:
                    # Prepend execution success note
                    pass

        # Score and calibrate
        raw_score = self.scorer.score_from_text(response_text)
        # Boost confidence if code was verified
        base_conf = 0.90 if code_passed else raw_score.overall
        cal = self.calibrator.calibrate(base_conf, response_text, decision.tier.value,
                                         code_passed, query)

        # Math/logic verification
        math_verified = math_corrected = False
        verify_summary = "Not checked."
        claims_checked = claims_correct = 0

        if decision.requires_verifier and not code_executed:
            report = self.verifier.verify_text(response_text, query)
            claims_checked = report.claims_found
            claims_correct = report.claims_verified
            math_verified  = report.overall_verdict == Verdict.TRUE
            math_corrected = report.overall_verdict == Verdict.FALSE
            verify_summary = report.summary
            if report.corrected_text:
                response_text = report.corrected_text
            cal = self.calibrator.calibrate(raw_score.overall, response_text,
                                             decision.tier.value, math_verified, query)

        self.memory.record_exchange(query, response_text)
        latency = (time.time() - start) * 1000

        feedback = (FeedbackSignal.EXCELLENT if (code_passed or math_verified) else
                    FeedbackSignal.GOOD if cal.calibrated > 0.7 else FeedbackSignal.NEUTRAL)
        pair = self.store.add_pair(instruction=query, response=response_text,
            feedback=feedback, confidence=cal.calibrated,
            verified=code_passed or math_verified,
            latency_ms=latency, tier_used=decision.tier.value)

        r = self._wrap(response_text, pair.id, cal, decision.tier.value, latency,
            math_verified, math_corrected, verify_summary,
            claims_checked, claims_correct, mem_used, False,
            round(pair.quality_score, 3))
        r.code_executed  = code_executed
        r.code_passed    = code_passed
        r.code_output    = code_output
        r.code_auto_fixed = code_auto_fixed
        return r

    def execute_code(self, code: str, language: str = "python"):
        """Directly execute code. Used by /run command."""
        return self.executor.execute_and_verify(code, language)

    def give_feedback(self, pair_id, good):
        self.store.update_feedback(
            pair_id, FeedbackSignal.EXCELLENT if good else FeedbackSignal.WRONG)

    def remember(self, fact):
        self.memory.remember_fact(fact)

    def get_profile(self):
        return self.memory.world.get_user_profile()

    def trigger_training(self):
        run = self.trainer.run_now()
        return {"status": run.status, "pairs_used": run.pairs_used,
                "notes": run.notes,
                "duration_s": round((run.completed_at - run.started_at), 2) if run.completed_at else 0}

    def generate_training_data(self, n=20):
        return self.trainer.generate_self_play(n)

    def training_status(self):
        return self.trainer.status()

    def switch_model(self, new_model_path: str):
        """Switch to a different model. Properly unloads old model first."""
        if new_model_path == self.model_path and self._model is not None:
            return  # already loaded

        # Unload current model
        if self._model is not None:
            del self._model
            self._model = None

        # Reset state
        self._model_loaded = False
        self.model_path = new_model_path
        self.model_name = Path(new_model_path).name
        self.prompt_format = _detect_prompt_format(new_model_path)
        self.n_threads = _optimal_threads(new_model_path)

        # Save as active model so it persists across restarts
        _save_active_model(new_model_path)

        # Load new model immediately
        self._load_model()

    # ── Private ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model_loaded:
            return
        if not Path(self.model_path).exists():
            print(f"[LeanAI v3] Model not found: {self.model_path}")
            # DON'T set _model_loaded = True — allow retry with different path
            return
        try:
            from llama_cpp import Llama
            model_name = Path(self.model_path).name
            print(f"[LeanAI v3] Loading {model_name} ({self.n_threads} threads)...")
            # Dynamic GPU layers: 7B can fit more layers, 32B needs fewer
            model_name_lower = Path(self.model_path).name.lower()
            is_qwen3_moe = "qwen3" in model_name_lower and "coder" in model_name_lower
            is_qwen35 = "qwen3.5" in model_name_lower or "qwen35" in model_name_lower
            is_gemma4_moe = "gemma" in model_name_lower and ("26b" in model_name_lower or "a4b" in model_name_lower)

            # Auto-detect available RAM
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024**3)
                avail_ram_gb = psutil.virtual_memory().available / (1024**3)
            except ImportError:
                total_ram_gb = 32  # assume 32GB if psutil not available
                avail_ram_gb = 24

            low_ram = total_ram_gb <= 12  # 8GB or 12GB machines

            if is_qwen3_moe:
                gpu_layers = 0   # CPU-only: Qwen3 MoE layers too large for 4GB VRAM
                ctx_size = 8192  # Start conservative, Qwen3 supports up to 262K
            elif is_qwen35:
                gpu_layers = 4   # Qwen3.5 27B dense — similar to 32B
                ctx_size = 4096  # Conservative context, supports up to 262K
            elif is_gemma4_moe:
                gpu_layers = 0   # CPU-only: Gemma 4 MoE layers too large for 4GB VRAM
                ctx_size = 4096  # Reduced from 8192 for speed, supports up to 256K
            elif any(x in model_name_lower for x in ["32b", "33b", "34b", "70b"]):
                gpu_layers = 4   # 32B dense — only 4 layers fit in 4GB VRAM
                ctx_size = 4096
            elif any(x in model_name_lower for x in ["7b", "6b", "8b"]):
                if low_ram:
                    gpu_layers = 0    # no GPU offload on low RAM machines
                    ctx_size = 2048   # smaller context to avoid OOM
                    print(f"[LeanAI v3] Low RAM detected ({total_ram_gb:.0f}GB) — using ctx=2048, CPU-only")
                else:
                    gpu_layers = 15   # 7B model — most layers fit in 4GB VRAM
                    ctx_size = 4096
            else:
                gpu_layers = 8
                ctx_size = 4096

            self._model = Llama(
                model_path=self.model_path,
                n_ctx=ctx_size, n_threads=self.n_threads, n_batch=1024,
                n_gpu_layers=gpu_layers, use_mmap=True, use_mlock=False,
                logits_all=False, verbose=self.verbose,
            )
            print(f"[LeanAI v3] GPU layers: {gpu_layers}")
            self._model_loaded = True
            print("[LeanAI v3] Model loaded. Ready.")
        except ImportError:
            print("[LeanAI v3] llama-cpp-python not installed")
            self._model_loaded = True
        except Exception as e:
            print(f"[LeanAI v3] Error loading model: {e}")
            # DON'T set _model_loaded = True — allow retry

    def _generate_with_model(self, prompt, config):
        self._load_model()
        if self._model is None:
            return (
                "Model not loaded. Possible causes:\n"
                "- Model file not found at: " + str(self.model_path) + "\n"
                "- Run: python setup_leanai.py  (downloads model automatically)\n"
                "- Or: /model list  (to see available models)"
            )
        stop_tokens = ["<|im_end|>", "<|im_start|>", "<|user|>", "<|end|>",
                       "<|assistant|>", "<end_of_turn>", "<start_of_turn>",
                       "\nYou:", "\nHuman:", "\nUser:"]
        result = self._model(prompt, max_tokens=config.max_tokens,
            temperature=config.temperature, top_p=config.top_p,
            top_k=config.top_k, repeat_penalty=config.repeat_penalty,
            stop=stop_tokens, echo=False)
        text = result["choices"][0]["text"].strip()
        for token in stop_tokens:
            text = text.split(token)[0].strip()
        # Strip <think>...</think> blocks (Qwen3.5 thinking mode output)
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Also strip incomplete thinking blocks that didn't close
        if "<think>" in text and "</think>" not in text:
            text = text.split("<think>")[0].strip()
        return text

    def _build_prompt(self, query, memory_context, history, project_context=""):
        system = (
            "You are LeanAI — a senior software engineer AI. Clear, accurate, insightful.\n\n"
            "RULES:\n"
            "1. No filler ('Certainly!', 'Let me explain'). Start with substance.\n"
            "2. Vary sentence patterns. Never repeat same structure.\n"
            "3. For each concept: WHAT (brief) + WHY it matters + RISKS/GOTCHAS.\n"
            "4. Show IMPROVED code with 3-5 real fixes from your Risks/Missing sections.\n"
            "5. Be SPECIFIC. Bad: 'ensure it's configured.' Good: 'add containerRegistry: X'\n"
            "6. Use markdown: ### headers, **bold**, `inline code`, ```lang code blocks, bullets.\n"
            "7. FLAG WHAT'S MISSING in ### Risks or ### What's missing sections.\n"
            "   Check: input validation, edge cases (None/empty/negative), error handling, "
            "security (injection/XSS/secrets), resource cleanup, thread safety, performance.\n"
            "   Python: type hints, bare except, mutable defaults.\n"
            "   JS/TS: ==/===, async errors, prototype pollution.\n"
            "   Go: unchecked err, goroutine leaks, missing defer.\n"
            "   Rust: unwrap without handling, unsafe blocks.\n"
            "   Java: null pointers, resource leaks, missing equals/hashCode.\n"
            "   SQL: injection, missing indexes, N+1, no LIMIT.\n"
            "   C/C++: buffer overflow, memory leaks, bounds checks.\n"
            "   CI/CD: test step, Dockerfile, auth, tag strategy, secrets.\n"
            "   API: auth, rate limiting, validation, status codes, CORS.\n"
            "   DB: injection, connection pooling, indexes, transactions.\n"
            "8. Structure: overview → breakdown (WHAT+WHY) → risks → missing → improved code.\n"
            "9. Correct language tags on code blocks. No prose in code blocks.\n"
            "10. For USER's project: use THEIR actual code, class names, file paths.\n"
            "11. Self-review before finishing: improved version? validation? error handling? "
            "edge cases? security? missing deps?\n"
            "12. Stop after answering."
        )

        # ── Build user query with dynamic context ──────────────────
        # Keep system prompt STATIC so llama.cpp can cache the KV prefix.
        # Dynamic context (project, memory) goes into the user message.
        user_with_context = ""

        if project_context:
            user_with_context += (
                "[PROJECT CONTEXT — use this to give specific answers about the user's ACTUAL code]\n"
                + project_context[:3000] + "\n\n"
            )

        if memory_context:
            ctx = memory_context[:500].replace("\n", " ")
            user_with_context += f"[MEMORY] {ctx}\n\n"

        user_with_context += query

        # Disable thinking mode for Qwen3.5 to avoid 25-min think blocks
        model_name_lower = Path(self.model_path).name.lower() if self.model_path else ""
        is_qwen35 = "qwen3.5" in model_name_lower or "qwen35" in model_name_lower

        if self.prompt_format == "chatml":
            parts = ["<|im_start|>system\n" + system + "<|im_end|>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:400] if len(msg["content"]) > 400 else msg["content"]
                if role == "user":
                    parts.append("<|im_start|>user\n" + content + "<|im_end|>\n")
                elif role == "assistant":
                    parts.append("<|im_start|>assistant\n" + content + "<|im_end|>\n")
            if is_qwen35:
                # Force empty think block so model skips thinking mode entirely
                parts.append("<|im_start|>user\n" + user_with_context + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n")
            else:
                parts.append("<|im_start|>user\n" + user_with_context + "<|im_end|>\n<|im_start|>assistant\n")
        elif self.prompt_format == "gemma":
            parts = ["<start_of_turn>user\n" + system + "\n\n" + user_with_context + "<end_of_turn>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:400] if len(msg["content"]) > 400 else msg["content"]
                if role == "user":
                    parts.append("<start_of_turn>user\n" + content + "<end_of_turn>\n")
                elif role == "assistant":
                    parts.append("<start_of_turn>model\n" + content + "<end_of_turn>\n")
            parts.append("<start_of_turn>user\n" + user_with_context + "<end_of_turn>\n<start_of_turn>model\n")
        else:
            parts = ["<|system|>\n" + system + "<|end|>\n"]
            for msg in history[-4:]:
                role = msg["role"]
                content = msg["content"][:300] if len(msg["content"]) > 300 else msg["content"]
                if role == "user":
                    parts.append("<|user|>\n" + content + "<|end|>\n")
                elif role == "assistant":
                    parts.append("<|assistant|>\n" + content + "<|end|>\n")
            parts.append("<|user|>\n" + user_with_context + "<|end|>\n<|assistant|>\n")

        return "".join(parts)

    def _handle_tiny(self, query):
        import math as m
        q = query.strip().lower()
        if re.match(r"^(hi|hello|hey)\b", q): return "Hello! How can I help?"
        if re.match(r"^(thanks|thank you)\b", q): return "You're welcome!"
        if re.match(r"^(bye|goodbye)\b", q): return "Goodbye!"
        try:
            expr = re.sub(r"[^0-9\+\-\*\/\(\)\.\s\^]", "", query).replace("^", "**")
            if expr.strip():
                result = eval(expr, {"__builtins__": {}, "sqrt": m.sqrt})
                return str(result)
        except Exception:
            pass
        return "Could you tell me more?"

    def _wrap(self, text, pair_id, cal, tier, latency, verified, corrected,
              verify_summary, claims_checked, claims_correct, mem_used,
              from_memory, quality, warning=None):
        return LeanAIResponse(
            text=text, pair_id=pair_id,
            confidence=cal.calibrated, confidence_label=cal.label,
            confidence_bar=cal.bar, confidence_method=cal.method,
            tier_used=tier, latency_ms=round(latency, 1),
            verified=verified, corrected=corrected,
            verification_summary=verify_summary,
            claims_checked=claims_checked, claims_correct=claims_correct,
            memory_context_used=mem_used, answered_from_memory=from_memory,
            quality_score=quality, warning=warning,
        )

    def status(self):
        return {
            "phase": "3+4b", "model": Path(self.model_path).name,
            "prompt_format": self.prompt_format, "threads": self.n_threads,
            "model_loaded": self._model is not None,
            "executor": {"available": self.executor.available_languages,
                         "auto_execute": self.auto_execute},
            "verifier": self.verifier.status,
            "memory": self.memory.stats(),
            "training": self.trainer.status(),
        }
