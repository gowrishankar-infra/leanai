#!/usr/bin/env python3
"""
LeanAI — Fully Integrated CLI
All phases connected: engine, memory, swarm, brain, git, TDD, editor, sessions.
Every query benefits from project context, session history, and adaptive routing.
"""

import os
import sys
import time
import re

# ══════════════════════════════════════════════════════════════════════
# WINDOWS CONSOLE BOOTSTRAP — must run BEFORE any other module is imported
# ══════════════════════════════════════════════════════════════════════
# This block fixes the PowerShell "cursor blinks but I can't type at the
# ❯❯❯ prompt" bug. Three root causes addressed:
#   1) PowerShell defaults to UTF-16 / CP-1252 — when Python writes the
#      ❯ Unicode character (U+276F) it desyncs the input echo
#   2) PowerShell's legacy console host doesn't fully parse 24-bit ANSI
#      color codes used in get_prompt(), corrupting cursor tracking
#   3) Background daemon threads (Trainer, Predictor, FileWatcher) can
#      print() while input() is active, scrambling the readline buffer
def _bootstrap_windows_console():
    """Force UTF-8 console + ANSI processing on Windows. Safe no-op on others."""
    if sys.platform != "win32":
        return
    # Force Python's stdio streams to UTF-8 so writing ❯ doesn't garble
    try:
        # Python 3.7+ supports reconfigure() on text streams
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        if hasattr(sys.stdin, 'reconfigure'):
            sys.stdin.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    # Set the Windows console code page to UTF-8 (65001) for both input and output
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)        # input
        kernel32.SetConsoleOutputCP(65001)  # output
    except Exception:
        pass

    # Enable virtual terminal processing on stdout (24-bit color, cursor moves)
    # AND restore stdin to cooked mode (line input + echo) — defends against
    # any prior session that left it in raw mode.
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32

        STD_INPUT_HANDLE = -10
        STD_OUTPUT_HANDLE = -11
        ENABLE_PROCESSED_INPUT = 0x0001
        ENABLE_LINE_INPUT = 0x0002
        ENABLE_ECHO_INPUT = 0x0004
        ENABLE_PROCESSED_OUTPUT = 0x0001
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200

        h_in = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if h_in and h_in != -1:
            kernel32.SetConsoleMode(
                h_in,
                ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT,
            )

        h_out = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        if h_out and h_out != -1:
            current = wintypes.DWORD()
            if kernel32.GetConsoleMode(h_out, ctypes.byref(current)):
                kernel32.SetConsoleMode(
                    h_out,
                    current.value | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
                )
    except Exception:
        pass

    # PYTHONIOENCODING and PYTHONUTF8 propagate to subprocesses
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')


# Run the bootstrap RIGHT NOW before anything else loads
_bootstrap_windows_console()

# Global lock so background-thread prints don't interleave with user input
import threading as _threading
_print_lock = _threading.RLock()
_user_at_prompt = _threading.Event()  # set when waiting on input()


def _safe_input(prompt: str) -> str:
    """
    Thread-safe input() wrapper.

    On Windows, when background daemon threads (Trainer, Predictor,
    FileWatcher) call print() while we're waiting on input(), the
    console output gets interleaved into the readline buffer and the
    cursor stops echoing what you type.

    This wrapper:
      - Signals to background threads that we're at the prompt
      - Flushes stdio before drawing the prompt
      - Re-flushes stdio after reading input
      - Re-asserts UTF-8 console mode if we're on Windows (in case
        a background thread's print() call changed the code page)
    """
    _user_at_prompt.set()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            value = input(prompt)
        finally:
            _user_at_prompt.clear()
            try:
                sys.stdout.flush()
            except Exception:
                pass
        return value
    except Exception:
        _user_at_prompt.clear()
        raise


# Patch the global print() so background threads serialize with each other
# (so two threads printing simultaneously don't shred each other's output)
_real_print = print
def _safe_print(*args, **kwargs):
    with _print_lock:
        _real_print(*args, **kwargs)
        try:
            sys.stdout.flush()
        except Exception:
            pass

# Make safe_print available as a builtin override only when explicitly called.
# We don't replace the global builtin — too risky for existing code.
# Background threads can opt in by importing _safe_print from main.

# ══════════════════════════════════════════════════════════════════════
# END Windows console bootstrap
# ══════════════════════════════════════════════════════════════════════


# ── Core engine ───────────────────────────────────────────────────
from core.engine_v3 import LeanAIEngineV3 as LeanAIEngine, GenerationConfig
from core.model_manager import ModelManager, classify_complexity
from core.reasoning_engine import ReasoningEngine
from core.writing_engine import WritingEngine
from core.speed_optimizer import SpeedOptimizer
from core.completer import AutoCompleter
from core.predictor import PredictivePreGenerator
from brain.semantic_bisect import SemanticGitBisect
from brain.evolution_tracker import EvolutionTracker
from tools.adversarial import AdversarialVerifier
from core.code_quality import CodeQualityEnhancer
from core.code_verifier import CodeGroundedVerifier
from core.agac import AGACEngine
from core.cascade import CascadeInference
from core.sentinel import SentinelEngine, Severity, format_findings_report
from core.react import ReActReasoner
from core.mixture_of_agents import MixtureOfAgents
from core.streaming import StreamingGenerator, StreamConfig, print_streaming_header, print_streaming_footer
from core.smart_context import SmartContext
from core.auto_recovery import AutoRecovery, RecoveryConfig
from tools.executor import CodeExecutor
from tools.indexer import ProjectIndexer
from agents.build_command import BuildHandler
from swarm import SwarmConsensus

# ── Phase 6: Advanced architectures ───────────────────────────────
from liquid import LiquidRouter
from hdc import HDKnowledgeStore

# ── Phase 7: Project-aware intelligence ───────────────────────────
from brain.project_brain import ProjectBrain
from brain.git_intel import GitIntel
from brain.tdd_loop import TDDLoop, TDDConfig
from brain.editor import MultiFileEditor
from brain.session_store import SessionStore
from training.finetune_pipeline import TrainingDataPipeline
from training.adapter_manager import AdapterManager
from training.finetune_runner import FineTuneRunner, TrainingConfig
from core.terminal_ui import (
    print_banner, print_status, print_commands, get_prompt,
    format_response, format_confidence, format_meta, separator,
    format_code_result, format_brain_scan, format_completions,
    print_response_header, C,
)


BANNER = ""  # handled by terminal_ui


def _truncate_repetition(text: str, max_word_repeats: int = 4, max_phrase_repeats: int = 3) -> str:
    """
    Detect and truncate repetitive model output.
    Catches cases like 'step-step-step-step...' or 'produce produce produce...'.
    Runs on ALL responses from ALL models as a safety net.
    """
    if not text or len(text) < 100:
        return text

    words = text.split()
    if len(words) < 20:
        return text

    # ── Check 1: Same word repeated N+ times consecutively ────────
    repeat_count = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeat_count += 1
            if repeat_count >= max_word_repeats:
                # Truncate at the start of the repetition
                cut_point = i - max_word_repeats + 1
                truncated = " ".join(words[:cut_point]).rstrip()
                if len(truncated) > 50:
                    return truncated + "\n\n*[Response truncated — repetition detected]*"
                return text  # too short after truncation, return original
        else:
            repeat_count = 1

    # ── Check 2: Same phrase (3-5 words) repeated N+ times ────────
    for phrase_len in [3, 4, 5]:
        if len(words) < phrase_len * max_phrase_repeats:
            continue
        for start in range(len(words) - phrase_len * max_phrase_repeats):
            phrase = " ".join(words[start:start + phrase_len])
            repeats = 1
            pos = start + phrase_len
            while pos + phrase_len <= len(words):
                next_phrase = " ".join(words[pos:pos + phrase_len])
                if next_phrase == phrase:
                    repeats += 1
                    pos += phrase_len
                else:
                    break
            if repeats >= max_phrase_repeats:
                truncated = " ".join(words[:start + phrase_len]).rstrip()
                if len(truncated) > 50:
                    return truncated + "\n\n*[Response truncated — repetition detected]*"

    # ── Check 3: Character-level repetition (e.g. "step-step-step-") ──
    # Look for a short pattern repeated many times
    for pattern_len in range(5, 30):
        if len(text) < pattern_len * 5:
            continue
        # Check last chunk of text for repetition
        tail = text[-500:] if len(text) > 500 else text
        for i in range(len(tail) - pattern_len * 4):
            pattern = tail[i:i + pattern_len]
            if pattern.strip() and tail.count(pattern) >= 8:
                # Find first occurrence in full text and truncate there
                first_pos = text.find(pattern)
                if first_pos > 50:
                    return text[:first_pos].rstrip() + "\n\n*[Response truncated — repetition detected]*"
                break

    return text


def _reset_windows_terminal():
    """
    Reset Windows console to clean line-input mode at startup.

    Required because:
      - A previous interrupted session (e.g. Ctrl+C during model load)
        may have left the console with ECHO_INPUT or LINE_INPUT disabled,
        causing the next `python main.py` to appear "frozen" — cursor
        blinks but typing doesn't show.
      - llama.cpp's stderr diagnostics can desync the input buffer.
      - Some PowerShell sessions need ENABLE_VIRTUAL_TERMINAL_PROCESSING
        explicitly set for ANSI color codes to work.
      - tqdm/colorama/huggingface progress bars call SetConsoleMode and
        do not always restore it cleanly on exit.

    Safe no-op on non-Windows.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32

        STD_INPUT_HANDLE = -10
        STD_OUTPUT_HANDLE = -11
        ENABLE_PROCESSED_INPUT = 0x0001
        ENABLE_LINE_INPUT = 0x0002
        ENABLE_ECHO_INPUT = 0x0004
        ENABLE_PROCESSED_OUTPUT = 0x0001
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

        # Restore stdin to default cooked mode (echo + line buffer + ctrl-c)
        h_in = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if h_in and h_in != -1:
            kernel32.SetConsoleMode(
                h_in,
                ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT,
            )

        # Ensure stdout supports ANSI escape codes (color, cursor moves)
        h_out = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        if h_out and h_out != -1:
            current = wintypes.DWORD()
            if kernel32.GetConsoleMode(h_out, ctypes.byref(current)):
                kernel32.SetConsoleMode(
                    h_out,
                    current.value | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
                )

        # Drain any leftover keystrokes from an interrupted prior session
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except ImportError:
            pass

    except Exception:
        # Never fail startup just because terminal reset didn't work
        pass


def _force_terminal_restore_on_exit():
    """
    Register an atexit handler that restores the Windows console to
    cooked mode on EVERY exit path — clean quit, sys.exit, uncaught
    exception, anything.

    This is the critical fix for the "second run can't type" bug:
    libraries like tqdm, colorama, and llama-cpp-python modify the
    console mode at runtime and don't always restore it on exit.
    Without this handler, the next `python main.py` in the same
    terminal inherits the broken state and `input()` doesn't echo.
    """
    if sys.platform != "win32":
        return
    try:
        import atexit

        def _on_exit():
            try:
                # Force terminal back to cooked mode
                _reset_windows_terminal()
                # Flush whatever's pending so the shell prompt renders cleanly
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
            except Exception:
                pass

        atexit.register(_on_exit)
    except Exception:
        pass


def main():
    # FIRST — reset terminal in case a prior session left it broken,
    # AND register an atexit hook so we restore it cleanly on this exit too
    _reset_windows_terminal()
    _force_terminal_restore_on_exit()

    print_banner()
    print(f"  {C.DIM}Initializing LeanAI (full integration)...{C.RESET}")

    # ══════════════════════════════════════════════════════════════
    # INITIALIZATION — all components
    # ══════════════════════════════════════════════════════════════

    engine = LeanAIEngine(verbose=False)
    executor = CodeExecutor()
    indexer = ProjectIndexer()

    # ── Model Manager (multi-model switching) ─────────────────────
    model_mgr = ModelManager()
    current_model_key = "qwen-7b"  # track which model is loaded

    # ── Model function (shared by build, swarm, TDD) ──────────────
    def model_fn(system_prompt: str, user_prompt: str) -> str:
        if not engine._model:
            engine._load_model()
        fmt = getattr(engine, "prompt_format", "chatml")
        model_name = os.path.basename(engine.model_path or "").lower()
        is_qwen35 = "qwen3.5" in model_name or "qwen35" in model_name
        if fmt == "chatml":
            if is_qwen35:
                prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n</think>\n"
                )
            else:
                prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            stop = ["<|im_end|>", "<|im_start|>"]
        else:
            prompt = (
                f"<|system|>\n{system_prompt}<|end|>\n"
                f"<|user|>\n{user_prompt}<|end|>\n"
                f"<|assistant|>\n"
            )
            stop = ["<|end|>", "<|user|>", "<|assistant|>"]
        result = engine._model(prompt, max_tokens=1024, temperature=0.1, stop=stop)
        text = result["choices"][0]["text"].strip()
        # Strip think/channel blocks from ALL model_fn callers (MoA, code_quality, etc.)
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in text:
            text = text.split("<think>")[0].strip()
        text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL).strip()
        return text

    def swarm_model_fn(prompt: str, temperature: float) -> str:
        if not engine._model:
            engine._load_model()
        fmt = getattr(engine, "prompt_format", "chatml")
        if fmt == "chatml":
            full = (
                f"<|im_start|>system\nYou are a helpful, accurate AI assistant.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            stop = ["<|im_end|>", "<|im_start|>"]
        else:
            full = (
                f"<|system|>\nYou are a helpful, accurate AI assistant.<|end|>\n"
                f"<|user|>\n{prompt}<|end|>\n"
                f"<|assistant|>\n"
            )
            stop = ["<|end|>", "<|user|>", "<|assistant|>"]
        result = engine._model(full, max_tokens=512, temperature=temperature, stop=stop)
        return result["choices"][0]["text"].strip()

    # ── Phase 4d: Agentic builder ─────────────────────────────────
    build_handler = BuildHandler(model_fn=model_fn, verbose=False)

    # ── Phase 5a: Swarm consensus ─────────────────────────────────
    swarm = SwarmConsensus(model_fn=swarm_model_fn, num_passes=3, verbose=True)

    # ── Phase 6b: Liquid adaptive router ──────────────────────────
    liquid_router = LiquidRouter()

    # ── Phase 6c: HDC fast memory ─────────────────────────────────
    hdc = HDKnowledgeStore()

    # ── Phase 7a: Project brain ───────────────────────────────────
    brain = None  # loaded when user runs /brain

    # ── Phase 7b: Git intelligence ────────────────────────────────
    git_intel = GitIntel(".")  # current directory

    # ── Phase 7c: TDD loop ────────────────────────────────────────
    tdd = TDDLoop(model_fn=model_fn, config=TDDConfig(max_attempts=5, verbose=True))

    # ── Phase 7d: Multi-file editor ───────────────────────────────
    editor = MultiFileEditor(".")

    # ── Reasoning Engine (chain-of-thought + self-critique) ───────
    reasoner = ReasoningEngine(model_fn=model_fn)

    # ── Writing Engine (outline → draft → edit) ──────────────────
    writer = WritingEngine(model_fn=model_fn)

    # ── Speed Optimizer (caching + optimization) ─────────────────
    speed = SpeedOptimizer()

    # ── Continuous Fine-Tuning ────────────────────────────────────
    ft_pipeline = TrainingDataPipeline()
    ft_adapters = AdapterManager()
    ft_runner = FineTuneRunner(pipeline=ft_pipeline, adapter_mgr=ft_adapters)

    # ── Streaming ─────────────────────────────────────────────────
    streamer = StreamingGenerator(model=None, prompt_format="chatml", config=StreamConfig())

    # ── Auto-Recovery ─────────────────────────────────────────────
    recovery = AutoRecovery(
        config=RecoveryConfig(max_retries=2, oom_fallback_enabled=True),
        on_fallback=lambda err: print(f"\n  [Recovery] {err[:80]}\n  [Recovery] Falling back to smaller model...", flush=True),
    )

    # ── Phase 7e: Session continuity ──────────────────────────────
    sessions = SessionStore()
    current_session = sessions.new_session(project_path=os.path.abspath("."))

    # ── Smart Context (after sessions is created) ─────────────────
    smart_ctx = SmartContext(brain=None, git_intel=git_intel, session_store=sessions, hdc=hdc)

    # ── Autocomplete ──────────────────────────────────────────────
    completer = AutoCompleter(brain=None)

    # ── Predictive Pre-Generation ─────────────────────────────────
    predictor = PredictivePreGenerator(generate_fn=None)

    # NOVEL: Anticipatory Generation — pre-generate predicted next answer in background
    import threading
    from core.predictor import predict_follow_ups
    _anticipatory_thread = None
    _anticipatory_cache = {}

    def anticipatory_generate(predicted_query):
        """Generate answer for predicted query in background thread."""
        try:
            if engine._model is None:
                return
            resp = engine.generate(predicted_query, project_context="")
            result_text = getattr(resp, "text", str(resp))
            if result_text and len(result_text.strip()) > 30:
                _anticipatory_cache[predicted_query.lower().strip()] = (result_text, 0.7)
        except Exception:
            pass

    def check_anticipatory(query):
        """Check if we pre-generated this query's answer."""
        key = query.lower().strip()
        for cached_q, (resp, conf) in list(_anticipatory_cache.items()):
            if key == cached_q or (len(key) > 10 and (key in cached_q or cached_q in key)):
                return resp, conf
        return None

    def stop_anticipatory():
        """Wait for background thread to finish before starting new generation."""
        nonlocal _anticipatory_thread
        if _anticipatory_thread and _anticipatory_thread.is_alive():
            _anticipatory_thread.join(timeout=2)

    def start_anticipatory(query, response):
        """Predict next query and start generating in background."""
        nonlocal _anticipatory_thread
        if _anticipatory_thread and _anticipatory_thread.is_alive():
            return
        predictions = predict_follow_ups(query, response, max_predictions=1)
        if predictions:
            _anticipatory_thread = threading.Thread(
                target=anticipatory_generate, args=(predictions[0],), daemon=True
            )
            _anticipatory_thread.start()

    # ── Semantic Git Bisect ───────────────────────────────────────
    semantic_bisect = SemanticGitBisect(repo_path=".", model_fn=model_fn)

    # ── Cross-Session Evolution ───────────────────────────────────
    evolution = EvolutionTracker()

    # ── Adversarial Verification ──────────────────────────────────
    adversarial = AdversarialVerifier()

    # ── Code Quality Enhancer (two-pass review) ───────────────────
    code_enhancer = CodeQualityEnhancer(model_fn=model_fn, enabled=True)

    # ── Code-Grounded Verification (fact-checks against AST) ──
    code_verifier = CodeGroundedVerifier(brain=None)
    agac = AGACEngine(brain=None)

    # ── Cascade Inference (7B draft → 32B review) ─────────────
    cascade = CascadeInference(draft_fn=model_fn, review_fn=model_fn, enabled=True)

    # ── Tool-Augmented Reasoning (ReAct) ──────────────────────
    react = ReActReasoner(model_fn=model_fn, max_steps=3)

    # ── Mixture of Agents ─────────────────────────────────────
    moa = MixtureOfAgents(model_fn=model_fn, perspectives=3, enabled=True)

    # ── DualPipe: Asymmetric GPU/CPU Speculative Decoding ──────
    from core.dual_pipe import DualPipeEngine, DualPipeConfig
    dual_pipe = None
    dual_pipe_enabled = False  # opt-in: user activates with /dualpipe on

    # ══════════════════════════════════════════════════════════════
    # STARTUP DISPLAY
    # ══════════════════════════════════════════════════════════════

    mem_backend = "unknown"
    mem_count = 0
    try:
        mem_count = engine.memory.episodic.count()
        mem_backend = engine.memory.episodic.backend
    except Exception:
        pass

    profile_count = 0
    try:
        profile_count = len(engine.memory.world.get_user_profile())
    except Exception:
        pass

    training_total = 0
    try:
        ts = engine.trainer.status()
        training_total = ts.get("total_pairs", 0)
    except Exception:
        pass

    print_status(
        model_name="not loaded (loads on first question)",
        model_mode=model_mgr.mode,
        memory_count=mem_count,
        mem_backend=f"{mem_backend} | HDC: {hdc.count}",
        profile_count=profile_count,
        training_pairs=training_total,
        session_count=sessions.total_sessions,
        exchange_count=sessions.total_exchanges,
        git_branch=git_intel.current_branch() if git_intel.is_available else "not a repo",
        finetune_pairs=ft_pipeline.count,
    )
    print_commands()

    # First-run guidance — show if brain hasn't been scanned yet
    if not brain:
        print()
        print(f"  {C.fg(220)}{'━' * 58}{C.RESET}")
        print(f"  {C.fg(220)}⚡ FIRST TIME? Run these 2 commands to unlock LeanAI's power:{C.RESET}")
        print(f"  {C.fg(220)}  1. /brain .        → scans your project (AST + dependency graph){C.RESET}")
        print(f"  {C.fg(220)}  2. /model auto     → smart 4-model routing by query type{C.RESET}")
        print(f"  {C.fg(220)}  Then ask about YOUR code — that's where LeanAI beats cloud AI.{C.RESET}")
        print(f"  {C.fg(220)}{'━' * 58}{C.RESET}")
        print()

    # ══════════════════════════════════════════════════════════════
    # COMMAND LOOP
    # ══════════════════════════════════════════════════════════════

    ctrl_c_count = 0  # require two consecutive Ctrl+C to actually exit

    while True:
        try:
            user_input = _safe_input(get_prompt()).strip()
            ctrl_c_count = 0  # reset on any successful input
        except KeyboardInterrupt:
            ctrl_c_count += 1
            if ctrl_c_count >= 2:
                print(f"\n  {C.DIM}Goodbye!{C.RESET}")
                sessions.end_session(current_session.id)
                sessions.save_all()
                try:
                    speed.cache._save()
                except Exception:
                    pass
                break
            print(f"\n  {C.DIM}(Press Ctrl+C again or type /quit to exit){C.RESET}")
            continue
        except EOFError:
            print(f"\n  {C.DIM}Goodbye!{C.RESET}")
            sessions.end_session(current_session.id)
            sessions.save_all()
            try:
                speed.cache._save()
            except Exception:
                pass
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        # ── Exit ──────────────────────────────────────────────────
        # Recognize both slash-commands AND bare words so people don't
        # accidentally trigger chat generation when they type "exit"
        if cmd in ("/quit", "/exit", "/q", "exit", "quit", "bye", ":q", ":quit"):
            print(f"  {C.DIM}Goodbye!{C.RESET}")
            sessions.end_session(current_session.id)
            sessions.save_all()
            speed.cache._save()  # Save response cache to disk
            _reset_windows_terminal()  # belt-and-suspenders: restore terminal before exit
            break

        # ── Help ──────────────────────────────────────────────────
        elif cmd == "/help":
            print("""
╔═══════════════════════════════════════════════════════╗
║                    LeanAI Commands                     ║
╠═══════════════════════════════════════════════════════╣
║ CHAT                                                   ║
║  (just type)       Normal AI query                     ║
║  /swarm <question> 3-pass consensus (highest accuracy) ║
║  /run <code>       Execute Python code                 ║
║  /explain <error> Explain error + how to fix          ║
║  /test <func>     Auto-generate unit tests            ║
║  /diff            Explain last git commit             ║
║  /security <file> Scan code for vulnerabilities       ║
║                                                        ║
║ BUILD                                                  ║
║  /build <task>     Multi-step project builder           ║
║  /tdd <tests>      Write code until tests pass         ║
║  /tdd-desc <text>  Generate tests + code from desc     ║
║                                                        ║
║ REASONING (3-pass: think → critique → refine)          ║
║  /reason <question> Deep reasoning with self-critique  ║
║  /plan <task>       Structured plan with phases        ║
║  /decompose <prob>  Break into sub-problems            ║
║                                                        ║
║ WRITING (4-pass: analyze → outline → draft → edit)     ║
║  /write <doc>      Any document type (auto-detected)   ║
║  /essay <topic>    Academic essay                      ║
║  /report <topic>   Professional report                 ║
║                                                        ║
║ PROJECT BRAIN                                          ║
║  /brain <path>     Scan & index a project              ║
║  /onboard          AI summary of your project          ║
║  /describe <file>  Describe a file's contents          ║
║  /deps <file>      Show what depends on a file         ║
║  /impact <file>    Impact analysis of changing file     ║
║  /find <function>  Find a function's details            ║
║                                                        ║
║ GIT                                                    ║
║  /git activity     Recent commits (7 days)             ║
║  /git hotspots     Most frequently changed files       ║
║  /git history <f>  History of a specific file          ║
║  /git why <file>   Why was a file changed              ║
║  /git changelog    Auto-generated changelog            ║
║  /git func <name>  When was a function last changed    ║
║                                                        ║
║ REFACTOR                                               ║
║  /refs <symbol>    Find all references to a symbol     ║
║  /rename <old> <n> Rename across entire project        ║
║                                                        ║
║ MEMORY & SESSIONS                                      ║
║  /remember <fact>  Store a fact in memory              ║
║  /profile          Show what LeanAI knows about you    ║
║  /sessions         List past sessions                  ║
║  /continue         Load context from last session      ║
║  /search <query>   Search past conversations           ║
║                                                        ║
║ SYSTEM                                                 ║
║  /model             List/switch/download models        ║
║  /model auto        Auto-switch (7B/Gemma4/Qwen3.5)    ║
║  /model quality     Always use best model              ║
║  /model download X  Download a model                   ║
║  /speed            Speed optimization report           ║
║  /echo             CodeEcho acceleration stats         ║
║  /dualpipe         DualPipe speculative decoding       ║
║  /status           Full system status                  ║
║  /index <path>     Index for semantic search           ║
║  /ask <question>   Search indexed codebase             ║
║  /generate <n>     Generate training pairs             ║
║  /train            Run training cycle                  ║
║  /quit             Exit LeanAI                         ║
╚═══════════════════════════════════════════════════════╝
""")

        # ══════════════════════════════════════════════════════════
        # BUILD COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/build"):
            task = user_input[6:].strip()
            if not task:
                print("Usage: /build <task description>")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            build_handler.execute_build(task)
            sessions.add_exchange(query=user_input, response="[build completed]", tier="build")

        elif cmd.startswith("/tdd-desc"):
            desc = user_input[9:].strip()
            if not desc:
                print("Usage: /tdd-desc <description of what to build>")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            print(f"[TDD] Generating tests + code for: {desc}", flush=True)
            result = tdd.run_with_description(desc)
            print(f"\n{result.summary()}")
            if result.success:
                print(f"\n── Implementation ──")
                print(result.implementation)
            sessions.add_exchange(query=user_input, response=result.summary(), tier="tdd", code_generated=result.success)

        elif cmd.startswith("/tdd"):
            test_code = user_input[4:].strip()
            if not test_code:
                print("Usage: /tdd <paste your test code>")
                print("  Or use /tdd-desc <description> to generate tests automatically")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            result = tdd.run(test_code)
            print(f"\n{result.summary()}")
            if result.success:
                print(f"\n── Implementation ({result.module_name}.py) ──")
                print(result.implementation)
            sessions.add_exchange(query=user_input, response=result.summary(), tier="tdd", code_generated=result.success)

        # ══════════════════════════════════════════════════════════
        # PROJECT BRAIN COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/brain"):
            path = user_input[6:].strip() or "."
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                print(f"Not a directory: {path}")
                continue
            print(f"[Brain] Scanning {path}...", flush=True)
            brain = ProjectBrain(path)
            result = brain.scan()
            editor = MultiFileEditor(path)  # update editor too
            git_intel = GitIntel(path)  # update git too
            smart_ctx = SmartContext(brain=brain, git_intel=git_intel, session_store=sessions, hdc=hdc)
            completer.update_brain(brain)  # update autocomplete index
            code_verifier.update_brain(brain)  # update fact-checker
            agac.update_brain(brain)  # update auto-corrector

            # Register ReAct tools with brain data
            react.register_tool("brain", brain.find_function, "Look up function/file in project")
            if git_intel and git_intel.is_available:
                react.register_tool("git", lambda q: git_intel.recent_activity(days=7), "Check git history")
            print(f"[Brain] Scanned {result['files_found']} files in {result['scan_time_ms']}ms")
            print(brain.project_summary())

        elif cmd == "/onboard":
            if not brain:
                print("Run /brain <path> first to scan a project.")
                continue
            print("[Onboard] Generating project summary...", flush=True)
            summary = brain.project_summary()
            # Get REAL filenames to prevent hallucination
            real_files = list(brain._file_analyses.keys())[:30]
            file_list = ", ".join(real_files)
            onboard_prompt = (
                f"Generate a concise project onboarding summary for a new developer.\n\n"
                f"Project data:\n{summary}\n"
                f"ACTUAL files in the project (use ONLY these names, do NOT invent filenames):\n{file_list}\n\n"
                f"Write 5-8 sentences covering: what this project does, main entry point, "
                f"key technologies, architecture pattern, and where a new developer should start. "
                f"ONLY reference files from the list above. Never invent file names."
            )
            onboard_resp = engine.generate(onboard_prompt, config=GenerationConfig(max_tokens=512),
                                            project_context="")
            onboard_text = getattr(onboard_resp, "text", str(onboard_resp))
            print(f"\n  {'─' * 55}")
            print(f"  📋 Project Onboarding Summary")
            print(f"  {'─' * 55}")
            print(format_response(onboard_text))
            separator()

        elif cmd.startswith("/describe"):
            filepath = user_input[9:].strip()
            if not filepath:
                print("Usage: /describe <filename>")
                continue
            if brain:
                print(brain.describe_file(filepath))
            else:
                print("Run /brain <path> first to scan a project.")

        elif cmd.startswith("/deps"):
            filepath = user_input[5:].strip()
            if not filepath:
                print("Usage: /deps <filename>")
                continue
            if brain:
                print(brain.what_depends_on(filepath))
            else:
                print("Run /brain <path> first.")

        elif cmd.startswith("/impact"):
            filepath = user_input[7:].strip()
            if not filepath:
                print("Usage: /impact <filename>")
                continue
            if brain:
                print(brain.impact_of_changing(filepath))
            else:
                print("Run /brain <path> first.")

        elif cmd.startswith("/find"):
            name = user_input[5:].strip()
            if not name:
                print("Usage: /find <function_name>")
                continue
            if brain:
                print(brain.find_function(name))
            else:
                print("Run /brain <path> first.")

        # ══════════════════════════════════════════════════════════
        # GIT COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/git"):
            git_cmd = user_input[4:].strip().lower()
            if not git_intel.is_available:
                print("Not a git repository. Run from a git project directory.")
                continue

            if git_cmd.startswith("activity"):
                days = 7
                parts = git_cmd.split()
                if len(parts) > 1 and parts[1].isdigit():
                    days = int(parts[1])
                print(git_intel.recent_activity(days=days))

            elif git_cmd.startswith("hotspot"):
                print(git_intel.hotspots())

            elif git_cmd.startswith("history"):
                filepath = git_cmd[7:].strip()
                if not filepath:
                    print("Usage: /git history <filename>")
                else:
                    print(git_intel.file_history(filepath))

            elif git_cmd.startswith("why"):
                filepath = git_cmd[3:].strip()
                if not filepath:
                    print("Usage: /git why <filename>")
                else:
                    print(git_intel.why_changed(filepath))

            elif git_cmd.startswith("changelog"):
                print(git_intel.generate_changelog())

            elif git_cmd.startswith("func"):
                name = git_cmd[4:].strip()
                if not name:
                    print("Usage: /git func <function_name>")
                else:
                    print(git_intel.function_last_changed(name))

            else:
                print("Git commands: activity, hotspots, history <file>, why <file>, changelog, func <name>")

        # ══════════════════════════════════════════════════════════
        # REFACTOR COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/refs"):
            symbol = user_input[5:].strip()
            if not symbol:
                print("Usage: /refs <symbol_name>")
                continue
            print(editor.find_references_summary(symbol))

        elif cmd.startswith("/rename"):
            parts = user_input[7:].strip().split()
            if len(parts) < 2:
                print("Usage: /rename <old_name> <new_name>")
                continue
            old_name, new_name = parts[0], parts[1]
            plan = editor.rename(old_name, new_name)
            print(plan.preview())
            if plan.num_edits > 0:
                confirm = input("\nApply these changes? (y/n): ").strip().lower()
                if confirm == "y":
                    result = editor.apply(plan)
                    print(f"Applied {result['applied']} edits across {result['files']} files.")
                    if brain:
                        brain.scan(force=True)
                else:
                    print("Cancelled.")

        # ══════════════════════════════════════════════════════════
        # MEMORY & SESSION COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/remember"):
            fact = user_input[9:].strip()
            if not fact:
                print("Usage: /remember <fact>")
                continue
            try:
                engine.remember(fact)
                hdc.add(fact, {"type": "user_fact", "time": time.time()})
                print(f'Stored in memory + HDC: "{fact}"')
            except Exception as e:
                print(f"Memory error: {e}")

        elif cmd == "/profile":
            try:
                profile = engine.get_profile()
                if profile:
                    print("What I know about you:")
                    for k, v in profile.items():
                        print(f"  {k}: {v}")
                else:
                    print("No profile data yet.")
            except Exception as e:
                print(f"Profile error: {e}")

        elif cmd == "/sessions":
            print(sessions.list_sessions_summary(limit=10))

        elif cmd == "/continue":
            ctx = sessions.get_continuation_context(max_exchanges=5)
            if ctx:
                print("Loaded context from last session:")
                print(ctx[:500])
            else:
                print("No previous sessions found.")

        elif cmd.startswith("/search"):
            query = user_input[7:].strip()
            if not query:
                print("Usage: /search <query>")
                continue
            print(sessions.search_summary(query))

        # ══════════════════════════════════════════════════════════
        # SWARM & RUN
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/swarm"):
            query = user_input[6:].strip()
            if not query:
                print("Usage: /swarm <question>")
                continue
            if not engine._model:
                print("[Swarm] Loading model...", flush=True)
                engine._load_model()
            print(f"[Swarm] Running 3-pass consensus...", flush=True)
            result = swarm.query(query)
            print(f"\nLeanAI (swarm):\n{result.best_answer}")
            print("───────────────────────────────────────────────────────")
            print(result.summary())
            sessions.add_exchange(query=user_input, response=result.best_answer,
                                 tier="swarm", confidence=result.confidence)

        elif cmd.startswith("/run"):
            code = user_input[4:].strip()
            if not code:
                print("Usage: /run <python code>")
                continue
            result = executor.execute(code)
            stdout = getattr(result, "stdout", "") or getattr(result, "output", "")
            stderr = getattr(result, "stderr", "") or getattr(result, "error", "")
            success = getattr(result, "success", False)
            if success:
                print(f"PASSED ({int(getattr(result, 'execution_time_ms', 0))}ms)")
                if stdout:
                    print(f"Output: {stdout}")
            else:
                print(f"FAILED: {stderr}")

        # ══════════════════════════════════════════════════════════
        # EXPLAIN — paste an error, get a fix
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/explain"):
            error_text = user_input[8:].strip()
            if not error_text:
                print("Usage: /explain <error message or traceback>")
                print("  Paste any error and LeanAI explains what went wrong + how to fix it.")
                continue

            print(f"  {C.DIM}Analyzing error...{C.RESET}", flush=True)

            # Build a focused prompt for error explanation
            explain_prompt = (
                "A developer got this error. Explain:\n"
                "1. What went wrong (in plain English, one sentence)\n"
                "2. Why it happened (the root cause)\n"
                "3. How to fix it (show the corrected code)\n"
                "4. How to prevent it in the future (one tip)\n\n"
                "Keep it concise. No filler.\n\n"
                f"Error:\n{error_text}"
            )

            # Add project context if brain is loaded
            if brain:
                # Try to find relevant files from the error traceback
                error_lower = error_text.lower()
                file_context = ""
                for rel_path in list(brain._file_analyses.keys())[:200]:
                    fname = os.path.basename(rel_path).lower()
                    if fname in error_lower:
                        desc = brain.describe_file(rel_path)
                        if desc:
                            file_context += f"\n[File: {rel_path}]\n{desc[:400]}"
                            break
                if file_context:
                    explain_prompt += f"\n\nProject context:{file_context}"

            config = GenerationConfig(max_tokens=1024, temperature=0.1)
            resp = engine.generate(explain_prompt, config=config)
            text = getattr(resp, "text", str(resp))

            print_response_header()
            print(format_response(text))
            separator()

            confidence = getattr(resp, "confidence", 0.7)
            if isinstance(confidence, float) and 0 < confidence <= 1.0:
                confidence *= 100
            print(format_confidence(confidence, "Explanation"))
            sessions.add_exchange(query=user_input, response=text, tier="explain", confidence=confidence)

        # ══════════════════════════════════════════════════════════
        # TEST — auto-generate unit tests for a function
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/test"):
            code_or_func = user_input[5:].strip()
            if not code_or_func:
                print("Usage: /test <function code or function name>")
                print("  Generates unit tests for a function.")
                continue

            print(f"  {C.DIM}Generating tests...{C.RESET}", flush=True)

            # Check if it's a function name from brain
            func_context = ""
            if brain and not code_or_func.startswith("def "):
                info = brain.find_function(code_or_func)
                if info and "not found" not in info.lower():
                    func_context = info
                    # Try to read the actual source
                    for rel_path in brain._file_analyses.keys():
                        analysis = brain._file_analyses[rel_path]
                        for func in analysis.get("functions", []):
                            if func.get("name", "").lower() == code_or_func.lower():
                                try:
                                    full_path = os.path.join(brain.config.project_path, rel_path)
                                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                        lines = f.readlines()
                                    start = func.get("line", 1) - 1
                                    end = min(start + 30, len(lines))
                                    func_context = "".join(lines[start:end])
                                except Exception:
                                    pass
                                break

            test_prompt = (
                "Generate comprehensive pytest unit tests for this function. Include:\n"
                "- Happy path tests (normal inputs)\n"
                "- Edge cases (empty, None, negative, zero, very large)\n"
                "- Error cases (wrong types, invalid inputs)\n"
                "- Boundary conditions\n\n"
                "Show ONLY the test code in a single ```python block. "
                "Include necessary imports. Make tests runnable as-is.\n\n"
            )

            if func_context:
                test_prompt += f"Function:\n```python\n{func_context}\n```"
            else:
                test_prompt += f"Function:\n```python\n{code_or_func}\n```"

            config = GenerationConfig(max_tokens=1536, temperature=0.1)
            resp = engine.generate(test_prompt, config=config)
            text = getattr(resp, "text", str(resp))

            print_response_header()
            print(format_response(text))
            separator()

            confidence = getattr(resp, "confidence", 0.7)
            if isinstance(confidence, float) and 0 < confidence <= 1.0:
                confidence *= 100
            print(format_confidence(confidence, "Tests Generated"))
            sessions.add_exchange(query=user_input, response=text, tier="test", confidence=confidence)

        # ══════════════════════════════════════════════════════════
        # DIFF — explain what changed in last commit
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/diff"):
            if not git_intel or not git_intel.is_available:
                print("Not a git repository.")
                continue

            print(f"  {C.DIM}Analyzing last commit...{C.RESET}", flush=True)

            try:
                import subprocess as sp
                result = sp.run(["git", "diff", "HEAD~1", "--stat"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=".")
                diff_stat = result.stdout.strip()
                result2 = sp.run(["git", "log", "-1", "--pretty=%s"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=".")
                commit_msg = result2.stdout.strip()
                result3 = sp.run(["git", "diff", "HEAD~1", "--", "*.py"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=".")
                diff_content = result3.stdout[:3000]  # first 3000 chars of actual diff

                diff_prompt = (
                    "Explain this git commit in plain English. What changed, why it matters, "
                    "and any risks or things to watch out for.\n\n"
                    f"Commit message: {commit_msg}\n\n"
                    f"Files changed:\n{diff_stat}\n\n"
                    f"Code diff (first 3000 chars):\n{diff_content}"
                )

                config = GenerationConfig(max_tokens=1024, temperature=0.1)
                resp = engine.generate(diff_prompt, config=config)
                text = getattr(resp, "text", str(resp))

                print_response_header()
                print(f"  {C.fg(75)}Commit:{C.RESET} {commit_msg}")
                print(f"  {C.DIM}{diff_stat}{C.RESET}")
                separator()
                print(format_response(text))
                separator()

                confidence = getattr(resp, "confidence", 0.7)
                if isinstance(confidence, float) and 0 < confidence <= 1.0:
                    confidence *= 100
                print(format_confidence(confidence, "Diff Analysis"))
                sessions.add_exchange(query=user_input, response=text, tier="diff", confidence=confidence)

            except Exception as e:
                print(f"Error: {e}")

        # ══════════════════════════════════════════════════════════
        # SECURITY — scan code for vulnerabilities
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/sentinel") or cmd.startswith("/security"):
            # Parse args: /sentinel [target] [--model] [--severity LEVEL]
            tokens = user_input.split()
            target = None
            use_model = False
            severity_floor = Severity.LOW

            i = 1
            while i < len(tokens):
                tok = tokens[i]
                if tok == "--model":
                    use_model = True
                elif tok == "--severity" and i + 1 < len(tokens):
                    sev = tokens[i + 1].upper()
                    if sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
                        severity_floor = Severity[sev]
                    i += 1
                elif not tok.startswith("--"):
                    target = tok
                i += 1

            if brain is None:
                print(f"  {C.DIM}Sentinel needs the project brain. Run /brain . first.{C.RESET}")
                continue

            print(f"  {C.DIM}Sentinel: AST-grounded security analysis...{C.RESET}", flush=True)

            # Build a model_fn wrapper if --model requested
            model_fn = None
            if use_model:
                def _sentinel_model(prompt: str) -> str:
                    cfg = GenerationConfig(max_tokens=128, temperature=0.1)
                    r = engine.generate(prompt, config=cfg)
                    return getattr(r, "text", str(r))
                model_fn = _sentinel_model

            try:
                sentinel = SentinelEngine(brain, model_fn=model_fn)
                findings, stats = sentinel.scan(
                    target=target,
                    severity_floor=severity_floor,
                    use_model=use_model,
                    verbose=True,
                )
                print(format_findings_report(findings, stats, color=True))

                # Save to session
                summary = (
                    f"Sentinel: {len(findings)} findings "
                    f"({', '.join(f'{c} {s}' for s, c in stats.by_severity.items())}) "
                    f"in {stats.time_ms:.0f}ms"
                )
                sessions.add_exchange(query=user_input, response=summary, tier="sentinel", confidence=85)
            except Exception as e:
                print(f"  {C.DIM}Sentinel error: {e}{C.RESET}")

        # ══════════════════════════════════════════════════════════
        # INDEX & ASK
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/index"):
            path = user_input[6:].strip()
            if not path:
                print("Usage: /index <path>")
                continue
            path = os.path.abspath(path)
            if not os.path.isdir(path):
                print(f"Not a directory: {path}")
                continue
            print(f"Indexing {path}...", flush=True)
            start = time.time()
            stats = indexer.index_project(path)
            elapsed = time.time() - start
            print(f"Indexed {stats.get('files_indexed', 0)} files, {stats.get('chunks_created', 0)} chunks in {elapsed:.1f}s")

        elif cmd.startswith("/ask"):
            query = user_input[4:].strip()
            if not query:
                print("Usage: /ask <question>")
                continue
            results = indexer.search(query, top_k=5)
            if not results:
                print("No results. Run /index <path> first.")
                continue
            for r in results:
                score = r.get("score", 0)
                filepath = r.get("filepath", "?")
                chunk = r.get("chunk", "")[:150].replace("\n", " ")
                print(f"  [{score:.0%}] {filepath}: {chunk}")

        # ══════════════════════════════════════════════════════════
        # TRAINING
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/generate"):
            n = int(user_input[9:].strip() or "10")
            print(f"Generating {n} self-play training pairs...", flush=True)
            try:
                engine.generate_training_data(n)
                ts = engine.trainer.status()
                print(f"Total: {ts.get('total_pairs', '?')} | Quality: {ts.get('quality_pairs', '?')}")
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == "/train":
            print("Running training cycle...", flush=True)
            try:
                result = engine.trigger_training()
                print(f"Status: {result.get('status', '?')} | Pairs: {result.get('pairs', 0)}")
            except Exception as e:
                print(f"Error: {e}")

        elif cmd == "/trainstatus":
            try:
                ts = engine.training_status()
                print(f"Total pairs: {ts.get('total_pairs', 0)} | Quality: {ts.get('quality_pairs', 0)}")
                print(f"Runs: {ts.get('runs_completed', 0)} | Last: {ts.get('last_run_status', 'n/a')}")
            except Exception as e:
                print(f"Error: {e}")

        # ══════════════════════════════════════════════════════════
        # REASONING COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/reason") or cmd.startswith("/think"):
            query = user_input.split(None, 1)[1] if " " in user_input else ""
            if not query:
                print("Usage: /reason <complex question>")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            print("[Reasoning] 3-pass: think → critique → refine...", flush=True)
            result = reasoner.reason(query, verbose=True)
            print(f"\nLeanAI (reasoned):\n{result.final_answer}")
            print(f"───────────────────────────────────────────────────────")
            print(f"{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.final_answer, tier="reasoning")

        elif cmd.startswith("/plan"):
            query = user_input[5:].strip()
            if not query:
                print("Usage: /plan <what to plan>")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            print("[Planning] Generating structured plan...", flush=True)
            result = reasoner.plan(query, verbose=True)
            print(f"\nLeanAI (plan):\n{result.final_answer}")
            print(f"───────────────────────────────────────────────────────")
            print(f"{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.final_answer, tier="planning")

        elif cmd.startswith("/decompose"):
            query = user_input[10:].strip()
            if not query:
                print("Usage: /decompose <complex problem>")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            print("[Decompose] Breaking into sub-problems...", flush=True)
            result = reasoner.decompose(query, verbose=True)
            print(f"\nLeanAI (decomposed):\n{result.final_answer}")
            print(f"───────────────────────────────────────────────────────")
            print(f"{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.final_answer, tier="decompose")

        # ══════════════════════════════════════════════════════════
        # WRITING COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/write"):
            query = user_input[6:].strip()
            if not query:
                print("Usage: /write <what to write>")
                print("  /write blog post about AI trends")
                print("  /write proposal for new CI/CD pipeline")
                print("  /write README for my project")
                continue
            if not engine._model:
                print("[LeanAI] Loading model...", flush=True)
                engine._load_model()
            print("[Writing] 4-pass: analyze → outline → draft → edit...", flush=True)
            result = writer.write(query, verbose=True)
            print(f"\nLeanAI ({result.doc_type}):\n{result.final_text}")
            print(f"───────────────────────────────────────────────────────")
            print(f"{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.final_text, tier="writing")

        elif cmd.startswith("/essay"):
            topic = user_input[6:].strip()
            if not topic:
                print("Usage: /essay <topic>")
                continue
            if not engine._model:
                engine._load_model()
            print("[Writing] Essay: outline → draft → edit...", flush=True)
            result = writer.write(topic, doc_type="essay", verbose=True)
            print(f"\nLeanAI (essay):\n{result.final_text}")
            print(f"───────────────────────────────────────────────────────")
            print(f"{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.final_text, tier="writing")

        elif cmd.startswith("/report"):
            topic = user_input[7:].strip()
            if not topic:
                print("Usage: /report <topic>")
                continue
            if not engine._model:
                engine._load_model()
            print("[Writing] Report: outline → draft → edit...", flush=True)
            result = writer.write(topic, doc_type="report", verbose=True)
            print(f"\nLeanAI (report):\n{result.final_text}")
            print(f"───────────────────────────────────────────────────────")
            print(f"{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.final_text, tier="writing")

        # ══════════════════════════════════════════════════════════
        # MODEL MANAGEMENT
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/model"):
            subcmd = user_input[6:].strip().lower()

            if not subcmd or subcmd == "list":
                print(model_mgr.list_models())

            elif subcmd == "auto":
                model_mgr.set_mode("auto")
                print("Mode set to AUTO — 7B simple, Gemma 4 medium, Qwen3.5 complex.")

            elif subcmd == "fast":
                model_mgr.set_mode("fast")
                print("Mode set to FAST — always use smallest model.")

            elif subcmd == "quality":
                model_mgr.set_mode("quality")
                print("Mode set to QUALITY — always use best model.")

            elif subcmd.startswith("download"):
                target = subcmd.replace("download", "").strip()
                if not target:
                    print("Usage: /model download qwen-32b")
                    print(f"Available: {list(model_mgr.models.keys())}")
                    continue
                print(f"Downloading {target}... (this takes a while)")
                success, msg = model_mgr.download(target)
                print(msg)

            elif subcmd in model_mgr.models:
                # Switch to a specific model
                model_key = subcmd
                model_info = model_mgr.get_model_info(model_key)
                model_path = model_mgr.get_model_path(model_key)
                if not model_path:
                    print(f"Model {model_key} not downloaded. Run: /model download {model_key}")
                    continue
                print(f"Switching to {model_info.name}...")
                try:
                    engine.switch_model(model_path)
                    current_model_key = model_key
                    print(f"Loaded {model_info.name} ({model_info.speed_label}, quality: {model_info.quality_score}%)")
                except Exception as e:
                    print(f"Error loading model: {e}")

            else:
                print("Model commands:")
                print("  /model              List available models")
                print("  /model auto         Auto-switch by query complexity")
                print("  /model fast         Always use fastest model")
                print("  /model quality      Always use best model")
                print("  /model gemma4-26b   Use Gemma 4 (fast, reliable)")
                print("  /model qwen35-27b   Use Qwen3.5 (best quality)")
                print("  /model qwen3-coder  Use Qwen3 Coder 30B MoE")
                print("  /model download X   Download a model")

        # ══════════════════════════════════════════════════════════
        # STATUS
        # ══════════════════════════════════════════════════════════

        elif cmd == "/speed":
            print(speed.optimization_report())
            cs = speed.cache.stats()
            print(f"\nCache: {cs['entries']} responses cached | {cs['hits']} hits | {cs['hit_rate']} hit rate")

        elif cmd == "/echo":
            es = engine.code_echo.stats()
            print("═══ CodeEcho: Source-Grounded Speculative Decoding ═══")
            print(f"  API available: {'yes' if es['api_available'] else 'no (run a query with file content to check)'}")
            print(f"  Generations:   {es['generations']}")
            print(f"  Total tokens:  {es['total_tokens']}")
            print(f"  Echoed tokens: {es['echoed_tokens']} ({es['echo_ratio']})")
            print(f"  Echo events:   {es['echo_events']}")
            print(f"  Avg speedup:   {es['avg_speedup']}")
            if es['generations'] == 0:
                print("\n  Tip: Ask about a file in your project (e.g. 'review engine_v3.py')")
                print("  CodeEcho activates when the model reproduces your source code.")

        elif cmd.startswith("/dualpipe"):
            arg = user_input[9:].strip().lower()
            if arg == "on":
                # Initialize DualPipe with 7B (draft) and 27B (target)
                draft_path = model_mgr.get_model_path("qwen-7b")
                target_path = model_mgr.get_model_path("qwen35-27b")
                if not draft_path or not target_path:
                    print("DualPipe requires both 7B and 27B models downloaded.")
                    print(f"  7B:  {'found' if draft_path else 'MISSING — /model download qwen-7b'}")
                    print(f"  27B: {'found' if target_path else 'MISSING — /model download qwen35-27b'}")
                    continue
                # Unload engine's model to free RAM
                if engine._model is not None:
                    del engine._model
                    engine._model = None
                    engine._model_loaded = False
                dual_pipe = DualPipeEngine(draft_path, target_path,
                    DualPipeConfig(verbose=True, n_threads=engine.n_threads))
                if dual_pipe.load():
                    dual_pipe_enabled = True
                    print("DualPipe ON — 7B drafts on GPU, 27B verifies on CPU")
                else:
                    dual_pipe = None
                    dual_pipe_enabled = False
                    print("DualPipe failed to load. Falling back to normal generation.")
            elif arg == "off":
                dual_pipe_enabled = False
                if dual_pipe:
                    dual_pipe.unload()
                    dual_pipe = None
                print("DualPipe OFF — back to normal generation.")
                print("Note: run a query to reload the model.")
            else:
                # Show stats
                print("═══ DualPipe: Asymmetric GPU/CPU Speculative Decoding ═══")
                if dual_pipe and dual_pipe_enabled:
                    ds = dual_pipe.stats()
                    print(f"  Status:          ON")
                    print(f"  Generations:     {ds['generations']}")
                    print(f"  Total tokens:    {ds['total_tokens']}")
                    print(f"  Accepted chunks: {ds['accepted_chunks']} ({ds['acceptance_rate']})")
                    print(f"  Rejected chunks: {ds['rejected_chunks']}")
                    print(f"  Draft time:      {ds['draft_time_ms']}ms")
                    print(f"  Target time:     {ds['target_time_ms']}ms")
                    print(f"  Effective speed: {ds['effective_tok_s']} tok/s")
                    print(f"  Speedup:         {ds['speedup']}")
                else:
                    print(f"  Status: OFF")
                    print(f"\n  Turn on:  /dualpipe on")
                    print(f"  Turn off: /dualpipe off")
                    print(f"  Requires: 7B + 27B models (both ~21GB in RAM via mmap)")

        elif cmd.startswith("/complete"):
            prefix = user_input[9:].strip()
            if not prefix:
                print("Usage: /complete <prefix>")
                print("  /complete gen          → functions starting with 'gen'")
                print("  /complete Model        → classes starting with 'Model'")
                print("  /complete engine.gen   → methods on engine")
                continue
            start_time = time.time()
            results = completer.complete(prefix)
            elapsed = (time.time() - start_time) * 1000
            if results:
                print(f"Completions for '{prefix}' ({elapsed:.1f}ms):")
                for r in results:
                    kind_icon = {"function": "ƒ", "class": "◆", "keyword": "⚡", "snippet": "✂"}.get(r.kind, "·")
                    print(f"  {kind_icon} {r.label:<40} {r.detail}")
            else:
                print(f"No completions for '{prefix}'")
            cs = completer.stats()
            print(f"Index: {cs['functions_indexed']} functions, {cs['classes_indexed']} classes")

        # ══════════════════════════════════════════════════════════
        # SEMANTIC GIT BISECT
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/bisect"):
            query = user_input[7:].strip()
            if not query:
                print("Usage: /bisect <bug description>")
                print("  /bisect authentication stopped working")
                print("  /bisect tests failing after refactor")
                continue
            semantic_bisect.repo_path = os.path.abspath(".")
            result = semantic_bisect.find_bug(query, num_commits=20, verbose=True)
            print(f"\n{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.summary(), tier="bisect")

        # ══════════════════════════════════════════════════════════
        # CROSS-SESSION EVOLUTION
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/evolution") or cmd.startswith("/evolve"):
            subcmd = user_input.split(None, 1)[1] if " " in user_input else "narrative"
            if subcmd == "narrative":
                print(evolution.get_narrative())
            elif subcmd == "insights":
                insights = evolution.get_insights()
                if insights:
                    for ins in insights:
                        print(f"\n{ins.summary()}")
                else:
                    print("No evolution insights yet. Keep using LeanAI!")
            elif subcmd == "predict":
                preds = evolution.predict_next_topics()
                if preds:
                    print("Predicted next topics:")
                    for p in preds:
                        print(f"  → {p}")
                else:
                    print("Not enough data to predict yet.")
            elif subcmd == "stats":
                s = evolution.stats()
                print(f"Themes tracked: {s['themes_tracked']}")
                print(f"Total occurrences: {s['total_occurrences']}")
                for t in s.get("active_themes", []):
                    print(f"  {t}")
            else:
                print("Evolution commands:")
                print("  /evolution narrative  — project story")
                print("  /evolution insights   — evolution insights")
                print("  /evolution predict    — predict next topics")
                print("  /evolution stats      — tracking stats")

        # ══════════════════════════════════════════════════════════
        # ADVERSARIAL CODE VERIFICATION
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/fuzz"):
            code = user_input[5:].strip()
            if not code:
                print("Usage: /fuzz <python code>")
                print("  /fuzz def sort(arr): return sorted(arr)")
                print("  /fuzz def add(a, b): return a + b")
                continue
            print("[Fuzz] Running adversarial verification...", flush=True)
            result = adversarial.fuzz(code, verbose=True)
            print(f"\n{result.summary()}")
            sessions.add_exchange(query=user_input, response=result.summary(), tier="fuzz")

        # ══════════════════════════════════════════════════════════
        # FINE-TUNING COMMANDS
        # ══════════════════════════════════════════════════════════

        elif cmd.startswith("/finetune") or cmd.startswith("/ft"):
            subcmd = user_input.split(None, 1)[1] if " " in user_input else "status"

            if subcmd == "status" or subcmd == "check":
                print(ft_runner.check_readiness())

            elif subcmd == "collect":
                # Import from session store + training exports
                n1 = ft_pipeline.add_from_session_store(sessions)
                n2 = ft_pipeline.add_from_training_exports()
                ft_pipeline.save()
                print(f"Collected {n1} from sessions + {n2} from exports = {ft_pipeline.count} total")

            elif subcmd.startswith("train"):
                adapter_name = subcmd.replace("train", "").strip() or "default"
                print(f"[FineTune] Starting training for adapter '{adapter_name}'...")
                # First collect latest data
                ft_pipeline.add_from_session_store(sessions)
                ft_pipeline.add_from_training_exports()
                ft_pipeline.save()
                run = ft_runner.train(TrainingConfig(adapter_name=adapter_name))
                print(f"\n{run.summary()}")

            elif subcmd == "history":
                print(ft_runner.list_runs())

            elif subcmd == "adapters":
                print(ft_adapters.list_adapters())

            elif subcmd.startswith("create"):
                name = subcmd.replace("create", "").strip()
                if not name:
                    print("Usage: /finetune create <adapter_name>")
                else:
                    ft_adapters.create(name)
                    print(f"Created adapter '{name}'")

            elif subcmd.startswith("activate"):
                name = subcmd.replace("activate", "").strip()
                if ft_adapters.set_active(name):
                    print(f"Activated adapter '{name}'")
                else:
                    print(f"Adapter '{name}' not found")

            elif subcmd == "deactivate":
                ft_adapters.deactivate()
                print("Adapter deactivated — using base model")

            elif subcmd == "schedule":
                ft_runner.start_nightly_schedule(hour=2)

            elif subcmd == "export":
                path = ft_pipeline.export_sharegpt()
                print(f"Exported to {path}")

            else:
                print("Fine-tuning commands:")
                print("  /finetune status     Check readiness")
                print("  /finetune collect    Collect training data from sessions")
                print("  /finetune train      Start training")
                print("  /finetune history    Show training runs")
                print("  /finetune adapters   List adapters")
                print("  /finetune activate X Use an adapter")
                print("  /finetune export     Export data for external training")
                print("  /finetune schedule   Enable nightly auto-training")

        elif cmd == "/status":
            print(f"═══ LeanAI v7 Full Status ═══")
            print(f"Model     : {current_model_key} ({getattr(engine, 'model_name', 'unknown')})")
            print(f"Mode      : {model_mgr.mode}")
            mm_stats = model_mgr.stats()
            print(f"Models    : {', '.join(mm_stats['downloaded_models'])} downloaded")
            print(f"Routing   : {mm_stats['fast_count']} fast / {mm_stats['quality_count']} quality")
            print(f"Format    : {getattr(engine, 'prompt_format', 'unknown')}")
            print(f"Threads   : {getattr(engine, 'n_threads', '?')}")
            print(f"Memory    : {mem_count} episodes | {mem_backend}")
            print(f"HDC Store : {hdc.count} entries | {hdc.stats().get('memory_kb', 0)} KB")
            print(f"Profile   : {profile_count} fields")
            print(f"Training  : {training_total} pairs")
            print(f"Sessions  : {sessions.total_sessions} total | {sessions.total_exchanges} exchanges")
            print(f"Git       : {git_intel.current_branch()} | {git_intel.stats().get('total_commits', 0)} commits")
            print(f"Brain     : {'active' if brain else 'not loaded'}")
            if brain:
                bs = brain.stats()
                print(f"  Files: {bs['files_indexed']} | Graph: {bs['graph']}")
            print(f"Router    : liquid adaptive")
            rs = liquid_router.stats()
            print(f"  Queries: {rs['queries_routed']} | Learned bins: {sum(t.get('learned_bins', 0) for t in rs['tiers'].values())}")
            if build_handler.last_result:
                r = build_handler.last_result
                print(f"Last build: {'SUCCESS' if r.success else 'FAILED'} — {r.plan.completed_steps}/{r.plan.total_steps}")

        # ══════════════════════════════════════════════════════════
        # NORMAL QUERY (with full integration)
        # ══════════════════════════════════════════════════════════

        else:
            start = time.time()
            enriched_context = ""

            # Speed: Check response cache first
            cached = speed.should_use_cache(user_input)
            if cached:
                text, confidence = cached
                print_response_header()
                print(f"  {C.DIM}⚡ from cache{C.RESET}")
                print(format_response(text))
                separator()
                print(format_confidence(confidence, "Cached"))
                print(f"  {C.BGREEN}⚡ CACHED — instant{C.RESET}")
                sessions.add_exchange(query=user_input, response=text, tier="cache", confidence=confidence)
                evolution.track_query(user_input, session_id=str(current_session.id))
                continue

            # Predictive: Check if we pre-generated this answer
            predicted = predictor.check_prediction(user_input)
            if predicted:
                text, confidence = predicted
                print_response_header()
                print(f"  {C.DIM}🔮 predicted{C.RESET}")
                print(format_response(text))
                separator()
                print(format_confidence(confidence * 100, "Predicted"))
                print(f"  {C.BMAGENTA}🔮 PREDICTED — pre-generated{C.RESET}")
                sessions.add_exchange(query=user_input, response=text, tier="predicted", confidence=confidence*100)
                evolution.track_query(user_input, session_id=str(current_session.id))
                continue

            # NOVEL: Semantic Speculative Caching — find similar cached response as draft
            semantic_draft = None
            semantic_draft_query = None
            try:
                embedder = getattr(engine, 'memory', None)
                embed_model = getattr(embedder, 'embedder', None) if embedder else None
                draft_result = speed.cache.get_semantic_draft(user_input, embedder=embed_model)
                if draft_result:
                    semantic_draft_query, semantic_draft, sim = draft_result
                    print(f"  {C.DIM}🧠 Semantic draft found (similarity: {sim:.0%}) — adapting...{C.RESET}")
            except Exception:
                pass

            # NOVEL: Anticipatory Generation — check if background thread pre-generated this
            antic = check_anticipatory(user_input)
            if antic:
                text, confidence = antic
                print_response_header()
                print(f"  {C.DIM}⚡ Anticipatory — pre-generated in background{C.RESET}")
                print(format_response(text))
                separator()
                print(format_confidence(confidence * 100, "Anticipated"))
                print(f"  {C.fg(213)}⚡ ANTICIPATED — generated while you were reading{C.RESET}")
                sessions.add_exchange(query=user_input, response=text, tier="anticipated", confidence=confidence*100)
                evolution.track_query(user_input, session_id=str(current_session.id))
                speed.cache_response(user_input, text, confidence * 100)
                speed.cache._save()
                continue

            # Phase 6b: Liquid router decides tier
            # SAFETY: Stop any background anticipatory generation before using the model
            stop_anticipatory()
            tier_suggestion = liquid_router.route(user_input)

            # Auto model selection: switch if needed
            downloaded = model_mgr.get_downloaded_models()
            if model_mgr.mode == "auto" and len(downloaded) >= 1:
                best_model = model_mgr.select_model(user_input)

                # Check if cascade will handle this query (skip auto-switch to 32B)
                # Skip cascade if auto-router selected a high-quality model (Gemma 4, Qwen3.5, Qwen3 Coder)
                high_quality_selected = best_model in ("gemma4-26b", "qwen35-27b", "qwen3-coder")
                will_cascade = (
                    cascade.enabled
                    and len(downloaded) >= 2
                    and cascade.should_cascade(user_input)
                    and not high_quality_selected
                )

                if will_cascade:
                    pass  # cascade handles model switching itself
                elif best_model != current_model_key and best_model in downloaded:
                    best_info = model_mgr.get_model_info(best_model)
                    best_path = model_mgr.get_model_path(best_model)
                    if best_path:
                        complexity = classify_complexity(user_input)
                        print(f"  [Auto] {complexity} query → switching to {best_info.name}...", flush=True)
                        try:
                            engine.switch_model(best_path)
                            current_model_key = best_model
                        except Exception:
                            pass  # stay on current model
                elif best_model == current_model_key:
                    pass  # already on the right model
                elif best_model not in downloaded and current_model_key not in downloaded:
                    # Need to load any available model
                    if downloaded:
                        fallback = downloaded[0]
                        fb_info = model_mgr.get_model_info(fallback)
                        fb_path = model_mgr.get_model_path(fallback)
                        if fb_path and fallback != current_model_key:
                            print(f"  [Auto] Loading {fb_info.name}...", flush=True)
                            try:
                                engine.switch_model(fb_path)
                                current_model_key = fallback
                            except Exception:
                                pass

            # Phase 7e: Add session context for continuity
            session_ctx = ""
            if current_session and current_session.num_exchanges > 0:
                # Include last 3 exchanges as context
                recent = current_session.exchanges[-3:]
                session_ctx = "\n".join(
                    f"Previous Q: {e.query[:100]}\nPrevious A: {e.response[:100]}"
                    for e in recent
                )

            # Phase 7a: Add project brain context if available
            brain_ctx = ""
            if brain:
                brain_ctx = brain.get_context_for_query(user_input)

            # ── ReAct: Gather tool data to enrich context ─────
            react_data = ""
            if brain and react.tools:
                tool_results = react._gather_tool_data(user_input)
                if tool_results:
                    react_parts = []
                    for tool_name, tool_query, tool_result in tool_results:
                        react_parts.append(f"[{tool_name}: {tool_query}]\n{tool_result[:500]}")
                    react_data = "\n".join(react_parts)
                    print(f"  {C.DIM}ReAct: gathered data from {len(tool_results)} tool(s){C.RESET}")

            # ── READ ACTUAL FILES (for MoA and standard paths) ──
            import re as _re
            file_patterns = _re.findall(r'[\w/\\]+\.(?:py|js|ts|jsx|tsx|go|rs|java|c|cpp|h|rb|php|sql|yaml|yml|json|toml|sh|css|html|svelte|vue|kt|swift|dart|ex|lua|zig|nim)\b', user_input)
            file_content_block = ""
            if file_patterns and brain and brain.config.project_path:
                for fpath in file_patterns[:3]:
                    fname = fpath.split('/')[-1].split('\\')[-1]
                    full = os.path.join(brain.config.project_path, fpath)
                    if not os.path.exists(full):
                        for root, dirs, files in os.walk(brain.config.project_path):
                            if fname in files:
                                full = os.path.join(root, fname)
                                break
                    if os.path.exists(full):
                        try:
                            with open(full, 'r', encoding='utf-8', errors='ignore') as _f:
                                content = _f.read()[:8000]
                            file_content_block += f"\n\n[FILE CONTENT: {fpath}]\n{content}\n[END FILE]\n"
                            print(f"  {C.DIM}📄 Reading {fpath} ({len(content)} chars){C.RESET}")
                        except Exception:
                            pass

            # ── Mixture of Agents: multi-perspective for reviews ──
            if moa.should_use_moa(user_input):
                print(f"  {C.fg(222)}⚙ Multi-perspective analysis...{C.RESET}", flush=True)
                moa_context = brain_ctx or ""
                if react_data:
                    moa_context += "\n" + react_data
                if file_content_block:
                    moa_context += file_content_block
                moa_result = moa.analyze(user_input, context=moa_context)

                if moa_result.final_answer and len(moa_result.final_answer.strip()) > 30:
                    # Truncate repetitive MoA output
                    moa_text = _truncate_repetition(moa_result.final_answer)
                    print(f"  {C.DIM}  {moa_result.summary()}{C.RESET}")

                    # Create a response-like object
                    class _MoAResp:
                        def __init__(self, text):
                            self.text = text
                            self.confidence = 0.9
                            self.confidence_label = "High"
                            self.tier_used = "moa"
                            self.latency_ms = moa_result.elapsed_ms
                            self.answered_from_memory = False
                            self.memory_context_used = False
                            self.verified = False
                            self.code_executed = False
                            self.code_passed = False
                            self.code_output = ""

                    resp = _MoAResp(moa_text)
                    elapsed = time.time() - start
                    text = resp.text
                    confidence = 90
                    conf_label = "High"
                    tier = "moa"
                    latency_ms = moa_result.elapsed_ms
                    from_mem = False
                    mem_active = False
                    verified = False
                    code_exec = False
                    code_passed = False
                    code_output = ""

                    # Code verification
                    if code_verifier.should_verify(user_input, text):
                        text = code_verifier.verify(text, query=user_input)

                    # AGAC: Auto-correct + enrich
                    if agac.brain:
                        text, _agac_s = agac.process(text, query=user_input)

                    print_response_header()
                    print(format_response(text))
                    separator()
                    print(format_confidence(confidence, conf_label))
                    meta_parts = [f"⏱ {elapsed:.0f}s", f"⧫ {moa_result.num_perspectives} perspectives"]
                    print(f"  {C.DIM}{' · '.join(meta_parts)}{C.RESET}")
                    sessions.add_exchange(query=user_input, response=text, tier=tier, confidence=confidence)
                    continue

            # Generate response with smart context + auto-recovery
            try:
                # Speed: Use optimal max_tokens for this query
                smart_tokens = speed.get_max_tokens(user_input)

                # Smart context: build enriched context
                enriched_context = smart_ctx.build(user_input)

                # Inject file content read earlier
                if file_content_block:
                    enriched_context = (enriched_context or "") + file_content_block

                # Inject ReAct tool data into context
                if react_data:
                    enriched_context = react_data + "\n\n" + enriched_context if enriched_context else react_data

                # NOVEL: Inject semantic draft for adaptation (Speculative Caching)
                if semantic_draft:
                    enriched_context = (enriched_context or "") + (
                        f"\n\n[DRAFT FROM SIMILAR QUERY: '{semantic_draft_query}']\n"
                        f"Adapt this draft for the current query. Keep what's relevant, modify what's different.\n"
                        f"{semantic_draft[:2000]}\n[END DRAFT]\n"
                    )

                # ── CASCADE INFERENCE ──────────────────────────────
                # For complex queries: 7B drafts fast, 32B reviews and corrects
                # Skip when Gemma 4, Qwen3.5, or Qwen3 Coder is loaded (they're good enough alone)
                current_is_high_quality = any(x in os.path.basename(engine.model_path or "").lower()
                    for x in ["gemma", "qwen3.5", "qwen35", "qwen3-coder-30b"])
                use_cascade = (
                    cascade.enabled
                    and model_mgr.mode == "auto"
                    and len(model_mgr.get_downloaded_models()) >= 2
                    and cascade.should_cascade(user_input)
                    and not current_is_high_quality
                )

                if use_cascade:
                    print(f"  {C.fg(141)}⚡ Cascade: 7B drafting → 32B reviewing...{C.RESET}", flush=True)

                    # Step 1: Ensure 7B is loaded for fast draft
                    small_path = model_mgr.get_model_path("qwen-7b")
                    if small_path and current_model_key != "qwen-7b":
                        engine.switch_model(small_path)
                        current_model_key = "qwen-7b"

                    # Step 2: 7B generates draft (fast — ~30 sec)
                    draft_start = time.time()
                    draft_resp = engine.generate(
                        user_input,
                        config=GenerationConfig(max_tokens=smart_tokens),
                        project_context=enriched_context,
                    )
                    draft_text = getattr(draft_resp, "text", str(draft_resp))
                    draft_ms = (time.time() - draft_start) * 1000
                    print(f"  {C.DIM}  Draft done ({draft_ms/1000:.0f}s). Reviewing...{C.RESET}", flush=True)

                    # Step 3: Switch to 32B for review
                    big_path = model_mgr.get_model_path("qwen-32b")
                    if big_path:
                        engine.switch_model(big_path)
                        current_model_key = "qwen-32b"

                        # Step 4: 32B reviews draft (focused — generates corrections only)
                        review_prompt = (
                            f"A junior developer wrote this response to the question: {user_input}\n\n"
                            f"Draft:\n{draft_text}\n\n"
                            "Fix any errors, add anything important that's missing, "
                            "improve unclear explanations. Keep what's good. "
                            "Output ONLY the improved response."
                        )
                        review_start = time.time()
                        review_resp = engine.generate(
                            review_prompt,
                            config=GenerationConfig(max_tokens=smart_tokens),
                            project_context=enriched_context,
                        )
                        review_text = getattr(review_resp, "text", str(review_resp))
                        review_ms = (time.time() - review_start) * 1000

                        # Use reviewed version if substantive
                        if review_text and len(review_text.strip()) > 30:
                            resp = review_resp
                        else:
                            resp = draft_resp

                        total_ms = draft_ms + review_ms
                        print(f"  {C.fg(114)}  Cascade complete: {draft_ms/1000:.0f}s draft + {review_ms/1000:.0f}s review = {total_ms/1000:.0f}s total{C.RESET}")
                    else:
                        # 32B not available, use draft
                        resp = draft_resp
                else:
                    # ── DUALPIPE: Asymmetric GPU/CPU Speculative Decoding ──
                    if dual_pipe_enabled and dual_pipe and dual_pipe.is_available:
                        mem_ctx = engine.memory.prepare_context(user_input)
                        history = engine.memory.working.get_context_window(max_tokens=512)
                        prompt = engine._build_prompt(user_input, mem_ctx, history,
                                                       project_context=enriched_context)

                        stop_strings = [
                            "<|im_end|>", "<|im_start|>", "<|user|>", "<|end|>",
                            "<|assistant|>", "<end_of_turn>", "<start_of_turn>",
                            "\nYou:", "\nHuman:", "\nUser:"
                        ]

                        print_response_header()
                        import sys
                        dp_text, dp_stats = dual_pipe.generate(
                            prompt, max_tokens=smart_tokens,
                            stop_strings=stop_strings,
                            callback=lambda token: (sys.stdout.write(token), sys.stdout.flush())
                        )
                        print()
                        dp_text = _truncate_repetition(dp_text)

                        if dp_stats and dp_stats.total_tokens > 0:
                            print(f"  {C.fg(208)}⚡ {dp_stats.summary()}{C.RESET}")

                        engine.memory.record_exchange(user_input, dp_text)

                        class DualPipeResp:
                            def __init__(self, t):
                                self.text = t
                                self.confidence = 0.85
                                self.confidence_label = "High"
                                self.tier_used = "dualpipe"
                                self.latency_ms = dp_stats.total_time_ms if dp_stats else 0
                                self.answered_from_memory = False
                                self.memory_context_used = bool(mem_ctx)
                                self.verified = False
                                self.code_executed = False
                                self.code_passed = False
                                self.code_output = ""
                        resp = DualPipeResp(dp_text)

                    else:
                        # ── STANDARD GENERATION ────────────────────────
                        # For high-quality models: use STREAMING (tokens appear as generated)
                        # CodeEcho also uses this path for source-grounded acceleration
                        model_basename = os.path.basename(engine.model_path or "").lower()
                        use_streaming = any(x in model_basename for x in ["qwen3.5", "qwen35", "qwen3-coder"])

                        if use_streaming:
                            # Build prompt same way engine does
                            mem_ctx = engine.memory.prepare_context(user_input)
                            history = engine.memory.working.get_context_window(max_tokens=512)
                            prompt = engine._build_prompt(user_input, mem_ctx, history,
                                                           project_context=enriched_context)
                            config = GenerationConfig(max_tokens=smart_tokens)

                            # Print header and stream tokens
                            print_response_header()
                            import sys

                            # ── CODEECHO: Source-Grounded Speculative Decoding ──
                            # Collect source material for echo detection
                            echo_sources = []
                            if file_content_block:
                                # Extract raw file contents (strip [FILE CONTENT:] markers)
                                import re as _echo_re
                                for fc_match in _echo_re.finditer(
                                    r'\[FILE CONTENT: [^\]]+\]\n(.*?)\n\[END FILE\]',
                                    file_content_block, _echo_re.DOTALL
                                ):
                                    echo_sources.append(fc_match.group(1))
                            if brain_ctx and len(brain_ctx) > 100:
                                echo_sources.append(brain_ctx)

                            echo_stats = None
                            if echo_sources:
                                streamed_text, echo_stats = engine.generate_with_codeecho(
                                    prompt, config, echo_sources,
                                    callback=lambda token: (sys.stdout.write(token), sys.stdout.flush())
                                )
                            else:
                                streamed_text = engine.generate_streaming(
                                    prompt, config,
                                    callback=lambda token: (sys.stdout.write(token), sys.stdout.flush())
                                )

                            print()  # newline after streaming

                            # Show CodeEcho stats if it was used
                            if echo_stats and echo_stats.echo_events > 0:
                                print(f"  {C.fg(81)}⚡ {echo_stats.summary()}{C.RESET}")

                            # Record in memory
                            # Truncate repetitive streamed output
                            streamed_text = _truncate_repetition(streamed_text)

                            engine.memory.record_exchange(user_input, streamed_text)

                            # Create a simple response object
                            class StreamResp:
                                def __init__(self, t):
                                    self.text = t
                                    self.confidence = 0.72
                                    self.confidence_label = "Moderate"
                                    self.tier_used = "stream"
                                    self.latency_ms = (time.time() - start) * 1000
                                    self.answered_from_memory = False
                                    self.memory_context_used = bool(mem_ctx)
                                    self.verified = False
                                    self.code_executed = False
                                    self.code_passed = False
                                    self.code_output = ""
                            resp = StreamResp(streamed_text)
                        else:
                            # Non-streaming fallback (7B, legacy 32B)
                            def do_generate(max_tokens, **kw):
                                return engine.generate(user_input,
                                                       config=GenerationConfig(max_tokens=max_tokens),
                                                       project_context=enriched_context)

                            def do_fallback(max_tokens, **kw):
                                return engine.generate(user_input,
                                                       config=GenerationConfig(max_tokens=max(max_tokens // 2, 128)),
                                                       project_context=enriched_context)

                            rec_result = recovery.safe_generate(
                                generate_fn=do_generate,
                                max_tokens=smart_tokens,
                                fallback_fn=do_fallback,
                            )

                            if rec_result.success:
                                resp = rec_result.text
                                if rec_result.recovered:
                                    print(f"  [Recovery] Recovered: {rec_result.recovery_note}")
                            else:
                                print(f"\nError: {rec_result.error}")
                                continue
            except KeyboardInterrupt:
                # User hit Ctrl+C during model loading or generation.
                # Cancel cleanly and return to the prompt instead of crashing.
                print(f"\n  {C.DIM}[Cancelled — back to prompt]{C.RESET}")
                # Stop any background pre-generation that may still be running
                try:
                    stop_anticipatory()
                except Exception:
                    pass
                # Re-flush stdio so the next prompt renders cleanly
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
                # Reset terminal mode in case llama.cpp's stderr scrambled it
                _reset_windows_terminal()
                continue
            except Exception as e:
                print(f"\nError: {e}")
                continue

            elapsed = time.time() - start

            # Read response attributes
            text = getattr(resp, "text", str(resp))
            confidence = getattr(resp, "confidence", 0.5)
            if isinstance(confidence, (int, float)) and 0 < confidence <= 1.0:
                confidence *= 100
            conf_label = getattr(resp, "confidence_label", "")
            tier = getattr(resp, "tier_used", "unknown")
            latency_ms = getattr(resp, "latency_ms", elapsed * 1000)
            from_mem = getattr(resp, "answered_from_memory", False)
            mem_active = getattr(resp, "memory_context_used", False)
            verified = getattr(resp, "verified", False)
            code_exec = getattr(resp, "code_executed", False)
            code_passed = getattr(resp, "code_passed", False)
            code_output = getattr(resp, "code_output", "")

            # ── Repetition detector (catches ALL models) ──────────────
            text = _truncate_repetition(text)

            # Print response with formatting
            # Two-pass quality: if response contains code, run a review pass
            # Skip for high-quality models (Gemma 4, Qwen3.5) — they're good enough without it
            model_name = os.path.basename(engine.model_path).lower() if engine.model_path else ""
            is_high_quality_model = any(x in model_name for x in ["gemma", "qwen3.5", "qwen35"])
            if not is_high_quality_model and code_enhancer.should_review(user_input, text):
                print(f"  {C.DIM}Running code review...{C.RESET}", end="", flush=True)
                text = code_enhancer.enhance(text, query=user_input)
                print(f"\r{' ' * 40}\r", end="", flush=True)

            # Code-Grounded Verification: check claims against actual AST
            if code_verifier.should_verify(user_input, text):
                text = code_verifier.verify(text, query=user_input)

            # AGAC: Auto-correct hallucinated identifiers + enrich with project references
            if agac.brain:
                text, agac_stats = agac.process(text, query=user_input)
                agac_summary = agac_stats.summary()
                if agac_summary:
                    print(f"  {C.fg(114)}✓ {agac_summary}{C.RESET}")

            # Print response (skip if already streamed)
            was_streamed = getattr(resp, "tier_used", "") == "stream"
            if not was_streamed:
                print_response_header()
                print(format_response(text))

            # Code execution results
            if code_exec:
                if code_passed:
                    output_lines = [l for l in code_output.strip().split("\n") if l.strip()][:10] if code_output else []
                    format_code_result(True, output="\n".join(output_lines))
                else:
                    error_msg = ""
                    if code_output:
                        for line in reversed(code_output.strip().split("\n")):
                            if line.strip() and not line.startswith(" "):
                                error_msg = line.strip()
                                break
                    format_code_result(False, error=error_msg or "execution error")

            # Confidence bar
            if code_exec and code_passed:
                confidence = max(confidence, 96)
                conf_label = "High"
            if not conf_label:
                conf_label = "High" if confidence >= 90 else "Good" if confidence >= 70 else "Moderate" if confidence >= 50 else "Low"

            # Human-readable latency
            if latency_ms < 1000:
                latency_str = f"{latency_ms:.0f}ms"
            elif latency_ms < 60000:
                latency_str = f"{latency_ms/1000:.1f}s"
            else:
                mins = int(latency_ms // 60000)
                secs = int((latency_ms % 60000) // 1000)
                latency_str = f"{mins}m {secs}s"

            separator()
            print(format_confidence(confidence, conf_label))
            print(format_meta(tier, latency_str, from_mem, verified,
                             mem_active, code_exec and code_passed, enriched_context))

            # Phase 6b: Feed back to liquid router (learns from every query)
            liquid_router.feedback(tier, confidence=confidence, latency_ms=latency_ms)

            # Phase 6c: Store in HDC for fast future retrieval
            if text and len(text) > 10:
                hdc.add(f"Q: {user_input[:100]} A: {text[:100]}", {
                    "type": "exchange", "confidence": confidence,
                })

            # Phase 7e: Record exchange in session
            sessions.add_exchange(
                query=user_input, response=text,
                tier=tier, confidence=confidence,
                code_generated=code_exec,
            )

            # Speed: Cache this response for future
            speed.cache_response(user_input, text, confidence)
            speed.cache._save()  # Save immediately so cache persists across restarts

            # Continuous fine-tuning: collect high-quality pairs
            ft_pipeline.add_example(
                instruction=user_input,
                response=text,
                quality_score=confidence / 100 if confidence > 1 else confidence,
                source=tier,
                verified=code_exec and code_passed,
            )

            # Evolution: track themes across sessions
            evolution.track_query(user_input, session_id=str(current_session.id))

            # Predictor: predict follow-ups and start pre-generating
            if predictor.generate_fn:
                predictor.on_query_complete(user_input, text)
            # Anticipatory generation disabled — llama-cpp-python is not thread-safe
            # Will be enabled when llama.cpp adds async/concurrent inference support
            # try:
            #     start_anticipatory(user_input, text)
            # except Exception:
            #     pass


if __name__ == "__main__":
    main()
