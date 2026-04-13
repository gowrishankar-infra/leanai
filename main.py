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
from core.cascade import CascadeInference
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


def main():
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
        if fmt == "chatml":
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
        return result["choices"][0]["text"].strip()

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

    # ── Cascade Inference (7B draft → 32B review) ─────────────
    cascade = CascadeInference(draft_fn=model_fn, review_fn=model_fn, enabled=True)

    # ── Tool-Augmented Reasoning (ReAct) ──────────────────────
    react = ReActReasoner(model_fn=model_fn, max_steps=3)

    # ── Mixture of Agents ─────────────────────────────────────
    moa = MixtureOfAgents(model_fn=model_fn, perspectives=3, enabled=True)

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

    # ══════════════════════════════════════════════════════════════
    # COMMAND LOOP
    # ══════════════════════════════════════════════════════════════

    while True:
        try:
            user_input = input(get_prompt()).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.DIM}Goodbye!{C.RESET}")
            sessions.end_session(current_session.id)
            sessions.save_all()
            break

        if not user_input:
            continue

        cmd = user_input.lower().strip()

        # ── Exit ──────────────────────────────────────────────────
        if cmd in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            sessions.end_session(current_session.id)
            sessions.save_all()
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

            # Register ReAct tools with brain data
            react.register_tool("brain", brain.find_function, "Look up function/file in project")
            if git_intel and git_intel.is_available:
                react.register_tool("git", lambda q: git_intel.recent_activity(days=7), "Check git history")
            print(f"[Brain] Scanned {result['files_found']} files in {result['scan_time_ms']}ms")
            print(brain.project_summary())

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

        elif cmd.startswith("/security"):
            target = user_input[9:].strip()
            if not target:
                print("Usage: /security <filename or code>")
                print("  Scans code for security vulnerabilities.")
                continue

            print(f"  {C.DIM}Scanning for vulnerabilities...{C.RESET}", flush=True)

            # If it's a filename and brain is loaded, read the file
            code_to_scan = target
            if brain and not target.startswith("def ") and not target.startswith("import "):
                for rel_path in brain._file_analyses.keys():
                    if target.lower() in rel_path.lower():
                        try:
                            full_path = os.path.join(brain.config.project_path, rel_path)
                            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                                code_to_scan = f.read(5000)
                            code_to_scan = f"File: {rel_path}\n\n{code_to_scan}"
                        except Exception:
                            pass
                        break

            security_prompt = (
                "Perform a security audit on this code. Check for:\n"
                "- SQL injection\n"
                "- XSS (cross-site scripting)\n"
                "- Command injection\n"
                "- Path traversal\n"
                "- Hardcoded secrets/credentials\n"
                "- Insecure deserialization\n"
                "- Missing input validation\n"
                "- Insecure file operations\n"
                "- CORS misconfiguration\n"
                "- Missing authentication/authorization\n\n"
                "For each issue found, state:\n"
                "- Severity (Critical/High/Medium/Low)\n"
                "- What the vulnerability is\n"
                "- Where it is (line or function)\n"
                "- How to fix it (show corrected code)\n\n"
                "If no issues found, say 'No security issues detected.'\n\n"
                f"Code to audit:\n```\n{code_to_scan}\n```"
            )

            config = GenerationConfig(max_tokens=1536, temperature=0.1)
            resp = engine.generate(security_prompt, config=config)
            text = getattr(resp, "text", str(resp))

            print_response_header()
            print(format_response(text))
            separator()

            confidence = getattr(resp, "confidence", 0.7)
            if isinstance(confidence, float) and 0 < confidence <= 1.0:
                confidence *= 100
            print(format_confidence(confidence, "Security Audit"))
            sessions.add_exchange(query=user_input, response=text, tier="security", confidence=confidence)

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

            # Phase 6b: Liquid router decides tier
            tier_suggestion = liquid_router.route(user_input)

            # Auto model selection: switch if needed
            downloaded = model_mgr.get_downloaded_models()
            if model_mgr.mode == "auto" and len(downloaded) >= 1:
                best_model = model_mgr.select_model(user_input)

                # Check if cascade will handle this query (skip auto-switch to 32B)
                will_cascade = (
                    cascade.enabled
                    and len(downloaded) >= 2
                    and cascade.should_cascade(user_input)
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

            # ── Mixture of Agents: multi-perspective for reviews ──
            if moa.should_use_moa(user_input):
                print(f"  {C.fg(222)}⚙ Multi-perspective analysis...{C.RESET}", flush=True)
                moa_context = brain_ctx or ""
                if react_data:
                    moa_context += "\n" + react_data
                moa_result = moa.analyze(user_input, context=moa_context)

                if moa_result.final_answer and len(moa_result.final_answer.strip()) > 30:
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

                    resp = _MoAResp(moa_result.final_answer)
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

                # Inject ReAct tool data into context
                if react_data:
                    enriched_context = react_data + "\n\n" + enriched_context if enriched_context else react_data

                # ── CASCADE INFERENCE ──────────────────────────────
                # For complex queries: 7B drafts fast, 32B reviews and corrects
                # Result: ~2-3 min instead of ~7 min for 32B from scratch
                use_cascade = (
                    cascade.enabled
                    and model_mgr.mode == "auto"
                    and len(model_mgr.get_downloaded_models()) >= 2
                    and cascade.should_cascade(user_input)
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
                    # ── STANDARD GENERATION ────────────────────────
                    # Auto-recovery wrapped generation
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
                        resp = rec_result.text  # this is the LeanAIResponse object
                        if rec_result.recovered:
                            print(f"  [Recovery] Recovered: {rec_result.recovery_note}")
                    else:
                        print(f"\nError: {rec_result.error}")
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

            # Print response with formatting
            # Two-pass quality: if response contains code, run a review pass
            if code_enhancer.should_review(user_input, text):
                print(f"  {C.DIM}Running code review...{C.RESET}", end="", flush=True)
                text = code_enhancer.enhance(text, query=user_input)
                print(f"\r{' ' * 40}\r", end="", flush=True)

            # Code-Grounded Verification: check claims against actual AST
            if code_verifier.should_verify(user_input, text):
                text = code_verifier.verify(text, query=user_input)

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


if __name__ == "__main__":
    main()
