#!/usr/bin/env python3
"""
LeanAI — Main CLI
Phase 4d: Agentic Multi-Step Coding
All phases integrated: router, watchdog, verifier, memory, world model,
confidence, training, executor, indexer, and the agentic builder.
"""

import os
import sys
import time
import re

from core.engine_v3 import LeanAIEngineV3 as LeanAIEngine, GenerationConfig
from tools.executor import CodeExecutor
from tools.indexer import ProjectIndexer
from agents.build_command import BuildHandler
from swarm import SwarmConsensus


BANNER = """
╔══════════════════════════════════════════════════════════╗
║                        LeanAI  v5                        ║
║   Fast · Lightweight · Runs anywhere · Gets smarter      ║
║   Phase 5a: Swarm Consensus — 3-pass verified answers    ║
╚══════════════════════════════════════════════════════════╝"""


def main():
    print(BANNER)
    print("Initializing LeanAI v4...")

    # ── Initialize engine ──────────────────────────────────────────
    engine = LeanAIEngine(verbose=False)

    # ── Initialize tools ───────────────────────────────────────────
    executor = CodeExecutor()
    indexer = ProjectIndexer()

    # ── Build handler (Phase 4d) ───────────────────────────────────
    def model_fn(system_prompt: str, user_prompt: str) -> str:
        """Call the loaded model with custom system/user prompts."""
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

        result = engine._model(
            prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=stop,
        )
        text = result["choices"][0]["text"].strip()
        return text

    build_handler = BuildHandler(model_fn=model_fn, verbose=False)

    # ── Swarm consensus (Phase 5a) ─────────────────────────────────
    def swarm_model_fn(prompt: str, temperature: float) -> str:
        """Model function for swarm — takes prompt and temperature."""
        if not engine._model:
            engine._load_model()
        fmt = getattr(engine, "prompt_format", "chatml")
        if fmt == "chatml":
            full_prompt = (
                f"<|im_start|>system\nYou are a helpful, accurate AI assistant. Answer concisely.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            stop = ["<|im_end|>", "<|im_start|>"]
        else:
            full_prompt = (
                f"<|system|>\nYou are a helpful, accurate AI assistant. Answer concisely.<|end|>\n"
                f"<|user|>\n{prompt}<|end|>\n"
                f"<|assistant|>\n"
            )
            stop = ["<|end|>", "<|user|>", "<|assistant|>"]

        result = engine._model(
            full_prompt,
            max_tokens=512,
            temperature=temperature,
            stop=stop,
        )
        return result["choices"][0]["text"].strip()

    swarm = SwarmConsensus(model_fn=swarm_model_fn, num_passes=3, verbose=True)

    # ── Display status ─────────────────────────────────────────────
    # ── Display status (reading actual engine internals) ──────────
    mem_backend = "unknown"
    mem_count = 0
    try:
        mem_count = engine.memory.episodic.count()
        mem_backend = engine.memory.episodic.backend
    except Exception:
        pass

    world_count = 0
    profile_count = 0
    try:
        world_count = len(getattr(engine.memory.world, "entities", {}))
        profile_count = len(engine.memory.world.get_user_profile())
    except Exception:
        pass

    training_total = 0
    training_quality = 0
    try:
        ts = engine.trainer.status()
        training_total = ts.get("total_pairs", 0)
        training_quality = ts.get("quality_pairs", 0)
    except Exception:
        pass

    print(f"Model    : not loaded (loads on first question)")
    print(f"Memory   : {mem_count} episodes | {mem_backend}")
    print(f"World    : {world_count} entities | {profile_count} profile fields")
    print(f"Training : {training_total} pairs | {training_quality} quality | background: on")
    print(f"Commands: /help  /swarm  /build  /run  /index  /ask  /remember  /profile  /world  /quit")

    # ── Command loop ───────────────────────────────────────────────
    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        elif cmd == "/help":
            print("""
Commands:
  /swarm <question> Run 3-pass consensus for highest accuracy answers
  /build <task>     Build a complete project (agentic multi-step)
  /plan             Show the last build plan
  /run <code>       Execute Python code directly
  /index <path>     Index a project directory for semantic search
  /ask <question>   Search indexed codebase
  /indexstats       Show indexer statistics
  /remember <fact>  Store a fact in memory
  /profile          Show what LeanAI knows about you
  /world            Show world model entities
  /generate <n>     Generate n self-play training pairs
  /train            Run training cycle
  /trainstatus      Show training statistics
  /export           Export training data
  /status           Show system status
  /quit             Exit LeanAI
""")

        elif cmd.startswith("/build"):
            task = user_input[6:].strip()
            if not task:
                print("Usage: /build <task description>")
                print("Example: /build Build a Flask REST API with user auth and SQLite")
                continue
            if not engine._model:
                print("[LeanAI v4] Loading model for build...", flush=True)
                try:
                    engine._load_model()
                    print("[LeanAI v4] Model loaded. Starting build...", flush=True)
                except Exception as e:
                    print(f"[LeanAI v4] Failed to load model: {e}")
                    continue
            build_handler.execute_build(task)

        elif cmd == "/plan":
            build_handler.show_last_result()

        elif cmd.startswith("/run"):
            code = user_input[4:].strip()
            if not code:
                print("Usage: /run <python code>")
                continue
            result = executor.execute(code)
            if result["success"]:
                print(f"Code verified working ({result['attempts']} attempt(s))")
                if result["output"]:
                    print(f"Output: {result['output']}")
                print(f"Execution time: {result['execution_time_ms']}ms")
            else:
                print(f"Code failed: {result['error']}")

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
            langs = stats.get("languages", {})
            if langs:
                print(f"Languages: {', '.join(f'{k}({v})' for k, v in langs.items())}")

        elif cmd.startswith("/ask"):
            query = user_input[4:].strip()
            if not query:
                print("Usage: /ask <question about your codebase>")
                continue
            results = indexer.search(query, top_k=5)
            if not results:
                print("No results. Run /index <path> first.")
                continue
            print(f"\nFound {len(results)} relevant chunks:\n")
            for r in results:
                score = r.get("score", 0)
                filepath = r.get("filepath", "?")
                chunk = r.get("chunk", "")
                lang = r.get("language", "")
                print(f"  [{score:.0%}] {filepath} ({lang})")
                preview = chunk[:200].replace("\n", "\n    ")
                print(f"    {preview}")
                print()

        elif cmd == "/indexstats":
            stats = indexer.stats()
            print(f"Total chunks: {stats.get('total_chunks', 0)}")
            print(f"Projects indexed: {stats.get('projects', 0)}")

        elif cmd.startswith("/remember"):
            fact = user_input[9:].strip()
            if not fact:
                print("Usage: /remember <fact>")
                continue
            try:
                engine.remember(fact)
                print(f'Stored in memory: "{fact}"')
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
                    print("No profile data yet. Tell me about yourself!")
            except Exception as e:
                print(f"Profile error: {e}")

        elif cmd == "/world":
            try:
                wm = engine.memory.world
                ent_count = len(getattr(wm, "entities", {}))
                rel_count = len(getattr(wm, "relations", {}))
                print(f"World model — {ent_count} entities, {rel_count} relations")
                if hasattr(wm, "entities") and wm.entities:
                    types = {}
                    ents = wm.entities.values() if isinstance(wm.entities, dict) else wm.entities
                    for e in ents:
                        t = e.get("type", "unknown") if isinstance(e, dict) else getattr(e, "type", "unknown")
                        types[t] = types.get(t, 0) + 1
                    print("Entity types:")
                    for t, c in types.items():
                        print(f"  {t}: {c}")
            except Exception as e:
                print(f"World error: {e}")

        elif cmd.startswith("/generate"):
            n_str = user_input[9:].strip()
            n = int(n_str) if n_str.isdigit() else 10
            print(f"Generating {n} self-play training pairs...", flush=True)
            try:
                engine.generate_training_data(n)
                ts = engine.trainer.status()
                print(f"Added {n} pairs. Total: {ts.get('total_pairs', '?')} | Quality: {ts.get('quality_pairs', '?')}")
            except Exception as e:
                print(f"Generate error: {e}")

        elif cmd == "/train":
            print("Running training cycle...", flush=True)
            try:
                result = engine.trigger_training()
                print(f"Status : {result.get('status', 'unknown')}")
                print(f"Pairs  : {result.get('pairs', 0)}")
                if result.get("notes"):
                    print(f"Notes  : {result['notes']}")
                print(f"Time   : {result.get('time', 0):.1f}s")
            except Exception as e:
                print(f"Training error: {e}")

        elif cmd == "/trainstatus":
            try:
                ts = engine.training_status()
                print(f"Background trainer : running")
                print(f"Total pairs        : {ts.get('total_pairs', 0)}")
                print(f"Quality passes     : {ts.get('quality_pairs', 0)}")
                print(f"Training runs done : {ts.get('runs_completed', 0)}")
                print(f"Last run           : {ts.get('last_run_status', 'n/a')}")
            except Exception as e:
                print(f"Status error: {e}")

        elif cmd == "/export":
            try:
                result = engine.trigger_training()
                path = result.get("export_path", "unknown")
                print(f"Exported to: {path}")
            except Exception as e:
                print(f"Export error: {e}")

        elif cmd == "/status":
            print(f"Engine   : LeanAI v5")
            print(f"Model    : {getattr(engine, 'model_name', 'not loaded')}")
            print(f"Format   : {getattr(engine, 'prompt_format', 'unknown')}")
            print(f"Threads  : {getattr(engine, 'n_threads', '?')}")
            print(f"Memory   : {mem_backend}")
            print(f"Swarm    : 3-pass consensus enabled")
            if build_handler.last_result:
                r = build_handler.last_result
                print(f"Last build: {'SUCCESS' if r.success else 'FAILED'} — {r.plan.completed_steps}/{r.plan.total_steps} steps")

        elif cmd.startswith("/swarm"):
            query = user_input[6:].strip()
            if not query:
                print("Usage: /swarm <question>")
                print("Runs 3 inference passes and picks the best answer via consensus.")
                continue
            if not engine._model:
                print("[Swarm] Loading model...", flush=True)
                try:
                    engine._load_model()
                except Exception as e:
                    print(f"[Swarm] Failed to load model: {e}")
                    continue
            print(f"[Swarm] Running 3-pass consensus...", flush=True)
            result = swarm.query(query)
            print(f"\nLeanAI (swarm):\n{result.best_answer}")
            print("───────────────────────────────────────────────────────")
            print(result.summary())

        else:
            # ── Normal query — route through engine ────────────────
            start = time.time()
            try:
                resp = engine.generate(user_input, config=GenerationConfig())
            except Exception as e:
                print(f"\nError: {e}")
                if "model" in str(e).lower():
                    print("Try: python setup.py --download-model")
                continue

            elapsed = time.time() - start

            # ── Read LeanAIResponse attributes ─────────────────────
            text         = getattr(resp, "text", str(resp))
            confidence   = getattr(resp, "confidence", 50)
            conf_label   = getattr(resp, "confidence_label", "")
            tier         = getattr(resp, "tier_used", "unknown")
            latency_ms   = getattr(resp, "latency_ms", int(elapsed * 1000))
            from_mem     = getattr(resp, "answered_from_memory", False)
            mem_active   = getattr(resp, "memory_context_used", False)
            verified     = getattr(resp, "verified", False)
            code_exec    = getattr(resp, "code_executed", False)
            code_passed  = getattr(resp, "code_passed", False)
            code_output  = getattr(resp, "code_output", "")
            warning      = getattr(resp, "warning", None)

            # Normalize confidence: engine returns 0.0-1.0, display wants 0-100
            if isinstance(confidence, (int, float)) and 0 < confidence <= 1.0:
                confidence = confidence * 100

            # Print response
            print(f"\nLeanAI:\n{text}")

            # Show code execution results (engine already ran the code)
            if code_exec:
                print(f"\n--- Code execution ---")
                if code_passed:
                    print(f"Code executed: PASSED")
                    if code_output:
                        print(f"Output: {code_output[:500]}")
                else:
                    print(f"Code executed: FAILED")
                    if code_output:
                        print(f"Error: {code_output[:300]}")

            # Display metadata bar
            print("───────────────────────────────────────────────────────")

            # Boost confidence if code verified
            if code_exec and code_passed:
                confidence = max(confidence, 96)
                conf_label = "High"

            if not conf_label:
                conf_label = "High" if confidence >= 90 else "Good" if confidence >= 70 else "Moderate" if confidence >= 50 else "Low"

            conf_bars = int(confidence / 5)
            conf_display = "█" * conf_bars + "░" * (20 - conf_bars)

            meta = f"Confidence  [{conf_display}] {confidence:.0f}%  {conf_label}"
            meta += f"\nTier: {tier}  Latency: {latency_ms:.0f}ms"

            if from_mem:
                meta += "  From memory: yes"
            if verified:
                meta += "  Verified: yes"
            if mem_active:
                meta += "  Memory: active"
            if code_exec and code_passed:
                meta += "  Code: verified"

            if warning:
                meta += f"\n⚠ {warning}"

            print(meta)

            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": text})


if __name__ == "__main__":
    main()
