"""
LeanAI v3 + Phase 4b + Phase 4c
Commands: /run /index /ask /remember /profile /world /train /generate /status /quit
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.engine_v3 import LeanAIEngineV3 as LeanAIEngine, GenerationConfig


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                        LeanAI  v3                        ║
║   Fast · Lightweight · Runs anywhere · Gets smarter      ║
║   Phase 4c: Project Indexer — knows your entire codebase ║
╚══════════════════════════════════════════════════════════╝""")


def print_response(r):
    print(f"\n{r.text}\n")
    if r.code_executed:
        if r.code_passed:
            print("Code executed: PASSED", end="")
            if r.code_auto_fixed:
                print(" (auto-fixed)", end="")
            print()
            if r.code_output:
                for line in r.code_output.strip().split("\n"):
                    print(f"  {line}")
        else:
            print("Code executed: FAILED")
            if r.code_output:
                for line in r.code_output.strip().split("\n"):
                    print(f"  {line}")
        print()
    print("─" * 55)
    pct = int(r.confidence * 100)
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"Confidence  [{bar}] {pct}%  {r.confidence_label}")
    stats = [f"Tier: {r.tier_used}", f"Latency: {r.latency_ms:.0f}ms"]
    if r.memory_context_used:   stats.append("Memory: active")
    if r.answered_from_memory:  stats.append("From memory: yes")
    if r.code_passed:           stats.append("Code: verified")
    if r.verified:              stats.append("Math: verified")
    if r.corrected:             stats.append("Auto-corrected")
    print("  ".join(stats))
    if r.warning:
        print(f"Note: {r.warning}")
    print()


def main():
    print_banner()
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("\nInitializing LeanAI...")
    engine = LeanAIEngine(verbose=verbose, auto_train=True, auto_execute=True)

    # Attach project indexer with the same embedder as memory
    from tools.indexer import ProjectIndexer
    embedder = engine.memory.episodic._embedder
    indexer  = ProjectIndexer(embedder=embedder)
    engine.indexer = indexer

    s   = engine.status()
    mem = s["memory"]
    tr  = s["training"]
    ex  = s["executor"]
    print(f"\nModel    : {'loaded' if s['model_loaded'] else 'not loaded (loads on first question)'}")
    print(f"Executor : {ex.get('available', ex.get('available_languages', []))}")
    print(f"Indexer  : {indexer.count()} chunks indexed")
    print(f"Memory   : {mem['episodic_entries']} episodes | {mem['episodic_backend']}")
    print(f"World    : {mem['semantic_entities']} entities | {mem['user_profile_fields']} profile fields")
    print(f"Training : {tr['total_pairs']} pairs | {tr['quality_filter']['passed']} quality")
    print("\nCommands: /help  /index  /ask  /run  /remember  /profile  /train  /status  /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            engine.trainer.stop()
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            engine.trainer.stop()
            break

        if user_input.lower() == "/help":
            print("""
Commands:
  /index <path>     — index a project directory
                      e.g. /index C:\\Users\\adity\\myproject
                      e.g. /index .  (index current folder)
  /ask <question>   — search indexed codebase
                      e.g. /ask how does the router work
                      e.g. /ask find all database functions
  /indexstats       — show indexer statistics
  /indexclear       — clear the project index

  /run <code>       — directly execute Python code
  /remember <fact>  — store fact in long-term memory
  /profile          — show what LeanAI knows about you
  /world            — show world model entities
  /memory           — memory layer stats
  /train            — manually trigger training cycle
  /generate [n]     — generate N self-play training pairs
  /trainstatus      — training loop status
  /export           — export training data to JSONL
  /status           — full engine status
  /clear            — clear conversation history
  /quit             — exit
""")
            continue

        # ── Indexer commands ──────────────────────────────────────────

        if user_input.lower().startswith("/index "):
            path_str = user_input[7:].strip()
            if path_str == ".":
                path_str = os.getcwd()
            from pathlib import Path
            path = Path(path_str).expanduser()
            if not path.exists():
                print(f"Path not found: {path}\n")
                continue
            print(f"Indexing: {path}")
            print("This may take a moment for large projects...\n")
            try:
                stats = indexer.index_project(str(path))
                print(f"Done!")
                print(f"  Files   : {stats.indexed_files} indexed, {stats.skipped_files} unchanged")
                print(f"  Chunks  : {stats.total_chunks} total")
                print(f"  Languages: {', '.join(f'{k}:{v}' for k,v in stats.languages.items())}")
                print(f"  Time    : {stats.index_time_s}s\n")
            except Exception as e:
                print(f"Indexing failed: {e}\n")
            continue

        if user_input.lower().startswith("/ask "):
            query = user_input[5:].strip()
            if not query:
                print("Usage: /ask <question about your codebase>\n")
                continue
            if indexer.count() == 0:
                print("No project indexed yet. Run: /index <path>\n")
                continue
            print(f"Searching {indexer.count()} chunks...")
            results = indexer.search(query, top_k=5)
            if not results:
                print("No relevant code found.\n")
                continue
            print(f"\nFound {len(results)} relevant sections:\n")
            for i, r in enumerate(results, 1):
                rel   = r.get("relative_path", "?")
                name  = r.get("name", "")
                lang  = r.get("language", "")
                line  = r.get("start_line", 0)
                score = r.get("relevance", 0)
                print(f"  {i}. {rel}", end="")
                if name: print(f" — {name}", end="")
                if line: print(f" (line {line})", end="")
                print(f"  [{score:.0%}]")
                # Show first 3 lines of content
                preview = r.get("content", "")[:200].split("\n")[:3]
                for pline in preview:
                    print(f"     {pline}")
                print()
            continue

        if user_input.lower() == "/indexstats":
            s = indexer.stats()
            print(f"\nProject index stats:")
            print(f"  Total chunks  : {s.get('total_chunks', 0)}")
            print(f"  Indexed files : {s.get('indexed_files', 0)}")
            if "project_root" in s:
                print(f"  Project root  : {s['project_root']}")
            if "languages" in s:
                print(f"  Languages     : {s['languages']}")
            if "last_updated" in s:
                import time
                age = time.time() - s["last_updated"]
                print(f"  Last indexed  : {int(age/60)} minutes ago")
            print()
            continue

        if user_input.lower() == "/indexclear":
            indexer.clear()
            print("Project index cleared.\n")
            continue

        # ── Other commands ────────────────────────────────────────────

        if user_input.lower().startswith("/run "):
            code = user_input[5:].strip()
            if not code:
                print("Usage: /run <python code>\n")
                continue
            print("Running...")
            verified = engine.execute_code(code)
            result = engine.executor.format_result(verified)
            print(f"\n{result}\n")
            if verified.passed and verified.final_result.stdout:
                print(f"Output: {verified.final_result.stdout.strip()}\n")
            continue

        if user_input.lower().startswith("/remember "):
            fact = user_input[10:].strip()
            if fact:
                engine.remember(fact)
                print(f"Stored: \"{fact}\"\n")
            continue

        if user_input.lower() == "/profile":
            profile = engine.get_profile()
            if not profile:
                print("\nNo profile data yet.\n")
            else:
                print("\nWhat I know about you:")
                for k, v in profile.items():
                    print(f"  {k}: {v if not isinstance(v, list) else ', '.join(str(x) for x in v)}")
                print()
            continue

        if user_input.lower() == "/world":
            ws = engine.memory.world.stats()
            print(f"\nEntities: {ws['entities']} | Relations: {ws['relations']}")
            entities = sorted(engine.memory.world._entities.values(),
                              key=lambda e: e.mention_count, reverse=True)
            for e in entities[:10]:
                print(f"  {e.name} ({e.entity_type.value}) — {e.mention_count}x")
            print()
            continue

        if user_input.lower() == "/memory":
            m = engine.memory.stats()
            print(f"\nWorking   : {m['working_messages']} messages")
            print(f"Episodic  : {m['episodic_entries']} [{m['episodic_backend']}]")
            print(f"World     : {m['semantic_entities']} entities")
            print(f"Profile   : {m['user_profile_fields']} fields\n")
            continue

        if user_input.lower() == "/train":
            print("Running training cycle...")
            result = engine.trigger_training()
            print(f"Status: {result['status']} | Pairs: {result['pairs_used']}")
            print(f"Notes: {result['notes']}\n")
            continue

        if user_input.lower().startswith("/generate"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            print(f"Generating {n} self-play pairs...")
            added = engine.generate_training_data(n)
            ts = engine.trainer.status()
            print(f"Added {added}. Total: {ts['total_pairs']} | Quality: {ts['quality_filter']['passed']}\n")
            continue

        if user_input.lower() == "/trainstatus":
            ts = engine.trainer.status()
            qf = ts["quality_filter"]
            print(f"\nTrainer : {'running' if ts['running'] else 'stopped'}")
            print(f"Pairs   : {ts['total_pairs']} total, {qf['passed']} quality ({qf['pass_rate']:.0%})")
            print(f"Avg score: {qf['avg_score']} | Runs: {ts['training_runs']}\n")
            continue

        if user_input.lower() == "/export":
            print("Exporting...")
            path = engine.trainer.export_training_data()
            if path:
                print(f"Exported: {path}\n")
            continue

        if user_input.lower() == "/status":
            s = engine.status()
            m, tr = s["memory"], s["training"]
            print(f"\nPhase  : {s['phase']} | Model: {s['model']} | Format: {s['prompt_format']}")
            print(f"Indexer: {indexer.count()} chunks")
            print(f"Memory : {m['episodic_entries']} episodes, {m['semantic_entities']} entities")
            print(f"Training: {tr['total_pairs']} pairs, {tr['quality_filter']['passed']} quality\n")
            continue

        if user_input.lower() == "/clear":
            engine.memory.working.clear()
            print("Cleared.\n")
            continue

        # ── Generate with codebase context if indexed ─────────────────
        query = user_input

        # If project is indexed, inject relevant code context into query
        if indexer.count() > 0:
            code_context = indexer.search(query, top_k=3)
            if code_context:
                formatted = indexer.format_search_results(code_context, max_chars=800)
                if formatted:
                    # Prepend codebase context to query for the engine
                    query = f"[Relevant codebase context:]\n{formatted}\n\n[Question:] {user_input}"

        print("\nLeanAI: ", end="", flush=True)
        response = engine.generate(query)
        print_response(response)


if __name__ == "__main__":
    main()
