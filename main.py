"""
LeanAI v3+4b — Code executor integrated
New: /run <code>  to directly execute code
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.engine_v3 import LeanAIEngineV3 as LeanAIEngine, GenerationConfig


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                        LeanAI  v3                        ║
║   Fast · Lightweight · Runs anywhere · Gets smarter      ║
║   Phase 4b: Code Executor — generates + runs + verifies  ║
╚══════════════════════════════════════════════════════════╝""")


def print_response(r):
    print(f"\n{r.text}\n")

    # Show code execution result if any
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

    s = engine.status()
    mem = s["memory"]
    tr  = s["training"]
    ex  = s["executor"]
    print(f"\nModel    : {'loaded' if s['model_loaded'] else 'not loaded (loads on first question)'}")
    print(f"Format   : {s['prompt_format']}")
    print(f"Executor : {ex['available_languages'] if 'available_languages' in ex else ex['available']}")
    print(f"Memory   : {mem['episodic_entries']} episodes | {mem['episodic_backend']}")
    print(f"World    : {mem['semantic_entities']} entities | {mem['user_profile_fields']} profile fields")
    print(f"Training : {tr['total_pairs']} pairs | {tr['quality_filter']['passed']} quality")
    print("\nCommands: /help  /run  /remember  /profile  /world  /train  /generate  /status  /quit\n")

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
  /run <code>       — directly execute Python code
                      e.g. /run print([x**2 for x in range(5)])
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
            stats = engine.memory.world.stats()
            print(f"\nEntities: {stats['entities']} | Relations: {stats['relations']}")
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
            print(f"\nTrainer    : {'running' if ts['running'] else 'stopped'}")
            print(f"Total pairs: {ts['total_pairs']}")
            print(f"Quality    : {qf['passed']}/{qf['total']} ({qf['pass_rate']:.0%})")
            print(f"Avg score  : {qf['avg_score']}")
            print(f"Runs done  : {ts['training_runs']}\n")
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
            print(f"\nPhase    : {s['phase']}")
            print(f"Model    : {s['model']}")
            print(f"Format   : {s['prompt_format']}")
            print(f"Threads  : {s['threads']}")
            print(f"Executor : {s['executor']}")
            print(f"Memory   : {m['episodic_entries']} episodes, {m['semantic_entities']} entities")
            print(f"Training : {tr['total_pairs']} pairs, {tr['quality_filter']['passed']} quality\n")
            continue

        if user_input.lower() == "/clear":
            engine.memory.working.clear()
            print("Cleared.\n")
            continue

        print("\nLeanAI: ", end="", flush=True)
        response = engine.generate(user_input)
        print_response(response)


if __name__ == "__main__":
    main()
