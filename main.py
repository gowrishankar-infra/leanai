"""
LeanAI · Phase 3 Main
New commands: /train  /generate  /trainstatus  /export
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine_v3 import LeanAIEngineV3 as LeanAIEngine, GenerationConfig


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                        LeanAI  v3                        ║
║   Fast · Lightweight · Runs anywhere · Gets smarter      ║
║   Phase 3: Continual Learning · Self-Play · Calibrated   ║
╚══════════════════════════════════════════════════════════╝""")


def print_response(r):
    print(f"\n{r.text}\n")
    print("─" * 55)
    print(f"Confidence  [{r.confidence_bar}] {int(r.confidence*100)}%  {r.confidence_label}")
    stats = [f"Tier: {r.tier_used}", f"Latency: {r.latency_ms:.0f}ms"]
    if r.memory_context_used:   stats.append("Memory: active")
    if r.answered_from_memory:  stats.append("From memory: yes")
    if r.verified:              stats.append("Verified: yes")
    if r.corrected:             stats.append("Auto-corrected: yes")
    print("  ".join(stats))
    if r.claims_checked > 0:
        print(f"Verification: {r.verification_summary}")
    if r.warning:
        print(f"Note: {r.warning}")
    print()


def main():
    print_banner()
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print("\nInitializing LeanAI v3...")
    engine = LeanAIEngine(verbose=verbose, auto_train=True)

    s = engine.status()
    mem = s["memory"]
    tr  = s["training"]
    print(f"\nModel    : {'loaded' if s['model_loaded'] else 'not loaded (loads on first question)'}")
    print(f"Memory   : {mem['episodic_entries']} episodes | {mem['episodic_backend']}")
    print(f"World    : {mem['semantic_entities']} entities | {mem['user_profile_fields']} profile fields")
    print(f"Training : {tr['total_pairs']} pairs | {tr['quality_filter']['passed']} quality | background: {'on' if tr['running'] else 'off'}")
    print("\nCommands: /help  /remember  /profile  /world  /train  /generate  /trainstatus  /export  /quit\n")

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
  /remember <fact>  — store fact in long-term memory
  /profile          — show what LeanAI knows about you
  /world            — show world model entities
  /memory           — show memory layer stats
  /train            — manually trigger training cycle
  /generate [n]     — generate N self-play training pairs (default 20)
  /trainstatus      — show training loop status
  /export           — export training data to JSONL file
  /selfplay         — generate 10 self-play pairs (quick)
  /status           — full engine status
  /clear            — clear conversation history
  /quit             — exit
""")
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
                print(f"  • {e.name} ({e.entity_type.value}) — {e.mention_count}x")
            print()
            continue

        if user_input.lower() == "/train":
            print("Running training cycle...")
            result = engine.trigger_training()
            print(f"Status : {result['status']}")
            print(f"Pairs  : {result['pairs_used']}")
            print(f"Notes  : {result['notes']}")
            print(f"Time   : {result['duration_s']}s\n")
            continue

        if user_input.lower().startswith("/generate"):
            parts = user_input.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
            print(f"Generating {n} self-play training pairs...")
            added = engine.generate_training_data(n)
            ts = engine.trainer.status()
            print(f"Added {added} pairs. Total: {ts['total_pairs']} | Quality: {ts['quality_filter']['passed']}\n")
            continue

        if user_input.lower() == "/trainstatus":
            ts = engine.trainer.status()
            qf = ts["quality_filter"]
            print(f"\nBackground trainer : {'running' if ts['running'] else 'stopped'}")
            print(f"Total pairs        : {ts['total_pairs']}")
            print(f"Quality passes     : {qf['passed']} / {qf['total']} ({qf['pass_rate']:.0%})")
            print(f"Avg quality score  : {qf['avg_score']}")
            print(f"Training runs done : {ts['training_runs']}")
            if ts['last_run']:
                lr = ts['last_run']
                print(f"Last run           : {lr['status']} — {lr['notes']}")
            print(f"Saved adapters     : {len(ts['adapters'])}\n")
            continue

        if user_input.lower() == "/export":
            print("Exporting training data...")
            path = engine.trainer.export_training_data()
            if path:
                print(f"Exported to: {path}\n")
            else:
                print("Nothing to export yet.\n")
            continue

        if user_input.lower() == "/selfplay":
            print("Generating 10 self-play pairs...")
            n = engine.generate_training_data(10)
            print(f"Done. Added {n} pairs.\n")
            continue

        if user_input.lower() == "/memory":
            m = engine.memory.stats()
            print(f"\nWorking   : {m['working_messages']} messages")
            print(f"Episodic  : {m['episodic_entries']} [{m['episodic_backend']}]")
            print(f"World     : {m['semantic_entities']} entities")
            print(f"Profile   : {m['user_profile_fields']} fields\n")
            continue

        if user_input.lower() == "/status":
            s = engine.status()
            m, tr = s["memory"], s["training"]
            print(f"\nPhase    : {s['phase']}")
            print(f"Model    : {'loaded' if s['model_loaded'] else 'not loaded'}")
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
