"""
LeanAI · Phase 2 Main Entry Point
New commands: /remember  /profile  /world
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine_v2 import LeanAIEngineV2 as LeanAIEngine, GenerationConfig


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                        LeanAI  v2                        ║
║   Fast · Lightweight · Runs anywhere · Gets smarter      ║
║   Phase 2: Vector Memory · World Model · Learns You      ║
╚══════════════════════════════════════════════════════════╝""")


def print_response(r):
    print(f"\n{r.text}\n")
    print("─" * 55)
    pct = int(r.confidence * 100)
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"Confidence  [{bar}] {pct}%  {r.confidence_label}")
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
    verbose   = "--verbose" in sys.argv or "-v" in sys.argv
    test_mode = "--test" in sys.argv

    print("\nInitializing LeanAI v2...")
    engine = LeanAIEngine(verbose=verbose)

    status = engine.status()
    mem = status["memory"]
    print(f"\nModel  : {'loaded' if status['model_loaded'] else 'not loaded (will load on first question)'}")
    print(f"Memory : {mem['episodic_entries']} episodes  |  backend: {mem['episodic_backend']}")
    print(f"World  : {mem['semantic_entities']} entities  |  {mem['user_profile_fields']} profile fields")
    t = status["training"]
    print(f"Training: {t['total']} pairs  |  {t['high_quality']} high quality")
    print("\nCommands: /help  /remember  /profile  /world  /memory  /status  /selfplay  /quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────

        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print("""
Commands:
  /remember <fact>  — store something in long-term memory
                      e.g. /remember my name is Aditya
                      e.g. /remember I work as a DevOps engineer
  /profile          — show what LeanAI knows about you
  /world            — show all known entities and facts
  /memory           — show memory layer stats
  /status           — show full engine status
  /selfplay         — generate 10 synthetic training pairs
  /clear            — clear conversation history
  /quit             — exit
""")
            continue

        if user_input.lower().startswith("/remember "):
            fact = user_input[10:].strip()
            if fact:
                engine.remember(fact)
                print(f"Stored in memory: \"{fact}\"\n")
            else:
                print("Usage: /remember <fact>\n")
            continue

        if user_input.lower() == "/profile":
            profile = engine.get_profile()
            if not profile:
                print("\nNo profile data yet. Tell me about yourself!")
                print("Try: \"My name is...\" or \"I work as...\" or \"I live in...\"\n")
            else:
                print("\nWhat I know about you:")
                for key, value in profile.items():
                    if isinstance(value, list):
                        print(f"  {key}: {', '.join(str(v) for v in value)}")
                    else:
                        print(f"  {key}: {value}")
                print()
            continue

        if user_input.lower() == "/world":
            stats = engine.memory.world.stats()
            print(f"\nWorld model — {stats['entities']} entities, {stats['relations']} relations")
            print("\nEntity types:")
            for etype, count in stats["entity_types"].items():
                if count > 0:
                    print(f"  {etype}: {count}")
            entities = list(engine.memory.world._entities.values())
            entities.sort(key=lambda e: e.mention_count, reverse=True)
            if entities:
                print("\nTop entities by mention count:")
                for e in entities[:10]:
                    print(f"  • {e.name} ({e.entity_type.value}) — mentioned {e.mention_count}x")
            print()
            continue

        if user_input.lower() == "/memory":
            m = engine.memory.stats()
            print(f"\nWorking   : {m['working_messages']} messages ({m['working_tokens']} tokens)")
            print(f"Episodic  : {m['episodic_entries']} entries  [{m['episodic_backend']}]")
            print(f"World     : {m['semantic_entities']} entities, {m['semantic_relations']} relations")
            print(f"Profile   : {m['user_profile_fields']} fields known")
            print(f"Procedural: {m['procedural_solutions']} solutions\n")
            continue

        if user_input.lower() == "/status":
            s = engine.status()
            m = s["memory"]
            t = s["training"]
            print(f"\nPhase      : {s['phase']}")
            print(f"Model      : {'loaded' if s['model_loaded'] else 'not loaded'}")
            print(f"Verifier   : {s['verifier']}")
            print(f"Memory     : {m['episodic_entries']} episodes, {m['semantic_entities']} entities")
            print(f"Training   : {t['total']} pairs, {t['high_quality']} high quality, {t['verified']} verified\n")
            continue

        if user_input.lower() == "/selfplay":
            print("Generating 10 synthetic training pairs...")
            n = engine.generate_self_play_batch(10)
            print(f"Added {n} verified pairs to training store.\n")
            continue

        if user_input.lower() == "/clear":
            engine.memory.working.clear()
            print("Conversation cleared.\n")
            continue

        # ── Generate ──────────────────────────────────────────────────
        print("\nLeanAI: ", end="", flush=True)
        response = engine.generate(user_input)
        print_response(response)


if __name__ == "__main__":
    main()
