# LeanAI — Phase 0

> Lightweight · Fast · Runs anywhere · Gets smarter over time

A genuinely novel AI architecture combining:
- **BitNet 1.58** — 1-bit quantized weights (10x memory reduction)
- **Mamba SSM** — linear-time sequence modeling (no quadratic attention)
- **Metacognitive router** — routes queries to the right model tier
- **Neurosymbolic verifier** — mathematically verifies logic/math outputs
- **Hierarchical memory** — remembers everything across sessions

## Phase 0 delivers today
- Runs on ANY machine (CPU only, no GPU needed)
- ~512MB RAM for 1B param model
- 50–200 tokens/sec on a basic laptop CPU
- Fully offline after first download
- Foundation for all future phases

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download a base model (Phase 0 uses GGUF quantized model)
python setup.py --download-model

# 3. Run the AI
python main.py

# 4. Run tests
python -m pytest tests/
```

## Project structure
```
leanai/
├── main.py              # Entry point — chat interface
├── setup.py             # Model downloader + environment check
├── requirements.txt     # All dependencies
├── core/
│   ├── engine.py        # Main inference engine (BitNet + Mamba hybrid)
│   ├── router.py        # Smart task router — picks model tier per query
│   ├── watchdog.py      # Metacognitive watchdog — monitors confidence
│   └── tokenizer.py     # Unified tokenizer wrapper
├── memory/
│   ├── hierarchy.py     # 4-layer memory system
│   ├── episodic.py      # Vector-based episodic memory (ChromaDB)
│   └── semantic.py      # Knowledge graph (NetworkX)
├── tools/
│   ├── verifier.py      # Neurosymbolic verifier (Z3 + SymPy)
│   ├── calculator.py    # Exact arithmetic tool
│   └── executor.py      # Safe sandboxed code execution
└── tests/
    ├── test_engine.py
    ├── test_router.py
    ├── test_memory.py
    └── test_verifier.py
```

## Roadmap
| Phase | Timeline | Feature |
|-------|----------|---------|
| 0 | Week 1–2 | BitNet+Mamba core, smart router, watchdog |
| 1 | Month 1–2 | Neurosymbolic verifier, confidence scoring |
| 2 | Month 2–4 | Hierarchical memory, causal world model |
| 3 | Month 4–6 | Self-play, continual on-device learning |
| 4 | Month 6–9 | Federated learning mesh |
| 5 | Month 9–12 | Swarm consensus, HDC knowledge store |
