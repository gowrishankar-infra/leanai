"""
LeanAI · Setup Script
Downloads models and verifies your environment.
Run: python setup.py --download-model --model qwen25-coder
"""

import sys
import os
import argparse
from pathlib import Path

MODEL_DIR = Path.home() / ".leanai" / "models"

MODELS = {
    "qwen25-coder": {
        "name": "Qwen2.5 Coder 7B (BEST for coding — 88% HumanEval)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "filename": "qwen25-coder-7b-q4.gguf",
        "size_gb": 4.5,
        "ram_gb": 6.0,
        "prompt_format": "chatml",
        "notes": "Best local coding model. Beats GPT-3.5 on code. Recommended.",
    },
    "phi3-mini": {
        "name": "Phi-3 Mini 4K (general purpose)",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "phi3-mini-q4.gguf",
        "size_gb": 2.3,
        "ram_gb": 3.0,
        "prompt_format": "phi3",
        "notes": "Good general model. Already downloaded.",
    },
    "tinyllama": {
        "name": "TinyLlama 1.1B (smallest — runs on 1GB RAM)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-q4.gguf",
        "size_gb": 0.7,
        "ram_gb": 1.0,
        "prompt_format": "chatml",
        "notes": "Ultra-light fallback.",
    },
    "llama3-1b": {
        "name": "Llama 3.2 1B (Meta)",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "llama3-1b-q4.gguf",
        "size_gb": 0.8,
        "ram_gb": 1.2,
        "prompt_format": "llama3",
        "notes": "Small Meta model.",
    },
}


def check_environment():
    print("\n[Setup] Checking environment...\n")
    checks = []

    py_ver = sys.version_info
    checks.append((py_ver >= (3, 10), f"Python {py_ver.major}.{py_ver.minor}", "3.10+ required"))

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        checks.append((ram_gb >= 1.0, f"RAM: {ram_gb:.1f} GB", "6GB+ needed for Qwen2.5 Coder"))
    except ImportError:
        checks.append((None, "RAM check skipped", "install psutil for RAM check"))

    import os as _os
    cores = _os.cpu_count()
    checks.append((cores >= 2, f"CPU cores: {cores}", "2+ recommended"))

    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9
    checks.append((free_gb >= 5.0, f"Free disk: {free_gb:.1f} GB", "5GB+ needed for Qwen2.5 Coder"))

    packages = [
        ("llama_cpp", "llama-cpp-python", "required for model inference"),
        ("sympy", "sympy", "required for math verification"),
        ("chromadb", "chromadb", "required for vector memory"),
        ("rich", "rich", "recommended for better output"),
    ]
    for module, package, note in packages:
        try:
            __import__(module)
            checks.append((True, f"Package: {package}", note))
        except ImportError:
            checks.append((False, f"Package: {package} NOT INSTALLED", f"pip install {package}"))

    all_ok = True
    for status, label, note in checks:
        icon = "✓" if status is True else ("✗" if status is False else "?")
        if status is False:
            all_ok = False
        print(f"  {icon}  {label}  ({note})")

    print()
    if all_ok:
        print("[Setup] Environment looks good!\n")
    else:
        print("[Setup] Some issues found. Install missing packages with:")
        print("        pip install -r requirements.txt\n")
    return all_ok


def download_model(model_key: str = "qwen25-coder"):
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return

    model = MODELS[model_key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / model["filename"]

    if dest.exists():
        print(f"\n[Setup] Already downloaded: {dest}")
        print(f"[Setup] Delete it to re-download.\n")
        _update_active_model(model_key, model)
        return

    print(f"\n[Setup] Downloading: {model['name']}")
    print(f"[Setup] Size: ~{model['size_gb']:.1f} GB | RAM needed: ~{model['ram_gb']:.1f} GB")
    print(f"[Setup] Note: {model['notes']}")
    print(f"[Setup] Destination: {dest}\n")

    try:
        import httpx
        print("[Setup] Downloading... (this will take 5-15 minutes for 4.5GB)")
        with httpx.stream("GET", model["url"], follow_redirects=True, timeout=3600) as r:
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                        print(f"\r  [{bar}] {pct:.1f}%  {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end="", flush=True)
        print(f"\n\n[Setup] Download complete: {dest}")
        _update_active_model(model_key, model)
    except ImportError:
        print("[Setup] httpx not installed. Run: pip install httpx")
    except Exception as e:
        print(f"\n[Setup] Download failed: {e}")
        print(f"[Setup] Manual download URL:\n  {model['url']}")
        print(f"[Setup] Save to: {dest}\n")


def _update_active_model(model_key: str, model: dict):
    """Update the engine config to use the newly downloaded model."""
    config_dir = Path.home() / ".leanai"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "active_model.txt"

    model_path = MODEL_DIR / model["filename"]
    config_file.write_text(str(model_path))

    print(f"\n[Setup] Active model set to: {model['name']}")
    print(f"[Setup] Path: {model_path}")
    print(f"[Setup] Prompt format: {model['prompt_format']}")
    print(f"\n[Setup] LeanAI will now use {model['name']} automatically.")
    print(f"[Setup] Run: python main.py\n")


def list_models():
    print("\nAvailable models:\n")

    # Check active model
    config_file = Path.home() / ".leanai" / "active_model.txt"
    active_path = config_file.read_text().strip() if config_file.exists() else ""

    for key, model in MODELS.items():
        dest = MODEL_DIR / model["filename"]
        status = "downloaded" if dest.exists() else "not downloaded"
        is_active = str(dest) == active_path
        active_marker = " ← ACTIVE" if is_active else ""
        print(f"  {key}{active_marker}")
        print(f"    {model['name']}")
        print(f"    Size: {model['size_gb']:.1f} GB | RAM: {model['ram_gb']:.1f} GB | Status: {status}")
        print(f"    {model['notes']}")
        print()


def set_model(model_key: str):
    """Switch to a different already-downloaded model."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        return
    model = MODELS[model_key]
    dest = MODEL_DIR / model["filename"]
    if not dest.exists():
        print(f"Model not downloaded yet. Run: python setup.py --download-model --model {model_key}")
        return
    _update_active_model(model_key, model)


def main():
    parser = argparse.ArgumentParser(description="LeanAI Setup")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--download-model", action="store_true", help="Download a model")
    parser.add_argument("--model", default="qwen25-coder", help="Model key (default: qwen25-coder)")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--set-model", type=str, help="Switch active model")
    args = parser.parse_args()

    if args.list_models:
        list_models()
    elif args.download_model:
        check_environment()
        download_model(args.model)
    elif args.set_model:
        set_model(args.set_model)
    elif args.check:
        check_environment()
    else:
        check_environment()
        print("Commands:")
        print("  python setup.py --download-model              # download Qwen2.5 Coder 7B")
        print("  python setup.py --download-model --model phi3-mini  # download specific model")
        print("  python setup.py --list-models                 # list all models")
        print("  python setup.py --set-model qwen25-coder      # switch active model")


if __name__ == "__main__":
    main()
