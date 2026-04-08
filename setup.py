"""
LeanAI · Setup Script
Downloads the base model and verifies your environment.
Run: python setup.py --download-model
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

MODEL_DIR = Path.home() / ".leanai" / "models"

MODELS = {
    "phi3-mini": {
        "name": "Phi-3 Mini 4K (recommended — best quality/size ratio)",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "phi3-mini-q4.gguf",
        "size_gb": 2.3,
        "ram_gb": 3.0,
    },
    "tinyllama": {
        "name": "TinyLlama 1.1B (smallest — runs on 1GB RAM)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-q4.gguf",
        "size_gb": 0.7,
        "ram_gb": 1.0,
    },
    "llama3-1b": {
        "name": "Llama 3.2 1B (Meta — great quality at 1B)",
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "llama3-1b-q4.gguf",
        "size_gb": 0.8,
        "ram_gb": 1.2,
    },
}


def check_environment():
    print("\n[Setup] Checking environment...\n")

    checks = []

    # Python version
    py_ver = sys.version_info
    ok = py_ver >= (3, 10)
    checks.append((ok, f"Python {py_ver.major}.{py_ver.minor}", "3.10+ required"))

    # Available RAM (rough)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        ok = ram_gb >= 1.0
        checks.append((ok, f"RAM: {ram_gb:.1f} GB", "1GB+ recommended"))
    except ImportError:
        checks.append((None, "RAM check skipped", "install psutil for RAM check"))

    # CPU cores
    import os
    cores = os.cpu_count()
    checks.append((cores >= 2, f"CPU cores: {cores}", "2+ recommended"))

    # Disk space
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9
    ok = free_gb >= 3.0
    checks.append((ok, f"Free disk: {free_gb:.1f} GB", "3GB+ needed for model"))

    # Key packages
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

    # Print results
    all_ok = True
    for status, label, note in checks:
        if status is True:
            icon = "✓"
        elif status is False:
            icon = "✗"
            all_ok = False
        else:
            icon = "?"
        print(f"  {icon}  {label}  ({note})")

    print()
    if all_ok:
        print("[Setup] Environment looks good!\n")
    else:
        print("[Setup] Some issues found. Install missing packages with:")
        print("        pip install -r requirements.txt\n")

    return all_ok


def download_model(model_key: str = "phi3-mini"):
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return

    model = MODELS[model_key]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODEL_DIR / model["filename"]

    if dest.exists():
        print(f"\n[Setup] Model already downloaded: {dest}")
        print(f"[Setup] Delete it to re-download.\n")
        return

    print(f"\n[Setup] Downloading: {model['name']}")
    print(f"[Setup] Size: ~{model['size_gb']:.1f} GB | RAM needed: ~{model['ram_gb']:.1f} GB")
    print(f"[Setup] Destination: {dest}\n")

    try:
        import httpx
        print("[Setup] Downloading... (this may take a few minutes)")
        with httpx.stream("GET", model["url"], follow_redirects=True) as r:
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                        print(f"\r  [{bar}] {pct:.1f}%  {downloaded/1e6:.0f}/{total/1e6:.0f} MB", end="")
        print(f"\n\n[Setup] Download complete: {dest}\n")
    except ImportError:
        print("[Setup] httpx not installed. Trying wget...")
        os.system(f'wget -O "{dest}" "{model["url"]}"')
    except Exception as e:
        print(f"\n[Setup] Download failed: {e}")
        print(f"[Setup] Manual download URL:\n  {model['url']}")
        print(f"[Setup] Save to: {dest}\n")


def list_models():
    print("\nAvailable models:\n")
    for key, model in MODELS.items():
        dest = MODEL_DIR / model["filename"]
        status = "downloaded" if dest.exists() else "not downloaded"
        print(f"  {key}")
        print(f"    {model['name']}")
        print(f"    Size: {model['size_gb']:.1f} GB | RAM: {model['ram_gb']:.1f} GB | Status: {status}")
        print()


def main():
    parser = argparse.ArgumentParser(description="LeanAI Setup")
    parser.add_argument("--check", action="store_true", help="Check environment")
    parser.add_argument("--download-model", action="store_true", help="Download default model")
    parser.add_argument("--model", default="phi3-mini", help="Model to download")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        list_models()
    elif args.download_model:
        check_environment()
        download_model(args.model)
    elif args.check:
        check_environment()
    else:
        check_environment()
        print("Run with --download-model to download a model.")
        print("Run with --list-models to see available models.")


if __name__ == "__main__":
    main()
