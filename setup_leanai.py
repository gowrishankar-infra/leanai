"""
LeanAI — One-Command Setup
Run: python setup_leanai.py

This script:
  1. Checks Python version
  2. Installs all dependencies
  3. Downloads the Qwen2.5 Coder 7B model (4.4 GB)
  4. Creates necessary directories
  5. Launches LeanAI

That's it. One command, fully working.
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path


# ── Config ────────────────────────────────────────────────────

MODEL_DIR = Path.home() / ".leanai" / "models"
MODEL_NAME = "qwen25-coder-7b-q4.gguf"
MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
MODEL_SIZE_GB = 4.4

REQUIRED_PYTHON = (3, 9)
REQUIRED_RAM_GB = 8


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              LeanAI — One-Command Setup                 ║")
    print("║     Local AI that understands your entire codebase      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def print_step(num, total, msg):
    print(f"  [{num}/{total}] {msg}", flush=True)


def print_ok(msg):
    print(f"       ✓ {msg}")


def print_warn(msg):
    print(f"       ⚠ {msg}")


def print_fail(msg):
    print(f"       ✗ {msg}")


def check_python():
    """Check Python version."""
    v = sys.version_info
    if (v.major, v.minor) < REQUIRED_PYTHON:
        print_fail(f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required. You have {v.major}.{v.minor}.")
        print(f"\n       Download Python: https://www.python.org/downloads/")
        sys.exit(1)
    print_ok(f"Python {v.major}.{v.minor}.{v.micro}")


def check_git():
    """Check git is installed."""
    if shutil.which("git"):
        print_ok("Git found")
    else:
        print_warn("Git not found — git features will be limited")
        print(f"       Download Git: https://git-scm.com/downloads")


def check_ram():
    """Check available RAM."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        if ram_gb < REQUIRED_RAM_GB:
            print_warn(f"Only {ram_gb:.1f} GB RAM — minimum {REQUIRED_RAM_GB} GB recommended")
        else:
            print_ok(f"{ram_gb:.1f} GB RAM")
    except ImportError:
        print_ok("RAM check skipped (psutil not installed)")


def check_disk_space():
    """Check if enough disk space for model."""
    try:
        free_gb = shutil.disk_usage(str(Path.home())).free / (1024 ** 3)
        if free_gb < MODEL_SIZE_GB + 1:
            print_fail(f"Only {free_gb:.1f} GB free disk space. Need ~{MODEL_SIZE_GB + 1:.0f} GB.")
            sys.exit(1)
        print_ok(f"{free_gb:.1f} GB free disk space")
    except Exception:
        print_ok("Disk space check skipped")


def install_dependencies():
    """Install Python dependencies."""
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print_fail("requirements.txt not found. Are you in the LeanAI directory?")
        sys.exit(1)

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file),
             "--break-system-packages", "-q"],
            check=True,
            capture_output=True,
            text=True,
        )
        print_ok("All dependencies installed")
    except subprocess.CalledProcessError as e:
        # Try without --break-system-packages (older pip)
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
                check=True,
                capture_output=True,
                text=True,
            )
            print_ok("All dependencies installed")
        except subprocess.CalledProcessError as e2:
            print_fail(f"Failed to install dependencies")
            print(f"       Try manually: pip install -r requirements.txt")
            print(f"       Error: {e2.stderr[:200]}")
            sys.exit(1)


def download_model():
    """Download Qwen2.5 Coder 7B model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / MODEL_NAME

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
        if size_gb > 1:
            print_ok(f"Model already downloaded ({size_gb:.1f} GB)")
            return
        else:
            print_warn(f"Model file exists but seems incomplete ({size_gb:.2f} GB). Re-downloading...")
            model_path.unlink()

    print(f"       Downloading Qwen2.5 Coder 7B ({MODEL_SIZE_GB} GB)...")
    print(f"       This will take 5-15 minutes depending on your internet speed.")
    print(f"       Saving to: {model_path}")
    print()

    # Try huggingface-cli first
    try:
        subprocess.run(
            ["huggingface-cli", "download",
             "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
             "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
             "--local-dir", str(MODEL_DIR)],
            check=True,
        )
        # Rename if needed
        downloaded = MODEL_DIR / "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        if downloaded.exists() and not model_path.exists():
            downloaded.rename(model_path)
        print_ok(f"Model downloaded to {model_path}")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback: use Python urllib
    try:
        import urllib.request

        def progress_hook(count, block_size, total_size):
            downloaded = count * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            downloaded_gb = downloaded / (1024 ** 3)
            total_gb = total_size / (1024 ** 3)
            bar = "█" * int(percent / 5) + "░" * (20 - int(percent / 5))
            print(f"\r       [{bar}] {percent:.0f}% ({downloaded_gb:.1f}/{total_gb:.1f} GB)", end="", flush=True)

        urllib.request.urlretrieve(MODEL_URL, str(model_path), reporthook=progress_hook)
        print()
        print_ok(f"Model downloaded to {model_path}")
    except Exception as e:
        print()
        print_fail(f"Download failed: {e}")
        print(f"\n       Download manually from:")
        print(f"       {MODEL_URL}")
        print(f"       Save to: {model_path}")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    dirs = [
        Path.home() / ".leanai",
        Path.home() / ".leanai" / "models",
        Path.home() / ".leanai" / "sessions",
        Path.home() / ".leanai" / "training_exports",
        Path.home() / ".leanai" / "evolution",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print_ok(f"Data directory: {Path.home() / '.leanai'}")


def verify_installation():
    """Quick verification that everything works."""
    try:
        import llama_cpp
        print_ok("llama-cpp-python loaded")
    except ImportError:
        print_warn("llama-cpp-python not found — may need manual install")

    try:
        import chromadb
        print_ok("ChromaDB loaded")
    except ImportError:
        print_warn("ChromaDB not found — memory features limited")

    try:
        import fastapi
        print_ok("FastAPI loaded")
    except ImportError:
        print_warn("FastAPI not found — web server unavailable")

    model_path = MODEL_DIR / MODEL_NAME
    if model_path.exists():
        print_ok(f"Model ready: {MODEL_NAME}")
    else:
        print_warn("Model not found — will be downloaded on first run")


def launch_leanai():
    """Ask user if they want to launch LeanAI."""
    print()
    print("═══════════════════════════════════════════════════════════")
    print("  Setup complete! LeanAI is ready.")
    print()
    print("  To start LeanAI:")
    print(f"    python main.py")
    print()
    print("  First thing to do after starting:")
    print("    /brain .    (scan your project)")
    print()
    print("  Then just ask questions about your code!")
    print("═══════════════════════════════════════════════════════════")
    print()

    try:
        answer = input("  Launch LeanAI now? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            print()
            os.execv(sys.executable, [sys.executable, "main.py"])
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Run 'python main.py' when you're ready.")


def main():
    print_banner()

    total_steps = 7

    print_step(1, total_steps, "Checking system requirements...")
    check_python()
    check_git()
    check_ram()
    check_disk_space()

    print()
    print_step(2, total_steps, "Creating directories...")
    create_directories()

    print()
    print_step(3, total_steps, "Installing Python dependencies...")
    install_dependencies()

    print()
    print_step(4, total_steps, "Downloading AI model...")
    download_model()

    print()
    print_step(5, total_steps, "Verifying installation...")
    verify_installation()

    print()
    print_step(6, total_steps, "Setup complete!")
    
    print()
    print_step(7, total_steps, "Ready to launch")
    launch_leanai()


if __name__ == "__main__":
    main()
