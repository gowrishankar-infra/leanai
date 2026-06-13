#!/usr/bin/env python3
"""
LeanAI — One-Command Setup
Checks system, installs dependencies, downloads model, and launches LeanAI.

Usage:
    python setup_leanai.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    print()
    print("=" * 60)
    print("  LeanAI Setup")
    print("  Project-Aware AI Coding System")
    print("=" * 60)
    print()


def check_python():
    """Check Python version."""
    v = sys.version_info
    print(f"  Python: {v.major}.{v.minor}.{v.micro}", end="")
    if v.major == 3 and v.minor >= 10:
        print(" ✓")
        return True
    else:
        print(" ✗ (need 3.10+)")
        return False


def check_pip():
    """Check pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      capture_output=True, check=True)
        print("  pip: installed ✓")
        return True
    except Exception:
        print("  pip: not found ✗")
        return False


def check_gpu():
    """Check for GPU."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract GPU name
            for line in result.stdout.split("\n"):
                if "NVIDIA" in line and "GeForce" in line or "RTX" in line or "GTX" in line:
                    gpu = line.strip().split("|")[1].strip() if "|" in line else "NVIDIA GPU"
                    print(f"  GPU: {gpu} ✓")
                    return True
            print("  GPU: NVIDIA detected ✓")
            return True
    except FileNotFoundError:
        pass
    print("  GPU: not detected (will use CPU — still works, just slower)")
    return False


def check_ram():
    """Check available RAM."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"  RAM: {ram_gb:.0f} GB", end="")
        if ram_gb >= 8:
            print(" ✓")
        else:
            print(" ⚠ (8 GB recommended)")
        return ram_gb
    except ImportError:
        print("  RAM: unknown (psutil not installed)")
        return 0


def check_disk():
    """Check available disk space."""
    try:
        models_dir = Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        free = shutil.disk_usage(str(models_dir)).free / (1024**3)
        print(f"  Disk: {free:.0f} GB free", end="")
        if free >= 5:
            print(" ✓")
        else:
            print(" ⚠ (5 GB needed for 7B model)")
        return free
    except Exception:
        print("  Disk: unknown")
        return 0


def install_dependencies(local_models=True):
    """Install Python dependencies — auto-creates venv if needed.

    local_models=False installs the lighter remote-only set (no
    llama-cpp-python), for machines that only talk to a remote endpoint.
    """
    print("\n  Installing dependencies...")
    req_name = "requirements.txt" if local_models else "requirements-remote.txt"
    req_file = Path(__file__).parent / req_name
    project_dir = Path(__file__).parent
    venv_dir = project_dir / ".venv"

    # Check if we're already in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if not in_venv:
        # Try a test install first
        test = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "numpy", "-q"],
            capture_output=True, text=True
        )
        if "externally-managed-environment" in (test.stderr or ""):
            # Ubuntu 23.04+ blocks system-wide pip installs — need a venv
            print("  System Python detected (externally managed).")
            print(f"  Creating virtual environment at {venv_dir}...")
            try:
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
                # Determine the pip/python paths inside the venv
                if platform.system() == "Windows":
                    venv_python = str(venv_dir / "Scripts" / "python.exe")
                    venv_pip = str(venv_dir / "Scripts" / "pip.exe")
                else:
                    venv_python = str(venv_dir / "bin" / "python")
                    venv_pip = str(venv_dir / "bin" / "pip")

                print(f"  Virtual environment created ✓")
                print(f"  Installing dependencies inside venv...")

                if req_file.exists():
                    result = subprocess.run(
                        [venv_pip, "install", "-r", str(req_file), "-q"],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        print("  Dependencies installed ✓")
                    else:
                        print(f"  Some packages failed. Trying individually...")
                        _install_individual(venv_pip, local_models)
                else:
                    _install_individual(venv_pip, local_models)

                print(f"\n  ⚠ IMPORTANT: Activate the venv before running LeanAI:")
                if platform.system() == "Windows":
                    print(f"    .venv\\Scripts\\activate")
                else:
                    print(f"    source .venv/bin/activate")
                print(f"    python main.py")
                return True

            except Exception as e:
                print(f"  Could not create venv: {e}")
                print("  Try manually:")
                print(f"    python3 -m venv .venv")
                print(f"    source .venv/bin/activate")
                print(f"    pip install -r requirements.txt")
                return False

    # Normal install (already in venv or system allows pip)
    if req_file.exists():
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  Dependencies installed ✓")
            return True
        else:
            print(f"  Some packages failed. Trying individually...")

    _install_individual(sys.executable + " -m pip", local_models)
    return True


def _install_individual(pip_cmd, local_models=True):
    """Install packages one by one. Used when bulk install fails."""
    packages = [
        "chromadb",
        "sentence-transformers",
        "rank-bm25",           # M6 InfiniteContext — hybrid retrieval
        "pyyaml",              # remote endpoints config
        "fastapi",
        "uvicorn",
        "numpy",
        "pydantic",
        "psutil",
        "pytest",
    ]
    if local_models:
        # llama-cpp-python + huggingface-hub are only needed for LOCAL models.
        packages = ["llama-cpp-python", "huggingface-hub"] + packages

    for pkg in packages:
        try:
            if isinstance(pip_cmd, str) and " " in pip_cmd:
                subprocess.run(
                    pip_cmd.split() + ["install", pkg, "-q"],
                    capture_output=True, check=True
                )
            else:
                subprocess.run(
                    [pip_cmd, "install", pkg, "-q"],
                    capture_output=True, check=True
                )
            print(f"    {pkg} ✓")
        except Exception:
            print(f"    {pkg} ✗ (install manually: pip install {pkg})")


def download_model():
    """Download the 7B model."""
    models_dir = Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Check if any model already exists
    existing = list(models_dir.glob("*.gguf"))
    if existing:
        model_path = str(existing[0])
        print(f"\n  Model found: {existing[0].name} ✓")
        _set_active_model(model_path)
        return True

    print("\n  Downloading Qwen2.5 Coder 7B (4.5 GB)...")
    print("  This may take 5-15 minutes depending on your connection.\n")

    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "from huggingface_hub import hf_hub_download; "
             "import os; "
             "dest = os.path.join(os.environ.get('LEANAI_HOME', os.path.join(os.path.expanduser('~'), '.leanai')), 'models'); "
             "os.makedirs(dest, exist_ok=True); "
             "hf_hub_download('Qwen/Qwen2.5-Coder-7B-Instruct-GGUF', "
             "'qwen2.5-coder-7b-instruct-q4_k_m.gguf', local_dir=dest); "
             "print('DONE')"],
            capture_output=False, text=True, timeout=3600
        )

        # Find the downloaded model
        downloaded = list(models_dir.glob("*.gguf"))
        if downloaded:
            model_path = str(downloaded[0])
            print(f"\n  Model downloaded: {downloaded[0].name} ✓")
            _set_active_model(model_path)
            return True
        else:
            print("\n  Download may have failed. Try manually:")
            print("    python download_models.py qwen-7b")
            return False

    except subprocess.TimeoutExpired:
        print("\n  Download timed out. Try manually:")
        print("    python download_models.py qwen-7b")
        return False
    except Exception as e:
        print(f"\n  Download error: {e}")
        print("  Try manually:")
        print("    python download_models.py qwen-7b")
        return False


def _set_active_model(model_path: str):
    """Save the active model path so LeanAI knows which model to use."""
    try:
        config_file = Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "active_model.txt"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(model_path)
        print(f"  Active model set: {Path(model_path).name} ✓")
    except Exception as e:
        print(f"  Warning: could not save active model config: {e}")


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Prompt y/n. Non-interactive (no stdin) falls back to `default`."""
    suffix = " (Y/n): " if default else " (y/N): "
    try:
        ans = input(question + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


def _leanai_home() -> Path:
    return Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai")))


def setup_remote_endpoint() -> bool:
    """Guided remote-model connect: auto-discovers the server's models so the
    user only types an address and picks one. Delegates to the shared wizard in
    core.endpoints so the same flow is available in-app via '/model connect'.
    """
    try:
        from core.endpoints import connect_interactive
    except Exception as e:
        print(f"  Could not load remote setup ({e}).")
        print("  You can configure it later — see endpoints.example.yaml / REMOTE_MODELS.md.")
        return False
    alias = connect_interactive(home=_leanai_home())
    return alias is not None


def main():
    print_header()

    # System checks
    print("  System Check")
    print("  " + "-" * 40)
    
    py_ok = check_python()
    if not py_ok:
        print("\n  Python 3.10+ is required. Please upgrade.")
        sys.exit(1)

    pip_ok = check_pip()
    if not pip_ok:
        print("\n  pip is required. Install it first.")
        sys.exit(1)

    has_gpu = check_gpu()
    check_ram()
    check_disk()

    print(f"\n  Platform: {platform.system()} {platform.machine()}")

    # Local models vs remote endpoint
    print("\n  Model location")
    print("  " + "-" * 40)
    print("  LeanAI can run a model locally (needs RAM/CPU/GPU + a few GB on disk)")
    print("  OR talk to one served elsewhere (Ollama / OpenAI-compatible API).")
    if not has_gpu:
        print("  No GPU was detected — a remote endpoint is a good fit for this machine.")
    local_models = ask_yes_no("  Run models LOCALLY on this machine?", default=has_gpu)

    # Install dependencies
    print("\n  Dependencies")
    print("  " + "-" * 40)
    install_dependencies(local_models=local_models)

    # Model — local download OR remote endpoint config
    print("\n  Model")
    print("  " + "-" * 40)
    remote_ready = False
    if local_models:
        download_model()
    else:
        print("  Skipping local model download (using a remote endpoint).")
        remote_ready = setup_remote_endpoint()

    # GPU acceleration hint
    if has_gpu and local_models:
        print("\n  GPU Acceleration (optional)")
        print("  " + "-" * 40)
        print("  Your GPU was detected! For 3.5x faster responses:")
        print("    1. Install Vulkan SDK: https://vulkan.lunarg.com/sdk/home")
        if platform.system() == "Windows":
            print('    2. $env:CMAKE_ARGS="-DGGML_VULKAN=ON"')
        else:
            print('    2. export CMAKE_ARGS="-DGGML_VULKAN=ON"')
        print("    3. pip install llama-cpp-python --no-cache-dir --force-reinstall")

    # Ready
    print("\n" + "=" * 60)
    print("  Setup complete! ✓")
    print("=" * 60)
    print()
    print("  To start LeanAI:")
    print("    python main.py")
    print()
    print("  First thing to do (unlocks project-aware features):")
    print("    /brain .          # scan + auto-build AST-grounded semantic index")
    print("    /model auto       # auto-routes queries to the best-suited model")
    print()
    print("  Key LeanAI features:")
    print("    /ask <question>   # hybrid semantic + BM25 + graph retrieval (M6)")
    print("    /sentinel         # security analysis — 12 OWASP vuln classes (M1)")
    print("    /chainbreak       # multi-stage attack chain detection (M2)")
    print("    /exploit --all    # benign proof-of-concept demos (M3)")
    print("    /forensics <fn>   # deterministic git+AST archaeology (M4)")
    print()
    if local_models:
        print("  Upgrade models (recommended, ~17 GB each):")
        print("    python download_models.py gemma4-26b    # best for frontend/UI")
        print("    python download_models.py qwen35-27b    # best for backend/reasoning")
    else:
        print("  Using a remote endpoint:")
        print(f"    /model              # your remote alias is listed here")
        print(f"    /model test         # check the endpoint is reachable")
        print(f"    Edit ~/.leanai/endpoints.yaml to add more remote models.")
    print()
    print("  Full command reference: COMMANDS.md")
    print("  Project README:         README.md")
    print()

    # Ask if they want to launch
    venv_dir = Path(__file__).parent / ".venv"
    if platform.system() == "Windows":
        venv_python = str(venv_dir / "Scripts" / "python.exe")
    else:
        venv_python = str(venv_dir / "bin" / "python")

    # Use venv python if it exists, otherwise system python
    python_cmd = venv_python if Path(venv_python).exists() else sys.executable

    try:
        answer = input("  Launch LeanAI now? (y/n): ").strip().lower()
        if answer in ("y", "yes", ""):
            print()
            # SECURITY: subprocess.run with a list and shell=False — no
            # injection risk even if python_cmd contains spaces or special
            # chars (VULN-2026-0002 / "VULN-2026-0011" in earlier scan).
            # Replaces: os.system(f"{python_cmd} main.py")
            try:
                subprocess.run([python_cmd, "main.py"], shell=False, check=False)
            except FileNotFoundError:
                print(f"  Could not launch: {python_cmd} not found")
                print("  Run 'python main.py' manually when ready.")
            except KeyboardInterrupt:
                pass
    except (EOFError, KeyboardInterrupt):
        print("\n  Run 'python main.py' when ready.")


if __name__ == "__main__":
    main()
