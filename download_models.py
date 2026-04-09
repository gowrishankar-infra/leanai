#!/usr/bin/env python3
"""
LeanAI — Model Downloader
Downloads AI models for local inference.

Usage:
    python download_models.py                  # show available models
    python download_models.py qwen-32b         # download the 32B model
    python download_models.py qwen-14b         # download the 14B model
    python download_models.py all              # download all models
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_manager import ModelManager


def main():
    manager = ModelManager()

    if len(sys.argv) < 2:
        print("LeanAI Model Downloader")
        print("=" * 50)
        print(manager.list_models())
        print("\nUsage:")
        print("  python download_models.py qwen-32b    # download 32B model (18 GB)")
        print("  python download_models.py qwen-14b    # download 14B model (8.5 GB)")
        print("  python download_models.py all          # download all models")
        print("\nManual download commands:")
        for key in manager.models:
            if not manager.models[key].is_downloaded:
                print(f"\n{manager.download_command(key)}")
        return

    target = sys.argv[1].lower()

    if target == "all":
        for key in manager.models:
            if not manager.models[key].is_downloaded:
                success, msg = manager.download(key)
                print(msg)
        return

    if target == "list":
        print(manager.list_models())
        return

    success, msg = manager.download(target)
    print(msg)

    if success:
        print(f"\nTo use this model, run LeanAI and type:")
        print(f"  /model {target}")
        print(f"\nOr set auto mode (uses big model for complex queries):")
        print(f"  /model auto")


if __name__ == "__main__":
    main()
