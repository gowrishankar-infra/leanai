"""
M11 — Lightweight persistent config.

LeanAI had no settings store, so user choices (like the file-injection
character limit) were hardcoded. This adds a tiny JSON config at
``$LEANAI_HOME/config.json`` (default ``~/.leanai/config.json``) with safe
defaults, robust to a missing or corrupt file.

Deliberately minimal: load/get/set/all. No schema migrations, no startup
mutation of anything else — it only reads/writes its own JSON file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    # Max characters of an attached file injected into a prompt (main.py).
    "snippet_limit": 8000,
    # M11: default cross-file mode for incremental Sentinel.
    "incremental_cross_file": False,
}

# Per-key validation so a bad value can't poison behaviour.
_VALIDATORS = {
    "snippet_limit": lambda v: isinstance(v, int) and 200 <= v <= 200_000,
    "incremental_cross_file": lambda v: isinstance(v, bool),
}


def _home() -> str:
    return os.environ.get("LEANAI_HOME", os.path.join(str(Path.home()), ".leanai"))


def _config_path() -> str:
    return os.path.join(_home(), "config.json")


class LeanAIConfig:
    """Read/write the JSON settings file. Always returns valid values."""

    def __init__(self, path: str = None):
        self.path = path or _config_path()
        self._data: Dict[str, Any] = dict(DEFAULTS)
        self.load()

    def load(self) -> "LeanAIConfig":
        self._data = dict(DEFAULTS)
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    stored = json.load(fh)
                if isinstance(stored, dict):
                    for k, v in stored.items():
                        if k in DEFAULTS and self._valid(k, v):
                            self._data[k] = v
        except Exception:
            # Corrupt/unreadable -> fall back to defaults silently.
            self._data = dict(DEFAULTS)
        return self

    def _valid(self, key: str, value: Any) -> bool:
        validator = _VALIDATORS.get(key)
        return validator(value) if validator else True

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self._data[key]
        return DEFAULTS.get(key, default)

    def set(self, key: str, value: Any) -> bool:
        """Set + persist. Returns False (no write) if key/value is invalid."""
        if key not in DEFAULTS or not self._valid(key, value):
            return False
        self._data[key] = value
        return self._save()

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def _save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp, self.path)   # atomic
            return True
        except Exception:
            return False
