"""
LeanAI — External model repository loader.

Reads ``$LEANAI_HOME/endpoints.yaml`` (default ``~/.leanai/endpoints.yaml``) and
turns it into a registry of remote model aliases the rest of LeanAI can switch
to exactly like a local model. See ``endpoints.example.yaml`` for the schema.

Robust by design: a missing file, a missing pyyaml, or a malformed entry never
crashes LeanAI — it logs a short note and returns whatever parsed cleanly.

API-key handling:
  * ``api_key:``      literal value (fine for Ollama, which ignores it).
  * ``api_key_env:``  name of an environment variable to read the key from.
  * ``${VAR}``        any string field may interpolate an env var.
Keys are never printed.
"""

from __future__ import annotations

import ipaddress as _ipaddress
import json
import os
import platform
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.remote_model import RemoteModel


def _home() -> Path:
    return Path(os.environ.get("LEANAI_HOME", str(Path.home() / ".leanai")))


def config_path() -> Optional[Path]:
    """Return the first existing endpoints config (.yaml, .yml, or .json)."""
    for name in ("endpoints.yaml", "endpoints.yml", "endpoints.json"):
        p = _home() / name
        if p.exists():
            return p
    return None


@dataclass
class RemoteSpec:
    alias: str
    endpoint_name: str
    base_url: str
    model_id: str
    api_key: Optional[str] = None
    mode: str = "completions"
    prompt_format: str = "chatml"
    context: int = 8192
    timeout: int = 180
    quality_score: int = 60
    speed_label: str = "remote"
    description: str = ""


def _interp(value: Any) -> Any:
    """Interpolate ${ENV_VAR} inside strings."""
    if not isinstance(value, str) or "${" not in value:
        return value
    out = value
    for var in {m for m in _ENV_RE.findall(value)}:
        out = out.replace("${" + var + "}", os.environ.get(var, ""))
    return out


import re as _re
_ENV_RE = _re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_key(ep: Dict[str, Any]) -> Optional[str]:
    if ep.get("api_key_env"):
        return os.environ.get(str(ep["api_key_env"])) or None
    key = ep.get("api_key")
    return _interp(key) if key else None


def _parse(doc: Dict[str, Any]) -> Dict[str, RemoteSpec]:
    specs: Dict[str, RemoteSpec] = {}
    endpoints = doc.get("endpoints") or []
    if not isinstance(endpoints, list):
        print("[LeanAI] endpoints.yaml: 'endpoints' must be a list — ignoring.")
        return specs

    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        name = str(ep.get("name", "endpoint"))
        base_url = _interp(ep.get("base_url", ""))
        if not base_url:
            print(f"[LeanAI] endpoints.yaml: endpoint '{name}' has no base_url — skipped.")
            continue
        mode = str(ep.get("mode", "completions")).lower()
        api_key = _resolve_key(ep)
        ep_timeout = int(ep.get("timeout", 180))

        for m in ep.get("models", []) or []:
            if not isinstance(m, dict):
                continue
            alias = str(m.get("alias") or m.get("id") or "").strip().lower()
            model_id = str(m.get("id") or m.get("alias") or "").strip()
            if not alias or not model_id:
                print(f"[LeanAI] endpoints.yaml: a model under '{name}' is missing id/alias — skipped.")
                continue
            if alias in specs:
                print(f"[LeanAI] endpoints.yaml: duplicate alias '{alias}' — keeping the first.")
                continue
            specs[alias] = RemoteSpec(
                alias=alias,
                endpoint_name=name,
                base_url=base_url,
                model_id=model_id,
                api_key=api_key,
                mode=mode,
                prompt_format=str(m.get("prompt_format", "chatml")).lower(),
                context=int(m.get("context", 8192)),
                timeout=int(m.get("timeout", ep_timeout)),
                quality_score=int(m.get("quality", 60)),
                speed_label=str(m.get("speed", "remote")),
                description=str(m.get("description", "")) or f"{name}: {model_id}",
            )
    return specs


def load_endpoints(path: Optional[str] = None) -> Dict[str, RemoteSpec]:
    """Load remote specs keyed by alias. Returns {} on any problem."""
    p = Path(path) if path else config_path()
    if not p or not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            doc = json.loads(text)
        else:
            try:
                import yaml  # PyYAML
            except ImportError:
                print("[LeanAI] endpoints.yaml found but PyYAML is not installed. "
                      "Run: pip install pyyaml  (or use endpoints.json instead).")
                return {}
            doc = yaml.safe_load(text)
        if not isinstance(doc, dict):
            print(f"[LeanAI] {p.name}: top level must be a mapping — ignoring.")
            return {}
        return _parse(doc)
    except Exception as e:
        print(f"[LeanAI] Could not read {p}: {e}")
        return {}


def make_client(spec: RemoteSpec) -> RemoteModel:
    return RemoteModel(
        base_url=spec.base_url,
        model=spec.model_id,
        api_key=spec.api_key,
        mode=spec.mode,
        prompt_format=spec.prompt_format,
        timeout=spec.timeout,
    )


# ══════════════════════════════════════════════════════════════════
# Smooth setup / auto-discovery
#
# Goal: someone with no technical knowledge points LeanAI at a server by
# typing an address (or just an IP). LeanAI figures out the rest — queries
# the server for its model list, lets them pick one, guesses the prompt
# format, picks the right mode, and writes endpoints.yaml for them. They
# never have to learn the schema.
# ══════════════════════════════════════════════════════════════════

OLLAMA_DEFAULT = "http://localhost:11434/v1"


def normalize_base_url(raw: str) -> Tuple[str, bool]:
    """Turn loose user input into a usable OpenAI-style base_url.

    Accepts what a non-technical user is likely to type::

        192.168.1.50            -> http://192.168.1.50:11434/v1
        192.168.1.50:11434      -> http://192.168.1.50:11434/v1
        http://host:11434       -> http://host:11434/v1
        https://api.openai.com  -> https://api.openai.com/v1

    Returns ``(base_url, is_openai)``.
    """
    s = (raw or "").strip().rstrip("/")
    if not s:
        return OLLAMA_DEFAULT, False
    if "://" not in s:
        host = s
        if ":" not in host:           # bare host/IP -> assume Ollama's port
            host = host + ":11434"
        s = "http://" + host
    is_openai = "api.openai.com" in s
    tail = s.split("://", 1)[1]
    if "/v1" not in tail:             # don't double-append if already versioned
        s = s + "/v1"
    return s, is_openai


def is_local_url(base_url: str) -> bool:
    """True if the URL points at this machine or a private LAN address.

    Used to decide whether to warn that code will leave the machine and
    whether to bother asking for an API key.
    """
    try:
        host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    except Exception:
        return False
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        return True
    if host.endswith(".local"):
        return True
    try:
        ip = _ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except ValueError:
        return False  # a hostname like api.openai.com -> external


def guess_prompt_format(model_id: str) -> str:
    """Best-effort LeanAI prompt format from a model id/tag."""
    low = (model_id or "").lower()
    if "gemma" in low:
        return "gemma"
    if "llama" in low:
        return "llama3"
    if "phi" in low:
        return "phi3"
    return "chatml"


def _slug(model_id: str) -> str:
    s = "".join(c if (c.isalnum()) else "-" for c in (model_id or "remote").lower())
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "remote"


@dataclass
class ProbeResult:
    reachable: bool
    models: List[str]
    status: str            # human-readable one-liner
    auth_failed: bool = False


def _extract_model_ids(doc: Any) -> List[str]:
    """Pull model ids from an OpenAI /v1/models (or Ollama) response."""
    out: List[str] = []
    if isinstance(doc, dict):
        items = doc.get("data") or doc.get("models") or []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    mid = it.get("id") or it.get("name") or it.get("model")
                    if mid:
                        out.append(str(mid))
                elif isinstance(it, str):
                    out.append(it)
    return out


def probe_endpoint(base_url: str, api_key: Optional[str] = None,
                   timeout: int = 10) -> ProbeResult:
    """GET ``{base_url}/models`` and report reachability + discovered ids.

    Never raises. Powers both the connect wizard (list models to pick) and a
    richer ``/model test``.
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        doc = json.loads(body) if body.strip() else {}
        ids = _extract_model_ids(doc)
        return ProbeResult(True, ids, f"reachable — {len(ids)} model(s) found")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return ProbeResult(False, [], "authentication failed (check the API key)",
                               auth_failed=True)
        return ProbeResult(True, [], f"reachable (server returned HTTP {e.code})")
    except urllib.error.URLError as e:
        return ProbeResult(False, [], f"could not connect ({getattr(e, 'reason', e)})")
    except Exception as e:  # pragma: no cover - defensive
        return ProbeResult(False, [], f"error: {e}")


def _dump_yaml(doc: Dict[str, Any]) -> str:
    header = "# LeanAI remote endpoints — edit freely. Docs: REMOTE_MODELS.md\n"
    try:
        import yaml  # PyYAML
        return header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    except Exception:
        return _hand_yaml(doc)


def _hand_yaml(doc: Dict[str, Any]) -> str:
    """Fallback serializer for our known shape when PyYAML isn't available."""
    lines = ["# LeanAI remote endpoints — edit freely. Docs: REMOTE_MODELS.md",
             "endpoints:"]
    for ep in doc.get("endpoints", []):
        lines.append(f"  - name: {ep.get('name', 'endpoint')}")
        lines.append(f"    base_url: {ep.get('base_url', '')}")
        lines.append(f"    mode: {ep.get('mode', 'completions')}")
        if ep.get("api_key"):
            lines.append(f"    api_key: {ep['api_key']}")
        if ep.get("api_key_env"):
            lines.append(f"    api_key_env: {ep['api_key_env']}")
        lines.append(f"    timeout: {ep.get('timeout', 180)}")
        lines.append("    models:")
        for m in ep.get("models", []):
            lines.append(f"      - id: {m.get('id', '')}")
            lines.append(f"        alias: {m.get('alias', '')}")
            lines.append(f"        prompt_format: {m.get('prompt_format', 'chatml')}")
            if m.get("quality") is not None:
                lines.append(f"        quality: {m['quality']}")
            if m.get("speed"):
                lines.append(f"        speed: {m['speed']}")
    return "\n".join(lines) + "\n"


def add_remote_model(home: Any, base_url: str, model_id: str, alias: str,
                     mode: str, prompt_format: str,
                     api_key: Optional[str] = None,
                     api_key_env: Optional[str] = None,
                     make_active: bool = True) -> Tuple[Path, str]:
    """Append (or create) a remote model in endpoints.yaml and optionally set
    it active. Never clobbers a file it can't safely parse — in that case it
    writes ``endpoints.generated.yaml`` alongside instead. Returns (path, alias).
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    path = home / "endpoints.yaml"

    doc: Dict[str, Any] = {"endpoints": []}
    if path.exists():
        parsed = None
        try:
            import yaml
            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("endpoints"), list):
            doc = parsed
        else:
            # Couldn't parse the existing file — don't destroy it.
            path = home / "endpoints.generated.yaml"
            doc = {"endpoints": []}

    # Ensure alias uniqueness across all existing models.
    existing = {
        str(m.get("alias", "")).lower()
        for ep in doc.get("endpoints", []) if isinstance(ep, dict)
        for m in (ep.get("models") or []) if isinstance(m, dict)
    }
    alias = alias.lower()
    if alias in existing:
        i = 2
        while f"{alias}-{i}" in existing:
            i += 1
        alias = f"{alias}-{i}"

    model_block: Dict[str, Any] = {
        "id": model_id, "alias": alias,
        "prompt_format": prompt_format, "quality": 72, "speed": "remote",
    }
    ep_block: Dict[str, Any] = {
        "name": f"{alias}-endpoint", "base_url": base_url, "mode": mode,
        "timeout": 180, "models": [model_block],
    }
    if api_key:
        ep_block["api_key"] = api_key
    if api_key_env:
        ep_block["api_key_env"] = api_key_env
    doc["endpoints"].append(ep_block)

    path.write_text(_dump_yaml(doc), encoding="utf-8")
    if make_active:
        (home / "active_model.txt").write_text(f"remote:{alias}", encoding="utf-8")
    return path, alias


def _os_key(os_name: Optional[str] = None) -> str:
    name = (os_name or platform.system() or "").lower()
    if "windows" in name:
        return "windows"
    if "darwin" in name or "mac" in name:
        return "macos"
    return "linux"


def ollama_host_command(os_name: Optional[str] = None) -> str:
    """The exact command (for the user's OS) that makes Ollama listen on all
    interfaces, so another machine can reach it. Restart Ollama afterwards."""
    k = _os_key(os_name)
    if k == "windows":
        return 'setx OLLAMA_HOST "0.0.0.0"'
    if k == "macos":
        return "launchctl setenv OLLAMA_HOST 0.0.0.0"
    return "export OLLAMA_HOST=0.0.0.0"


def remote_prereq_lines(os_name: Optional[str] = None) -> List[str]:
    """Short 'your server must be running first' checklist, shown up front so a
    user without a server learns the prerequisites before hitting a wall."""
    return [
        "  First, a model server must be running on the machine with the GPU",
        "  (that can be this same PC for a local test). On THAT machine:",
        "    1. Install Ollama       ->  https://ollama.com/download",
        "    2. Pull a model         ->  ollama pull qwen2.5-coder:7b",
        "    3. Allow remote access  ->  " + ollama_host_command(os_name) + "   (then restart Ollama)",
        "       (step 3 is only needed when LeanAI runs on a different machine)",
        "  Full walkthrough: REMOTE_MODELS.md",
    ]


class _WizardAbort(Exception):
    """Raised internally when the user cancels the connect wizard (Ctrl-C/EOF)."""


def connect_interactive(home: Any = None, in_fn=input, out_fn=print,
                        os_name: Optional[str] = None) -> Optional[str]:
    """Guided, step-by-step 'connect a remote model' wizard. Auto-discovers the
    server's models so the user only types an address and picks from a list.

    Friendly by design: explains prerequisites up front, labels each step, and
    if the user cancels at any point (Ctrl-C / EOF / declining) it says clearly
    that nothing was saved. Driveable in tests via ``in_fn``/``out_fn``.
    Returns the active alias on success, or None if the user cancelled.
    """
    p = out_fn
    home = Path(home) if home else _home()

    def ask(prompt: str = "") -> str:
        try:
            return in_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            raise _WizardAbort()

    def nothing_saved(msg: str = "Re-run /model connect when your server is ready.") -> None:
        p("")
        p("  Nothing saved. " + msg)

    try:
        # ---- Intro + prerequisites ----------------------------------
        p("")
        p("  Connect a remote model (Ollama / OpenAI-compatible)")
        p("  " + "-" * 48)
        p("  Point LeanAI at a model running on another machine — no GPU needed here.")
        p("")
        for line in remote_prereq_lines(os_name):
            p(line)
        p("")
        p("  Examples:  192.168.1.50     (your Ollama box on the LAN)")
        p("             http://localhost:11434/v1")
        p("  (press Ctrl-C any time to cancel — nothing is saved until the last step)")

        # ---- Step 1: where is the server? ---------------------------
        p("")
        p("  Step 1 of 3 — where is the server?")
        raw = ask("  Server address [localhost:11434]: ").strip()
        base_url, is_openai = normalize_base_url(raw)
        external = is_openai or not is_local_url(base_url)

        api_key: Optional[str] = None
        api_key_env: Optional[str] = None
        if external:
            p("")
            p("  NOTE: that address is outside your machine/LAN, so code you send will")
            p("  leave your computer. For private code, prefer your own Ollama box.")
            envname = ask("  Env var holding the API key (blank to type the key): ").strip()
            if envname:
                api_key_env = envname
                api_key = os.environ.get(envname)  # used only to probe
            else:
                typed = ask("  API key (blank if none): ").strip()
                api_key = typed or None

        # ---- Step 2: check the server and find models ---------------
        p("")
        p("  Step 2 of 3 — checking the server and finding its models")
        p(f"  Checking {base_url} ...")
        res = probe_endpoint(base_url, api_key=api_key)
        p(f"  -> {res.status}")

        model_id: Optional[str] = None
        if res.reachable and res.models:
            p("")
            p("  Models available on that server:")
            for i, mid in enumerate(res.models, 1):
                p(f"    {i}. {mid}")
            choice = ask(f"  Pick one [1-{len(res.models)}, default 1]: ").strip()
            try:
                idx = (int(choice) - 1) if choice else 0
                model_id = res.models[idx] if 0 <= idx < len(res.models) else res.models[0]
            except (ValueError, IndexError):
                model_id = res.models[0]
        else:
            p("")
            if res.auth_failed:
                p("  The server answered but rejected the API key — double-check it.")
            elif not res.reachable:
                p("  Could not reach that server. Most common causes:")
                p("    - The model server (e.g. Ollama) isn't running yet — start it first.")
                p("    - If it's on another machine: Ollama only listens on localhost until")
                p("      you allow remote access. On the SERVER machine, run:")
                p("        " + ollama_host_command(os_name) + "   (then restart Ollama)")
                p("    - Wrong IP/port, or a firewall is blocking the connection.")
            else:
                p("  The server is reachable but didn't return a model list.")
            p("  More help: REMOTE_MODELS.md")
            cont = ask("  Enter the model id/tag manually instead? (y/N): ").strip().lower()
            if cont not in ("y", "yes"):
                nothing_saved()
                return None
            model_id = ask("  Model id/tag (e.g. qwen2.5-coder:7b): ").strip()
            if not model_id:
                nothing_saved("No model id was given.")
                return None

        # ---- Step 3: name it and save -------------------------------
        p("")
        p("  Step 3 of 3 — name it and save")
        fmt = guess_prompt_format(model_id)
        default_alias = _slug(model_id)
        alias = (ask(f"  Short name to type in LeanAI [{default_alias}]: ").strip()
                 or default_alias).lower()
        mode = "chat" if is_openai else "completions"
        literal_key = api_key if (api_key and not api_key_env) else None

        path, alias = add_remote_model(
            home, base_url, model_id, alias, mode, fmt,
            api_key=literal_key, api_key_env=api_key_env, make_active=True,
        )
        p("")
        p(f"  Done! Saved to {path}")
        p(f"  Active model set: remote:{alias}")
        p(f"  Use it now with:  /model {alias}    (check it any time with /model test)")
        return alias

    except _WizardAbort:
        nothing_saved()
        return None
