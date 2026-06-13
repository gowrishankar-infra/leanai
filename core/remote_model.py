"""
LeanAI — Remote model client (OpenAI-compatible / Ollama).

This lets LeanAI use a model served over HTTP instead of a local GGUF, so a
machine with no GPU (e.g. a dev laptop) can point at an Ollama / llama.cpp /
vLLM / LM Studio / LocalAI / OpenAI endpoint running elsewhere.

KEY DESIGN: ``RemoteModel`` is a *drop-in* for ``llama_cpp.Llama``. The engine
calls the model like ``model(prompt, max_tokens=..., stop=[...], stream=...)``
and reads ``result["choices"][0]["text"]``. The OpenAI **/v1/completions**
endpoint returns that exact shape, so in "completions" mode this object is a
1:1 substitute and the engine needs no special-casing of generation.

Two modes:
  * ``completions`` (default) — POSTs the raw, already-formatted prompt to
    ``/v1/completions``. The server treats it as a raw completion (no extra
    chat template), so LeanAI's own chatml/gemma/llama3 formatting is honoured.
    Works with Ollama, llama.cpp server, vLLM, LM Studio, LocalAI, OpenRouter.
  * ``chat`` — for endpoints that only expose ``/v1/chat/completions``
    (OpenAI proper). LeanAI's formatted prompt is parsed back into role
    messages and the server applies its own template. Best-effort; prefer
    ``completions`` for local servers.

Stdlib only (urllib) — no new hard dependency. Network/HTTP errors are turned
into a visible error string in the llama-shaped response rather than a crash,
EXCEPT ``ping()`` which returns a bool for honest load-status reporting.
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


class RemoteModelError(RuntimeError):
    pass


# ── Prompt → messages (chat mode only) ────────────────────────────

def prompt_to_messages(prompt: str, prompt_format: str = "chatml") -> List[Dict[str, str]]:
    """Parse a LeanAI-formatted prompt string back into OpenAI chat messages.

    LeanAI emits these templates deterministically (see engine_v3._build_prompt),
    so marker-based parsing is reliable. Falls back to a single user message if
    nothing parses.
    """
    fmt = (prompt_format or "chatml").lower()
    messages: List[Dict[str, str]] = []

    if fmt == "gemma":
        # <start_of_turn>user\n...<end_of_turn>  (model -> assistant)
        for role, content in re.findall(
            r"<start_of_turn>(user|model)\n(.*?)<end_of_turn>", prompt, re.DOTALL
        ):
            messages.append({
                "role": "assistant" if role == "model" else "user",
                "content": content.strip(),
            })
    elif fmt in ("phi3", "llama3"):
        # <|system|>\n...<|end|>  /  <|user|>  /  <|assistant|>
        for role, content in re.findall(
            r"<\|(system|user|assistant)\|>\n(.*?)<\|end\|>", prompt, re.DOTALL
        ):
            messages.append({"role": role, "content": content.strip()})
    else:  # chatml (default)
        for role, content in re.findall(
            r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", prompt, re.DOTALL
        ):
            text = content.strip()
            # Qwen3.5 primes an empty <think></think> block — drop empties.
            text = re.sub(r"<think>\s*</think>", "", text).strip()
            if text:
                messages.append({"role": role, "content": text})

    # Drop a trailing empty assistant turn (the generation cue).
    if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages.pop()

    if not messages:
        # Couldn't parse — strip any leftover template tokens and send as one
        # user message. Works, just not ideal; documented in REMOTE_MODELS.md.
        cleaned = re.sub(r"<\|[^>]*\|>|<start_of_turn>|<end_of_turn>", " ", prompt)
        messages = [{"role": "user", "content": cleaned.strip()}]

    return messages


# ── The drop-in client ────────────────────────────────────────────

@dataclass
class RemoteModel:
    base_url: str                       # e.g. http://host:11434/v1
    model: str                          # e.g. qwen2.5-coder:7b  /  gpt-4o-mini
    api_key: Optional[str] = None
    mode: str = "completions"           # "completions" | "chat"
    prompt_format: str = "chatml"       # used to parse prompt in chat mode
    timeout: int = 180
    extra_headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.base_url = self.base_url.rstrip("/")
        if self.mode not in ("completions", "chat"):
            self.mode = "completions"

    # -- HTTP plumbing -------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = "Bearer " + self.api_key
        h.update(self.extra_headers or {})
        return h

    def _post(self, path: str, payload: Dict[str, Any], stream: bool):
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        # Caller is responsible for closing when streaming.
        return urllib.request.urlopen(req, timeout=self.timeout)

    @staticmethod
    def _wrap_text(text: str) -> Dict[str, Any]:
        """Shape a plain string like llama_cpp's non-stream return value."""
        return {"choices": [{"text": text, "finish_reason": "stop"}]}

    # -- Reachability (honest load status) -----------------------------

    def ping(self) -> bool:
        """True if the endpoint is reachable AND auth is accepted.

        Tries GET {base_url}/models. 200 -> ok. 401/403 -> bad key (False).
        Any other HTTP status still means the server answered -> ok. Connection
        errors / timeouts -> False. Used by the engine for True/False load.
        """
        url = self.base_url + "/models"
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 15)) as resp:
                return 200 <= resp.status < 500 and resp.status not in (401, 403)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return False
            return True  # server answered (e.g. 404) -> reachable
        except Exception:
            return False

    # -- Generation (the llama_cpp.Llama call interface) ---------------

    def __call__(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,                 # accepted, not sent (not OpenAI-standard)
        repeat_penalty: float = 1.05,    # accepted, not sent
        stop: Optional[List[str]] = None,
        echo: bool = False,              # accepted, ignored
        stream: bool = False,
        **kwargs: Any,
    ):
        # OpenAI caps `stop` at 4 sequences; keep the most specific ones.
        stop_list = list(stop)[:4] if stop else None

        if self.mode == "chat":
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": prompt_to_messages(prompt, self.prompt_format),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "stream": bool(stream),
            }
            if stop_list:
                payload["stop"] = stop_list
            return self._chat(payload, stream)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": bool(stream),
        }
        if stop_list:
            payload["stop"] = stop_list
        return self._completions(payload, stream)

    # -- /v1/completions -----------------------------------------------

    def _completions(self, payload: Dict[str, Any], stream: bool):
        if not stream:
            try:
                with self._post("/completions", payload, stream=False) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                text = body.get("choices", [{}])[0].get("text", "")
                return self._wrap_text(text)
            except Exception as e:
                return self._wrap_text(self._err(e))
        return self._stream_completions(payload)

    def _stream_completions(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        try:
            resp = self._post("/completions", payload, stream=True)
        except Exception as e:
            yield self._wrap_text(self._err(e))
            return
        try:
            for piece in self._iter_sse(resp):
                text = piece.get("choices", [{}])[0].get("text", "")
                if text:
                    yield {"choices": [{"text": text}]}
        finally:
            resp.close()

    # -- /v1/chat/completions ------------------------------------------

    def _chat(self, payload: Dict[str, Any], stream: bool):
        if not stream:
            try:
                with self._post("/chat/completions", payload, stream=False) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                msg = body.get("choices", [{}])[0].get("message", {})
                return self._wrap_text(msg.get("content", "") or "")
            except Exception as e:
                return self._wrap_text(self._err(e))
        return self._stream_chat(payload)

    def _stream_chat(self, payload: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        try:
            resp = self._post("/chat/completions", payload, stream=True)
        except Exception as e:
            yield self._wrap_text(self._err(e))
            return
        try:
            for piece in self._iter_sse(resp):
                delta = piece.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield {"choices": [{"text": text}]}
        finally:
            resp.close()

    # -- shared helpers ------------------------------------------------

    @staticmethod
    def _iter_sse(resp) -> Iterator[Dict[str, Any]]:
        """Yield parsed JSON objects from an OpenAI-style SSE stream."""
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue

    def _err(self, e: Exception) -> str:
        if isinstance(e, urllib.error.HTTPError):
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            return (f"[Remote model error] HTTP {e.code} from {self.base_url} "
                    f"(model={self.model}). {detail}").strip()
        if isinstance(e, urllib.error.URLError):
            return (f"[Remote model error] Could not reach {self.base_url} "
                    f"(model={self.model}): {getattr(e, 'reason', e)}. "
                    f"Is the server running and reachable from this machine?")
        return f"[Remote model error] {type(e).__name__}: {e}"
