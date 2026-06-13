"""
Tests for remote (OpenAI-compatible / Ollama) model support.

These run WITHOUT any real model or GPU: a tiny in-process HTTP server emulates
the OpenAI /v1/completions, /v1/chat/completions and /v1/models endpoints
(both streaming and non-streaming). They verify the drop-in client shape, the
endpoints.yaml loader, and ModelManager integration.

Run:  python -m pytest tests/test_remote_models.py -q
 or:  python tests/test_remote_models.py
"""

import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

# Allow `python tests/test_remote_models.py` from the repo root (not just pytest,
# which injects rootdir automatically): put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.remote_model import RemoteModel, prompt_to_messages
from core import endpoints as ep


# ── Mock OpenAI-compatible server ─────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in chunks:
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send_json({"data": [{"id": "mock-model"}]})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        stream = bool(payload.get("stream"))

        if self.path.endswith("/completions") and "chat" not in self.path:
            # echo back so tests can assert the prompt arrived raw
            text = f"COMPLETION[{payload.get('prompt','')[:20]}]"
            if stream:
                self._send_sse([{"choices": [{"text": "COMPLETION"}]},
                                {"choices": [{"text": "[stream]"}]}])
            else:
                self._send_json({"choices": [{"text": text}]})

        elif self.path.endswith("/chat/completions"):
            msgs = payload.get("messages", [])
            last = msgs[-1]["content"] if msgs else ""
            text = f"CHAT[{len(msgs)}msgs:{last[:15]}]"
            if stream:
                self._send_sse([{"choices": [{"delta": {"content": "CHAT"}}]},
                                {"choices": [{"delta": {"content": "[stream]"}}]}])
            else:
                self._send_json({"choices": [{"message": {"role": "assistant",
                                                           "content": text}}]})
        else:
            self._send_json({"error": "not found"}, 404)


@pytest.fixture(scope="module")
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/v1"
    httpd.shutdown()


# ── RemoteModel: completions mode (drop-in shape) ─────────────────

def test_completions_nonstream(server):
    m = RemoteModel(base_url=server, model="mock-model", mode="completions")
    out = m("hello-prompt", max_tokens=16, stream=False)
    assert isinstance(out, dict)
    text = out["choices"][0]["text"]
    assert "COMPLETION" in text and "hello-prompt" in text  # prompt sent raw


def test_completions_stream(server):
    m = RemoteModel(base_url=server, model="mock-model", mode="completions")
    chunks = list(m("p", stream=True))
    joined = "".join(c["choices"][0]["text"] for c in chunks)
    assert joined == "COMPLETION[stream]"


def test_chat_nonstream_maps_message_to_text(server):
    m = RemoteModel(base_url=server, model="mock-model", mode="chat")
    prompt = "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n"
    out = m(prompt, stream=False)
    text = out["choices"][0]["text"]          # chat content remapped to .text
    assert text.startswith("CHAT[2msgs")       # system + user parsed


def test_chat_stream_maps_delta_to_text(server):
    m = RemoteModel(base_url=server, model="mock-model", mode="chat")
    chunks = list(m("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", stream=True))
    joined = "".join(c["choices"][0]["text"] for c in chunks)
    assert joined == "CHAT[stream]"


def test_ping_true_when_reachable(server):
    assert RemoteModel(base_url=server, model="m").ping() is True


def test_ping_false_when_unreachable():
    # An unused port → connection refused → False (honest load status).
    assert RemoteModel(base_url="http://127.0.0.1:9", model="m", timeout=2).ping() is False


def test_stop_is_capped_to_four(server):
    # >4 stop tokens must not blow up; client trims to 4 before sending.
    m = RemoteModel(base_url=server, model="m")
    out = m("p", stop=["a", "b", "c", "d", "e", "f"], stream=False)
    assert out["choices"][0]["text"]  # request succeeded


def test_http_error_returns_visible_string():
    # No server → URLError → friendly error text, not an exception.
    m = RemoteModel(base_url="http://127.0.0.1:9", model="m", timeout=2)
    out = m("p", stream=False)
    assert "[Remote model error]" in out["choices"][0]["text"]


# ── prompt_to_messages parsing ────────────────────────────────────

def test_parse_chatml():
    p = ("<|im_start|>system\nYou are X<|im_end|>\n"
         "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n")
    msgs = prompt_to_messages(p, "chatml")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "hello"


def test_parse_gemma_maps_model_to_assistant():
    p = "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nyo<end_of_turn>\n"
    msgs = prompt_to_messages(p, "gemma")
    assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"


def test_parse_phi3():
    p = "<|system|>\nS<|end|>\n<|user|>\nU<|end|>\n<|assistant|>\n"
    msgs = prompt_to_messages(p, "phi3")
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_parse_fallback_to_single_user():
    msgs = prompt_to_messages("just some text no markers", "chatml")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"


# ── endpoints.yaml loader ─────────────────────────────────────────

def _write(tmp, text):
    p = Path(tmp) / "endpoints.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_endpoints_basic():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, """
endpoints:
  - name: home
    base_url: http://h:11434/v1
    mode: completions
    api_key: secret
    models:
      - id: qwen2.5-coder:7b
        alias: remote-coder
        prompt_format: chatml
        quality: 72
""")
        specs = ep.load_endpoints(path)
        assert "remote-coder" in specs
        s = specs["remote-coder"]
        assert s.model_id == "qwen2.5-coder:7b"
        assert s.base_url == "http://h:11434/v1"
        assert s.mode == "completions" and s.api_key == "secret"
        client = ep.make_client(s)
        assert client.model == "qwen2.5-coder:7b" and client.api_key == "secret"


def test_load_endpoints_api_key_env_and_interp():
    os.environ["MY_TEST_KEY"] = "envkey123"
    os.environ["MY_HOST"] = "myhost:1234"
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, """
endpoints:
  - name: openai
    base_url: https://${MY_HOST}/v1
    mode: chat
    api_key_env: MY_TEST_KEY
    models:
      - id: gpt-4o-mini
        alias: gpt4o-mini
""")
        specs = ep.load_endpoints(path)
        s = specs["gpt4o-mini"]
        assert s.api_key == "envkey123"
        assert s.base_url == "https://myhost:1234/v1"
        assert s.mode == "chat"


def test_load_endpoints_missing_file_returns_empty():
    assert ep.load_endpoints("/nonexistent/endpoints.yaml") == {}


def test_load_endpoints_malformed_entry_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, """
endpoints:
  - name: broken
    # no base_url
    models:
      - id: x
        alias: x
  - name: good
    base_url: http://h/v1
    models:
      - id: m
        alias: good-one
""")
        specs = ep.load_endpoints(path)
        assert "good-one" in specs and "x" not in specs


# ── ModelManager integration ──────────────────────────────────────

def test_manager_lists_and_routes_remote(monkeypatch):
    with tempfile.TemporaryDirectory() as home:
        # Point LEANAI_HOME at a temp dir with an endpoints.yaml and no models.
        Path(home, "endpoints.yaml").write_text("""
endpoints:
  - name: home
    base_url: http://h:11434/v1
    mode: completions
    models:
      - id: qwen2.5-coder:7b
        alias: remote-coder
        quality: 72
""", encoding="utf-8")
        monkeypatch.setenv("LEANAI_HOME", home)

        from core.model_manager import ModelManager
        mgr = ModelManager()

        # Listed + switchable + reports available (no download needed)
        assert "remote-coder" in mgr.models
        assert mgr.models["remote-coder"].is_remote is True
        assert mgr.get_model_path("remote-coder") == "remote:remote-coder"
        assert "remote-coder" in mgr.get_downloaded_models()

        # With no local GGUFs present, auto-routing falls through to remote.
        mgr.set_mode("auto")
        assert mgr.select_model("design a distributed system") == "remote-coder"
        assert mgr.select_model("what is my name") == "remote-coder"

        # /model list shows the Remote section
        assert "Remote endpoints" in mgr.list_models()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))


# ── Auto-discovery / smooth setup ─────────────────────────────────

def test_normalize_base_url_variants():
    assert ep.normalize_base_url("192.168.1.50") == ("http://192.168.1.50:11434/v1", False)
    assert ep.normalize_base_url("192.168.1.50:11434") == ("http://192.168.1.50:11434/v1", False)
    assert ep.normalize_base_url("http://host:11434") == ("http://host:11434/v1", False)
    assert ep.normalize_base_url("http://host:11434/v1") == ("http://host:11434/v1", False)
    url, is_openai = ep.normalize_base_url("https://api.openai.com")
    assert url == "https://api.openai.com/v1" and is_openai is True
    assert ep.normalize_base_url("")[0] == ep.OLLAMA_DEFAULT


def test_is_local_url():
    assert ep.is_local_url("http://localhost:11434/v1") is True
    assert ep.is_local_url("http://127.0.0.1:11434/v1") is True
    assert ep.is_local_url("http://192.168.1.50:11434/v1") is True
    assert ep.is_local_url("http://10.0.0.5:11434/v1") is True
    assert ep.is_local_url("https://api.openai.com/v1") is False


def test_guess_prompt_format():
    assert ep.guess_prompt_format("qwen2.5-coder:7b") == "chatml"
    assert ep.guess_prompt_format("gemma2:9b") == "gemma"
    assert ep.guess_prompt_format("llama3.1:8b") == "llama3"
    assert ep.guess_prompt_format("phi3:mini") == "phi3"


def test_slug():
    assert ep._slug("qwen2.5-coder:7b") == "qwen2-5-coder-7b"
    assert ep._slug("") == "remote"


def test_probe_endpoint_lists_models(server):
    res = ep.probe_endpoint(server)
    assert res.reachable is True
    assert res.models == ["mock-model"]


def test_probe_endpoint_unreachable():
    res = ep.probe_endpoint("http://127.0.0.1:9", timeout=2)
    assert res.reachable is False
    assert res.models == []


def test_connect_interactive_end_to_end(server, monkeypatch):
    """The whole smooth flow with no real server: point at the mock, it
    discovers the one model, user accepts defaults, endpoints.yaml is written
    and load_endpoints() can resolve the new alias."""
    d = tempfile.mkdtemp()
    monkeypatch.setenv("LEANAI_HOME", d)

    # Inputs in order: server address, pick (default), alias (default).
    answers = iter([server, "", ""])
    out = []
    alias = ep.connect_interactive(
        home=d,
        in_fn=lambda prompt="": next(answers),
        out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
    )
    assert alias == "mock-model"

    # The written file round-trips through the real loader.
    specs = ep.load_endpoints()
    assert alias in specs
    s = specs[alias]
    assert s.base_url == server
    assert s.mode == "completions"
    assert s.model_id == "mock-model"
    assert s.prompt_format == "chatml"
    # active model was set
    assert (Path(d) / "active_model.txt").read_text().strip() == f"remote:{alias}"


def test_connect_interactive_abort_when_unreachable(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setenv("LEANAI_HOME", d)
    # Unreachable server, then decline manual entry -> abort, nothing written.
    answers = iter(["http://127.0.0.1:9", "n"])
    alias = ep.connect_interactive(
        home=d,
        in_fn=lambda prompt="": next(answers),
        out_fn=lambda *a: None,
    )
    assert alias is None
    assert not (Path(d) / "endpoints.yaml").exists()


def test_add_remote_model_dedupes_alias():
    d = tempfile.mkdtemp()
    p1, a1 = ep.add_remote_model(d, "http://h:11434/v1", "m", "coder", "completions", "chatml")
    p2, a2 = ep.add_remote_model(d, "http://h:11434/v1", "m", "coder", "completions", "chatml")
    assert a1 == "coder" and a2 == "coder-2"


# ── Prerequisite guidance (teach-as-you-go) ───────────────────────

def test_ollama_host_command_per_os():
    assert 'setx OLLAMA_HOST "0.0.0.0"' == ep.ollama_host_command("Windows")
    assert ep.ollama_host_command("Darwin") == "launchctl setenv OLLAMA_HOST 0.0.0.0"
    assert ep.ollama_host_command("Linux") == "export OLLAMA_HOST=0.0.0.0"


def test_remote_prereq_lines_cover_the_three_steps():
    text = "\n".join(ep.remote_prereq_lines("Linux"))
    assert "ollama.com/download" in text          # install
    assert "ollama pull" in text                   # pull a model
    assert "OLLAMA_HOST" in text                    # allow remote access
    assert "REMOTE_MODELS.md" in text               # where to read more


def test_connect_shows_prereqs_up_front(server):
    """The checklist is printed before the user is asked anything."""
    d = tempfile.mkdtemp()
    out = []
    answers = iter([server, "", ""])
    ep.connect_interactive(
        home=d,
        in_fn=lambda prompt="": next(answers),
        out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
        os_name="Windows",
    )
    blob = "\n".join(out)
    assert "Install Ollama" in blob and "ollama pull" in blob


def test_connect_unreachable_shows_os_specific_hint():
    d = tempfile.mkdtemp()
    out = []
    # Unreachable, then decline manual entry. os_name drives the hint shown.
    answers = iter(["http://127.0.0.1:9", "n"])
    alias = ep.connect_interactive(
        home=d,
        in_fn=lambda prompt="": next(answers),
        out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
        os_name="Windows",
    )
    blob = "\n".join(out)
    assert alias is None
    assert 'setx OLLAMA_HOST "0.0.0.0"' in blob      # Windows-specific fix shown
    assert "export OLLAMA_HOST" not in blob           # not the Linux one


# ── Friendly signposting (fixes A & B) ────────────────────────────

def test_list_models_shows_none_yet_when_no_remote(monkeypatch):
    """Fix A: /model explains the empty remote section instead of hiding it."""
    import tempfile as _tf
    from core import model_manager as mm
    d = _tf.mkdtemp()
    monkeypatch.setenv("LEANAI_HOME", d)            # no endpoints.yaml here
    monkeypatch.setattr(mm, "get_models_dir", lambda: Path(d))
    mgr = mm.ModelManager()
    out = mgr.list_models()
    assert "Remote endpoints: none yet." in out
    assert "/model connect" in out


def test_wizard_cancel_via_ctrl_c_says_nothing_saved():
    """Fix B: cancelling (KeyboardInterrupt) is not silent."""
    d = tempfile.mkdtemp()
    out = []

    def boom(prompt=""):
        raise KeyboardInterrupt()

    alias = ep.connect_interactive(
        home=d, in_fn=boom, out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
    )
    assert alias is None
    assert any("Nothing saved" in line for line in out)
    assert not (Path(d) / "endpoints.yaml").exists()


def test_wizard_cancel_via_eof_says_nothing_saved():
    d = tempfile.mkdtemp()
    out = []

    def eof(prompt=""):
        raise EOFError()

    alias = ep.connect_interactive(
        home=d, in_fn=eof, out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
    )
    assert alias is None
    assert any("Nothing saved" in line for line in out)


def test_wizard_shows_numbered_steps(server):
    d = tempfile.mkdtemp()
    out = []
    answers = iter([server, "", ""])
    ep.connect_interactive(
        home=d,
        in_fn=lambda prompt="": next(answers),
        out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
    )
    blob = "\n".join(out)
    assert "Step 1 of 3" in blob and "Step 2 of 3" in blob and "Step 3 of 3" in blob


def test_wizard_decline_manual_says_nothing_saved():
    """Unreachable + declining manual entry -> clear 'nothing saved'."""
    d = tempfile.mkdtemp()
    out = []
    answers = iter(["http://127.0.0.1:9", "n"])
    alias = ep.connect_interactive(
        home=d,
        in_fn=lambda prompt="": next(answers),
        out_fn=lambda *a: out.append(" ".join(str(x) for x in a)),
    )
    assert alias is None
    assert any("Nothing saved" in line for line in out)
