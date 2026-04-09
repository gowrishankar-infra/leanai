"""
Tests for LeanAI Phase 5b — FastAPI REST API
Tests endpoint contracts, request/response models, and web UI.
"""

import pytest
import json

# Import only the Pydantic models and HTML — these don't trigger engine loading
import sys
import importlib

# We need to import from api.server but it imports heavy engine modules.
# On the actual machine this works fine. In limited environments, we skip.
try:
    from api.server import (
        ChatRequest, ChatResponse,
        SwarmRequest, SwarmResponse,
        BuildRequest, BuildResponse,
        RunRequest, RunResponse,
        IndexRequest, AskRequest,
        RememberRequest,
        StatusResponse,
        WEB_UI_HTML,
    )
    API_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    API_AVAILABLE = False

pytestmark = pytest.mark.skipif(not API_AVAILABLE, reason="API module requires full LeanAI engine")


# ── Request model tests ───────────────────────────────────────────

class TestChatRequest:
    def test_basic(self):
        r = ChatRequest(message="hello")
        assert r.message == "hello"
        assert r.max_tokens == 512
        assert r.temperature == 0.1

    def test_custom_params(self):
        r = ChatRequest(message="test", max_tokens=256, temperature=0.5)
        assert r.max_tokens == 256
        assert r.temperature == 0.5

    def test_empty_message_allowed(self):
        r = ChatRequest(message="")
        assert r.message == ""


class TestSwarmRequest:
    def test_basic(self):
        r = SwarmRequest(message="What is AI?")
        assert r.message == "What is AI?"
        assert r.num_passes == 3

    def test_custom_passes(self):
        r = SwarmRequest(message="test", num_passes=5)
        assert r.num_passes == 5


class TestBuildRequest:
    def test_basic(self):
        r = BuildRequest(task="Build a calculator")
        assert r.task == "Build a calculator"
        assert r.workspace is None

    def test_with_workspace(self):
        r = BuildRequest(task="Build API", workspace="/tmp/project")
        assert r.workspace == "/tmp/project"


class TestRunRequest:
    def test_basic(self):
        r = RunRequest(code="print(42)")
        assert r.code == "print(42)"


class TestIndexRequest:
    def test_basic(self):
        r = IndexRequest(path="/home/user/project")
        assert r.path == "/home/user/project"
        assert r.force is False

    def test_force(self):
        r = IndexRequest(path="/tmp", force=True)
        assert r.force is True


class TestAskRequest:
    def test_basic(self):
        r = AskRequest(query="find all functions")
        assert r.query == "find all functions"
        assert r.top_k == 5


class TestRememberRequest:
    def test_basic(self):
        r = RememberRequest(fact="I work at Google")
        assert r.fact == "I work at Google"


# ── Response model tests ──────────────────────────────────────────

class TestChatResponse:
    def test_creation(self):
        r = ChatResponse(
            text="Hello!", confidence=95.0, confidence_label="High",
            tier="memory", latency_ms=1.0,
        )
        assert r.text == "Hello!"
        assert r.confidence == 95.0
        assert r.code_executed is False

    def test_with_code(self):
        r = ChatResponse(
            text="def add(a,b): return a+b", confidence=72.0,
            confidence_label="Moderate", tier="medium", latency_ms=5000,
            code_executed=True, code_passed=True, code_output="3",
        )
        assert r.code_executed is True
        assert r.code_passed is True


class TestSwarmResponse:
    def test_creation(self):
        r = SwarmResponse(
            text="Paris", consensus_score=1.0, confidence=99.0,
            unanimous=True, num_passes=3, latency_ms=5000.0,
        )
        assert r.unanimous is True
        assert r.consensus_score == 1.0

    def test_with_candidates(self):
        r = SwarmResponse(
            text="Paris", consensus_score=0.8, confidence=85.0,
            unanimous=False, num_passes=3, latency_ms=6000.0,
            candidates=[
                {"text": "Paris", "temperature": 0.1, "agreement": 0.9},
                {"text": "Paris", "temperature": 0.3, "agreement": 0.8},
                {"text": "Lyon", "temperature": 0.5, "agreement": 0.5},
            ],
        )
        assert len(r.candidates) == 3


class TestBuildResponse:
    def test_success(self):
        r = BuildResponse(
            success=True, steps_passed=4, steps_total=4,
            files_created=["main.py", "test_main.py"],
            errors=[], workspace="/tmp/project", time_seconds=120.0,
        )
        assert r.success is True
        assert len(r.files_created) == 2

    def test_failure(self):
        r = BuildResponse(
            success=False, steps_passed=2, steps_total=4,
            files_created=["main.py"], errors=["Step 3 failed"],
            workspace="/tmp/project", time_seconds=60.0,
        )
        assert r.success is False
        assert len(r.errors) == 1


class TestRunResponse:
    def test_success(self):
        r = RunResponse(success=True, output="42", execution_time_ms=5)
        assert r.success is True
        assert r.output == "42"

    def test_failure(self):
        r = RunResponse(success=False, output="", error="SyntaxError", execution_time_ms=1)
        assert r.error == "SyntaxError"


class TestStatusResponse:
    def test_creation(self):
        r = StatusResponse(
            version="5.0", model="qwen25-coder-7b", prompt_format="chatml",
            threads=16, memory_episodes=20, memory_backend="chromadb+vectors",
            world_entities=13, profile_fields=3, training_pairs=86,
            model_loaded=False,
        )
        assert r.version == "5.0"
        assert r.threads == 16
        assert r.model_loaded is False


# ── Web UI tests ──────────────────────────────────────────────────

class TestWebUI:
    def test_html_is_valid(self):
        assert "<!DOCTYPE html>" in WEB_UI_HTML
        assert "<html" in WEB_UI_HTML
        assert "</html>" in WEB_UI_HTML

    def test_html_has_title(self):
        assert "<title>LeanAI</title>" in WEB_UI_HTML

    def test_html_has_chat_div(self):
        assert 'id="chat"' in WEB_UI_HTML

    def test_html_has_input(self):
        assert 'id="input"' in WEB_UI_HTML

    def test_html_has_send_button(self):
        assert "send()" in WEB_UI_HTML

    def test_html_has_mode_buttons(self):
        assert "btn-chat" in WEB_UI_HTML
        assert "btn-swarm" in WEB_UI_HTML
        assert "btn-run" in WEB_UI_HTML

    def test_html_fetches_chat(self):
        assert "fetch('/chat'" in WEB_UI_HTML

    def test_html_fetches_swarm(self):
        assert "fetch('/swarm'" in WEB_UI_HTML

    def test_html_fetches_run(self):
        assert "fetch('/run'" in WEB_UI_HTML

    def test_html_fetches_status(self):
        assert "fetch('/status')" in WEB_UI_HTML

    def test_html_is_responsive(self):
        assert "viewport" in WEB_UI_HTML

    def test_html_has_dark_theme(self):
        assert "#0f1117" in WEB_UI_HTML  # bg color


# ── Serialization tests ───────────────────────────────────────────

class TestSerialization:
    def test_chat_response_json(self):
        r = ChatResponse(
            text="Hi", confidence=90.0, confidence_label="High",
            tier="tiny", latency_ms=1.0,
        )
        d = r.model_dump()
        assert d["text"] == "Hi"
        assert d["confidence"] == 90.0

    def test_swarm_response_json(self):
        r = SwarmResponse(
            text="Answer", consensus_score=0.9, confidence=95.0,
            unanimous=True, num_passes=3, latency_ms=5000.0,
        )
        j = r.model_dump_json()
        parsed = json.loads(j)
        assert parsed["unanimous"] is True

    def test_status_response_json(self):
        r = StatusResponse(
            version="5.0", model="test", prompt_format="chatml",
            threads=8, memory_episodes=10, memory_backend="chroma",
            world_entities=5, profile_fields=2, training_pairs=50,
            model_loaded=True,
        )
        d = r.model_dump()
        assert d["model_loaded"] is True
        assert d["threads"] == 8
