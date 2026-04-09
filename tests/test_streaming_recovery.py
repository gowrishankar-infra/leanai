"""
Tests for LeanAI Streaming, Smart Context, and Auto-Recovery.
"""

import os
import time
import shutil
import tempfile
import pytest

from core.streaming import StreamingGenerator, StreamConfig, print_streaming_footer
from core.smart_context import SmartContext
from core.auto_recovery import AutoRecovery, RecoveryConfig, RecoveryResult


# ══════════════════════════════════════════════════════════════════
# Streaming Tests
# ══════════════════════════════════════════════════════════════════

class TestStreamConfig:
    def test_defaults(self):
        c = StreamConfig()
        assert c.enabled is True
        assert c.flush_every == 1

    def test_custom(self):
        c = StreamConfig(enabled=False)
        assert c.enabled is False


class TestStreamingGenerator:
    def test_build_chatml_prompt(self):
        sg = StreamingGenerator(prompt_format="chatml")
        prompt, stop = sg._build_prompt("system", "user")
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_end|>" in stop

    def test_build_phi3_prompt(self):
        sg = StreamingGenerator(prompt_format="phi3")
        prompt, stop = sg._build_prompt("system", "user")
        assert "<|system|>" in prompt
        assert "<|end|>" in stop

    def test_no_model_returns_empty(self):
        sg = StreamingGenerator(model=None)
        result = sg.generate_streaming("sys", "user")
        assert result == ""

    def test_non_streaming_no_model(self):
        sg = StreamingGenerator(model=None)
        result = sg.generate_non_streaming("sys", "user")
        assert result == ""

    def test_generate_auto_selects(self):
        sg = StreamingGenerator(model=None, config=StreamConfig(enabled=True))
        result = sg.generate("sys", "user", stream=True)
        assert result == ""  # no model, returns empty safely


class TestStreamingFooter:
    def test_footer_prints(self, capsys):
        print_streaming_footer(1000, 85, "medium")
        captured = capsys.readouterr()
        assert "85%" in captured.out
        assert "medium" in captured.out

    def test_footer_cached(self, capsys):
        print_streaming_footer(0, 90, "cache", cached=True)
        captured = capsys.readouterr()
        assert "CACHED" in captured.out

    def test_footer_verified(self, capsys):
        print_streaming_footer(500, 96, "small", verified=True)
        captured = capsys.readouterr()
        assert "Verified" in captured.out


# ══════════════════════════════════════════════════════════════════
# Smart Context Tests
# ══════════════════════════════════════════════════════════════════

class MockBrain:
    def __init__(self):
        self._file_analyses = {"main.py": type("A", (), {"total_lines": 100})()}

        class MockGraph:
            _function_lookup = {"run": "main.py:run"}
            def stats(self):
                return {"files": 5, "functions": 20, "classes": 3}

        self.graph = MockGraph()

    def describe_file(self, path):
        if "main" in path:
            return "main.py: Main entry point with run() function"
        return "File not indexed"

    def find_function(self, name):
        if name == "run":
            return "Function: run\nFile: main.py\nArgs: none"
        return "not found"


class MockGitIntel:
    is_available = True

    def file_history(self, filepath, limit=3):
        return f"History for {filepath}: 3 commits"

    def recent_activity(self, days=3):
        return "Activity in the last 3 days: 5 commits"


class MockSessionStore:
    def search(self, query, limit=2):
        if "auth" in query.lower():
            ex = type("E", (), {"query": "how to add auth", "response": "Use JWT tokens"})()
            return [(ex, "session1")]
        return []


class MockHDC:
    count = 5

    def search(self, query, top_k=2):
        return [("Q: what is Python A: A programming language", 0.7, {})]


class TestSmartContext:
    @pytest.fixture
    def ctx(self):
        return SmartContext(
            brain=MockBrain(),
            git_intel=MockGitIntel(),
            session_store=MockSessionStore(),
            hdc=MockHDC(),
        )

    def test_build_with_file_mention(self, ctx):
        result = ctx.build("fix the bug in main.py")
        assert "main.py" in result

    def test_build_with_function_mention(self, ctx):
        result = ctx.build("explain the main.py file structure")
        assert "main" in result.lower()

    def test_build_with_git_context(self, ctx):
        result = ctx.build("what changed recently")
        assert "commits" in result.lower() or "Activity" in result

    def test_build_with_session_context(self, ctx):
        result = ctx.build("how to add auth")
        assert "auth" in result.lower() or "JWT" in result

    def test_build_with_hdc(self, ctx):
        result = ctx.build("what is Python")
        assert "Python" in result or "Similar" in result

    def test_build_empty_query(self, ctx):
        result = ctx.build("hello")
        assert isinstance(result, str)

    def test_build_system_prompt(self, ctx):
        enriched = ctx.build_system_prompt("You are helpful.", "fix main.py")
        assert "You are helpful" in enriched
        assert "main.py" in enriched

    def test_no_brain(self):
        ctx = SmartContext(brain=None)
        result = ctx.build("anything")
        assert result == "" or isinstance(result, str)

    def test_no_git(self):
        ctx = SmartContext(git_intel=None)
        result = ctx.build("what changed")
        assert isinstance(result, str)

    def test_all_none(self):
        ctx = SmartContext()
        result = ctx.build("test query")
        assert result == ""


# ══════════════════════════════════════════════════════════════════
# Auto-Recovery Tests
# ══════════════════════════════════════════════════════════════════

class TestAutoRecovery:
    def test_success_no_recovery(self):
        recovery = AutoRecovery()
        result = recovery.safe_generate(
            generate_fn=lambda max_tokens, **kw: "Hello world",
        )
        assert result.success is True
        assert result.text == "Hello world"
        assert result.attempts == 1
        assert result.model_used == "primary"

    def test_retry_on_exception(self):
        call_count = [0]
        def flaky(max_tokens, **kw):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("Temporary failure")
            return "Recovered!"

        recovery = AutoRecovery(config=RecoveryConfig(max_retries=3))
        result = recovery.safe_generate(generate_fn=flaky)
        assert result.success is True
        assert result.text == "Recovered!"
        assert result.attempts == 2

    def test_fallback_on_total_failure(self):
        def always_fail(max_tokens, **kw):
            raise RuntimeError("Always fails")

        def fallback_fn(max_tokens, **kw):
            return "Fallback response"

        recovery = AutoRecovery(config=RecoveryConfig(max_retries=1))
        result = recovery.safe_generate(
            generate_fn=always_fail,
            fallback_fn=fallback_fn,
        )
        assert result.success is True
        assert result.text == "Fallback response"
        assert result.model_used == "fallback"
        assert result.recovered is True

    def test_total_failure(self):
        def always_fail(max_tokens, **kw):
            raise RuntimeError("Crash")

        recovery = AutoRecovery(config=RecoveryConfig(max_retries=1))
        result = recovery.safe_generate(generate_fn=always_fail)
        assert result.success is False
        assert "Crash" in result.error

    def test_memory_error_reduces_tokens(self):
        tokens_received = []
        call_count = [0]
        def oom_then_ok(max_tokens, **kw):
            tokens_received.append(max_tokens)
            call_count[0] += 1
            if call_count[0] == 1:
                raise MemoryError("OOM")
            return "OK with fewer tokens"

        recovery = AutoRecovery(config=RecoveryConfig(
            max_retries=2, reduce_tokens_on_retry=True, token_reduction_factor=0.5
        ))
        result = recovery.safe_generate(generate_fn=oom_then_ok, max_tokens=1024)
        assert result.success is True
        assert tokens_received[1] < tokens_received[0]  # tokens were reduced

    def test_safe_call_success(self):
        recovery = AutoRecovery()
        result = recovery.safe_call(lambda x: x * 2, 5)
        assert result == 10

    def test_safe_call_failure(self):
        recovery = AutoRecovery()
        result = recovery.safe_call(lambda: 1/0, default="safe")
        assert result == "safe"

    def test_stats(self):
        recovery = AutoRecovery()
        recovery.safe_generate(generate_fn=lambda max_tokens, **kw: "ok")
        s = recovery.stats()
        assert "total_recoveries" in s

    def test_recent_events_empty(self):
        recovery = AutoRecovery()
        s = recovery.recent_events()
        assert "No recovery" in s

    def test_recovery_events_recorded(self):
        def fail(max_tokens, **kw):
            raise RuntimeError("Test error")

        recovery = AutoRecovery(config=RecoveryConfig(max_retries=1))
        recovery.safe_generate(generate_fn=fail)
        assert len(recovery._events) >= 1
        s = recovery.recent_events()
        assert "RuntimeError" in s


class TestRecoveryResult:
    def test_success_result(self):
        r = RecoveryResult(success=True, text="Hello")
        assert r.success
        assert r.text == "Hello"

    def test_failure_result(self):
        r = RecoveryResult(success=False, text="", error="OOM")
        assert not r.success
        assert r.error == "OOM"

    def test_recovered_result(self):
        r = RecoveryResult(success=True, text="ok", recovered=True, model_used="fallback")
        assert r.recovered
        assert r.model_used == "fallback"
