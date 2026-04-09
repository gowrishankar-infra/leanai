"""
Tests for LeanAI Phase 7d (Multi-File Editor) and 7e (Session Continuity)
"""

import os
import time
import shutil
import tempfile
import pytest

from brain.editor import MultiFileEditor, RefactorPlan, FileEdit, Reference
from brain.session_store import SessionStore, Session, Exchange


# ══════════════════════════════════════════════════════════════════
# Phase 7d — Multi-File Editor Tests
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def project_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "main.py"), "w") as f:
        f.write("from utils import greet, format_name\n\ndef run():\n    name = format_name('A', 'B')\n    print(greet(name))\n")
    with open(os.path.join(d, "utils.py"), "w") as f:
        f.write("def greet(name):\n    return f'Hello, {name}!'\n\ndef format_name(first, last):\n    return f'{first} {last}'\n")
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)
    with open(os.path.join(d, "tests", "test_utils.py"), "w") as f:
        f.write("from utils import greet, format_name\n\ndef test_greet():\n    assert greet('X') == 'Hello, X!'\n\ndef test_format_name():\n    assert format_name('A', 'B') == 'A B'\n")
    yield d
    shutil.rmtree(d)


@pytest.fixture
def editor(project_dir):
    return MultiFileEditor(project_dir)


class TestFindReferences:
    def test_finds_definition(self, editor):
        refs = editor.find_references("greet")
        defs = [r for r in refs if r.ref_type == "definition"]
        assert len(defs) >= 1

    def test_finds_imports(self, editor):
        refs = editor.find_references("greet")
        imports = [r for r in refs if r.ref_type == "import"]
        assert len(imports) >= 1

    def test_finds_calls(self, editor):
        refs = editor.find_references("greet")
        calls = [r for r in refs if r.ref_type == "call"]
        assert len(calls) >= 1

    def test_finds_across_files(self, editor):
        refs = editor.find_references("greet")
        files = set(r.filepath for r in refs)
        assert len(files) >= 2  # utils.py + main.py or test_utils.py

    def test_no_references(self, editor):
        refs = editor.find_references("nonexistent_function")
        assert len(refs) == 0

    def test_summary(self, editor):
        s = editor.find_references_summary("greet")
        assert "greet" in s
        assert "DEFINITION" in s

    def test_summary_not_found(self, editor):
        s = editor.find_references_summary("xyz")
        assert "No references" in s


class TestRename:
    def test_rename_creates_plan(self, editor):
        plan = editor.rename("greet", "say_hello")
        assert plan.num_edits >= 3  # def + imports + calls
        assert plan.num_files >= 2

    def test_rename_preview(self, editor):
        plan = editor.rename("greet", "say_hello")
        preview = plan.preview()
        assert "greet" in preview
        assert "say_hello" in preview

    def test_rename_apply(self, editor, project_dir):
        plan = editor.rename("greet", "say_hello")
        result = editor.apply(plan)
        assert result["applied"] >= 3
        assert result["files"] >= 2
        # Verify the file was actually changed
        with open(os.path.join(project_dir, "utils.py")) as f:
            content = f.read()
        assert "say_hello" in content
        assert "greet" not in content

    def test_rename_no_match(self, editor):
        plan = editor.rename("nonexistent", "new_name")
        assert plan.num_edits == 0
        assert len(plan.warnings) >= 1


class TestUpdateSignature:
    def test_update_creates_plan(self, editor):
        plan = editor.update_signature("format_name", "first, last", "first, last, middle=''")
        assert plan.num_edits >= 1  # at least the definition

    def test_update_warns_about_callers(self, editor):
        plan = editor.update_signature("format_name", "first, last", "first, last, middle=''")
        # Should warn about callers in main.py and test_utils.py
        assert len(plan.warnings) >= 1

    def test_update_preview(self, editor):
        plan = editor.update_signature("greet", "name", "name, formal=False")
        preview = plan.preview()
        assert "greet" in preview


class TestRefactorPlan:
    def test_empty_plan(self):
        plan = RefactorPlan(operation="test", target="x")
        assert plan.num_edits == 0
        assert plan.num_files == 0

    def test_plan_with_edits(self):
        plan = RefactorPlan(operation="rename", target="x → y")
        plan.edits.append(FileEdit("a.py", 1, "old", "new"))
        plan.files_affected.add("a.py")
        assert plan.num_edits == 1
        assert plan.num_files == 1


class TestFileEdit:
    def test_preview(self):
        edit = FileEdit("main.py", 5, "  old_name()", "  new_name()")
        p = edit.preview()
        assert "main.py:5" in p
        assert "old_name" in p
        assert "new_name" in p


class TestEditorStats:
    def test_stats(self, editor):
        s = editor.stats()
        assert s["python_files"] >= 3


# ══════════════════════════════════════════════════════════════════
# Phase 7e — Session Continuity Tests
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def store():
    d = tempfile.mkdtemp()
    s = SessionStore(data_dir=d)
    yield s
    shutil.rmtree(d)


class TestSessionLifecycle:
    def test_new_session(self, store):
        session = store.new_session(project_path="/my/project")
        assert session.id
        assert session.project_path == "/my/project"

    def test_end_session(self, store):
        session = store.new_session()
        store.end_session(session.id)
        assert session.ended > 0

    def test_current_session(self, store):
        session = store.new_session()
        assert store.current_session.id == session.id

    def test_get_session(self, store):
        session = store.new_session(title="Test Session")
        retrieved = store.get_session(session.id)
        assert retrieved.title == "Test Session"


class TestExchangeRecording:
    def test_add_exchange(self, store):
        session = store.new_session()
        ex = store.add_exchange(query="what is Python?", response="A programming language.")
        assert ex.query == "what is Python?"
        assert session.num_exchanges == 1

    def test_auto_title(self, store):
        session = store.new_session()
        store.add_exchange(query="explain quicksort algorithm", response="...")
        assert "quicksort" in session.title.lower()

    def test_auto_create_session(self, store):
        # No session exists yet
        ex = store.add_exchange(query="hello", response="hi")
        assert store.current_session is not None

    def test_topic_detection(self, store):
        session = store.new_session()
        ex = store.add_exchange(query="write a pytest test for the API", response="...")
        assert "test" in ex.topics or "api" in ex.topics

    def test_file_detection(self, store):
        session = store.new_session()
        ex = store.add_exchange(
            query="fix the bug in main.py",
            response="The issue is in utils.py",
        )
        assert "main.py" in ex.files_mentioned or "utils.py" in ex.files_mentioned

    def test_multiple_exchanges(self, store):
        session = store.new_session()
        for i in range(5):
            store.add_exchange(query=f"question {i}", response=f"answer {i}")
        assert session.num_exchanges == 5


class TestSessionContext:
    def test_get_context(self, store):
        session = store.new_session(project_path="/proj")
        store.add_exchange(query="what is X?", response="X is Y.")
        store.add_exchange(query="explain Z", response="Z does W.")
        ctx = store.get_session_context(session.id)
        assert "what is X" in ctx
        assert "explain Z" in ctx

    def test_continuation_context(self, store):
        session = store.new_session()
        store.add_exchange(query="build a calculator", response="Here's the code...")
        ctx = store.get_continuation_context()
        assert "calculator" in ctx

    def test_empty_continuation(self, store):
        ctx = store.get_continuation_context()
        assert ctx == ""


class TestSessionSearch:
    def test_search_finds_match(self, store):
        session = store.new_session()
        store.add_exchange(query="how to deploy Docker containers", response="Use docker-compose...")
        results = store.search("Docker")
        assert len(results) >= 1

    def test_search_no_match(self, store):
        session = store.new_session()
        store.add_exchange(query="hello", response="hi")
        results = store.search("quantum physics")
        assert len(results) == 0

    def test_search_summary(self, store):
        session = store.new_session()
        store.add_exchange(query="fix database migration", response="Update the schema...")
        s = store.search_summary("database")
        assert "database" in s.lower()

    def test_search_summary_empty(self, store):
        s = store.search_summary("nonexistent")
        assert "No past conversations" in s


class TestSessionPersistence:
    def test_save_and_load(self):
        d = tempfile.mkdtemp()
        s1 = SessionStore(data_dir=d)
        session = s1.new_session(title="Persistent Test")
        s1.add_exchange(query="remember this", response="stored!")
        s1.save_all()

        s2 = SessionStore(data_dir=d)
        loaded = s2.get_session(session.id)
        assert loaded is not None
        assert loaded.title == "Persistent Test"
        assert loaded.num_exchanges == 1
        shutil.rmtree(d)

    def test_list_sessions(self, store):
        store.new_session(title="Session A")
        store.new_session(title="Session B")
        sessions = store.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_summary(self, store):
        store.new_session(title="Test Session")
        store.add_exchange(query="q", response="r")
        s = store.list_sessions_summary()
        assert "Test Session" in s


class TestSessionStats:
    def test_stats(self, store):
        store.new_session()
        store.add_exchange(query="q", response="r")
        s = store.stats()
        assert s["total_sessions"] >= 1
        assert s["total_exchanges"] >= 1

    def test_total_counts(self, store):
        store.new_session()
        store.add_exchange(query="a", response="b")
        store.add_exchange(query="c", response="d")
        assert store.total_sessions >= 1
        assert store.total_exchanges >= 2


class TestExchangeDataclass:
    def test_to_dict(self):
        ex = Exchange(id="123", timestamp=1.0, query="q", response="r")
        d = ex.to_dict()
        assert d["id"] == "123"
        assert d["query"] == "q"

    def test_from_dict(self):
        d = {"id": "abc", "timestamp": 2.0, "query": "q", "response": "r"}
        ex = Exchange.from_dict(d)
        assert ex.id == "abc"

    def test_preview(self):
        ex = Exchange(id="1", timestamp=1.0, query="hello world", response="hi there")
        assert "hello" in ex.preview


class TestSessionDataclass:
    def test_summary(self):
        s = Session(id="1", started=time.time(), title="My Session")
        s.exchanges.append(Exchange(id="e1", timestamp=time.time(), query="q", response="r"))
        assert "My Session" in s.summary()
        assert "1 exchange" in s.summary()

    def test_duration(self):
        s = Session(id="1", started=time.time() - 600)  # 10 min ago
        assert s.duration_minutes >= 9
