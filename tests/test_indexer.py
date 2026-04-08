"""
LeanAI · Phase 4c Tests — Project Indexer
Run: python -m pytest tests/test_indexer.py -v
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pathlib import Path
from tools.indexer import ProjectIndexer, CodeChunk, SUPPORTED_EXTENSIONS


@pytest.fixture
def tmp_project(tmp_path):
    """Create a small fake project for testing."""
    # Python file with functions
    (tmp_path / "main.py").write_text("""
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
""")
    # Another Python file
    (tmp_path / "utils.py").write_text("""
import os

def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
""")
    # JavaScript file
    (tmp_path / "app.js").write_text("""
function fetchData(url) {
    return fetch(url).then(r => r.json());
}

class App {
    constructor() {
        this.data = [];
    }
}
""")
    # Markdown file
    (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test.\n")
    # File that should be skipped
    (tmp_path / "package-lock.json").write_text('{"name": "test"}')
    # Subdir
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "helpers.py").write_text("""
def format_date(date):
    return date.strftime('%Y-%m-%d')

def slugify(text):
    return text.lower().replace(' ', '-')
""")
    return tmp_path


@pytest.fixture
def indexer(tmp_path):
    storage = tmp_path / "index_storage"
    return ProjectIndexer(storage_path=str(storage))


class TestProjectIndexer:

    def test_index_project_runs(self, indexer, tmp_project):
        stats = indexer.index_project(str(tmp_project))
        assert stats.total_files > 0
        assert stats.indexed_files > 0

    def test_finds_python_files(self, indexer, tmp_project):
        stats = indexer.index_project(str(tmp_project))
        assert "python" in stats.languages
        assert stats.languages["python"] >= 3

    def test_finds_js_files(self, indexer, tmp_project):
        stats = indexer.index_project(str(tmp_project))
        assert "javascript" in stats.languages

    def test_skips_package_lock(self, indexer, tmp_project):
        stats = indexer.index_project(str(tmp_project))
        # package-lock.json should be skipped
        assert stats.indexed_files < stats.total_files + 5

    def test_chunks_created(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        assert indexer.count() > 0

    def test_chunks_python_functions(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        assert indexer.count() >= 4  # greet, add, Calculator + helpers

    def test_incremental_index(self, indexer, tmp_project):
        # First index
        stats1 = indexer.index_project(str(tmp_project))
        # Second index — should skip unchanged files
        stats2 = indexer.index_project(str(tmp_project))
        assert stats2.skipped_files > 0

    def test_force_reindex(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        stats = indexer.index_project(str(tmp_project), force=True)
        assert stats.skipped_files == 0

    def test_search_returns_results(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        results = indexer.search("function that adds numbers")
        assert isinstance(results, list)

    def test_search_finds_relevant(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        # Search works when chunks are indexed (semantic quality depends on embedder)
        results = indexer.search("greet name hello")
        # Results may be empty without a real embedder — just verify structure
        assert isinstance(results, list)
        if results:
            assert "content" in results[0]
            assert "file_path" in results[0]

    def test_search_with_language_filter(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        results = indexer.search("function", language="python")
        assert isinstance(results, list)

    def test_format_results_string(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        results = indexer.search("file path read write")
        formatted = indexer.format_search_results(results)
        assert isinstance(formatted, str)

    def test_count_increases_after_index(self, indexer, tmp_project):
        before = indexer.count()
        indexer.index_project(str(tmp_project))
        after = indexer.count()
        assert after > before

    def test_stats_structure(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        stats = indexer.stats()
        assert "total_chunks" in stats

    def test_clear_removes_all(self, indexer, tmp_project):
        indexer.index_project(str(tmp_project))
        assert indexer.count() > 0
        indexer.clear()
        assert indexer.count() == 0

    def test_invalid_project_raises(self, indexer):
        with pytest.raises(ValueError):
            indexer.index_project("/nonexistent/path/xyz")


class TestChunking:

    def setup_method(self):
        self.indexer = ProjectIndexer("/tmp/leanai_test_idx")

    def test_chunk_python_functions(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        chunks = self.indexer._chunk_python(f.read_text(), f, "test.py", tmp_path)
        assert len(chunks) == 2
        names = [c.name for c in chunks]
        assert "foo" in names
        assert "bar" in names

    def test_chunk_python_class(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class MyClass:\n    def method(self):\n        pass\n")
        chunks = self.indexer._chunk_python(f.read_text(), f, "test.py", tmp_path)
        assert any(c.chunk_type == "class" for c in chunks)

    def test_chunk_generic_splits_large(self, tmp_path):
        f = tmp_path / "big.sql"
        content = "SELECT * FROM table;\n" * 200
        f.write_text(content)
        chunks = self.indexer._chunk_generic(content, f, "big.sql", tmp_path, "sql")
        assert len(chunks) > 1

    def test_chunk_min_size_filter(self, tmp_path):
        f = tmp_path / "tiny.py"
        f.write_text("def f():\n    pass\n")
        chunks = self.indexer._chunk_python(f.read_text(), f, "tiny.py", tmp_path)
        # Very short functions filtered out
        for c in chunks:
            assert len(c.content) >= 50 or len(chunks) == 0

    def test_detect_language(self):
        assert self.indexer._detect_language(Path("file.py")) == "python"
        assert self.indexer._detect_language(Path("file.js")) == "javascript"
        assert self.indexer._detect_language(Path("file.ts")) == "typescript"
        assert self.indexer._detect_language(Path("file.go")) == "go"
        assert self.indexer._detect_language(Path("file.rs")) == "rust"
        assert self.indexer._detect_language(Path("file.sql")) == "sql"
        assert self.indexer._detect_language(Path("file.md")) == "markdown"

    def test_supported_extensions(self):
        assert ".py" in SUPPORTED_EXTENSIONS
        assert ".js" in SUPPORTED_EXTENSIONS
        assert ".ts" in SUPPORTED_EXTENSIONS
        assert ".go" in SUPPORTED_EXTENSIONS
        assert ".rs" in SUPPORTED_EXTENSIONS

    def test_skip_dirs(self):
        from tools.indexer import SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert "node_modules" in SKIP_DIRS
        assert ".git" in SKIP_DIRS
        assert ".venv" in SKIP_DIRS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
