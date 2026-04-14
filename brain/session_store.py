"""
LeanAI Phase 7e — Session Continuity Engine
Persistent conversation history that survives across sessions.

Every conversation is stored with full context. "Continue where we left off"
actually works. Code from session 2 is available in session 5.

Features:
  - Save every exchange (query + response + metadata)
  - Load previous session context into new conversations
  - Search past conversations semantically
  - Track which topics/files were discussed
  - Resume any previous session
"""

import os
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class Exchange:
    """A single query-response exchange."""
    id: str
    timestamp: float
    query: str
    response: str
    tier: str = ""
    confidence: float = 0.0
    files_mentioned: List[str] = field(default_factory=list)
    code_generated: bool = False
    topics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Exchange":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def preview(self) -> str:
        q = self.query[:60].replace("\n", " ")
        r = self.response[:60].replace("\n", " ")
        return f"Q: {q}... → R: {r}..."


@dataclass
class Session:
    """A conversation session."""
    id: str
    started: float
    ended: float = 0.0
    exchanges: List[Exchange] = field(default_factory=list)
    title: str = ""               # auto-generated from first query
    project_path: str = ""        # which project was being worked on
    files_touched: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        end = self.ended or time.time()
        return (end - self.started) / 60

    @property
    def num_exchanges(self) -> int:
        return len(self.exchanges)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "started": self.started, "ended": self.ended,
            "title": self.title, "project_path": self.project_path,
            "files_touched": self.files_touched, "tags": self.tags,
            "exchanges": [e.to_dict() for e in self.exchanges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        exchanges = [Exchange.from_dict(e) for e in d.get("exchanges", [])]
        return cls(
            id=d["id"], started=d["started"], ended=d.get("ended", 0),
            title=d.get("title", ""), project_path=d.get("project_path", ""),
            files_touched=d.get("files_touched", []),
            tags=d.get("tags", []),
            exchanges=exchanges,
        )

    def summary(self) -> str:
        title = self.title or "Untitled"
        return (
            f"Session: {title} ({self.num_exchanges} exchanges, "
            f"{self.duration_minutes:.0f} min)"
        )


class SessionStore:
    """
    Persistent storage for conversation sessions.
    
    Usage:
        store = SessionStore()
        
        # Start a new session
        session = store.new_session(project_path="/my/project")
        
        # Record exchanges
        store.add_exchange(session.id, query="explain main.py", response="...")
        
        # End session
        store.end_session(session.id)
        
        # Later — resume or search
        sessions = store.list_sessions(limit=10)
        context = store.get_session_context(session.id)
        results = store.search("database migration")
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or str(Path(os.environ.get('LEANAI_HOME', str(Path.home() / '.leanai'))) / "sessions")
        os.makedirs(self.data_dir, exist_ok=True)
        self._sessions: Dict[str, Session] = {}
        self._current_session_id: Optional[str] = None
        self._load_index()

    def _generate_id(self) -> str:
        """Generate a unique session/exchange ID."""
        return hashlib.md5(f"{time.time()}-{os.getpid()}".encode()).hexdigest()[:12]

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.data_dir, f"session_{session_id}.json")

    def _index_path(self) -> str:
        return os.path.join(self.data_dir, "index.json")

    # ── Session lifecycle ─────────────────────────────────────────

    def new_session(self, project_path: str = "", title: str = "") -> Session:
        """Start a new conversation session."""
        session = Session(
            id=self._generate_id(),
            started=time.time(),
            project_path=project_path,
            title=title,
        )
        self._sessions[session.id] = session
        self._current_session_id = session.id
        self._save_index()
        return session

    def end_session(self, session_id: Optional[str] = None):
        """End a session and save it."""
        sid = session_id or self._current_session_id
        if sid and sid in self._sessions:
            session = self._sessions[sid]
            session.ended = time.time()
            self._save_session(session)
            self._save_index()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self._load_session(session_id)

    @property
    def current_session(self) -> Optional[Session]:
        if self._current_session_id:
            return self._sessions.get(self._current_session_id)
        return None

    # ── Exchange recording ────────────────────────────────────────

    def add_exchange(
        self,
        session_id: Optional[str] = None,
        query: str = "",
        response: str = "",
        tier: str = "",
        confidence: float = 0.0,
        files_mentioned: Optional[List[str]] = None,
        code_generated: bool = False,
    ) -> Exchange:
        """Record a query-response exchange in the current session."""
        sid = session_id or self._current_session_id
        if not sid or sid not in self._sessions:
            # Auto-create session if needed
            session = self.new_session()
            sid = session.id
        else:
            session = self._sessions[sid]

        # Auto-detect topics from query
        topics = self._extract_topics(query)

        # Auto-detect files mentioned
        if files_mentioned is None:
            files_mentioned = self._extract_files(query + " " + response)

        exchange = Exchange(
            id=self._generate_id(),
            timestamp=time.time(),
            query=query,
            response=response,
            tier=tier,
            confidence=confidence,
            files_mentioned=files_mentioned,
            code_generated=code_generated,
            topics=topics,
        )
        session.exchanges.append(exchange)

        # Auto-title from first query
        if not session.title and query:
            session.title = query[:50].strip()

        # Track files
        for f in files_mentioned:
            if f not in session.files_touched:
                session.files_touched.append(f)

        # Auto-save periodically
        if len(session.exchanges) % 5 == 0:
            self._save_session(session)

        return exchange

    # ── Context generation ────────────────────────────────────────

    def get_session_context(self, session_id: str, max_exchanges: int = 10) -> str:
        """
        Get conversation context from a session for injection into prompts.
        Returns the last N exchanges formatted as context.
        """
        session = self.get_session(session_id)
        if not session:
            return ""

        lines = [f"Previous session: {session.title or 'Untitled'}"]
        if session.project_path:
            lines.append(f"Project: {session.project_path}")
        if session.files_touched:
            lines.append(f"Files discussed: {', '.join(session.files_touched[:10])}")
        lines.append("")

        recent = session.exchanges[-max_exchanges:]
        for ex in recent:
            lines.append(f"User: {ex.query[:200]}")
            lines.append(f"AI: {ex.response[:200]}")
            lines.append("")

        return "\n".join(lines)

    def get_continuation_context(self, max_exchanges: int = 5) -> str:
        """
        Get context for "continue where we left off" — 
        combines the most recent session's last exchanges.
        """
        if not self._sessions:
            return ""

        # Find the most recent session with exchanges
        recent = sorted(
            self._sessions.values(),
            key=lambda s: s.started,
            reverse=True,
        )
        for session in recent:
            if session.exchanges:
                return self.get_session_context(session.id, max_exchanges)
        return ""

    # ── Search ────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 10) -> List[Tuple[Exchange, str]]:
        """
        Search across all sessions for relevant exchanges.
        Returns list of (exchange, session_id) tuples.
        """
        query_lower = query.lower()
        results = []

        for sid, session in self._sessions.items():
            for ex in session.exchanges:
                # Simple keyword matching (could be upgraded to semantic)
                score = 0
                for word in query_lower.split():
                    if word in ex.query.lower():
                        score += 2
                    if word in ex.response.lower():
                        score += 1
                if score > 0:
                    results.append((ex, sid, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return [(ex, sid) for ex, sid, _ in results[:limit]]

    def search_summary(self, query: str, limit: int = 5) -> str:
        """Human-readable search results."""
        results = self.search(query, limit)
        if not results:
            return f"No past conversations found about '{query}'"

        lines = [f"Found {len(results)} relevant exchanges:"]
        for ex, sid in results:
            session = self._sessions.get(sid)
            title = session.title if session else sid
            lines.append(f"\n  [{title}]")
            lines.append(f"  Q: {ex.query[:80]}")
            lines.append(f"  A: {ex.response[:80]}")
        return "\n".join(lines)

    # ── Session listing ───────────────────────────────────────────

    def list_sessions(self, limit: int = 10) -> List[Session]:
        """List recent sessions."""
        sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.started,
            reverse=True,
        )
        return sessions[:limit]

    def list_sessions_summary(self, limit: int = 10) -> str:
        """Human-readable session list."""
        sessions = self.list_sessions(limit)
        if not sessions:
            return "No sessions recorded yet."
        lines = [f"Recent sessions ({len(sessions)}):"]
        for s in sessions:
            lines.append(f"  {s.summary()}")
        return "\n".join(lines)

    @property
    def total_sessions(self) -> int:
        return len(self._sessions)

    @property
    def total_exchanges(self) -> int:
        return sum(s.num_exchanges for s in self._sessions.values())

    # ── Topic/file extraction ─────────────────────────────────────

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text."""
        topics = []
        keywords = {
            "code": ["code", "function", "class", "implement", "write", "debug"],
            "test": ["test", "pytest", "assert", "verify"],
            "deploy": ["deploy", "docker", "kubernetes", "ci/cd", "pipeline"],
            "database": ["database", "sql", "query", "migration", "schema"],
            "api": ["api", "endpoint", "rest", "http", "route"],
            "frontend": ["react", "html", "css", "component", "ui"],
            "config": ["config", "environment", "settings", "setup"],
        }
        text_lower = text.lower()
        for topic, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                topics.append(topic)
        return topics

    def _extract_files(self, text: str) -> List[str]:
        """Extract filenames mentioned in text."""
        import re
        pattern = r'\b[\w/\\]+\.\w{1,5}\b'
        matches = re.findall(pattern, text)
        files = []
        valid_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".yaml", ".json", ".toml", ".md"}
        for m in matches:
            ext = os.path.splitext(m)[1].lower()
            if ext in valid_exts and m not in files:
                files.append(m)
        return files[:10]

    # ── Persistence ───────────────────────────────────────────────

    def _save_session(self, session: Session):
        """Save a single session to disk."""
        path = self._session_path(session.id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception:
            pass

    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load a session from disk."""
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = Session.from_dict(data)
            self._sessions[session.id] = session
            return session
        except (json.JSONDecodeError, Exception):
            return None

    def _save_index(self):
        """Save session index (lightweight metadata for listing)."""
        index = {
            sid: {
                "title": s.title, "started": s.started,
                "ended": s.ended, "exchanges": s.num_exchanges,
                "project": s.project_path,
            }
            for sid, s in self._sessions.items()
        }
        try:
            with open(self._index_path(), "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass

    def _load_index(self):
        """Load session index and lazy-load sessions."""
        path = self._index_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                index = json.load(f)
            for sid in index:
                if sid not in self._sessions:
                    self._load_session(sid)
        except (json.JSONDecodeError, Exception):
            pass

    def save_all(self):
        """Save all sessions to disk."""
        for session in self._sessions.values():
            self._save_session(session)
        self._save_index()

    def stats(self) -> dict:
        return {
            "total_sessions": self.total_sessions,
            "total_exchanges": self.total_exchanges,
            "current_session": self._current_session_id,
        }
