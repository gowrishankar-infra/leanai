"""
LeanAI — Cross-Session Code Evolution Tracking
Tracks how your understanding of a problem evolves across sessions.

Session 1: "how to set up a database"
Session 3: "add caching layer for database queries"
Session 5: "optimize cache invalidation strategy"
Session 7: LeanAI predicts → "you might want to look at distributed caching"

It connects your sessions into a coherent project narrative:
  "You started with basic database setup, then added caching, then
   optimized invalidation. Your project is evolving toward a distributed
   data layer. Consider Redis cluster or memcached."

Nobody tracks this. Every AI tool treats each session as isolated.
"""

import os
import json
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import Counter


@dataclass
class ThemeOccurrence:
    """A single occurrence of a theme in a session."""
    session_id: str
    query: str
    timestamp: float
    confidence: float = 0.0


@dataclass
class ProjectTheme:
    """A recurring theme across sessions."""
    name: str
    keywords: List[str]
    occurrences: List[ThemeOccurrence] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    evolution_stage: str = "exploring"  # exploring → building → optimizing → maintaining

    @property
    def frequency(self) -> int:
        return len(self.occurrences)

    @property
    def sessions_count(self) -> int:
        return len(set(o.session_id for o in self.occurrences))

    def summary(self) -> str:
        return (
            f"{self.name} ({self.evolution_stage}) — "
            f"{self.frequency} mentions across {self.sessions_count} sessions"
        )


@dataclass
class EvolutionInsight:
    """An insight about how the project is evolving."""
    theme: str
    stage: str
    trajectory: str  # what's likely next
    confidence: float
    supporting_queries: List[str]

    def summary(self) -> str:
        return (
            f"Theme: {self.theme}\n"
            f"Stage: {self.stage}\n"
            f"Trajectory: {self.trajectory}\n"
            f"Based on: {len(self.supporting_queries)} queries across sessions"
        )


# ── Theme detection ──────────────────────────────────────────

THEME_PATTERNS = {
    "database": {
        "keywords": ["database", "sql", "query", "table", "migration", "schema",
                     "orm", "model", "postgres", "mysql", "sqlite", "mongo",
                     "index", "join", "transaction"],
        "stages": {
            "exploring": ["what is", "how to", "setup", "install", "basics"],
            "building": ["create", "implement", "add", "write", "build"],
            "optimizing": ["optimize", "performance", "slow", "index", "cache", "scale"],
            "maintaining": ["migrate", "backup", "monitor", "debug", "fix"],
        },
    },
    "authentication": {
        "keywords": ["auth", "login", "password", "token", "jwt", "session",
                     "oauth", "permission", "role", "security", "credential"],
        "stages": {
            "exploring": ["what is", "how does", "explain", "understand"],
            "building": ["implement", "create", "add", "setup", "configure"],
            "optimizing": ["refresh token", "session management", "rate limit", "secure"],
            "maintaining": ["fix", "debug", "vulnerability", "update", "rotate"],
        },
    },
    "api": {
        "keywords": ["api", "endpoint", "rest", "graphql", "route", "controller",
                     "request", "response", "middleware", "cors", "rate limit"],
        "stages": {
            "exploring": ["design", "plan", "structure"],
            "building": ["create", "implement", "add endpoint", "route"],
            "optimizing": ["cache", "pagination", "versioning", "throttle"],
            "maintaining": ["deprecate", "migrate", "document", "test"],
        },
    },
    "testing": {
        "keywords": ["test", "pytest", "unittest", "mock", "fixture", "coverage",
                     "assert", "tdd", "integration test", "e2e"],
        "stages": {
            "exploring": ["how to test", "testing strategy"],
            "building": ["write tests", "add tests", "test for"],
            "optimizing": ["coverage", "parametrize", "faster tests", "parallel"],
            "maintaining": ["flaky", "fix test", "update test", "refactor test"],
        },
    },
    "deployment": {
        "keywords": ["deploy", "docker", "kubernetes", "ci/cd", "pipeline",
                     "nginx", "aws", "cloud", "terraform", "ansible", "helm"],
        "stages": {
            "exploring": ["how to deploy", "deployment options", "infrastructure"],
            "building": ["dockerfile", "create pipeline", "setup", "configure"],
            "optimizing": ["scale", "auto-scale", "optimize", "reduce cost"],
            "maintaining": ["monitor", "logging", "alert", "rollback", "incident"],
        },
    },
    "caching": {
        "keywords": ["cache", "redis", "memcached", "ttl", "invalidation",
                     "lru", "cdn", "memoize", "in-memory"],
        "stages": {
            "exploring": ["what is caching", "caching strategy"],
            "building": ["add cache", "implement cache", "cache layer"],
            "optimizing": ["invalidation", "ttl", "cache miss", "warm cache"],
            "maintaining": ["clear cache", "cache bug", "stale data"],
        },
    },
    "architecture": {
        "keywords": ["architecture", "design pattern", "microservice", "monolith",
                     "event driven", "message queue", "pub sub", "clean architecture"],
        "stages": {
            "exploring": ["design", "architecture", "how to structure"],
            "building": ["implement", "refactor into", "split", "extract"],
            "optimizing": ["decouple", "async", "event driven", "scale"],
            "maintaining": ["technical debt", "legacy", "migrate", "simplify"],
        },
    },
}


class EvolutionTracker:
    """
    Tracks how project understanding evolves across sessions.
    
    Usage:
        tracker = EvolutionTracker()
        
        # Feed it queries from sessions
        tracker.track_query("how to set up postgres", session_id="s1")
        tracker.track_query("add caching for db queries", session_id="s3")
        tracker.track_query("optimize cache invalidation", session_id="s5")
        
        # Get evolution insights
        insights = tracker.get_insights()
        # → "Database theme: exploring → building → optimizing.
        #    Trajectory: consider connection pooling and read replicas."
        
        # Predict next questions
        predictions = tracker.predict_next_topics()
        # → ["distributed caching", "cache monitoring"]
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or str(Path.home() / ".leanai" / "evolution")
        os.makedirs(self.data_dir, exist_ok=True)
        self.themes: Dict[str, ProjectTheme] = {}
        self._load()

    def track_query(self, query: str, session_id: str = "",
                    timestamp: Optional[float] = None):
        """Track a query and detect themes."""
        ts = timestamp or time.time()
        query_lower = query.lower()

        for theme_name, pattern in THEME_PATTERNS.items():
            # Check if query matches this theme
            matches = sum(1 for kw in pattern["keywords"] if kw in query_lower)
            if matches == 0:
                continue

            # Create or update theme
            if theme_name not in self.themes:
                self.themes[theme_name] = ProjectTheme(
                    name=theme_name,
                    keywords=pattern["keywords"],
                    first_seen=ts,
                )

            theme = self.themes[theme_name]
            theme.last_seen = ts
            theme.occurrences.append(ThemeOccurrence(
                session_id=session_id,
                query=query[:200],
                timestamp=ts,
                confidence=min(matches * 0.3, 1.0),
            ))

            # Detect evolution stage
            theme.evolution_stage = self._detect_stage(theme, query_lower, pattern)

        # Periodic save
        if sum(t.frequency for t in self.themes.values()) % 10 == 0:
            self.save()

    def _detect_stage(self, theme: ProjectTheme, query_lower: str,
                      pattern: dict) -> str:
        """Detect which evolution stage a theme is in."""
        stages = pattern.get("stages", {})
        stage_scores = {}

        # Score each stage based on recent queries
        recent = theme.occurrences[-5:]  # last 5 occurrences
        for occ in recent:
            q = occ.query.lower()
            for stage_name, stage_keywords in stages.items():
                for kw in stage_keywords:
                    if kw in q:
                        stage_scores[stage_name] = stage_scores.get(stage_name, 0) + 1

        # Also check current query
        for stage_name, stage_keywords in stages.items():
            for kw in stage_keywords:
                if kw in query_lower:
                    stage_scores[stage_name] = stage_scores.get(stage_name, 0) + 2

        if not stage_scores:
            return theme.evolution_stage  # keep current

        return max(stage_scores, key=stage_scores.get)

    def get_insights(self) -> List[EvolutionInsight]:
        """Generate evolution insights across all themes."""
        insights = []

        for name, theme in self.themes.items():
            if theme.frequency < 2:
                continue  # need at least 2 occurrences for a pattern

            trajectory = self._predict_trajectory(theme)
            queries = [o.query for o in theme.occurrences[-5:]]

            insights.append(EvolutionInsight(
                theme=name,
                stage=theme.evolution_stage,
                trajectory=trajectory,
                confidence=min(theme.frequency * 0.15, 0.9),
                supporting_queries=queries,
            ))

        # Sort by frequency (most active themes first)
        insights.sort(key=lambda i: len(i.supporting_queries), reverse=True)
        return insights

    def _predict_trajectory(self, theme: ProjectTheme) -> str:
        """Predict what the user will likely need next based on evolution stage."""
        stage = theme.evolution_stage
        name = theme.name

        trajectories = {
            "database": {
                "exploring": "You'll likely need to set up an ORM and define models next",
                "building": "Consider adding migrations and seed data",
                "optimizing": "Look into connection pooling, read replicas, or query caching",
                "maintaining": "Set up automated backups and monitoring dashboards",
            },
            "authentication": {
                "exploring": "JWT with refresh tokens is the most common pattern to implement",
                "building": "Don't forget password hashing (bcrypt) and rate limiting on login",
                "optimizing": "Consider OAuth2 for third-party auth and session management",
                "maintaining": "Rotate secrets regularly and audit auth logs",
            },
            "api": {
                "exploring": "RESTful design with versioning is the safest starting point",
                "building": "Add input validation, error handling, and API documentation",
                "optimizing": "Implement response caching, pagination, and rate limiting",
                "maintaining": "Set up API monitoring, deprecation notices, and changelog",
            },
            "testing": {
                "exploring": "Start with unit tests for core business logic",
                "building": "Add integration tests for database and API layers",
                "optimizing": "Parallelize tests and add mutation testing for coverage gaps",
                "maintaining": "Fix flaky tests first — they erode trust in the suite",
            },
            "deployment": {
                "exploring": "Docker + CI/CD pipeline is the minimum viable deployment",
                "building": "Add health checks, environment configs, and staging environment",
                "optimizing": "Consider Kubernetes for auto-scaling and blue-green deployments",
                "maintaining": "Set up centralized logging, alerting, and incident runbooks",
            },
            "caching": {
                "exploring": "Start with application-level caching (in-memory or Redis)",
                "building": "Implement cache-aside pattern with appropriate TTLs",
                "optimizing": "Add cache warming, tiered caching, and monitor hit rates",
                "maintaining": "Watch for stale data bugs and implement cache versioning",
            },
            "architecture": {
                "exploring": "Start with a clean modular monolith before microservices",
                "building": "Extract domains into bounded contexts with clear interfaces",
                "optimizing": "Introduce event-driven communication between services",
                "maintaining": "Document architecture decisions (ADRs) and reduce coupling",
            },
        }

        return trajectories.get(name, {}).get(
            stage, "Continue building and iterate based on feedback"
        )

    def predict_next_topics(self, max_topics: int = 3) -> List[str]:
        """Predict what topics the user will likely ask about next."""
        predictions = []
        for theme in sorted(self.themes.values(),
                            key=lambda t: t.last_seen, reverse=True):
            trajectory = self._predict_trajectory(theme)
            if trajectory:
                predictions.append(trajectory)
            if len(predictions) >= max_topics:
                break
        return predictions

    def get_narrative(self) -> str:
        """Generate a coherent narrative of the project's evolution."""
        if not self.themes:
            return "No project evolution tracked yet. Keep using LeanAI to build a history."

        active_themes = sorted(
            [t for t in self.themes.values() if t.frequency >= 2],
            key=lambda t: t.first_seen,
        )

        if not active_themes:
            return "Still gathering data. Use LeanAI for a few more sessions."

        lines = ["Project Evolution:"]
        for theme in active_themes:
            duration_days = (theme.last_seen - theme.first_seen) / 86400
            lines.append(
                f"\n  {theme.name} ({theme.evolution_stage})"
                f" — {theme.frequency} queries across {theme.sessions_count} sessions"
                f" over {max(duration_days, 1):.0f} days"
            )
            # Show progression
            stages_seen = []
            for occ in theme.occurrences:
                q = occ.query.lower()
                for stage, keywords in THEME_PATTERNS.get(theme.name, {}).get("stages", {}).items():
                    if any(kw in q for kw in keywords):
                        if stage not in stages_seen:
                            stages_seen.append(stage)
            if stages_seen:
                lines.append(f"    Journey: {' → '.join(stages_seen)}")

            trajectory = self._predict_trajectory(theme)
            lines.append(f"    Next: {trajectory}")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────

    def save(self):
        path = os.path.join(self.data_dir, "evolution.json")
        data = {}
        for name, theme in self.themes.items():
            data[name] = {
                "name": theme.name,
                "keywords": theme.keywords,
                "first_seen": theme.first_seen,
                "last_seen": theme.last_seen,
                "evolution_stage": theme.evolution_stage,
                "occurrences": [
                    {"session_id": o.session_id, "query": o.query,
                     "timestamp": o.timestamp, "confidence": o.confidence}
                    for o in theme.occurrences
                ],
            }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load(self):
        path = os.path.join(self.data_dir, "evolution.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for name, d in data.items():
                theme = ProjectTheme(
                    name=d["name"],
                    keywords=d.get("keywords", []),
                    first_seen=d.get("first_seen", 0),
                    last_seen=d.get("last_seen", 0),
                    evolution_stage=d.get("evolution_stage", "exploring"),
                )
                for o in d.get("occurrences", []):
                    theme.occurrences.append(ThemeOccurrence(
                        session_id=o["session_id"],
                        query=o["query"],
                        timestamp=o["timestamp"],
                        confidence=o.get("confidence", 0.5),
                    ))
                self.themes[name] = theme
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "themes_tracked": len(self.themes),
            "total_occurrences": sum(t.frequency for t in self.themes.values()),
            "active_themes": [t.summary() for t in self.themes.values() if t.frequency >= 2],
        }
