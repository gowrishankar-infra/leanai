"""
LeanAI · Phase 2 — Causal World Model
The AI's persistent model of YOUR world.

Every conversation, it learns:
  - Who you are (name, job, preferences, goals)
  - What you're working on (projects, tasks, problems)
  - What happened (events, decisions, outcomes)
  - How things relate (A caused B, X leads to Y)

Then uses this to:
  - Answer questions about your specific situation
  - Reason about what would happen if you did X
  - Remember context across sessions automatically
  - Give personalized answers, not generic ones

No other local AI does this. This is what makes LeanAI feel like
it actually knows you instead of starting fresh every conversation.
"""

import json
import time
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
from enum import Enum


class EntityType(Enum):
    PERSON     = "person"
    PLACE      = "place"
    PROJECT    = "project"
    CONCEPT    = "concept"
    TOOL       = "tool"
    EVENT      = "event"
    PREFERENCE = "preference"
    GOAL       = "goal"
    FACT       = "fact"


class RelationType(Enum):
    WORKS_ON   = "works_on"
    KNOWS      = "knows"
    LOCATED_IN = "located_in"
    CAUSED     = "caused"
    LEADS_TO   = "leads_to"
    PART_OF    = "part_of"
    PREFERS    = "prefers"
    OWNS       = "owns"
    CREATED    = "created"
    IS_A       = "is_a"


@dataclass
class Entity:
    id: str
    name: str
    entity_type: EntityType
    properties: dict = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    mention_count: int = 1
    confidence: float = 0.8

    def update(self, properties: dict):
        self.properties.update(properties)
        self.last_updated = time.time()
        self.mention_count += 1


@dataclass
class Relation:
    subject_id: str
    relation_type: RelationType
    object_id: str
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    evidence: str = ""


@dataclass
class CausalEvent:
    id: str
    description: str
    cause: Optional[str]       # entity or event ID
    effect: Optional[str]      # entity or event ID
    timestamp: float
    confidence: float = 0.7


class WorldModel:
    """
    Persistent causal world model.
    Extracts entities and relationships from conversations automatically.
    """

    def __init__(self, storage_path: str = "~/.leanai/world"):
        self.path = Path(storage_path).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._events: list[CausalEvent] = []
        self._user_profile: dict = {}
        self._load()
        print(f"[World] Model loaded — {len(self._entities)} entities, "
              f"{len(self._relations)} relations")

    # ══════════════════════════════════════════════════
    # Learning from conversations
    # ══════════════════════════════════════════════════

    def learn_from_exchange(self, user_msg: str, ai_response: str):
        """
        Extract entities, facts, and relationships from a conversation turn.
        Called automatically after every exchange.
        """
        self._extract_user_facts(user_msg)
        self._extract_entities(user_msg)
        self._save()

    def learn_fact(self, fact: str, confidence: float = 0.9):
        """Explicitly store a fact about the user's world."""
        entity_id = f"fact_{hash(fact) % 100000}"
        entity = Entity(
            id=entity_id,
            name=fact[:60],
            entity_type=EntityType.FACT,
            properties={"full_text": fact, "confidence": confidence},
        )
        self._entities[entity_id] = entity
        self._save()

    # ══════════════════════════════════════════════════
    # Querying the world model
    # ══════════════════════════════════════════════════

    def get_context_for_query(self, query: str) -> str:
        """
        Build a context string from the world model relevant to the query.
        Injected into the AI prompt to ground responses in user's reality.
        """
        parts = []

        # User profile
        if self._user_profile:
            profile_items = []
            if "name" in self._user_profile:
                profile_items.append(f"User's name: {self._user_profile['name']}")
            if "job" in self._user_profile:
                profile_items.append(f"Works as: {self._user_profile['job']}")
            if "location" in self._user_profile:
                profile_items.append(f"Location: {self._user_profile['location']}")
            if "projects" in self._user_profile:
                profile_items.append(f"Current projects: {', '.join(self._user_profile['projects'][:3])}")
            if profile_items:
                parts.append("About the user:\n" + "\n".join(f"  • {p}" for p in profile_items))

        # Relevant entities
        query_lower = query.lower()
        relevant = []
        for entity in self._entities.values():
            if (entity.name.lower() in query_lower or
                any(str(v).lower() in query_lower
                    for v in entity.properties.values()
                    if isinstance(v, str))):
                relevant.append(entity)

        if relevant:
            entity_lines = []
            for e in relevant[:5]:
                props = ", ".join(f"{k}: {v}" for k, v in
                                  list(e.properties.items())[:3])
                entity_lines.append(f"  • {e.name} ({e.entity_type.value}): {props}")
            parts.append("Known context:\n" + "\n".join(entity_lines))

        # Recent facts
        facts = [e for e in self._entities.values()
                 if e.entity_type == EntityType.FACT]
        facts.sort(key=lambda e: e.last_updated, reverse=True)
        if facts:
            fact_lines = [f"  • {f.properties.get('full_text', f.name)}"
                          for f in facts[:3]]
            parts.append("Known facts:\n" + "\n".join(fact_lines))

        return "\n\n".join(parts)

    def answer_about_user(self, query: str) -> Optional[str]:
        """
        Try to answer a question directly from the world model.
        Returns None if we don't know.
        """
        q = query.lower()

        # Name queries
        if any(w in q for w in ["my name", "who am i", "what's my name"]):
            if "name" in self._user_profile:
                return f"Your name is {self._user_profile['name']}."

        # Location queries
        if any(w in q for w in ["where am i", "my location", "where do i live"]):
            if "location" in self._user_profile:
                return f"You're in {self._user_profile['location']}."

        # Job queries
        if any(w in q for w in ["my job", "what do i do", "where do i work"]):
            if "job" in self._user_profile:
                return f"You work as {self._user_profile['job']}."

        return None

    def get_user_profile(self) -> dict:
        return dict(self._user_profile)

    def stats(self) -> dict:
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "events": len(self._events),
            "user_profile_fields": len(self._user_profile),
            "entity_types": {
                t.value: sum(1 for e in self._entities.values()
                             if e.entity_type == t)
                for t in EntityType
            },
        }

    # ══════════════════════════════════════════════════
    # Entity extraction (Phase 2 — heuristic, Phase 3 — NER model)
    # ══════════════════════════════════════════════════

    def _extract_user_facts(self, text: str):
        """Extract facts the user states about themselves."""
        t = text.strip()
        tl = t.lower()

        # Name: "my name is X" / "I'm X" / "call me X"
        name_patterns = [
            r"my name is ([A-Za-z][a-z]+)",
            r"i['']m ([A-Za-z][a-z]+)(?:\s|$)",
            r"call me ([A-Za-z][a-z]+)",
            r"i am ([A-Za-z][a-z]+)(?:\s|$)",
        ]
        for pattern in name_patterns:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                if name.lower() not in ["a", "the", "not", "just", "also"]:
                    self._user_profile["name"] = name
                    break

        # Job: "I work as X" / "I'm a X" / "I'm an X"
        job_patterns = [
            r"i work as (?:a |an )?(.+?)(?:\.|,|$)",
            r"i['']m (?:a |an )(.+?)(?:\.|,|$)",
            r"i work (?:at|for|in) (.+?)(?:\.|,|$)",
            r"my job is (.+?)(?:\.|,|$)",
        ]
        for pattern in job_patterns:
            m = re.search(pattern, tl)
            if m:
                job = m.group(1).strip()
                if len(job) < 50 and len(job) > 2:
                    self._user_profile["job"] = job
                    break

        # Location: "I live in X" / "I'm in X" / "I'm from X"
        loc_patterns = [
            r"i live in ([A-Za-z][a-zA-Z\s]+?)(?:\.|,|$)",
            r"i['']m (?:in|from) ([A-Za-z][a-zA-Z\s]+?)(?:\.|,|$)",
            r"based in ([A-Za-z][a-zA-Z\s]+?)(?:\.|,|$)",
        ]
        for pattern in loc_patterns:
            m = re.search(pattern, t, re.IGNORECASE)
            if m:
                loc = m.group(1).strip()
                if len(loc) < 40:
                    self._user_profile["location"] = loc
                    break

        # Projects: "I'm working on X" / "my project is X"
        proj_patterns = [
            r"(?:working on|building|developing|creating) (.+?)(?:\.|,|$)",
            r"my project (?:is|called) (.+?)(?:\.|,|$)",
        ]
        for pattern in proj_patterns:
            m = re.search(pattern, tl)
            if m:
                project = m.group(1).strip()
                if len(project) < 60 and len(project) > 3:
                    projects = self._user_profile.get("projects", [])
                    if project not in projects:
                        projects.append(project)
                    self._user_profile["projects"] = projects[:10]
                    break

    def _extract_entities(self, text: str):
        """Extract named entities from text (heuristic NER)."""
        # Capitalized words that aren't at sentence start = likely entity names
        words = text.split()
        for i, word in enumerate(words):
            clean = re.sub(r'[^\w]', '', word)
            if (len(clean) > 2 and
                clean[0].isupper() and
                i > 0 and  # not sentence start
                clean.lower() not in {
                    "the", "a", "an", "is", "are", "was", "what",
                    "how", "why", "when", "where", "who", "which",
                    "this", "that", "these", "those", "can", "will",
                    "leanai", "i",
                }):
                entity_id = f"ent_{clean.lower()}"
                if entity_id not in self._entities:
                    self._entities[entity_id] = Entity(
                        id=entity_id,
                        name=clean,
                        entity_type=EntityType.CONCEPT,
                        properties={},
                    )
                else:
                    self._entities[entity_id].mention_count += 1

    # ══════════════════════════════════════════════════
    # Persistence
    # ══════════════════════════════════════════════════

    def _save(self):
        try:
            # Save entities
            entities_data = {}
            for eid, e in self._entities.items():
                d = asdict(e)
                d["entity_type"] = e.entity_type.value
                entities_data[eid] = d
            (self.path / "entities.json").write_text(
                json.dumps(entities_data, indent=2))

            # Save user profile
            (self.path / "profile.json").write_text(
                json.dumps(self._user_profile, indent=2))

            # Save relations
            relations_data = []
            for r in self._relations:
                d = asdict(r)
                d["relation_type"] = r.relation_type.value
                relations_data.append(d)
            (self.path / "relations.json").write_text(
                json.dumps(relations_data, indent=2))
        except Exception as e:
            pass  # Never crash on save failure

    def _load(self):
        try:
            # Load entities
            ent_file = self.path / "entities.json"
            if ent_file.exists():
                data = json.loads(ent_file.read_text())
                for eid, d in data.items():
                    d["entity_type"] = EntityType(d["entity_type"])
                    self._entities[eid] = Entity(**d)

            # Load user profile
            prof_file = self.path / "profile.json"
            if prof_file.exists():
                self._user_profile = json.loads(prof_file.read_text())

            # Load relations
            rel_file = self.path / "relations.json"
            if rel_file.exists():
                data = json.loads(rel_file.read_text())
                for d in data:
                    d["relation_type"] = RelationType(d["relation_type"])
                    self._relations.append(Relation(**d))
        except Exception:
            pass
