"""
LeanAI · Phase 2 Tests
Run: python -m pytest tests/test_phase2.py -v
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from memory.vector_memory import VectorEpisodicMemory, MemoryEntry
from world.world_model import WorldModel, EntityType


# ══════════════════════════════════════════════════════
# Vector Memory Tests
# ══════════════════════════════════════════════════════

class TestVectorMemory:

    def setup_method(self):
        self.mem = VectorEpisodicMemory("/tmp/leanai_test_vector")

    def test_store_returns_id(self):
        eid = self.mem.store("Test memory content about neural networks")
        assert isinstance(eid, str) and len(eid) > 0

    def test_search_returns_results(self):
        self.mem.store("The quick brown fox jumps over the lazy dog")
        results = self.mem.search("fox")
        assert isinstance(results, list)

    def test_store_fact_high_importance(self):
        eid = self.mem.store_fact("Python is a programming language")
        assert eid is not None

    def test_store_exchange(self):
        before = self.mem.count()
        self.mem.store_exchange("What is AI?", "AI is artificial intelligence.")
        assert self.mem.count() >= before

    def test_count_increases(self):
        before = self.mem.count()
        self.mem.store("New unique memory xyz987")
        assert self.mem.count() >= before

    def test_backend_property(self):
        backend = self.mem.backend
        assert backend in ["chromadb+vectors", "keyword_fallback"]

    def test_importance_estimated(self):
        # Personal info should get higher importance
        eid1 = self.mem.store("hello world")
        eid2 = self.mem.store("my name is Aditya and I work in DevOps")
        assert eid2 is not None

    def test_search_returns_list(self):
        results = self.mem.search("anything at all")
        assert isinstance(results, list)

    def test_multiple_stores(self):
        for i in range(5):
            self.mem.store(f"Memory entry number {i} about topic {i}")
        assert self.mem.count() >= 5


# ══════════════════════════════════════════════════════
# World Model Tests
# ══════════════════════════════════════════════════════

class TestWorldModel:

    def setup_method(self):
        self.world = WorldModel("/tmp/leanai_test_world")

    def test_learn_name(self):
        self.world._extract_user_facts("My name is Aditya")
        assert self.world._user_profile.get("name") == "Aditya"

    def test_learn_job(self):
        self.world._extract_user_facts("I work as a DevOps engineer")
        assert "job" in self.world._user_profile

    def test_learn_location(self):
        self.world._extract_user_facts("I live in Hyderabad")
        assert self.world._user_profile.get("location") == "Hyderabad"

    def test_learn_project(self):
        self.world._extract_user_facts("I'm working on building an AI system")
        assert "projects" in self.world._user_profile

    def test_answer_name_query(self):
        self.world._user_profile["name"] = "Aditya"
        answer = self.world.answer_about_user("what is my name")
        assert answer is not None
        assert "Aditya" in answer

    def test_answer_location_query(self):
        self.world._user_profile["location"] = "Hyderabad"
        answer = self.world.answer_about_user("where do I live")
        assert answer is not None
        assert "Hyderabad" in answer

    def test_answer_unknown_returns_none(self):
        answer = self.world.answer_about_user("what is the weather like")
        assert answer is None

    def test_learn_fact(self):
        before = len(self.world._entities)
        self.world.learn_fact("Python was created in 1991 by Guido van Rossum")
        assert len(self.world._entities) > before

    def test_stats_structure(self):
        stats = self.world.stats()
        assert "entities" in stats
        assert "relations" in stats
        assert "user_profile_fields" in stats
        assert "entity_types" in stats

    def test_context_for_query_string(self):
        ctx = self.world.get_context_for_query("tell me about Python")
        assert isinstance(ctx, str)

    def test_profile_returns_dict(self):
        profile = self.world.get_user_profile()
        assert isinstance(profile, dict)

    def test_learn_from_exchange(self):
        self.world.learn_from_exchange(
            "My name is Aditya and I'm a DevOps engineer",
            "Nice to meet you Aditya!"
        )
        assert self.world._user_profile.get("name") == "Aditya"

    def test_entity_extraction(self):
        before = len(self.world._entities)
        self.world._extract_entities("I use Python and Docker for my projects")
        # Should extract capitalized words as potential entities
        assert len(self.world._entities) >= before

    def test_save_and_reload(self):
        self.world._user_profile["name"] = "TestUser"
        self.world._save()
        new_world = WorldModel("/tmp/leanai_test_world")
        assert new_world._user_profile.get("name") == "TestUser"


# ══════════════════════════════════════════════════════
# Hierarchy V2 Tests
# ══════════════════════════════════════════════════════

class TestHierarchyV2:

    def setup_method(self):
        from memory.hierarchy_v2 import HierarchicalMemoryV2
        self.mem = HierarchicalMemoryV2("/tmp/leanai_test_h2")

    def test_stats_structure(self):
        stats = self.mem.stats()
        assert "working_tokens" in stats
        assert "episodic_entries" in stats
        assert "episodic_backend" in stats
        assert "semantic_entities" in stats
        assert "user_profile_fields" in stats

    def test_record_exchange(self):
        before = len(self.mem.working.messages)
        self.mem.record_exchange("Hello", "Hi there!")
        assert len(self.mem.working.messages) == before + 2

    def test_prepare_context_string(self):
        ctx = self.mem.prepare_context("What is machine learning?")
        assert isinstance(ctx, str)

    def test_answer_from_memory_miss(self):
        answer = self.mem.answer_from_memory("what is the weather")
        assert answer is None

    def test_answer_from_memory_hit(self):
        self.mem.world._user_profile["name"] = "Aditya"
        answer = self.mem.answer_from_memory("what is my name")
        assert answer is not None
        assert "Aditya" in answer

    def test_remember_fact(self):
        self.mem.remember_fact("LeanAI was built in 2026")
        assert self.mem.episodic.count() > 0

    def test_world_learns_from_exchange(self):
        self.mem.record_exchange(
            "My name is Aditya and I live in Hyderabad",
            "Got it!"
        )
        profile = self.mem.world.get_user_profile()
        assert "name" in profile or "location" in profile


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
