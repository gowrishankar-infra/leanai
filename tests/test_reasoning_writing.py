"""
Tests for LeanAI Reasoning Engine and Writing Engine.
"""

import pytest
from core.reasoning_engine import ReasoningEngine, ReasoningResult
from core.writing_engine import WritingEngine, WritingResult, _detect_doc_type, DOC_TYPES


# ── Mock model ────────────────────────────────────────────────────

def _mock_model(system: str, user: str) -> str:
    """Mock model that returns contextual responses."""
    s = system.lower()
    u = user.lower()
    if "critic" in s or "review" in s:
        return "No errors found. The reasoning is sound."
    if "outline" in s:
        return "1. Introduction\n2. Main Points\n  - Point A\n  - Point B\n3. Conclusion"
    if "editor" in s or "edit" in s:
        return "Improved version of the document with better clarity and flow."
    if "decompose" in s or "break" in s:
        return "Sub-problem 1: Define requirements\nSub-problem 2: Design architecture\nSub-problem 3: Implement"
    if "plan" in s:
        return "Goal: Complete the project\nPhase 1: Research\nPhase 2: Build\nPhase 3: Deploy"
    if "step by step" in u or "think" in u:
        return "Step 1: Analyze the problem\nStep 2: Consider options\nStep 3: The answer is X"
    if "write" in u or "draft" in u:
        return "This is a well-written document about the topic at hand. It covers the key points thoroughly."
    if "analyze" in u or "identify" in u:
        return "Type: article\nAudience: developers\nTone: professional\nKey points: AI, automation"
    return "A thoughtful response."


# ══════════════════════════════════════════════════════════════════
# Reasoning Engine Tests
# ══════════════════════════════════════════════════════════════════

class TestReasoningEngine:
    @pytest.fixture
    def engine(self):
        return ReasoningEngine(model_fn=_mock_model)

    def test_reason_3_pass(self, engine):
        result = engine.reason("Why do tides happen?")
        assert result.final_answer
        assert result.chain_of_thought
        assert result.critique
        assert result.num_passes >= 2  # may be 2 if "no errors found"

    def test_reason_1_pass(self, engine):
        result = engine.reason("Simple question", max_passes=1)
        assert result.num_passes == 1
        assert result.chain_of_thought
        assert result.technique == "chain-of-thought"

    def test_plan(self, engine):
        result = engine.plan("How to migrate to microservices")
        assert result.final_answer
        assert result.chain_of_thought  # has content
        assert result.num_passes >= 2

    def test_decompose(self, engine):
        result = engine.decompose("Design a distributed rate limiter")
        assert result.final_answer
        assert result.num_passes == 3
        assert result.technique == "decomposition + synthesis"

    def test_quick_reason(self, engine):
        result = engine.quick_reason("What is 2+2?")
        assert result.num_passes == 1

    def test_no_model_raises(self):
        engine = ReasoningEngine(model_fn=None)
        with pytest.raises(RuntimeError):
            engine.reason("test")

    def test_stats_tracking(self, engine):
        engine.reason("Q1")
        engine.plan("Q2")
        engine.decompose("Q3")
        s = engine.stats()
        assert s["total_queries"] == 3
        assert s["cot_queries"] == 1
        assert s["plan_queries"] == 1
        assert s["decompose_queries"] == 1

    def test_result_summary(self):
        r = ReasoningResult(
            final_answer="Answer", num_passes=3,
            total_time_ms=5000, technique="chain-of-thought + critique + refine",
        )
        s = r.summary()
        assert "3 passes" in s
        assert "5000ms" in s

    def test_critique_skips_refine_when_no_errors(self, engine):
        result = engine.reason("Simple correct question")
        # Mock critique returns "No errors found" so should skip pass 3
        assert result.num_passes == 2


# ══════════════════════════════════════════════════════════════════
# Writing Engine Tests
# ══════════════════════════════════════════════════════════════════

class TestDocTypeDetection:
    def test_essay(self):
        assert _detect_doc_type("write an essay about climate change") == "essay"

    def test_report(self):
        assert _detect_doc_type("create a report on Q3 performance") == "report"

    def test_blog(self):
        assert _detect_doc_type("write a blog post about Rust") == "blog"

    def test_email(self):
        assert _detect_doc_type("draft an email to the team") == "email"

    def test_proposal(self):
        assert _detect_doc_type("write a proposal for the new CI/CD pipeline") == "proposal"

    def test_readme(self):
        assert _detect_doc_type("create a README for the project") == "readme"

    def test_documentation(self):
        assert _detect_doc_type("write API documentation for the auth service") == "documentation"

    def test_general(self):
        assert _detect_doc_type("something random") == "general"


class TestDocTypes:
    def test_all_types_have_structure(self):
        for name, template in DOC_TYPES.items():
            assert "structure" in template
            assert "tone" in template
            assert "tips" in template


class TestWritingEngine:
    @pytest.fixture
    def engine(self):
        return WritingEngine(model_fn=_mock_model)

    def test_write_full_pipeline(self, engine):
        result = engine.write("Write an article about AI trends")
        assert result.final_text
        assert result.outline
        assert result.doc_type == "article"
        assert result.num_passes == 4
        assert result.word_count > 0

    def test_quick_write(self, engine):
        result = engine.quick_write("Write a README")
        assert result.final_text
        assert result.num_passes == 2

    def test_write_essay(self, engine):
        result = engine.write_essay("Impact of AI on education")
        assert result.doc_type == "essay"
        assert result.num_passes == 4

    def test_write_report(self, engine):
        result = engine.write_report("Q3 sales analysis")
        assert result.doc_type == "report"

    def test_write_email(self, engine):
        result = engine.write_email("Team standup cancellation")
        assert result.doc_type == "email"

    def test_write_proposal(self, engine):
        result = engine.write_proposal("New monitoring system")
        assert result.doc_type == "proposal"

    def test_write_readme(self, engine):
        result = engine.write_readme("My awesome project")
        assert result.doc_type == "readme"

    def test_write_docs(self, engine):
        result = engine.write_docs("Auth API endpoints")
        assert result.doc_type == "documentation"

    def test_no_model_raises(self):
        engine = WritingEngine(model_fn=None)
        with pytest.raises(RuntimeError):
            engine.write("test")

    def test_stats_tracking(self, engine):
        engine.write("article about X")
        engine.write_essay("topic Y")
        s = engine.stats()
        assert s["total_docs"] == 2
        assert s["total_words"] > 0

    def test_result_summary(self):
        r = WritingResult(
            final_text="Content here", doc_type="essay",
            num_passes=4, total_time_ms=10000, word_count=500,
        )
        s = r.summary()
        assert "essay" in s
        assert "500 words" in s
        assert "4 passes" in s

    def test_3_pass_mode(self, engine):
        result = engine.write("Short article", max_passes=3)
        assert result.num_passes == 3
