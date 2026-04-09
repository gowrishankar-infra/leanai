"""
LeanAI — Writing Engine
Produces high-quality documents, essays, reports, and articles.

Pipeline:
  1. Analyze: Understand the task, audience, tone, structure needed
  2. Outline: Create a detailed outline before writing
  3. Draft: Write the full document from the outline
  4. Edit: Self-review for clarity, accuracy, flow, and completeness

This 4-pass approach dramatically improves writing quality because
the model plans before writing (outline) and reviews after (edit).
Without this, models tend to ramble, lose structure, and miss points.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict


@dataclass
class WritingResult:
    """Result of the writing pipeline."""
    final_text: str
    outline: str = ""
    first_draft: str = ""
    edit_notes: str = ""
    doc_type: str = ""
    num_passes: int = 0
    total_time_ms: float = 0
    word_count: int = 0

    def summary(self) -> str:
        return (
            f"Writing: {self.doc_type} | {self.num_passes} passes | "
            f"{self.word_count} words | {self.total_time_ms:.0f}ms"
        )


# ── Document Type Templates ──────────────────────────────────────

DOC_TYPES = {
    "essay": {
        "structure": "Introduction → Thesis → Body paragraphs (each with topic sentence, evidence, analysis) → Counterargument → Conclusion",
        "tone": "academic, analytical, well-argued",
        "tips": "Use transition words. Support claims with evidence. Address counterarguments.",
    },
    "report": {
        "structure": "Executive Summary → Background → Methodology → Findings → Analysis → Recommendations → Conclusion",
        "tone": "professional, data-driven, objective",
        "tips": "Lead with key findings. Use specific numbers. Be actionable in recommendations.",
    },
    "article": {
        "structure": "Hook → Context → Main points → Supporting details → Conclusion with takeaway",
        "tone": "engaging, informative, accessible",
        "tips": "Start with a compelling hook. Use concrete examples. End with a clear takeaway.",
    },
    "documentation": {
        "structure": "Overview → Prerequisites → Installation/Setup → Usage → API Reference → Examples → Troubleshooting → FAQ",
        "tone": "clear, concise, technical but accessible",
        "tips": "Show code examples. Be precise about requirements. Include common pitfalls.",
    },
    "email": {
        "structure": "Subject → Greeting → Purpose → Details → Action items → Sign-off",
        "tone": "professional, concise, action-oriented",
        "tips": "State purpose in first sentence. Bold action items. Keep paragraphs short.",
    },
    "proposal": {
        "structure": "Problem Statement → Proposed Solution → Benefits → Implementation Plan → Timeline → Budget → Risks → Conclusion",
        "tone": "persuasive, professional, evidence-based",
        "tips": "Quantify benefits. Be specific about costs and timeline. Address objections proactively.",
    },
    "blog": {
        "structure": "Attention-grabbing title → Hook → Personal angle → Main content with subheadings → Practical takeaways → Call to action",
        "tone": "conversational, authentic, helpful",
        "tips": "Write like you talk. Use short paragraphs. Include personal experience.",
    },
    "readme": {
        "structure": "Project name + tagline → What it does → Quick start → Features → Installation → Usage → Configuration → Contributing → License",
        "tone": "clear, welcoming, developer-friendly",
        "tips": "Show, don't tell. First example should work in under 30 seconds. Use badges.",
    },
    "general": {
        "structure": "Introduction → Main points → Supporting details → Conclusion",
        "tone": "clear, well-organized, appropriate to topic",
        "tips": "Know your audience. Be concise. Use examples.",
    },
}


# ── Prompt Templates ──────────────────────────────────────────────

ANALYZE_SYSTEM = """You are a writing expert. Analyze the writing task and determine:
1. Document type (essay, report, article, documentation, email, proposal, blog, readme)
2. Target audience
3. Appropriate tone
4. Key points that must be covered
5. Approximate length needed

Be specific and concise."""

ANALYZE_USER = """Analyze this writing task:

{query}

Identify: document type, audience, tone, key points, and length."""

OUTLINE_SYSTEM = """You are a writing expert creating a detailed outline.
Structure: {structure}
Tone: {tone}
Tips: {tips}

Create a detailed outline with:
- Clear section headers
- Key points under each section (2-4 bullet points)
- Notes on what evidence/examples to include
- Estimated word count per section

The outline should be detailed enough that someone could write the full document from it."""

OUTLINE_USER = """Create a detailed outline for:

{query}

Audience: {audience}
Key points to cover: {key_points}

Write the outline:"""

DRAFT_SYSTEM = """You are a skilled writer producing a polished document.
Follow the outline exactly. Write with:
- {tone} tone
- Clear, well-structured paragraphs
- Smooth transitions between sections
- Concrete examples and evidence
- No filler or repetition
- Professional quality throughout

Write the complete document, not a summary."""

DRAFT_USER = """Write the complete document following this outline:

{outline}

Topic: {query}

Write the full document:"""

EDIT_SYSTEM = """You are a professional editor. Review the document and identify:
1. Unclear or awkward sentences — rewrite them
2. Logical gaps — what's missing
3. Repetitive content — what to cut
4. Weak transitions — improve flow
5. Grammar and style issues
6. Missing examples or evidence
7. Whether the conclusion is strong enough

Then rewrite the document with all improvements applied. Output ONLY the improved document."""

EDIT_USER = """Edit and improve this document. Fix all issues and produce the final version.

Original request: {query}

Document to edit:
{draft}

Produce the improved final version:"""


def _detect_doc_type(query: str) -> str:
    """Detect the document type from the query."""
    lower = query.lower()
    type_keywords = {
        "essay": ["essay", "argue", "thesis", "persuasive", "argumentative", "opinion piece"],
        "report": ["report", "analysis", "findings", "assessment", "evaluation"],
        "article": ["article", "write about", "piece on", "feature"],
        "documentation": ["documentation", "docs", "api docs", "technical docs", "manual", "guide"],
        "email": ["email", "message", "write to", "send to", "dear"],
        "proposal": ["proposal", "propose", "pitch", "business case", "recommend"],
        "blog": ["blog", "blog post", "post about"],
        "readme": ["readme", "read me", "project description"],
    }
    for doc_type, keywords in type_keywords.items():
        if any(kw in lower for kw in keywords):
            return doc_type
    return "general"


class WritingEngine:
    """
    Multi-pass writing engine for high-quality documents.
    
    Usage:
        writer = WritingEngine(model_fn=my_model)
        
        # Full 4-pass pipeline (best quality)
        result = writer.write("Write a blog post about why Rust is better than C++")
        
        # Quick mode (2 passes — outline + draft)
        result = writer.quick_write("Write a README for my project")
        
        # Specific document types
        result = writer.write_report("Q3 sales analysis")
        result = writer.write_essay("The impact of AI on education")
    """

    def __init__(self, model_fn: Optional[Callable] = None):
        self.model_fn = model_fn
        self._stats = {
            "total_docs": 0,
            "by_type": {},
            "avg_passes": 0,
            "total_words": 0,
        }

    def _call(self, system: str, user: str) -> str:
        if self.model_fn is None:
            raise RuntimeError("No model_fn provided to WritingEngine")
        return self.model_fn(system, user)

    def write(self, query: str, doc_type: Optional[str] = None,
              audience: str = "general", max_passes: int = 4,
              verbose: bool = False) -> WritingResult:
        """
        Full writing pipeline: analyze → outline → draft → edit.
        """
        start = time.time()
        self._stats["total_docs"] += 1

        # Step 1: Detect or use provided doc type
        if not doc_type:
            doc_type = _detect_doc_type(query)

        template = DOC_TYPES.get(doc_type, DOC_TYPES["general"])
        self._stats["by_type"][doc_type] = self._stats["by_type"].get(doc_type, 0) + 1

        # Step 2: Analyze (embedded in outline step for efficiency)
        if verbose:
            print("  [Step 1/4] Analyzing task...", flush=True)
        analysis = self._call(ANALYZE_SYSTEM, ANALYZE_USER.format(query=query))
        key_points = analysis  # use full analysis as context

        # Step 3: Outline
        if verbose:
            print(f"  [Step 2/4] Creating {doc_type} outline...", flush=True)
        outline = self._call(
            OUTLINE_SYSTEM.format(
                structure=template["structure"],
                tone=template["tone"],
                tips=template["tips"],
            ),
            OUTLINE_USER.format(
                query=query, audience=audience, key_points=key_points,
            ),
        )

        if max_passes <= 2:
            # Quick mode: draft from outline, no editing
            if verbose:
                print("  [Step 3/3] Writing draft...", flush=True)
            draft = self._call(
                DRAFT_SYSTEM.format(tone=template["tone"]),
                DRAFT_USER.format(outline=outline[:2000], query=query),
            )
            word_count = len(draft.split())
            self._stats["total_words"] += word_count
            return WritingResult(
                final_text=draft, outline=outline, first_draft=draft,
                doc_type=doc_type, num_passes=2,
                total_time_ms=(time.time() - start) * 1000,
                word_count=word_count,
            )

        # Step 4: Draft
        if verbose:
            print("  [Step 3/4] Writing full draft...", flush=True)
        draft = self._call(
            DRAFT_SYSTEM.format(tone=template["tone"]),
            DRAFT_USER.format(outline=outline[:2000], query=query),
        )

        if max_passes <= 3:
            word_count = len(draft.split())
            self._stats["total_words"] += word_count
            return WritingResult(
                final_text=draft, outline=outline, first_draft=draft,
                doc_type=doc_type, num_passes=3,
                total_time_ms=(time.time() - start) * 1000,
                word_count=word_count,
            )

        # Step 5: Edit and improve
        if verbose:
            print("  [Step 4/4] Self-editing and improving...", flush=True)
        final = self._call(
            EDIT_SYSTEM,
            EDIT_USER.format(query=query, draft=draft[:3000]),
        )

        elapsed = (time.time() - start) * 1000
        word_count = len(final.split())
        self._stats["total_words"] += word_count

        n = self._stats["total_docs"]
        self._stats["avg_passes"] = (self._stats["avg_passes"] * (n - 1) + 4) / n

        return WritingResult(
            final_text=final, outline=outline,
            first_draft=draft, edit_notes="Self-edited",
            doc_type=doc_type, num_passes=4,
            total_time_ms=elapsed, word_count=word_count,
        )

    def quick_write(self, query: str, doc_type: Optional[str] = None) -> WritingResult:
        """Quick 2-pass write: outline + draft, no editing."""
        return self.write(query, doc_type=doc_type, max_passes=2)

    def write_essay(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="essay")

    def write_report(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="report")

    def write_article(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="article")

    def write_docs(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="documentation")

    def write_email(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="email")

    def write_proposal(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="proposal")

    def write_readme(self, topic: str) -> WritingResult:
        return self.write(topic, doc_type="readme")

    def stats(self) -> dict:
        return dict(self._stats)
