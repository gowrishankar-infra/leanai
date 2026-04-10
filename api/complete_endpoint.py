"""
LeanAI — Autocomplete API endpoint
Add this to your existing api/server.py or run standalone.

Provides /complete endpoint for VS Code inline completions.
Response time target: <50ms using project brain lookup.
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import time


class CompletionRequest(BaseModel):
    prefix: str
    language: str = "python"
    file_path: str = ""
    line: str = ""
    max_results: int = 8


class CompletionItem(BaseModel):
    text: str
    label: str
    kind: str
    detail: str
    sortOrder: int
    insertText: str


class CompletionResponse(BaseModel):
    completions: List[CompletionItem]
    time_ms: float
    source: str  # "brain", "language", "mixed"


def register_completion_routes(app: FastAPI, completer):
    """Register completion routes on an existing FastAPI app."""

    @app.post("/complete", response_model=CompletionResponse)
    async def complete(req: CompletionRequest):
        start = time.time()
        results = completer.complete(
            prefix=req.prefix,
            language=req.language,
            file_path=req.file_path,
            line=req.line,
            max_results=req.max_results,
        )
        elapsed = (time.time() - start) * 1000

        source = "brain" if any(r.sort_order == 0 for r in results) else "language"
        if any(r.sort_order == 0 for r in results) and any(r.sort_order > 0 for r in results):
            source = "mixed"

        return CompletionResponse(
            completions=[CompletionItem(**r.to_dict()) for r in results],
            time_ms=round(elapsed, 1),
            source=source,
        )

    @app.get("/complete/stats")
    async def complete_stats():
        return completer.stats()
