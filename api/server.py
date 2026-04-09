"""
LeanAI Phase 5b — FastAPI REST API + Web UI
Exposes all LeanAI capabilities over HTTP at localhost:8000.

Endpoints:
  POST /chat          — normal query (routed through engine)
  POST /swarm         — 3-pass consensus query
  POST /build         — agentic multi-step project builder
  POST /run           — execute Python code
  POST /index         — index a project directory
  POST /ask           — semantic search across indexed codebase
  POST /remember      — store a fact in memory
  GET  /profile       — user profile from world model
  GET  /world         — world model entities
  GET  /status        — system status
  GET  /              — web UI
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── LeanAI imports ────────────────────────────────────────────────
from core.engine_v3 import LeanAIEngineV3, GenerationConfig
from tools.executor import CodeExecutor
from tools.indexer import ProjectIndexer
from agents.build_command import BuildHandler
from agents.pipeline import PipelineConfig
from swarm import SwarmConsensus


# ── Request/Response models ───────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    max_tokens: int = 512
    temperature: float = 0.1

class ChatResponse(BaseModel):
    text: str
    confidence: float
    confidence_label: str
    tier: str
    latency_ms: float
    from_memory: bool = False
    verified: bool = False
    code_executed: bool = False
    code_passed: bool = False
    code_output: str = ""

class SwarmRequest(BaseModel):
    message: str
    num_passes: int = 3

class SwarmResponse(BaseModel):
    text: str
    consensus_score: float
    confidence: float
    unanimous: bool
    num_passes: int
    latency_ms: float
    candidates: list = []

class BuildRequest(BaseModel):
    task: str
    workspace: Optional[str] = None

class BuildResponse(BaseModel):
    success: bool
    steps_passed: int
    steps_total: int
    files_created: list
    errors: list
    workspace: str
    time_seconds: float

class RunRequest(BaseModel):
    code: str

class RunResponse(BaseModel):
    success: bool
    output: str
    error: str = ""
    execution_time_ms: int = 0

class IndexRequest(BaseModel):
    path: str
    force: bool = False

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

class RememberRequest(BaseModel):
    fact: str

class StatusResponse(BaseModel):
    version: str
    model: str
    prompt_format: str
    threads: int
    memory_episodes: int
    memory_backend: str
    world_entities: int
    profile_fields: int
    training_pairs: int
    model_loaded: bool


# ── App initialization ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Initialize all LeanAI components on startup."""
    global engine, executor, indexer, build_handler, swarm

    print("[API] Initializing LeanAI engine...")
    engine = LeanAIEngineV3(verbose=False)

    executor = CodeExecutor()
    indexer = ProjectIndexer()
    build_handler = BuildHandler(model_fn=_model_fn, verbose=False)
    swarm = SwarmConsensus(model_fn=_swarm_model_fn, num_passes=3, verbose=False)

    print("[API] LeanAI ready at http://localhost:8000")
    print("[API] Web UI at http://localhost:8000/")
    print("[API] API docs at http://localhost:8000/docs")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="LeanAI",
    description="Fast, lightweight, self-improving AI — REST API",
    version="5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (set during lifespan startup)
engine: Optional[LeanAIEngineV3] = None
executor: Optional[CodeExecutor] = None
indexer: Optional[ProjectIndexer] = None
build_handler: Optional[BuildHandler] = None
swarm: Optional[SwarmConsensus] = None


def _model_fn(system_prompt: str, user_prompt: str) -> str:
    """Model function for build handler."""
    if not engine._model:
        engine._load_model()
    fmt = getattr(engine, "prompt_format", "chatml")
    if fmt == "chatml":
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        stop = ["<|im_end|>", "<|im_start|>"]
    else:
        prompt = (
            f"<|system|>\n{system_prompt}<|end|>\n"
            f"<|user|>\n{user_prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        stop = ["<|end|>", "<|user|>", "<|assistant|>"]
    result = engine._model(prompt, max_tokens=1024, temperature=0.1, stop=stop)
    return result["choices"][0]["text"].strip()


def _swarm_model_fn(prompt: str, temperature: float) -> str:
    """Model function for swarm consensus."""
    if not engine._model:
        engine._load_model()
    fmt = getattr(engine, "prompt_format", "chatml")
    if fmt == "chatml":
        full = (
            f"<|im_start|>system\nYou are a helpful, accurate AI assistant. Answer concisely.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        stop = ["<|im_end|>", "<|im_start|>"]
    else:
        full = (
            f"<|system|>\nYou are a helpful, accurate AI assistant. Answer concisely.<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            f"<|assistant|>\n"
        )
        stop = ["<|end|>", "<|user|>", "<|assistant|>"]
    result = engine._model(full, max_tokens=512, temperature=temperature, stop=stop)
    return result["choices"][0]["text"].strip()


# ── Endpoints ─────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message to LeanAI."""
    try:
        start = time.time()
        config = GenerationConfig(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        resp = engine.generate(req.message, config=config)
        elapsed = time.time() - start

        confidence = getattr(resp, "confidence", 0.5)
        if isinstance(confidence, float) and 0 < confidence <= 1.0:
            confidence *= 100

        return ChatResponse(
            text=getattr(resp, "text", str(resp)),
            confidence=confidence,
            confidence_label=getattr(resp, "confidence_label", ""),
            tier=getattr(resp, "tier_used", "unknown"),
            latency_ms=getattr(resp, "latency_ms", elapsed * 1000),
            from_memory=getattr(resp, "answered_from_memory", False),
            verified=getattr(resp, "verified", False),
            code_executed=getattr(resp, "code_executed", False),
            code_passed=getattr(resp, "code_passed", False),
            code_output=getattr(resp, "code_output", ""),
        )
    except Exception as e:
        return ChatResponse(
            text=f"Error: {str(e)}", confidence=0, confidence_label="Error",
            tier="error", latency_ms=0,
        )


@app.post("/swarm", response_model=SwarmResponse)
async def swarm_query(req: SwarmRequest):
    """Run multi-pass swarm consensus."""
    try:
        if not engine._model:
            engine._load_model()
        result = swarm.query(req.message, base_confidence=50)
        candidates = [
            {"text": c.text[:200], "temperature": c.temperature, "agreement": c.agreement_score}
            for c in result.candidates
        ]
        return SwarmResponse(
            text=result.best_answer,
            consensus_score=result.consensus_score,
            confidence=result.confidence,
            unanimous=result.unanimous,
            num_passes=result.num_passes,
            latency_ms=result.total_latency_ms,
            candidates=candidates,
        )
    except Exception as e:
        return SwarmResponse(
            text=f"Error: {str(e)}", consensus_score=0, confidence=0,
            unanimous=False, num_passes=0, latency_ms=0,
        )


@app.post("/build", response_model=BuildResponse)
async def build_project(req: BuildRequest):
    """Build a complete project using agentic multi-step coding."""
    if not engine._model:
        engine._load_model()

    result = build_handler.execute_build(req.task, workspace=req.workspace)
    if result is None:
        raise HTTPException(status_code=500, detail="Build failed to start")

    return BuildResponse(
        success=result.success,
        steps_passed=result.plan.completed_steps,
        steps_total=result.plan.total_steps,
        files_created=result.files_created,
        errors=result.errors[:5],
        workspace=result.plan.workspace,
        time_seconds=result.total_time,
    )


@app.post("/run", response_model=RunResponse)
async def run_code(req: RunRequest):
    """Execute Python code in a sandbox."""
    try:
        if executor is None:
            return RunResponse(success=False, output="", error="Executor not initialized", execution_time_ms=0)
        result = executor.execute(req.code)
        # Handle both dict and dataclass results
        if isinstance(result, dict):
            return RunResponse(
                success=result.get("success", False),
                output=result.get("output", ""),
                error=result.get("error", ""),
                execution_time_ms=int(result.get("execution_time_ms", 0)),
            )
        else:
            return RunResponse(
                success=getattr(result, "success", False),
                output=getattr(result, "stdout", "") or getattr(result, "output", ""),
                error=getattr(result, "stderr", "") or getattr(result, "error", ""),
                execution_time_ms=int(getattr(result, "execution_time_ms", 0)),
            )
    except Exception as e:
        return RunResponse(success=False, output="", error=str(e), execution_time_ms=0)


@app.post("/index")
async def index_project(req: IndexRequest):
    """Index a project directory for semantic search."""
    path = os.path.abspath(req.path)
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
    stats = indexer.index_project(path, force=req.force)
    return stats


@app.post("/ask")
async def ask_codebase(req: AskRequest):
    """Search indexed codebase semantically."""
    results = indexer.search(req.query, top_k=req.top_k)
    return {"results": results, "count": len(results)}


@app.post("/remember")
async def remember_fact(req: RememberRequest):
    """Store a fact in persistent memory."""
    try:
        engine.remember(req.fact)
        return {"stored": True, "fact": req.fact}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profile")
async def get_profile():
    """Get user profile from world model."""
    try:
        profile = engine.get_profile()
        return {"profile": profile}
    except Exception as e:
        return {"profile": {}, "error": str(e)}


@app.get("/world")
async def get_world():
    """Get world model entities."""
    try:
        wm = engine.memory.world
        entities = getattr(wm, "entities", {})
        return {
            "entity_count": len(entities),
            "relation_count": len(getattr(wm, "relations", {})),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status."""
    mem_count = 0
    mem_backend = "unknown"
    try:
        mem_count = engine.memory.episodic.count()
        mem_backend = engine.memory.episodic.backend
    except Exception:
        pass

    world_count = 0
    profile_count = 0
    try:
        world_count = len(getattr(engine.memory.world, "entities", {}))
        profile_count = len(engine.memory.world.get_user_profile())
    except Exception:
        pass

    training_pairs = 0
    try:
        ts = engine.trainer.status()
        training_pairs = ts.get("total_pairs", 0)
    except Exception:
        pass

    return StatusResponse(
        version="5.0",
        model=getattr(engine, "model_name", "unknown"),
        prompt_format=getattr(engine, "prompt_format", "unknown"),
        threads=getattr(engine, "n_threads", 0),
        memory_episodes=mem_count,
        memory_backend=mem_backend,
        world_entities=world_count,
        profile_fields=profile_count,
        training_pairs=training_pairs,
        model_loaded=engine._model is not None,
    )


# ── Web UI ────────────────────────────────────────────────────────

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LeanAI</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3a;
    --text: #e1e4ed; --muted: #8b8fa3; --accent: #6c63ff;
    --accent-dim: #4a43cc; --green: #4ade80; --red: #f87171;
    --yellow: #fbbf24;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    background: var(--bg); color: var(--text);
    height: 100vh; display: flex; flex-direction: column;
  }
  header {
    padding: 16px 24px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header .badge {
    font-size: 11px; padding: 3px 8px; border-radius: 4px;
    background: var(--accent-dim); color: white;
  }
  header .status {
    margin-left: auto; font-size: 12px; color: var(--muted);
  }
  header .status .dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; background: var(--green); margin-right: 6px;
  }
  #chat {
    flex: 1; overflow-y: auto; padding: 24px;
    display: flex; flex-direction: column; gap: 16px;
  }
  .msg {
    max-width: 85%; padding: 12px 16px; border-radius: 12px;
    font-size: 14px; line-height: 1.6; white-space: pre-wrap;
    word-wrap: break-word;
  }
  .msg.user {
    align-self: flex-end; background: var(--accent);
    color: white; border-bottom-right-radius: 4px;
  }
  .msg.ai {
    align-self: flex-start; background: var(--surface);
    border: 1px solid var(--border); border-bottom-left-radius: 4px;
  }
  .msg .meta {
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid var(--border);
    font-size: 11px; color: var(--muted);
    display: flex; gap: 12px; flex-wrap: wrap;
  }
  .msg .meta .tag {
    padding: 2px 6px; border-radius: 3px;
    background: rgba(108, 99, 255, 0.15);
  }
  .msg .meta .good { color: var(--green); }
  .msg .meta .warn { color: var(--yellow); }
  .msg code {
    background: rgba(0,0,0,0.3); padding: 2px 5px;
    border-radius: 3px; font-size: 13px;
  }
  .msg pre {
    background: rgba(0,0,0,0.4); padding: 12px;
    border-radius: 6px; margin: 8px 0; overflow-x: auto;
    font-size: 13px;
  }
  .loading { color: var(--muted); font-style: italic; }
  #input-area {
    padding: 16px 24px; border-top: 1px solid var(--border);
    display: flex; gap: 12px;
  }
  #input-area input {
    flex: 1; padding: 12px 16px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); font-family: inherit; font-size: 14px;
    outline: none; transition: border-color 0.2s;
  }
  #input-area input:focus { border-color: var(--accent); }
  #input-area button {
    padding: 12px 24px; border-radius: 8px; border: none;
    background: var(--accent); color: white; font-family: inherit;
    font-size: 14px; font-weight: 600; cursor: pointer;
    transition: background 0.2s;
  }
  #input-area button:hover { background: var(--accent-dim); }
  #input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  .mode-bar {
    padding: 8px 24px; display: flex; gap: 8px;
    border-bottom: 1px solid var(--border);
  }
  .mode-bar button {
    padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: transparent; color: var(--muted); font-family: inherit;
    font-size: 12px; cursor: pointer; transition: all 0.2s;
  }
  .mode-bar button:hover { border-color: var(--accent); color: var(--text); }
  .mode-bar button.active {
    background: var(--accent); color: white; border-color: var(--accent);
  }
</style>
</head>
<body>

<header>
  <h1>LeanAI</h1>
  <span class="badge">v5</span>
  <span class="badge">Local</span>
  <div class="status"><span class="dot"></span><span id="model-status">Loading...</span></div>
</header>

<div class="mode-bar">
  <button class="active" onclick="setMode('chat')" id="btn-chat">Chat</button>
  <button onclick="setMode('swarm')" id="btn-swarm">Swarm (3-pass)</button>
  <button onclick="setMode('run')" id="btn-run">Run Code</button>
</div>

<div id="chat"></div>

<div id="input-area">
  <input type="text" id="input" placeholder="Ask LeanAI anything..." autofocus
         onkeydown="if(event.key==='Enter')send()">
  <button onclick="send()" id="send-btn">Send</button>
</div>

<script>
let MODE = 'chat';
let BUSY = false;

function setMode(m) {
  MODE = m;
  document.querySelectorAll('.mode-bar button').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + m).classList.add('active');
  const ph = {chat: 'Ask LeanAI anything...', swarm: 'Ask with 3-pass consensus...', run: 'Enter Python code to execute...'};
  document.getElementById('input').placeholder = ph[m] || '';
}

function addMsg(role, html, meta) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'meta';
    m.innerHTML = meta;
    div.appendChild(m);
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatCode(text) {
  return text.replace(/```(\\w+)?\\n([\\s\\S]*?)```/g, (_, lang, code) => {
    return '<pre><code>' + escHtml(code.trim()) + '</code></pre>';
  });
}

async function send() {
  if (BUSY) return;
  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';

  addMsg('user', escHtml(msg));
  const loading = addMsg('ai', '<span class="loading">Thinking...</span>');
  BUSY = true;
  document.getElementById('send-btn').disabled = true;

  try {
    let data;
    if (MODE === 'swarm') {
      const res = await fetch('/swarm', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg})
      });
      data = await res.json();
      loading.innerHTML = formatCode(escHtml(data.text || ''));
      const u = data.unanimous ? '<span class="good">UNANIMOUS</span>' : Math.round(data.consensus_score*100)+'% agreement';
      loading.innerHTML += '<div class="meta">' +
        '<span class="tag">Swarm: ' + data.num_passes + ' passes</span>' +
        '<span>' + u + '</span>' +
        '<span>Confidence: ' + Math.round(data.confidence) + '%</span>' +
        '<span>' + Math.round(data.latency_ms) + 'ms</span></div>';
    } else if (MODE === 'run') {
      const res = await fetch('/run', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({code: msg})
      });
      data = await res.json();
      const status = data.success ? '<span class="good">PASSED</span>' : '<span class="warn">FAILED</span>';
      loading.innerHTML = '<pre><code>' + escHtml(data.output || data.error || 'No output') + '</code></pre>';
      loading.innerHTML += '<div class="meta">' + status +
        '<span>' + data.execution_time_ms + 'ms</span></div>';
    } else {
      const res = await fetch('/chat', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg})
      });
      data = await res.json();
      loading.innerHTML = formatCode(escHtml(data.text || ''));
      let metaHtml = '<span class="tag">' + data.tier + '</span>' +
        '<span>Confidence: ' + Math.round(data.confidence) + '%</span>' +
        '<span>' + Math.round(data.latency_ms) + 'ms</span>';
      if (data.from_memory) metaHtml += '<span class="good">from memory</span>';
      if (data.verified) metaHtml += '<span class="good">verified</span>';
      if (data.code_executed && data.code_passed) metaHtml += '<span class="good">code verified</span>';
      loading.innerHTML += '<div class="meta">' + metaHtml + '</div>';
    }
  } catch (err) {
    loading.innerHTML = '<span class="warn">Error: ' + escHtml(err.message) + '</span>';
  }

  BUSY = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
}

// Check status on load
fetch('/status').then(r => r.json()).then(d => {
  document.getElementById('model-status').textContent =
    d.model_loaded ? d.model : 'Ready (loads on first query)';
}).catch(() => {
  document.getElementById('model-status').textContent = 'Connecting...';
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI."""
    return WEB_UI_HTML
