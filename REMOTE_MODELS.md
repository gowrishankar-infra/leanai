# Remote models (OpenAI-compatible / Ollama)

LeanAI can use a model served over HTTP instead of (or alongside) a local GGUF.
This is meant for the **no-GPU laptop → hosted Ollama box** case: your dev
machine stays light, the model runs elsewhere.

It works with anything that speaks the OpenAI API: **Ollama**, **llama.cpp
server**, **vLLM**, **LM Studio**, **LocalAI**, **OpenRouter**, **Together**,
**Groq**, and **OpenAI** itself.

A remote model behaves exactly like a local one inside LeanAI — you `/model`
switch to it, it streams, CodeEcho still works (with a safe fallback), and
auto-routing can pick it.

---

## 1. Quick start (guided — easiest)

The fastest way, no YAML required. Inside LeanAI:

```
/model connect
```

It asks for your server's address (you can type just an IP like
`192.168.1.50`), then **queries the server, lists the models it found, and you
pick one by number**. It guesses the prompt format, picks the right mode, writes
`endpoints.yaml` for you, and makes the model active. Done.

The installer offers the same wizard: run `python setup_leanai.py`, answer *no*
to "run models locally", and it walks you through the identical steps.

What you'll see:

```
  Connect a remote model (Ollama / OpenAI-compatible)
  Server address [localhost:11434]: 192.168.1.50
  Checking http://192.168.1.50:11434/v1 ...
  -> reachable — 3 model(s) found
  Models available on that server:
    1. qwen2.5-coder:7b
    2. llama3.1:8b
    3. gemma2:9b
  Pick one [1-3, default 1]: 1
  Short name to type in LeanAI [qwen2-5-coder-7b]:
  Saved to C:\Users\<you>\.leanai\endpoints.yaml
  Active model set: remote:qwen2-5-coder-7b
```

If the server can't be reached it tells you the usual cause (most often: Ollama
only listens on localhost until you set `OLLAMA_HOST=0.0.0.0` on the **server**
and restart it).

---

## 2. Manual setup (Ollama on another machine)

On the **host** (the machine with the GPU), make sure Ollama is running and the
model is pulled:

```bash
ollama pull qwen2.5-coder:7b
ollama serve            # exposes http://<host>:11434
```

On your **laptop**, either run the installer and answer *no* to "run models
locally", or create the config by hand. The config lives at:

- **Windows:** `C:\Users\<you>\.leanai\endpoints.yaml`
- **macOS/Linux:** `~/.leanai/endpoints.yaml`

Minimal `endpoints.yaml`:

```yaml
endpoints:
  - name: home-ollama
    base_url: http://192.168.1.50:11434/v1   # your host's IP + /v1
    mode: completions
    models:
      - id: qwen2.5-coder:7b                  # the Ollama tag
        alias: remote-coder                   # what you type in LeanAI
        prompt_format: chatml
        quality: 72
        speed: fast
```

Then in LeanAI:

```
/model test            # check the endpoint is reachable
/model remote-coder    # switch to it
```

`endpoints.example.yaml` in the repo has a fuller, commented example.

---

## 3. The installer path

`python setup_leanai.py` now asks:

```
Run models LOCALLY on this machine?
```

- **Yes** → installs the full set (including `llama-cpp-python`) and offers to
  download a local GGUF, same as before.
- **No** → installs the lighter `requirements-remote.txt` (no
  `llama-cpp-python`, no `huggingface-hub`) and walks you through writing
  `endpoints.yaml` for one remote model, which it also sets active.

The default answer follows GPU detection, so on a GPU-less laptop the remote
path is the default.

---

## 4. `completions` vs `chat` mode

This is the one choice that matters. LeanAI builds its own prompt (chatml /
gemma / llama3 / phi3) before sending it to the model.

- **`completions`** (default, recommended for Ollama / llama.cpp / vLLM / LM
  Studio): the formatted prompt is sent verbatim to `/v1/completions`. LeanAI's
  prompt formatting is preserved exactly — this is what local mode does too, so
  behaviour matches.
- **`chat`**: for servers that only expose `/v1/chat/completions` (notably
  **OpenAI's current models**). LeanAI parses its formatted prompt back into
  role messages and sends those. This parsing is best-effort; prefer
  `completions` whenever the server supports it.

If a model's output looks doubly-wrapped in chat markup, the server is applying
its own template on `/v1/completions` — switch that endpoint to `mode: chat`.

---

## 5. Config reference

```yaml
endpoints:
  - name: <endpoint-name>          # label only
    base_url: <http://host:port/v1>   # must end in /v1 for most servers
    mode: completions | chat       # default: completions
    api_key: <literal>             # optional; Ollama ignores it
    api_key_env: OPENAI_API_KEY    # OR read the key from an env var
    timeout: 180                   # seconds; raise for big/slow models
    models:
      - id: <model id/tag>         # required — as the server knows it
        alias: <short-name>        # required — lowercase, what you type
        prompt_format: chatml      # chatml | gemma | llama3 | phi3
        context: 8192              # ctx window LeanAI assumes
        quality: 72                # 0-100, used by auto-routing
        speed: fast                # fast | medium | slow | remote
        description: "..."
```

Notes:
- **Aliases must be lowercase** and must not collide with built-in model keys
  (`qwen-7b`, `gemma4-26b`, `qwen35-27b`, …). Colliding aliases are skipped with
  a warning.
- Any string field may interpolate an env var with `${VAR}`.
- A missing file, missing PyYAML, or a malformed entry **never** crashes
  LeanAI — it prints a short note and continues with whatever parsed cleanly.
  You can also use `endpoints.json` with the same structure if you'd rather not
  install PyYAML.

---

## 6. API keys & privacy

- For **Ollama and other self-hosted servers**, no key is needed. Your code
  never leaves your network.
- For **OpenAI / hosted providers**, set `api_key_env` and export the variable
  rather than writing the key into the file. Keys are never printed by LeanAI.
- Sending source to a third-party API (OpenAI, Groq, etc.) means **your code
  leaves your machine** — unlike your own Ollama box. Keep that in mind for
  private repos. This is the opposite of LeanAI's local-first default, so it's
  opt-in and per-endpoint.

---

## 7. `/model test`

```
/model test            # ping every configured endpoint
/model test remote-coder   # ping just one
```

It does a `GET /v1/models` and reports each endpoint as reachable or
UNREACHABLE. A `401/403` counts as *not reachable for use* (bad/missing key);
a connection error or timeout means the host is down or the URL is wrong.

---

## 8. What is and isn't supported

- Sampling params sent to the remote: `max_tokens`, `temperature`, `top_p`,
  `stop` (capped at 4, per the OpenAI spec).
- `top_k` and `repeat_penalty` are **not** sent — they aren't part of the
  OpenAI API. If your server needs them, set its own defaults server-side.
- CodeEcho's structured path probes for a `tokenize()` method the remote client
  doesn't have, so it cleanly falls back to streaming generation. Output is
  unaffected; you just don't get the token-level echo on remote models.
- Streaming uses standard SSE (`data: {...}` / `data: [DONE]`).

---

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/model test` → UNREACHABLE | Wrong `base_url`, host down, or firewall. Confirm `curl http://<host>:11434/v1/models` from the laptop. |
| UNREACHABLE only for OpenAI | Missing/blank key — check the `api_key_env` variable is exported. |
| Output looks double-templated | Server templates `/v1/completions`; set that endpoint to `mode: chat`. |
| `PyYAML is not installed` note | `pip install pyyaml`, or rename the file to `endpoints.json`. |
| Alias doesn't show in `/model` | Alias collides with a built-in key, or isn't lowercase — rename it. |
