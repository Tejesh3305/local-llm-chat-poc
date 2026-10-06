# Local LLM Chat (POC) — Setup and Implementation Guide

### FastAPI backend · React frontend · Ollama local models

> **Scope:** a proof of concept that runs open-weight LLMs locally, exposes them through a standard API, and provides a simple chat app. You can switch between local models (and later other runtimes or hosted providers) by editing configuration, not code. The architecture is kept deliberately small; [§14](#14-production-considerations) lists what to add before real production use.
>
> **Researched and verified:** 6 October 2026, against the official documentation in [§16](#16-references). Every file in [Appendix A](#appendix-a--complete-source-code) was run before publication:
> - backend tests: `pytest`, 7 passing
> - frontend: type-checked and built
> - end-to-end in a real browser, through the backend, against an OpenAI-compatible test server: streaming, model switching, stop, error and retry, new chat
>
> **Reference hardware:** Windows laptop with an Intel Core i5 13th Gen and an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM). The instructions also cover macOS and Linux.

---

## Table of contents

- [0. Summary of decisions](#0-summary-of-decisions)
- [1. Local LLM setup](#1-local-llm-setup)
- [2. Model installation and management](#2-model-installation-and-management)
- [3. Local LLM API](#3-local-llm-api)
- [4. LLM switching architecture](#4-llm-switching-architecture)
- [5. AI library / abstraction layer](#5-ai-library--abstraction-layer)
- [6. Simple chat application — project plan](#6-simple-chat-application--project-plan)
- [7. Backend architecture](#7-backend-architecture)
- [8. API implementation](#8-api-implementation)
- [9. Local model switching (step by step)](#9-local-model-switching-step-by-step)
- [10. Project structure](#10-project-structure)
- [11. Configuration](#11-configuration)
- [12. Development workflow (fresh machine → running app)](#12-development-workflow-fresh-machine--running-app)
- [13. Testing](#13-testing)
- [14. Production considerations](#14-production-considerations)
- [15. Comparison of available approaches](#15-comparison-of-available-approaches)
- [16. References](#16-references)
- [Appendix A — Complete source code](#appendix-a--complete-source-code)
- [Appendix B — Troubleshooting](#appendix-b--troubleshooting)

---

## 0. Summary of decisions

The options are compared in [§15](#15-comparison-of-available-approaches). The selected POC stack:

| Concern | Decision | Why |
|---|---|---|
| Local runtime | **Ollama** (llama.cpp `llama-server` as an optional alternative) | Native Windows, macOS and Linux installers. Uses the NVIDIA GPU automatically. Downloads a model with one command. Loads whichever model a request names. Exposes an OpenAI-compatible API. |
| API standard | **OpenAI-compatible Chat Completions** (`/v1/chat/completions`, `/v1/models`) | Supported by Ollama, llama.cpp, LM Studio, vLLM and many hosted providers. Changing runtime means changing a URL. |
| Backend | **Python 3.12+ with FastAPI** (run by Uvicorn) | Little code. Pydantic validation and interactive API docs (`/docs`) come built in. Native Server-Sent Events (`EventSourceResponse`) for streaming replies. |
| AI abstraction | **Official OpenAI Python SDK** (`AsyncOpenAI`) behind a ~100-line `ModelRegistry` | One client type works with every runtime above. The registry maps short model names to a runtime and its model name, taken from `models.json`. Only `llm.py` would change if you later need a non-OpenAI provider (via LiteLLM or Pydantic AI). |
| Frontend | **React 19 + Vite + TypeScript** | Small app with one chat hook (`useChat.ts`) that reads the stream with `fetch`. No UI framework or AI-specific frontend library. |
| Configuration | **`models.json`** (which models exist) **+ `.env`** (runtime URLs, ports, limits), validated by Pydantic at startup | No model names, URLs or ports hard-coded in application code. |
| Storage | **In memory** (conversations are lost on restart) | Enough for a POC; [§14](#14-production-considerations) covers moving to a database. |

### Architecture at a glance

```mermaid
flowchart LR
  subgraph Browser
    UI["React app<br/>model dropdown · chat · Stop / Retry"]
  end
  subgraph Backend["FastAPI backend :8000"]
    API["main.py<br/>/api/chat · /api/models · /api/conversations · /api/health"]
    REG["llm.py · ModelRegistry<br/>short model name → runtime + model name"]
    ST[("store.py<br/>in-memory conversations")]
  end
  subgraph Runtime["Local LLM runtime"]
    OL["Ollama :11434/v1<br/>gemma4:e4b · nemotron-3-nano:4b"]
    LC["llama.cpp :8080/v1<br/>(optional)"]
  end
  UI -- "POST /api/chat {message, model, conversation_id}<br/>← SSE: start · delta… · done" --> API
  API --> REG
  API --> ST
  REG -- "OpenAI Chat Completions (stream)" --> OL
  REG -.-> LC
```

The browser never talks to Ollama directly and never sends runtime model names. It sends a **short model name** such as `gemma4-e4b`, which the backend looks up in `models.json`. Only models listed there can be used.

---

## 1. Local LLM setup

### 1.1 Recommended runtime: Ollama

**Why Ollama for this project** (see [§15](#15-comparison-of-available-approaches) for the full comparison):

- Native Windows application with NVIDIA and AMD GPU support; after install it runs in the background and serves its API on `http://localhost:11434`.
- One command downloads a model in the right quantization from the Ollama library (`ollama pull gemma4:e4b`); `ollama ls / ps / rm / stop` cover the rest of the lifecycle.
- Serves an **OpenAI-compatible API** (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses`) in addition to its native `/api/*` API.
- Loads the requested model on demand and unloads idle models after a keep-alive period — model switching is just a different `model` value in the request.
- The local API does not require authentication and binds to `127.0.0.1` by default, which is the safe default for a single developer machine.

**When to choose something else:** use **llama.cpp `llama-server`** when you need fine-grained control (exact GGUF file, text-only builds without the vision projector, precise GPU-layer/KV-cache flags) — the app supports it with a config change only ([§9.4](#94-switch-runtime-ollama--llamacpp)). Use **vLLM** for multi-user GPU servers on Linux ([§14](#14-production-considerations)).

### 1.2 Where to download

| OS | Official download / install |
|---|---|
| Windows 10 22H2+ / 11 | `OllamaSetup.exe` from <https://ollama.com/download> |
| macOS 14 (Sonoma)+ | `Ollama.dmg` from <https://ollama.com/download> |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

### 1.3 System requirements

| Item | Requirement (from the official Ollama docs) | Notes for the reference laptop |
|---|---|---|
| Windows | Windows 10 22H2 or newer (Home or Pro) | Installs in your user profile; **no Administrator rights needed**. |
| macOS | macOS Sonoma (v14) or newer; Apple Silicon (CPU+GPU) or Intel (CPU only) | |
| Linux | x86-64 / arm64; systemd service installed by the script | |
| NVIDIA GPU | Compute capability **5.0+** and driver **550+** (Windows docs: **551.61+**). Compute capability 5.0–6.2 needs driver 570+ | RTX 40-series = compute capability 8.9 ✔. Update the NVIDIA driver before installing. |
| AMD GPU | ROCm v7 driver stack, or Vulkan | |
| Disk | ≥ 4 GB for the binaries + space for models (3–10 GB each for the models in this guide) | Models can be moved with `OLLAMA_MODELS`. |
| RAM | 16 GB recommended for 4B–8B models | |

### 1.4 Installation steps

**Windows (PowerShell)**

```powershell
# 1. Update the NVIDIA driver (GeForce/Studio driver from nvidia.com), then check it:
nvidia-smi

# 2. Download and run OllamaSetup.exe from https://ollama.com/download
#    (optional: install somewhere other than your home folder)
#    .\OllamaSetup.exe /DIR="D:\Apps\Ollama"

# 3. Open a NEW terminal and verify
ollama -v
curl.exe http://localhost:11434/api/version
```

**macOS**

```bash
# Drag Ollama.app into /Applications, start it once, then:
ollama -v
curl http://localhost:11434/api/version
```

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
nvidia-smi                 # NVIDIA: driver must be visible
ollama -v
systemctl status ollama    # the installer registers a systemd service
curl http://localhost:11434/api/version
```

### 1.5 Where models and files are stored

| OS | Models (default) | Logs | Change model location |
|---|---|---|---|
| Windows | `C:\Users\<you>\.ollama\models` | `%LOCALAPPDATA%\Ollama\server.log` (also `app.log`, `upgrade.log`) | User env var `OLLAMA_MODELS`, then quit & restart Ollama from the Start menu |
| macOS | `~/.ollama/models` | `~/.ollama/logs/server.log` | `launchctl setenv OLLAMA_MODELS /path`, restart app |
| Linux | `/usr/share/ollama/.ollama/models` | `journalctl -e -u ollama` | `sudo systemctl edit ollama` → `Environment="OLLAMA_MODELS=/path"`; `sudo chown -R ollama:ollama /path` |

Binaries on Windows live in `%LOCALAPPDATA%\Programs\Ollama` (added to your user `PATH`).

### 1.6 Configure Ollama for a 6 GB GPU

Ollama is configured with **environment variables on the server process** (not in this project). Recommended values for the reference laptop:

| Variable | Value | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `127.0.0.1:11434` | Default. Keep the runtime reachable only from this machine. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Only one model in VRAM at a time; switching unloads the previous one (two 4B models do not fit together in 6 GB). |
| `OLLAMA_NUM_PARALLEL` | `1` | One generation at a time per model. Memory scales with `NUM_PARALLEL × CONTEXT_LENGTH`. |
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Default context is 4k on GPUs with < 24 GiB VRAM; 8k is a good chat default that still fits. |
| `OLLAMA_FLASH_ATTENTION` | `1` | Lower memory use as context grows (Ollama enables it automatically where supported; this forces it). |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Halves KV-cache memory with negligible quality loss (requires flash attention). |
| `OLLAMA_KEEP_ALIVE` | `10m` | Unload a model after 10 idle minutes, freeing the GPU. |
| `OLLAMA_MAX_QUEUE` | `64` | Requests beyond this get HTTP 503 instead of waiting forever (default 512). |
| `OLLAMA_NO_CLOUD` | `1` *(optional)* | Local-only mode: disables Ollama's cloud models and web search. |

**Windows:** quit Ollama from the tray → *Settings → "Edit environment variables for your account"* → add the variables → start Ollama from the Start menu. Or from PowerShell:

```powershell
setx OLLAMA_MAX_LOADED_MODELS 1
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_CONTEXT_LENGTH 8192
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
setx OLLAMA_KEEP_ALIVE 10m
# Quit Ollama from the system tray and start it again so it picks these up.
```

**macOS:** `launchctl setenv OLLAMA_CONTEXT_LENGTH 8192` (repeat per variable), then restart the app.
**Linux:** `sudo systemctl edit ollama`, add `Environment="OLLAMA_CONTEXT_LENGTH=8192"` lines under `[Service]`, then `sudo systemctl daemon-reload && sudo systemctl restart ollama`.

Run `ollama serve --help` to list every supported variable for your installed version.

---

## 2. Model installation and management

### 2.1 Command reference

| Task | Command | Notes |
|---|---|---|
| Download / install | `ollama pull gemma4:e4b` | `name:tag`; the tag selects size/quantization. |
| Update a model | `ollama pull gemma4:e4b` | Re-pulling fetches only changed layers. |
| List installed | `ollama ls` | Name, ID, size, modified. |
| Run interactively | `ollama run gemma4:e4b` | `/bye` to exit; `--verbose` prints tokens/s. |
| One-shot prompt | `ollama run gemma4:e4b "Explain TCP in 2 lines"` | |
| Show loaded models | `ollama ps` | `PROCESSOR` must say **`100% GPU`** for good speed. |
| Unload from memory | `ollama stop gemma4:e4b` | Frees VRAM immediately. |
| Inspect a model | `ollama show gemma4:e4b` | Parameters, context length, capabilities, license. |
| Remove | `ollama rm gemma4:e4b` | Deletes the files from disk. |
| Copy / alias | `ollama cp gemma4:e4b my-chat` | |
| Create a variant | `ollama create gemma4-e4b-8k -f Modelfile` | e.g. fixed `num_ctx` or system prompt. |
| Preload (warm-up) | `ollama run gemma4:e4b ""` | Avoids the first-request load delay. |
| Upgrade Ollama | Windows/macOS: tray icon → *Restart to update*; Linux: re-run the install script | |

### 2.2 Models for this project (6 GB VRAM)

The two models recommended for this hardware, both from non-Chinese vendors:

| Public id (app) | Ollama tag | Vendor / license | Download size | Use it for |
|---|---|---|---|---|
| `gemma4-e4b` | `gemma4:e4b` | Google, Apache 2.0 | 6.6 GB *(Ollama build includes image input)* | Default general chat |
| `nemotron-nano-4b` | `nemotron-3-nano:4b` | NVIDIA, NVIDIA Open Model License | 2.8 GB | Coding and step-by-step reasoning (thinking + tools capable) |

Lighter options for development, CI or weaker machines: `gemma4:e2b` (4.6 GB) and `gemma4:e2b-it-qat` (4.3 GB).

```bash
ollama pull gemma4:e4b
ollama pull nemotron-3-nano:4b
ollama ls
```

> **Important for 6 GB GPUs — verify full GPU offload.** Ollama's `gemma4:e4b` package bundles the vision encoder, so its footprint is close to the 6 GB limit. After the first chat run `ollama ps`:
> - `100% GPU` → good.
> - `xx%/yy% CPU/GPU` → part of the model runs on the CPU (slow, high CPU load). Fixes, in order: lower `OLLAMA_CONTEXT_LENGTH` (e.g. 4096), keep `OLLAMA_KV_CACHE_TYPE=q8_0`, close other GPU apps; or use `gemma4:e2b`; or serve a **text-only** Gemma 4 E4B Q4_K_M GGUF (≈5.0 GB) through llama.cpp ([§3.7](#37-alternative-runtime-llamacpp-llama-server)), which this app supports with a configuration change only.

### 2.3 Custom variants (fixed context or system prompt)

The OpenAI-compatible API has no field for context size, so the official way to pin one for a model is a derived model:

```text
# Modelfile
FROM gemma4:e4b
PARAMETER num_ctx 8192
```

```bash
ollama create gemma4-e4b-8k -f Modelfile
ollama run gemma4-e4b-8k "hello"
```

Then reference `gemma4-e4b-8k` as `provider_model` in `models.json` (see [§9](#9-local-model-switching-step-by-step)).

### 2.4 Switching between models (runtime level)

There is no "active model" setting in Ollama — **each request names its model**. With `OLLAMA_MAX_LOADED_MODELS=1`, a request for model B unloads model A and loads B (a few seconds); later requests to B are immediate until it idles out. The application exploits exactly this: the selected model id travels with each chat request.

---

## 3. Local LLM API

### 3.1 Which API standard

Use the **OpenAI-compatible Chat Completions API** for all inference:

- Ollama, llama.cpp `llama-server`, LM Studio and vLLM all implement it, as do most hosted providers. Changing runtime = changing the base URL.
- The official OpenAI Python SDK works with it unchanged, by setting `base_url`.
- Streaming uses standard Server-Sent Events.

Ollama's **native** API (`/api/*`) is used only for operational tasks the OpenAI spec doesn't cover (pull, delete, list running models, version).

### 3.2 Starting the API server

Ollama's API server **is** the Ollama background process:

- Windows / macOS: it starts with the app (tray / menu-bar icon).
- Linux: `sudo systemctl start ollama` (enabled at boot by the installer).
- Manual / foreground (any OS, e.g. to see logs): quit the app, then `ollama serve`.

Base URLs:

| API | Base URL |
|---|---|
| OpenAI-compatible | `http://localhost:11434/v1` |
| Native | `http://localhost:11434/api` |

### 3.3 Required configuration

- No API key is needed for the local server; OpenAI client libraries require *some* key value, which Ollama ignores.
- Keep `OLLAMA_HOST=127.0.0.1:11434` (default). Only the backend on the same machine should reach it.
- CORS (`OLLAMA_ORIGINS`) is **not** needed in this architecture because the browser never calls Ollama directly.
- Context size: `OLLAMA_CONTEXT_LENGTH` (server-wide) or a derived model ([§2.3](#23-custom-variants-fixed-context-or-system-prompt)).

### 3.4 Endpoints

| Endpoint | Purpose | Used by this app |
|---|---|---|
| `POST /v1/chat/completions` | Chat (streaming, JSON mode, tools, vision, `reasoning_effort`) | ✅ every chat turn |
| `GET /v1/models` | List installed models | ✅ availability check (`/api/models`, `/api/health`) |
| `GET /v1/models/{model}` | One model | |
| `POST /v1/embeddings` | Embeddings (for RAG later) | |
| `POST /v1/responses` | OpenAI Responses API (stateless only) | |
| `POST /v1/completions` | Legacy text completion | |
| `GET /api/version` | Runtime version | health checks / ops |
| `GET /api/tags` | Installed models (native format) | ops |
| `GET /api/ps` | Models currently loaded and where (GPU/CPU) | ops |
| `POST /api/pull` / `DELETE /api/delete` | Download / remove a model | ops |
| `POST /api/chat` / `POST /api/generate` | Native chat / generate (support `keep_alive`, `options.num_ctx`) | preload / unload |

Not supported by Ollama's OpenAI layer (as documented): `tool_choice`, `logit_bias`, `n`, `logprobs`, and stateful Responses (`previous_response_id`).

### 3.5 Authentication

- **Local Ollama API: none.** Security relies on binding to `127.0.0.1`. Never expose port 11434 to a network without a reverse proxy that adds authentication.
- **llama.cpp `llama-server`:** supports `--api-key KEY` (or `--api-key-file`); clients send `Authorization: Bearer KEY`. `GET /health` stays public.
- **This application's backend** has no authentication in the POC; user authentication belongs there before production ([§14](#14-production-considerations)).

### 3.6 Example requests

**List installed models**

```bash
curl http://localhost:11434/v1/models
```

**Chat completion (non-streaming)**

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e4b",
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": "Give me three uses of a hash map."}
    ],
    "temperature": 0.7,
    "max_tokens": 300
  }'
```

**Streaming (SSE)** — add `"stream": true`; usage totals arrive at the end when `stream_options.include_usage` is set:

```bash
curl -N http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nemotron-3-nano:4b","stream":true,"stream_options":{"include_usage":true},
       "messages":[{"role":"user","content":"Write a Python function that reverses a string."}]}'
```

**Windows PowerShell**

```powershell
$body = @{ model = "gemma4:e4b"; messages = @(@{ role = "user"; content = "Hello!" }) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://localhost:11434/v1/chat/completions -ContentType "application/json" -Body $body |
  Select-Object -ExpandProperty choices | ForEach-Object { $_.message.content }
```

**Official OpenAI Python SDK pointed at Ollama** (this is exactly what the backend does):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # key is required by the SDK, ignored by Ollama
reply = client.chat.completions.create(model="gemma4:e4b", messages=[{"role": "user", "content": "Hi"}])
print(reply.choices[0].message.content)
```

**Operational calls (native API)**

```bash
curl http://localhost:11434/api/ps                                   # what is loaded, on GPU or CPU
curl http://localhost:11434/api/generate -d '{"model":"gemma4:e4b","keep_alive":-1}'   # preload & keep loaded
curl http://localhost:11434/api/generate -d '{"model":"gemma4:e4b","keep_alive":0}'    # unload now
```

### 3.7 Alternative runtime: llama.cpp `llama-server`

Use this when you want an exact GGUF file (for example a **text-only** Gemma 4 E4B Q4_K_M that fits fully in 6 GB VRAM) or more low-level control. It exposes the same OpenAI-compatible endpoints plus a web UI.

```bash
# Install (pick one)
winget install llama.cpp             # Windows
brew install llama.cpp               # macOS / Linux
# or download a CUDA build from https://github.com/ggml-org/llama.cpp/releases

# Serve Gemma 4 E4B Q4_K_M (~5.0 GB) from Hugging Face, text-only, 8k context, all layers on GPU,
# q8_0 KV cache, API key required, listening on localhost only:
llama-server -hf unsloth/gemma-4-E4B-it-GGUF:Q4_K_M --no-mmproj \
  -c 8192 -ngl all -fa on -ctk q8_0 -ctv q8_0 -np 1 \
  --alias gemma4-e4b-gguf --host 127.0.0.1 --port 8080 --api-key "change-me-long-random-key"

curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/models -H "Authorization: Bearer change-me-long-random-key"
```

Notes: `-hf` downloads into llama.cpp's cache (`llama-server --cache-list` shows it); `--alias` sets the model name the API reports and expects. To serve several GGUF files from one process, llama-server also has a **router mode** (`--models-dir`, `--models-preset`, `--models-max`) — see the server README.

---

## 4. LLM switching architecture

### 4.1 Goals (kept simple for a POC)

1. **Switching models or runtimes is a configuration change.** Edit `models.json` or `.env`, never Python or React code.
2. **The frontend only knows short model names** (`gemma4-e4b`), never runtime model names (`gemma4:e4b`) or URLs.
3. **Only models listed in `models.json` can be called.** A request for anything else is rejected with `400`.

### 4.2 The three pieces

| Piece | File | What it holds |
|---|---|---|
| Model list | `backend/models.json` | Which models exist, which runtime ("provider") serves each one, the name that runtime uses, and per-model settings (system prompt, temperature, max tokens) |
| Runtime addresses | `backend/.env` | The base URL (and optional API key) of each runtime, e.g. `OLLAMA_BASE_URL=http://localhost:11434/v1` |
| Model registry | `backend/app/llm.py` | Reads both. Creates one OpenAI-compatible client per runtime. Turns a short model name into "this client + this runtime model name". |

```mermaid
flowchart LR
  A["Request: model = 'nemotron-nano-4b'"] --> B["models.json<br/>provider: ollama<br/>provider_model: nemotron-3-nano:4b"]
  B --> C[".env<br/>OLLAMA_BASE_URL = http://localhost:11434/v1"]
  C --> D["AsyncOpenAI(base_url)<br/>.chat.completions.create(model='nemotron-3-nano:4b', stream=True)"]
```

### 4.3 `models.json`

```json
{
  "default_model": "gemma4-e4b",
  "providers": {
    "ollama":   { "base_url_env": "OLLAMA_BASE_URL",   "api_key_env": "OLLAMA_API_KEY" },
    "llamacpp": { "base_url_env": "LLAMACPP_BASE_URL", "api_key_env": "LLAMACPP_API_KEY" }
  },
  "models": [
    { "id": "gemma4-e4b", "label": "Gemma 4 E4B", "description": "…", "provider": "ollama",
      "provider_model": "gemma4:e4b", "system_prompt": "You are a helpful, concise assistant.",
      "temperature": 0.7, "max_tokens": 1024 },
    { "id": "nemotron-nano-4b", "label": "Nemotron 3 Nano 4B", "description": "…", "provider": "ollama",
      "provider_model": "nemotron-3-nano:4b", "temperature": 0.6, "max_tokens": 2048 }
  ]
}
```

Rules, checked by Pydantic when the backend starts:

- `base_url_env` and `api_key_env` are **names of environment variables**, not URLs or keys.
- Model `id`s are unique and lowercase.
- Every model's `provider` exists, and `default_model` is one of the models.

A provider whose URL variable is not set is skipped. Its models then show as unavailable, and the app still starts.

### 4.4 Which model answers a message

1. The `model` sent with the message (the dropdown in the UI), otherwise
2. the model saved on the conversation, otherwise
3. `default_model` (which `DEFAULT_MODEL` in `.env` can override).

---

## 5. AI library / abstraction layer

### 5.1 Recommendation for this POC: the official OpenAI Python SDK plus a small registry

Every runtime this project may use (Ollama, llama.cpp, LM Studio, vLLM) speaks the OpenAI Chat Completions protocol. So one well-supported client covers all of them:

| Need | How `openai` (v3.x, `AsyncOpenAI`) covers it |
|---|---|
| Talk to any local runtime | `AsyncOpenAI(base_url=..., api_key=...)`. Only the URL differs between runtimes. |
| Streaming | `create(..., stream=True, stream_options={"include_usage": True})` returns an async stream of chunks plus token usage at the end. |
| Stop / cancel | `async with stream:` closes the connection to the runtime when the user presses Stop. |
| Errors | Typed exceptions (`APIConnectionError`, `APITimeoutError`, `NotFoundError`, `APIStatusError`) map cleanly to user-friendly messages. |
| Timeouts / retries | `timeout=` and `max_retries=` on the client. |
| Model discovery | `client.models.list()` shows which models the runtime has installed. |

The registry in `llm.py` (about 100 lines) adds what the SDK doesn't do:

- the mapping from short model names to runtime and runtime model name,
- per-model settings,
- one client per runtime,
- an availability check.

**Why not a bigger framework for a POC?** LiteLLM, Pydantic AI and LangChain all work with Ollama. But each adds a layer you don't need while every runtime already speaks the same protocol ([§15.2](#152-python-ai-libraries--abstraction-layers)). Because all model access sits in `llm.py`, you can adopt one later without touching routes or the frontend:

- **Need a provider without an OpenAI-compatible API** (e.g. Anthropic's native API)? Re-implement `stream_chat()` with **LiteLLM** (`litellm.acompletion(model="anthropic/...", stream=True)`) or **Pydantic AI**.
- **Need RAG or agents?** Add LangChain, LlamaIndex or Pydantic AI behind the same functions.

### 5.2 How it is integrated (excerpts — full file in Appendix A)

**Creating one client per configured runtime:**

```python
# backend/app/llm.py
for provider_id, provider in models_file.providers.items():
    base_url = os.getenv(provider.base_url_env, "").strip()
    if not base_url:
        continue  # runtime not configured -> its models show as unavailable
    api_key = os.getenv(provider.api_key_env, "") if provider.api_key_env else ""
    self.clients[provider_id] = AsyncOpenAI(
        base_url=base_url, api_key=api_key or "not-needed", timeout=timeout_seconds, max_retries=1
    )
```

**Streaming a reply:**

```python
# backend/app/llm.py
stream = await client.chat.completions.create(
    model=model.provider_model,           # e.g. "gemma4:e4b"
    messages=prompt,                      # system prompt + history + new message
    stream=True,
    stream_options={"include_usage": True},
    **options,                            # temperature / max_tokens from models.json
)
async with stream:
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

**Using it from the chat endpoint:**

```python
# backend/app/main.py
async for text in registry.stream_chat(ctx.model, history, result):
    yield {"type": "delta", "text": text}  # FastAPI sends each dict as one SSE event
```

---

## 6. Simple chat application — project plan

### 6.1 Features (POC scope)

| Feature | How |
|---|---|
| Chat interface | One React page: a header (title, model dropdown, **New chat**), the message list and an input box |
| Message history | Shown in the browser. The backend also keeps it per conversation and sends the last `MAX_HISTORY_MESSAGES` to the model as context. |
| User input | Textarea. **Enter** sends, **Shift+Enter** adds a new line. |
| AI responses | Streamed word by word over SSE. The model name and token count are shown under each reply. |
| Loading / streaming | "Thinking…" until the first word arrives. A **Stop** button while streaming; the partial answer is kept. |
| Error handling | Readable error banner with **Retry** and **Dismiss**. A warning when the selected model isn't installed, and a banner when the API is unreachable. |
| Model selection | Dropdown filled from `GET /api/models`; models that aren't installed are marked "(not installed)". The choice is sent with every message, so you can switch mid-conversation. |

### 6.2 Build plan

| Step | Deliverable | Done when |
|---|---|---|
| 1 | Ollama installed and configured, two models downloaded | `ollama run gemma4:e4b "hi"` answers; `ollama ps` shows `100% GPU` |
| 2 | FastAPI backend: config, model registry, `/api/health`, `/api/models` | `GET /api/models` lists both models with `"available": true` |
| 3 | Streaming `/api/chat` and in-memory conversations | `curl -N` shows `start`, `delta` and `done` events |
| 4 | React frontend | Chat works in the browser with both models; Stop, Retry and New chat work |
| 5 | Tests | `pytest` passes; manual checks in [§13](#13-testing) pass |

### 6.3 Screen layout

```text
┌───────────────────────────────────────────────────────────────────────┐
│ Local LLM Chat                  Model: [Gemma 4 E4B ▾]   [New chat]   │
├───────────────────────────────────────────────────────────────────────┤
│                                              ┌─────────────────────┐  │
│                                              │ user message        │  │
│                                              └─────────────────────┘  │
│ assistant reply streams here…                                         │
│ Gemma 4 E4B · 212 tokens                                              │
│ ⚠ error banner text                               Retry   Dismiss    │
├───────────────────────────────────────────────────────────────────────┤
│ [ Type a message…                                        ] [Send/Stop]│
└───────────────────────────────────────────────────────────────────────┘
```

---

## 7. Backend architecture

### 7.1 Who talks to whom

| Connection | How |
|---|---|
| Browser → FastAPI | JSON over HTTP. `POST /api/chat` answers with **Server-Sent Events** (`text/event-stream`). In development the browser calls the Vite dev server (`:5173`), which forwards `/api/*` to FastAPI (`:8000`), so CORS is not involved. |
| FastAPI → Ollama | OpenAI Chat Completions over HTTP with streaming (`POST {OLLAMA_BASE_URL}/chat/completions`), plus `GET {OLLAMA_BASE_URL}/models` to check which models are installed. Both stay on `localhost`. |
| Stop button | The browser aborts its request. FastAPI notices the disconnect and stops the stream, and `async with stream` closes the connection to Ollama, so generation stops. The partial answer is saved. |

### 7.2 One chat turn

```mermaid
sequenceDiagram
  participant B as Browser (useChat.ts)
  participant F as FastAPI /api/chat
  participant R as ModelRegistry (llm.py)
  participant O as Ollama /v1
  B->>F: POST {message, model:"nemotron-nano-4b", conversation_id}
  F->>F: prepare_chat(): validate body, look up conversation,<br/>pick model (request › conversation › default)
  Note over F: Unknown model → 400 JSON (before streaming starts)
  F-->>B: SSE start {conversation_id, model}
  F->>R: stream_chat(model, history + message)
  R->>O: chat.completions.create(model="nemotron-3-nano:4b", stream=True)
  O-->>R: chunks
  R-->>F: text pieces
  F-->>B: SSE delta {text} … (many)
  F-->>B: SSE done {finish_reason, input_tokens, output_tokens}
  F->>F: save user + assistant messages
```

### 7.3 How the selected model travels through the app

1. On load, the React app calls `GET /api/models` and preselects `default_model`.
2. The user picks a model in the dropdown, which sets a React state value.
3. `useChat.send(text, model)` posts `{ message, model, conversation_id }`.
4. The `prepare_chat` dependency validates the request and calls `registry.get(model)`. An unknown name returns **400**; a provider with no URL configured returns **503**. These are normal JSON errors, because they happen before streaming starts.
5. `registry.stream_chat()` calls the runtime with the runtime's own model name (`provider_model`).
6. The `start` event returns the model name. It is stored with the assistant message and becomes the conversation's model.

### 7.4 Backend files

| File | Responsibility |
|---|---|
| `app/__main__.py` | `python -m app` starts Uvicorn with host, port and reload taken from `.env` |
| `app/config.py` | Loads `.env`. Defines `Settings` and the `models.json` schema, both validated by Pydantic. |
| `app/llm.py` | `ModelRegistry`: one client per runtime, model lookup, availability check, streaming, error translation |
| `app/main.py` | FastAPI app, CORS and all routes. `prepare_chat` validates requests before streaming starts. |
| `app/schemas.py` | Request and response models (also used for the `/docs` page) |
| `app/store.py` | In-memory conversations |

---

## 8. API implementation

FastAPI generates interactive API docs at **<http://localhost:8000/docs>**.

### 8.1 Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| `POST` | `/api/chat` | Send a message and stream the reply. Optional `model` selects the model; optional `conversation_id` continues a conversation. | `200` SSE |
| `GET` | `/api/models` | Models from `models.json`, the default, and whether each is installed | `200` |
| `GET` | `/api/conversations` | Conversation summaries, newest first | `200` |
| `GET` | `/api/conversations/{id}` | One conversation with its messages | `200` / `404` |
| `PATCH` | `/api/conversations/{id}` | Select the model for the next turns (`{"model": "…"}`) and/or rename it (`{"title": "…"}`) | `200` |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation | `204` |
| `GET` | `/api/health` | Can the backend reach each runtime? | `200` ok / `503` degraded |

Errors use FastAPI's standard shape: `{"detail": "Unknown model 'gpt-9'. See GET /api/models."}`. Field validation errors return `422` with a list in `detail`.

### 8.2 `POST /api/chat`

Request:

```json
{ "message": "Write a Python function that reverses a string",
  "model": "nemotron-nano-4b",
  "conversation_id": "5d3f88c74cce4adc98a8426f60b9353a" }
```

- `message` (required): 1 to `MAX_MESSAGE_CHARS` characters.
- `model` (optional): a short model name from `GET /api/models`.
- `conversation_id` (optional): omit it to start a new conversation, then send back the id from the `start` event to continue.

Response — Server-Sent Events, one JSON object per event. This is real output from a test run:

```text
data: {"type":"start","conversation_id":"5d3f88c74cce4adc98a8426f60b9353a","model":"nemotron-nano-4b"}

data: {"type":"delta","text":"[nemotron-3-nano:4b] "}

data: {"type":"delta","text":"You "}

data: {"type":"delta","text":"said: "}

data: {"type":"delta","text":"Hello "}

data: {"type":"delta","text":"there"}

data: {"type":"done","finish_reason":"stop","input_tokens":10,"output_tokens":5}
```

(The test server echoes the message; a real model returns its answer in the `delta` events.) If the runtime fails after streaming has started, the last event is `{"type":"error","message":"<user-friendly text>"}`. FastAPI also sends `: ping` comment lines every 15 seconds on long streams; clients ignore them.

```bash
curl -N http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"Hello","model":"gemma4-e4b"}'
```

### 8.3 `GET /api/models`

```json
{
  "default_model": "gemma4-e4b",
  "models": [
    { "id": "gemma4-e4b", "label": "Gemma 4 E4B", "description": "…", "provider": "ollama", "is_default": true,  "available": true },
    { "id": "nemotron-nano-4b", "label": "Nemotron 3 Nano 4B", "description": "…", "provider": "ollama", "is_default": false, "available": false }
  ]
}
```

`available: false` means the runtime is unreachable or doesn't have that model installed.

### 8.4 Conversations and model selection

```bash
curl http://localhost:8000/api/conversations
curl http://localhost:8000/api/conversations/<id>
curl -X PATCH http://localhost:8000/api/conversations/<id> -H "Content-Type: application/json" -d '{"model":"nemotron-nano-4b"}'
curl -X DELETE http://localhost:8000/api/conversations/<id>
```

After the `PATCH`, any message sent to that conversation without a `model` field uses `nemotron-nano-4b`.

### 8.5 How the chat endpoint is built

The key point is the split into two steps. **Validation runs in a dependency** (`prepare_chat`), before the stream starts, so bad requests still get a proper `4xx` status. **Streaming** is an async generator with `response_class=EventSourceResponse`, available since FastAPI 0.135:

```python
@app.post("/api/chat", response_class=EventSourceResponse)
async def chat(ctx: ChatContext = Depends(prepare_chat), ...) -> AsyncIterable[dict]:
    yield {"type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id}
    parts: list[str] = []
    try:
        async for text in registry.stream_chat(ctx.model, history, result):
            parts.append(text)
            yield {"type": "delta", "text": text}
    except LLMError as error:
        yield {"type": "error", "message": error.user_message}
        return
    finally:
        if parts:  # also runs when the user presses Stop
            store.add_turn(ctx.conversation, user_message, assistant_message)
    yield {"type": "done", ...}
```

---

## 9. Local model switching (step by step)

None of these procedures touches a `.py` or `.tsx` file.

### 9.1 Switch per message, from the UI (no restart)

1. Check both models are installed: `ollama ls`.
2. Pick another model in the **Model** dropdown and send a message.
3. Ollama unloads the previous model and loads the new one, so the first reply takes a few extra seconds.
4. Check the line under the reply shows the new model, and `ollama ps` lists it at `100% GPU`.

### 9.2 Change the default model

- In `backend/.env`, set `DEFAULT_MODEL=nemotron-nano-4b` and restart the backend; or
- change `"default_model"` in `backend/models.json` and restart.

### 9.3 Add a new local model

1. `ollama pull gemma4:e2b`
2. Add an entry to `backend/models.json`:
   ```json
   { "id": "gemma4-e2b", "label": "Gemma 4 E2B (fast)", "description": "Smaller, faster Gemma 4.",
     "provider": "ollama", "provider_model": "gemma4:e2b", "temperature": 0.7, "max_tokens": 1024 }
   ```
3. Restart the backend. If the entry is invalid, startup stops with an error naming the bad field.
4. Reload the browser. The model is now in the dropdown.

To remove a model, delete its entry and restart. Run `ollama rm <tag>` too if you want the files gone.

### 9.4 Switch runtime: Ollama → llama.cpp

1. Start `llama-server` as in [§3.7](#37-alternative-runtime-llamacpp-llama-server), with alias `gemma4-e4b-gguf` and port 8080.
2. In `backend/.env`:
   ```dotenv
   LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1
   LLAMACPP_API_KEY=change-me-long-random-key
   DEFAULT_MODEL=gemma4-e4b-gguf
   ```
3. Add to `models` in `backend/models.json`:
   ```json
   { "id": "gemma4-e4b-gguf", "label": "Gemma 4 E4B (llama.cpp, text-only)", "provider": "llamacpp",
     "provider_model": "gemma4-e4b-gguf", "temperature": 0.7, "max_tokens": 1024 }
   ```
4. Restart the backend. Models from both runtimes appear in the dropdown, and the llama.cpp one is the default.

This was tested while writing the guide: with a second OpenAI-compatible server on port 11600, `GET /api/models` listed models from both providers. `/api/chat` then answered from the llama.cpp-configured model, with no code changes.

### 9.5 Later: a hosted provider

If the provider offers an OpenAI-compatible endpoint, add it as another entry under `providers` with its own `base_url_env` and `api_key_env`, plus its models. No code changes are needed.

If it only has its own API, change `stream_chat()` in `llm.py` to use LiteLLM or Pydantic AI for that provider ([§5.1](#51-recommendation-for-this-poc-the-official-openai-python-sdk-plus-a-small-registry)).

---

## 10. Project structure

```text
local-llm-chat-poc/
├── .gitignore
├── backend/                       # FastAPI (Python)
│   ├── requirements.txt           # pinned Python dependencies
│   ├── .env.example               # every backend setting, documented; copy to .env
│   ├── models.json                # model list + providers (no URLs, no secrets)
│   ├── pytest.ini                 # test settings
│   ├── app/
│   │   ├── __init__.py
│   │   ├── __main__.py            # `python -m app` starts the server
│   │   ├── config.py              # settings from .env + models.json schema
│   │   ├── llm.py                 # ModelRegistry: clients per runtime, streaming, errors
│   │   ├── main.py                # FastAPI app and all routes
│   │   ├── schemas.py             # request/response models
│   │   └── store.py               # in-memory conversations
│   ├── scripts/
│   │   └── mock_llm_server.py     # fake OpenAI-compatible runtime for testing without a GPU
│   └── tests/
│       └── test_api.py            # pytest tests (no model needed)
└── frontend/                      # React + Vite (TypeScript)
    ├── package.json
    ├── package-lock.json          # created by npm install
    ├── vite.config.ts             # dev server port + /api proxy, read from .env
    ├── tsconfig.json
    ├── .env.example               # copy to .env
    ├── index.html
    └── src/
        ├── main.tsx               # entry point
        ├── App.tsx                # page layout, model selection, banners
        ├── useChat.ts             # chat state: send, stream, stop, retry, new chat
        ├── api.ts                 # fetch helpers + SSE reader
        ├── types.ts               # API types
        ├── styles.css
        └── components/
            ├── ModelSelector.tsx
            ├── MessageList.tsx
            └── ChatInput.tsx
```

The backend is six small Python files and the frontend is about ten. Most changes have an obvious home:

| Change | Where |
|---|---|
| Add or swap a model or runtime | `models.json` / `.env` only |
| Use a different LLM library | `app/llm.py` only |
| Save conversations to a database | `app/store.py` only |
| Add an endpoint | `app/main.py` (move routes into `APIRouter` files when it grows) |
| Change the UI | `frontend/src` |

---

## 11. Configuration

### 11.1 Rules

1. **No hard-coded values in code.** Model names live in `models.json`. URLs, ports, CORS origins and limits live in `.env`.
2. **Checked at startup.** Pydantic validates `.env` (`Settings`) and `models.json` (`ModelsFile`). A mistake stops the backend with a clear message.
3. **Precedence:** real environment variables › `backend/.env` › defaults in `config.py`. `load_dotenv()` never overwrites variables that are already set.
4. **Secrets only in `.env`**, which is never committed. Commit `.env.example`.

### 11.2 Backend settings (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Address the API listens on |
| `PORT` | `8000` | API port |
| `RELOAD` | `false` (`true` in the example file) | Auto-restart on code changes (development only) |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |
| `MODELS_CONFIG_PATH` | `models.json` | Model list file (relative to `backend/`) |
| `DEFAULT_MODEL` | — | Overrides `default_model` from `models.json` |
| `OLLAMA_BASE_URL` | — | Ollama's OpenAI-compatible URL, e.g. `http://localhost:11434/v1` (named by `base_url_env` in `models.json`) |
| `LLAMACPP_BASE_URL` / `LLAMACPP_API_KEY` | — | Optional llama.cpp runtime |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated browser origins allowed to call the API directly |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Give up on a generation after this long |
| `MAX_MESSAGE_CHARS` | `16000` | Longest accepted message |
| `MAX_HISTORY_MESSAGES` | `40` | How many earlier messages are sent to the model as context |

Variable names for providers are not fixed in code. A provider with `"base_url_env": "MY_SERVER_URL"` makes the backend read `MY_SERVER_URL`.

### 11.3 Frontend settings (`frontend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `DEV_SERVER_HOST` | `127.0.0.1` | Vite dev server address |
| `DEV_SERVER_PORT` | `5173` | Vite dev server port. If you change it, update `CORS_ORIGINS` in the backend. |
| `API_PROXY_TARGET` | `http://127.0.0.1:8000` | Where Vite forwards `/api` requests |
| `VITE_API_BASE_URL` | *(empty)* | API address used by the browser. Leave empty to go through the proxy. Anything in `VITE_*` is visible in the browser, so never put secrets here. |

---

## 12. Development workflow (fresh machine → running app)

Commands are for **Windows PowerShell**; macOS/Linux variants are shown where they differ.

### Step 1 — Install the tools

1. **NVIDIA driver** (latest Game Ready or Studio driver). Check that `nvidia-smi` works.
2. **Python 3.12 or 3.13** from <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer. FastAPI needs ≥ 3.10; this guide was tested with 3.13.
3. **Node.js 22 LTS** from <https://nodejs.org/en/download>. It's only needed for the React frontend.
4. **Git** (optional) from <https://git-scm.com/downloads>.

```powershell
python --version     # 3.12.x or 3.13.x   (macOS/Linux: python3 --version)
node -v              # v22.x
npm -v
```

### Step 2 — Install and configure Ollama

Follow [§1.4](#14-installation-steps) and [§1.6](#16-configure-ollama-for-a-6-gb-gpu). Then check:

```powershell
ollama -v
curl.exe http://localhost:11434/api/version
```

### Step 3 — Download the models and check the GPU

```powershell
ollama pull gemma4:e4b
ollama pull nemotron-3-nano:4b
ollama run gemma4:e4b "Reply with one word: ready"
ollama ps            # PROCESSOR should say 100% GPU
```

### Step 4 — Check the OpenAI-compatible API

```powershell
curl.exe http://localhost:11434/v1/models

$body = @{ model = "nemotron-3-nano:4b"; messages = @(@{ role = "user"; content = "Say hi" }) } | ConvertTo-Json -Depth 5
(Invoke-RestMethod -Method Post -Uri http://localhost:11434/v1/chat/completions -ContentType "application/json" -Body $body).choices[0].message.content
```

### Step 5 — Create the project

Create the folders from [§10](#10-project-structure) and copy each file from [Appendix A](#appendix-a--complete-source-code), or unzip the project if you have it.

### Step 6 — Set up and start the backend (terminal 1)

```powershell
cd local-llm-chat-poc\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
Copy-Item .env.example .env           # macOS/Linux: cp .env.example .env
python -m app
```

If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

You should see `Uvicorn running on http://127.0.0.1:8000`. In another terminal:

```powershell
curl.exe http://localhost:8000/api/health     # {"status":"ok","providers":{"ollama":"ok"}}
curl.exe http://localhost:8000/api/models     # both models with "available": true
```

Then open <http://localhost:8000/docs> to try the API in the browser.

### Step 7 — Set up and start the frontend (terminal 2)

```powershell
cd local-llm-chat-poc\frontend
npm install
Copy-Item .env.example .env           # macOS/Linux: cp .env.example .env
npm run dev
```

Open <http://localhost:5173>.

### Step 8 — Try the whole app

1. Send "Hello" with **Gemma 4 E4B**. The reply streams in, and the line under it shows the model and token count.
2. Switch to **Nemotron 3 Nano 4B** and ask a coding question. The first reply is a few seconds slower while the model loads; `ollama ps` now shows `nemotron-3-nano:4b`.
3. Ask for something long and press **Stop**. Streaming stops and the partial answer stays.
4. Quit Ollama and send a message. A banner says *"The local model server is not reachable…"*. Start Ollama again and press **Retry**; it works.
5. Press **New chat**. The screen clears, and the earlier conversation is still listed by `GET /api/conversations`.

### Step 9 — Run the tests

```powershell
cd backend             # with the venv activated
pytest -q              # 7 passed
cd ..\frontend
npm run build          # type-check + production build
```

### Daily routine

| Terminal | Command |
|---|---|
| 1 | `cd backend` → activate venv → `python -m app` |
| 2 | `cd frontend` → `npm run dev` |

Ollama runs in the background. Stop each app with `Ctrl+C`.

---

## 13. Testing

### 13.1 Automated tests (pytest)

`backend/tests/test_api.py` swaps the real model registry for a fake one through FastAPI's `dependency_overrides`. No GPU, Ollama or network is needed.

| Test | Checks |
|---|---|
| `test_models_lists_default_and_availability` | `/api/models` returns the default and the installed / not-installed flags |
| `test_chat_streams_and_saves_history` | Event order `start → delta → delta → done`; the second turn sends earlier messages to the model; history is saved |
| `test_model_switch_per_request_and_per_conversation` | A `model` in the request is used; `PATCH` changes the conversation's model for later turns |
| `test_validation_errors` | Unknown model → 400; empty message → 422; message too long → 400; bad conversation id → 422 |
| `test_runtime_error_is_streamed_and_nothing_saved` | A runtime failure arrives as an `error` event and nothing is saved |
| `test_conversation_crud` | List, get, delete, then 404 |
| `test_models_file_validation` | An invalid `models.json` is rejected |

```bash
cd backend && pytest -q
```

### 13.2 Testing without a GPU: the mock runtime

`scripts/mock_llm_server.py` is a tiny OpenAI-compatible server. Each reply starts with `[<runtime model name>]`, so you can see at a glance which model answered.

```powershell
# terminal A (venv active)
python scripts/mock_llm_server.py          # http://127.0.0.1:11500/v1
# terminal B: point the backend at it, then start it
$env:OLLAMA_BASE_URL="http://127.0.0.1:11500/v1"; python -m app
# macOS/Linux: OLLAMA_BASE_URL=http://127.0.0.1:11500/v1 python -m app
```

Simulate a model that isn't installed by limiting what the mock offers: `$env:MOCK_MODELS="gemma4:e4b"; python scripts/mock_llm_server.py` (macOS/Linux: `MOCK_MODELS=gemma4:e4b python scripts/mock_llm_server.py`).

### 13.3 API checks with curl

```bash
B=http://localhost:8000/api
curl -s $B/health
curl -s $B/models
curl -sN $B/chat -H 'Content-Type: application/json' -d '{"message":"Hello","model":"gemma4-e4b"}'
curl -s $B/conversations
```

### 13.4 Model switching checks

| Check | How | Expected |
|---|---|---|
| Per message | Send with `"model":"gemma4-e4b"`, then `"model":"nemotron-nano-4b"` in the same conversation | Each `start` event names the requested model |
| Per conversation | `PATCH /api/conversations/<id> {"model":"gemma4-e4b"}`, then send without `model` | `start` says `gemma4-e4b` |
| Default | Start the backend with `DEFAULT_MODEL=nemotron-nano-4b` | `GET /api/models` shows it as `default_model` |
| New model via config | Add an entry to `models.json`, restart | It appears in `/api/models` and the dropdown |
| Other runtime | Follow [§9.4](#94-switch-runtime-ollama--llamacpp) | Models from both runtimes are listed and both answer |

### 13.5 Error handling checks

All of these were run against this code:

| Scenario | How | Result |
|---|---|---|
| Unknown model | `"model":"gpt-9"` | `400` `Unknown model 'gpt-9'. See GET /api/models.` |
| Empty message | `"message":""` | `422` validation error |
| Message too long | over `MAX_MESSAGE_CHARS` | `400` |
| Model not installed | chat with a model the runtime doesn't have | SSE `error`: "Model 'nemotron-3-nano:4b' is not installed in the runtime. Run: ollama pull nemotron-3-nano:4b" |
| Runtime down | stop Ollama, then chat | `/api/health` → `503 degraded`; SSE `error`: "The local model server is not reachable…"; UI banner with Retry |
| Stop pressed | Stop during a long reply | Streaming ends; the partial answer is shown and saved |
| Unknown conversation | `GET /api/conversations/<bad id>` | `404` |

### 13.6 Performance checks

| What | How | Typical on an RTX 4050 6 GB (4B-class model fully on GPU) |
|---|---|---|
| Generation speed | `ollama run gemma4:e4b --verbose "Write 200 words about rivers"` → `eval rate` | roughly 25–50 tokens/s, depending on the laptop's power limit |
| Model fully on GPU | `ollama ps` while it answers | `100% GPU` |
| VRAM / temperature | `nvidia-smi -l 1` | VRAM below ~5.8 GB; no sustained thermal throttling |
| Reply time through the app | command below | `total` is about the model's generation time; the backend adds only milliseconds |
| Model switch delay | first message after switching models | a few seconds (model loads from disk) |

```bash
curl -sN -o /dev/null -w "total %{time_total}s\n" \
  http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"message":"Write 100 words about rivers.","model":"gemma4-e4b"}'
```

---

## 14. Production considerations

This is a POC, and a few things were deliberately left out to keep it small. The table shows what exists today and what to add before real users rely on it.

| Area | In the POC | Before production |
|---|---|---|
| **Security — runtime** | Ollama on `127.0.0.1` (default); the backend calls it server-side | Keep the runtime off the public network. The local Ollama API has no authentication, so never expose port 11434. If you use llama.cpp, start it with `--api-key`. |
| **Security — app** | CORS restricted to configured origins; system prompts come from server config only; only models in `models.json` can be used; model output shown as plain text | Serve frontend and API from one origin behind HTTPS (Nginx, Caddy, cloud load balancer). If you render Markdown, sanitise HTML. Treat model output as untrusted. |
| **Authentication** | None | Add login (OIDC/OAuth via your identity provider) with FastAPI security dependencies. Store a `user_id` on each conversation and check ownership on every `/api/conversations/{id}` call. |
| **Input validation** | Pydantic models, max message length, conversation id format, model allowlist | Add a total request size limit at the proxy. Count tokens if you raise context sizes. |
| **Rate limiting** | None | Per user or IP: `slowapi` in the app, or `limit_req` in Nginx / your API gateway. Use Redis when running more than one process. |
| **Logging** | Standard Python logging; prompts are not logged | Structured JSON logs (e.g. `structlog`) with a request id. Log model, duration and tokens per reply. Keep prompts out of logs unless users have consented. |
| **Error handling** | Runtime errors turned into readable messages; validation errors returned before streaming | Add a global exception handler returning one error shape. Add alerting on `/api/health` = 503. |
| **Resource management** | Request timeout; `max_tokens` per model; history capped at `MAX_HISTORY_MESSAGES` | Size `OLLAMA_CONTEXT_LENGTH × OLLAMA_NUM_PARALLEL` so models stay `100% GPU`. Add a limit on simultaneous generations (e.g. an `asyncio.Semaphore` returning 503 when busy). |
| **Model loading** | Loaded on first use, unloaded after `OLLAMA_KEEP_ALIVE` | Preload at startup (`curl http://localhost:11434/api/generate -d '{"model":"gemma4:e4b","keep_alive":-1}'`). On one GPU, avoid users constantly alternating models; each switch reloads. |
| **Concurrency** | One Uvicorn process; conversations in memory | In-memory storage only works with **one** worker. Move conversations to SQLite or Postgres (re-implement `store.py`) before running several workers (`uvicorn --workers N` or Gunicorn). Raise `OLLAMA_NUM_PARALLEL` only if VRAM allows. |
| **Scalability** | Single laptop | For many users, serve models with **vLLM** (Linux, batching) or **llama.cpp `llama-server`** with parallel slots on dedicated GPUs. Only the URL in `.env` changes. |
| **Persistence** | Lost on restart | Database plus a retention policy and backups. |
| **Observability** | `/api/health` | Metrics (request rate, errors, time to first token, tokens/s). llama.cpp offers `--metrics`. |
| **Model governance** | Models chosen by `models.json` | Record model versions (`ollama ls` IDs) and check licences: Gemma 4 is Apache 2.0; Nemotron uses the NVIDIA Open Model License. |
| **Dependencies** | Pinned in `requirements.txt` and `package-lock.json` | Regular updates and vulnerability scans (`pip-audit`, `npm audit`). Update Ollama regularly. |

**Deployment sketch for later:**

```text
Browser ─HTTPS─► Reverse proxy (TLS, auth, rate limits)
                   ├─ /        → frontend/dist (static files from `npm run build`)
                   └─ /api/*   → uvicorn app.main:app --host 127.0.0.1 --port 8000
                                    └─► Ollama on localhost / private network (never public)
```

---

## 15. Comparison of available approaches

Ratings: ●●● strong · ●● adequate · ● weak — judged **for this POC** (Windows laptop, 6 GB NVIDIA GPU, simple chat, model switching).

### 15.1 Local LLM runtimes

| Criterion | **Ollama** | llama.cpp `llama-server` | LM Studio | vLLM |
|---|---|---|---|---|
| Ease of setup | ●●● installer, GPU auto-detected | ●● winget/brew/binaries; you pick files and flags | ●●● desktop app | ● Linux only (Windows via WSL), Python/CUDA stack |
| API support | ●●● OpenAI-compatible + native API | ●●● OpenAI-compatible, web UI, metrics | ●●● OpenAI-compatible + own REST API | ●●● OpenAI-compatible |
| Model management | ●●● `pull / ls / ps / rm / stop` | ●● downloads from Hugging Face (`-hf`) | ●●● GUI + `lms` CLI | ● manual |
| Model switching | ●●● per request, loads/unloads automatically | ●● one model per process, or router mode | ●●● loads on demand, auto-unloads | ● usually one model per server |
| Performance on 6 GB | ●● good; some packages bundle vision weights | ●●● most control (text-only files, KV cache options) | ●● | ● built for data-centre GPUs |
| Many users at once | ●● | ●●● | ●● | ●●● |
| Production suitability | ●● single machine / internal | ●●● lean, API key, metrics | ● desktop-oriented | ●●● |
| Licence | MIT | MIT | Free to use, closed source | Apache 2.0 |

**Pick:** Ollama for this POC. llama.cpp when you need exact control or a lean server; vLLM for multi-user GPU servers. Moving between them is a `.env` + `models.json` change ([§9.4](#94-switch-runtime-ollama--llamacpp)).

### 15.2 Python AI libraries / abstraction layers

Versions on PyPI on 6 Oct 2026: `openai` 3.24.0 · `litellm` 1.104.0 · `pydantic-ai` 2.54.0 · `langchain` 1.4.3.

| Criterion | **OpenAI Python SDK + small registry** | LiteLLM | Pydantic AI | LangChain |
|---|---|---|---|---|
| Ease of setup | ●●● one package | ●● | ●● | ● many packages and concepts |
| Works with all local runtimes | ●●● (all speak the OpenAI protocol) | ●●● | ●●● | ●●● |
| Non-OpenAI hosted providers (e.g. Anthropic native API) | ● only via OpenAI-compatible endpoints | ●●● 100+ providers, one call style | ●●● many providers | ●●● many integrations |
| Model switching by config | ●●● `models.json` → registry | ●●● model strings like `provider/model` | ●●● | ●● |
| Streaming | ●●● | ●●● | ●●● | ●● |
| Code you must understand | ●●● ~100 lines of your own | ●● | ●● | ● large abstraction surface |
| Growth path (tools, structured output, agents, RAG) | ●● you build it | ●● gateway features (keys, budgets, fallbacks) | ●●● typed agents and tools | ●●● biggest RAG/agent ecosystem |

**Pick:** the **OpenAI Python SDK** for the POC. It is the least to learn and covers every local runtime. Because model access is isolated in `llm.py`, switch to **LiteLLM** when you need many hosted providers. Choose **Pydantic AI** or **LangChain** when the project grows into tools, agents or RAG.

### 15.3 Backend framework

| Option | Notes |
|---|---|
| **FastAPI** (chosen) | Async, so it handles long streaming responses well. Pydantic validation, auto-generated `/docs`, native SSE since 0.135. Large Python AI ecosystem. |
| Flask | Simple, but synchronous by default; streaming and async need more work. |
| Django | Excellent for full web apps with admin and ORM; heavy for a chat POC. |
| Node.js (Hono / Express) + Vercel AI SDK | Strong option if the team prefers one language (TypeScript) everywhere; similar architecture. |

### 15.4 Frontend

| Option | Notes |
|---|---|
| **React + Vite** (chosen) | Standard, fast dev server with an `/api` proxy, simple static build. |
| Next.js | Adds server rendering and routing; more than a single chat page needs. |
| Streamlit / Gradio (Python only) | Fastest for demos, but limited UI control and a different path to a real product. |

### 15.5 Final recommendation

**Ollama + OpenAI-compatible API + FastAPI (OpenAI Python SDK) + React/Vite**, with `models.json` and `.env` for configuration. It's the smallest setup that still lets you switch models and runtimes without code changes, and each piece can be upgraded on its own later.

---

## 16. References

**Ollama (runtime)**
- Download: <https://ollama.com/download>
- Docs: <https://docs.ollama.com> · full index: <https://docs.ollama.com/llms.txt>
- Windows: <https://docs.ollama.com/windows> · macOS: <https://docs.ollama.com/macos> · Linux: <https://docs.ollama.com/linux>
- GPU support: <https://docs.ollama.com/gpu>
- FAQ (environment variables, model storage, keep-alive, concurrency): <https://docs.ollama.com/faq>
- Context length: <https://docs.ollama.com/context-length>
- CLI: <https://docs.ollama.com/cli>
- OpenAI compatibility: <https://docs.ollama.com/api/openai-compatibility>
- API authentication: <https://docs.ollama.com/api/authentication>
- Modelfile: <https://docs.ollama.com/modelfile>

**Models**
- Gemma 4 in Ollama: <https://ollama.com/library/gemma4/tags> · model card: <https://ai.google.dev/gemma/docs/core/model_card_4>
- Nemotron 3 Nano in Ollama: <https://ollama.com/library/nemotron-3-nano/tags> · NVIDIA: <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF>
- Gemma 4 E4B GGUF (for llama.cpp): <https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF>

**Other runtimes**
- llama.cpp server: <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md> · install: <https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md>
- LM Studio OpenAI compatibility: <https://lmstudio.ai/docs/developer/openai-compat>
- vLLM installation: <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>

**API standard**
- OpenAI Chat Completions reference: <https://platform.openai.com/docs/api-reference/chat>

**Backend (Python)**
- Python downloads: <https://www.python.org/downloads/>
- FastAPI: <https://fastapi.tiangolo.com> · Server-Sent Events: <https://fastapi.tiangolo.com/tutorial/server-sent-events/> · CORS: <https://fastapi.tiangolo.com/tutorial/cors/> · Testing: <https://fastapi.tiangolo.com/tutorial/testing/> · Dependencies: <https://fastapi.tiangolo.com/tutorial/dependencies/>
- Uvicorn: <https://www.uvicorn.org>
- OpenAI Python SDK: <https://github.com/openai/openai-python>
- Pydantic: <https://docs.pydantic.dev> · pydantic-settings: <https://docs.pydantic.dev/latest/concepts/pydantic_settings/>
- python-dotenv: <https://github.com/theskumar/python-dotenv>
- pytest: <https://docs.pytest.org>

**Alternative AI libraries**
- LiteLLM: <https://docs.litellm.ai>
- Pydantic AI: <https://ai.pydantic.dev>
- LangChain (Python): <https://docs.langchain.com/oss/python/langchain/overview>

**Frontend**
- React: <https://react.dev>
- Vite (config, env variables, dev proxy): <https://vite.dev/config/> · <https://vite.dev/guide/env-and-mode> · <https://vite.dev/config/server-options>
- Node.js downloads: <https://nodejs.org/en/download>
- MDN — reading streamed responses with `fetch`: <https://developer.mozilla.org/en-US/docs/Web/API/Streams_API/Using_readable_streams>

---

## Appendix A — Complete source code

Every file below is copied verbatim from the project that was tested for this guide (`pytest`: 7 passed; frontend type-checked and built; end-to-end browser test passed). `frontend/package-lock.json` is created by `npm install` and isn't reproduced here.

### Root

#### `.gitignore`

Keeps virtual environments, dependencies, builds and real `.env` files out of Git.

```text
# Python
.venv/
__pycache__/
.pytest_cache/

# Node
node_modules/
dist/

# Secrets / local config
.env
!.env.example
```

### Backend

#### `backend/requirements.txt`

Pinned Python dependencies (`pip install -r requirements.txt`).

```text
fastapi==0.142.2
uvicorn[standard]==0.54.0
openai==3.24.0
pydantic==2.13.5
pydantic-settings==2.15.0
python-dotenv==1.2.4

# tests
pytest==9.1.1
httpx==0.28.1
```

#### `backend/.env.example`

All backend settings. Copy to `.env`.

```dotenv
# Copy to backend/.env  (never commit the real .env)

# --- Server ---
HOST=127.0.0.1
PORT=8000
RELOAD=true
LOG_LEVEL=info

# --- Models ---
# Which models exist and which provider serves them
MODELS_CONFIG_PATH=models.json
# Optional: override "default_model" from models.json
# DEFAULT_MODEL=nemotron-nano-4b

# Provider endpoints, referenced by "base_url_env" in models.json.
# Must be an OpenAI-compatible base URL ending in /v1.
OLLAMA_BASE_URL=http://localhost:11434/v1
# LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1
# LLAMACPP_API_KEY=change-me

# --- Frontend access ---
# Comma-separated browser origins allowed to call the API
CORS_ORIGINS=http://localhost:5173

# --- Limits ---
REQUEST_TIMEOUT_SECONDS=120
MAX_MESSAGE_CHARS=16000
# How many previous messages are sent to the model as context
MAX_HISTORY_MESSAGES=40
```

#### `backend/models.json`

Providers and the list of usable models.

```json
{
  "default_model": "gemma4-e4b",
  "providers": {
    "ollama": {
      "base_url_env": "OLLAMA_BASE_URL",
      "api_key_env": "OLLAMA_API_KEY"
    },
    "llamacpp": {
      "base_url_env": "LLAMACPP_BASE_URL",
      "api_key_env": "LLAMACPP_API_KEY"
    }
  },
  "models": [
    {
      "id": "gemma4-e4b",
      "label": "Gemma 4 E4B",
      "description": "Google Gemma 4 (effective 4B). Default for general chat.",
      "provider": "ollama",
      "provider_model": "gemma4:e4b",
      "system_prompt": "You are a helpful, concise assistant.",
      "temperature": 0.7,
      "max_tokens": 1024
    },
    {
      "id": "nemotron-nano-4b",
      "label": "Nemotron 3 Nano 4B",
      "description": "NVIDIA Nemotron 3 Nano 4B. Good for coding and step-by-step reasoning.",
      "provider": "ollama",
      "provider_model": "nemotron-3-nano:4b",
      "system_prompt": "You are a precise assistant for programming and reasoning tasks. Put code in fenced blocks.",
      "temperature": 0.6,
      "max_tokens": 2048
    }
  ]
}
```

#### `backend/pytest.ini`

Lets `pytest` find the `app` package.

```ini
[pytest]
pythonpath = .
testpaths = tests
```

#### `backend/app/__init__.py`

Marks `app` as a Python package.

```python
"""Local LLM Chat POC backend."""
```

#### `backend/app/__main__.py`

`python -m app` starts Uvicorn using settings from `.env`.

```python
"""Start the API with `python -m app` (host, port and reload come from .env)."""

import uvicorn

from .config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.reload, log_level=settings.log_level)
```

#### `backend/app/config.py`

Settings from `.env` and the `models.json` schema.

```python
"""Settings (from environment / .env) and the model catalogue (from models.json)."""

import json
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env into os.environ. Variables already set in the real
# environment win, because load_dotenv does not override by default.
load_dotenv(BACKEND_DIR / ".env")


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    models_config_path: str = "models.json"
    default_model: str | None = None

    cors_origins: str = "http://localhost:5173"  # comma-separated

    request_timeout_seconds: float = 120
    max_message_chars: int = 16_000
    max_history_messages: int = 40

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


class ProviderConfig(BaseModel):
    # Names of environment variables, never literal URLs or keys.
    base_url_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")


class ModelConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")  # public id used by the UI/API
    label: str
    description: str = ""
    provider: str  # key into "providers"
    provider_model: str  # name the runtime expects, e.g. "gemma4:e4b"
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class ModelsFile(BaseModel):
    default_model: str
    providers: dict[str, ProviderConfig]
    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> "ModelsFile":
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        for model in self.models:
            if model.provider not in self.providers:
                raise ValueError(f"model '{model.id}' uses unknown provider '{model.provider}'")
        if self.default_model not in ids:
            raise ValueError(f"default_model '{self.default_model}' is not in models")
        return self


def load_models_file(path: str, default_override: str | None = None) -> ModelsFile:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = BACKEND_DIR / file_path
    models_file = ModelsFile.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
    if default_override:
        if default_override not in {model.id for model in models_file.models}:
            raise ValueError(f"DEFAULT_MODEL '{default_override}' is not defined in {file_path}")
        models_file.default_model = default_override
    return models_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

#### `backend/app/llm.py`

Model registry: one OpenAI-compatible client per runtime, streaming, error messages.

```python
"""Model registry: maps public model ids to an OpenAI-compatible client + runtime model name.

Every local runtime used here (Ollama, llama.cpp, LM Studio, vLLM) speaks the
OpenAI Chat Completions API, so one client type covers all of them. Switching
models or runtimes is a change to models.json / .env only.
"""

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

from .config import ModelConfig, ModelsFile

logger = logging.getLogger("app.llm")


class UnknownModelError(Exception):
    pass


class ProviderNotConfiguredError(Exception):
    pass


class LLMError(Exception):
    """An error from the runtime, with a message that is safe to show to users."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class StreamResult:
    """Filled in while streaming; read after the stream ends."""

    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelRegistry:
    def __init__(self, models_file: ModelsFile, timeout_seconds: float = 120):
        self.config = models_file
        self.clients: dict[str, AsyncOpenAI] = {}
        for provider_id, provider in models_file.providers.items():
            base_url = os.getenv(provider.base_url_env, "").strip()
            if not base_url:
                logger.info("provider '%s' skipped: %s is not set", provider_id, provider.base_url_env)
                continue
            api_key = os.getenv(provider.api_key_env, "") if provider.api_key_env else ""
            self.clients[provider_id] = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed",  # local runtimes ignore the key; the SDK requires a value
                timeout=timeout_seconds,
                max_retries=1,
            )

    @property
    def default_model_id(self) -> str:
        return self.config.default_model

    def get(self, model_id: str) -> ModelConfig:
        model = next((m for m in self.config.models if m.id == model_id), None)
        if model is None:
            raise UnknownModelError(model_id)
        if model.provider not in self.clients:
            raise ProviderNotConfiguredError(model.provider)
        return model

    async def installed_models(self) -> dict[str, set[str] | None]:
        """Per provider: the model names the runtime reports, or None if unreachable/not configured."""
        result: dict[str, set[str] | None] = {provider_id: None for provider_id in self.config.providers}
        for provider_id, client in self.clients.items():
            try:
                page = await client.with_options(timeout=3, max_retries=0).models.list()
                result[provider_id] = {model.id for model in page.data}
            except Exception as error:  # runtime down, wrong URL, etc.
                logger.warning("provider '%s' unreachable: %s", provider_id, error)
        return result

    async def stream_chat(
        self, model: ModelConfig, messages: list[dict[str, str]], result: StreamResult
    ) -> AsyncIterator[str]:
        """Yields text chunks. Raises LLMError with a user-friendly message on failure."""
        client = self.clients[model.provider]
        prompt = ([{"role": "system", "content": model.system_prompt}] if model.system_prompt else []) + messages

        options: dict = {}
        if model.temperature is not None:
            options["temperature"] = model.temperature
        if model.max_tokens is not None:
            options["max_tokens"] = model.max_tokens

        try:
            stream = await client.chat.completions.create(
                model=model.provider_model,
                messages=prompt,
                stream=True,
                stream_options={"include_usage": True},
                **options,
            )
            async with stream:  # closes the upstream connection if the client disconnects
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if choice.delta and choice.delta.content:
                            yield choice.delta.content
                        if choice.finish_reason:
                            result.finish_reason = choice.finish_reason
                    if chunk.usage:
                        result.input_tokens = chunk.usage.prompt_tokens
                        result.output_tokens = chunk.usage.completion_tokens
        except openai.APIConnectionError as error:
            if isinstance(error, openai.APITimeoutError):
                raise LLMError("The model took too long to respond. Try a shorter prompt or a smaller model.") from error
            raise LLMError("The local model server is not reachable. Make sure Ollama (or your runtime) is running.") from error
        except openai.NotFoundError as error:
            raise LLMError(
                f"Model '{model.provider_model}' is not installed in the runtime. Run: ollama pull {model.provider_model}"
            ) from error
        except openai.APIStatusError as error:
            logger.error("runtime returned HTTP %s: %s", error.status_code, error.message)
            raise LLMError("The model failed to generate a response. Please try again.") from error
```

#### `backend/app/schemas.py`

Request and response models.

```python
"""Request and response models (validated by Pydantic, documented in /docs)."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    model: str | None = None  # which model wrote an assistant message


class Conversation(BaseModel):
    id: str
    title: str
    model: str
    messages: list[Message]
    created_at: str
    updated_at: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    model: str
    message_count: int
    updated_at: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    # Omit to start a new conversation; the server returns its id in the "start" event.
    # Send that id back on the next turn to continue the conversation.
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    # Public model id from GET /api/models. Omit to use the conversation's model or the default.
    model: str | None = Field(default=None, max_length=64)


class UpdateConversationRequest(BaseModel):
    model: str | None = Field(default=None, max_length=64)
    title: str | None = Field(default=None, min_length=1, max_length=120)


class ModelInfo(BaseModel):
    id: str
    label: str
    description: str
    provider: str
    is_default: bool
    available: bool  # True when the runtime reports the model as installed


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]
```

#### `backend/app/store.py`

In-memory conversation storage.

```python
"""In-memory conversation storage. Good enough for a POC; data is lost on restart.

To persist conversations, re-implement these five methods on top of SQLite/Postgres.
"""

import uuid
from datetime import datetime, timezone

from .schemas import Conversation, Message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationStore:
    def __init__(self) -> None:
        self._items: dict[str, Conversation] = {}

    def list(self) -> list[Conversation]:
        return sorted(self._items.values(), key=lambda c: c.updated_at, reverse=True)

    def get(self, conversation_id: str) -> Conversation | None:
        return self._items.get(conversation_id)

    def new(self, model: str, conversation_id: str | None = None) -> Conversation:
        """A conversation object that is only stored once its first turn completes."""
        return Conversation(
            id=conversation_id or uuid.uuid4().hex, title="New conversation", model=model, messages=[], created_at=_now(), updated_at=_now()
        )

    def add_turn(self, conversation: Conversation, user: Message, assistant: Message) -> None:
        if not conversation.messages:
            conversation.title = user.content[:60]
        conversation.messages.extend([user, assistant])
        conversation.model = assistant.model or conversation.model
        conversation.updated_at = _now()
        self._items[conversation.id] = conversation

    def delete(self, conversation_id: str) -> bool:
        return self._items.pop(conversation_id, None) is not None
```

#### `backend/app/main.py`

FastAPI app and all routes.

```python
"""FastAPI application: routes for models, chat (streaming) and conversations."""

import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse

from .config import ModelConfig, get_settings, load_models_file
from .llm import LLMError, ModelRegistry, ProviderNotConfiguredError, StreamResult, UnknownModelError
from .schemas import (
    ChatRequest,
    Conversation,
    ConversationSummary,
    Message,
    ModelInfo,
    ModelsResponse,
    UpdateConversationRequest,
)
from .store import ConversationStore

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

registry = ModelRegistry(
    load_models_file(settings.models_config_path, settings.default_model),
    timeout_seconds=settings.request_timeout_seconds,
)
store = ConversationStore()

app = FastAPI(title="Local LLM Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


# Dependencies (overridable in tests)
def get_registry() -> ModelRegistry:
    return registry


def get_store() -> ConversationStore:
    return store


def resolve_model(registry: ModelRegistry, model_id: str) -> ModelConfig:
    try:
        return registry.get(model_id)
    except UnknownModelError:
        raise HTTPException(400, f"Unknown model '{model_id}'. See GET /api/models.")
    except ProviderNotConfiguredError as error:
        raise HTTPException(503, f"Provider '{error}' is not configured (missing base URL in .env).")


# ---------- health & models ----------


@app.get("/api/health")
async def health(registry: ModelRegistry = Depends(get_registry)):
    installed = await registry.installed_models()
    providers = {pid: ("ok" if models is not None else "unreachable") for pid, models in installed.items() if pid in registry.clients}
    healthy = any(v == "ok" for v in providers.values())
    return JSONResponse({"status": "ok" if healthy else "degraded", "providers": providers}, status_code=200 if healthy else 503)


@app.get("/api/models", response_model=ModelsResponse)
async def list_models(registry: ModelRegistry = Depends(get_registry)):
    installed = await registry.installed_models()
    return ModelsResponse(
        default_model=registry.default_model_id,
        models=[
            ModelInfo(
                id=m.id,
                label=m.label,
                description=m.description,
                provider=m.provider,
                is_default=m.id == registry.default_model_id,
                available=m.provider_model in (installed.get(m.provider) or set()),
            )
            for m in registry.config.models
        ],
    )


# ---------- chat ----------


@dataclass
class ChatContext:
    conversation: Conversation
    model: ModelConfig
    message: str


def prepare_chat(
    body: ChatRequest,
    registry: ModelRegistry = Depends(get_registry),
    store: ConversationStore = Depends(get_store),
) -> ChatContext:
    """Runs BEFORE streaming starts, so validation errors become normal 4xx JSON responses."""
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(400, f"Message is longer than {settings.max_message_chars} characters.")

    # An unknown id is treated as a new conversation (e.g. the first turn failed before it was saved).
    conversation = store.get(body.conversation_id) if body.conversation_id else None

    # Model resolution: request > conversation > default
    model_id = body.model or (conversation.model if conversation else registry.default_model_id)
    model = resolve_model(registry, model_id)

    if conversation is None:
        conversation = store.new(model.id, body.conversation_id)
    return ChatContext(conversation=conversation, model=model, message=body.message)


@app.post("/api/chat", response_class=EventSourceResponse)
async def chat(
    ctx: ChatContext = Depends(prepare_chat),
    registry: ModelRegistry = Depends(get_registry),
    store: ConversationStore = Depends(get_store),
) -> AsyncIterable[dict]:
    """Streams Server-Sent Events: start -> delta... -> done (or error)."""
    yield {"type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id}

    history = [{"role": m.role, "content": m.content} for m in ctx.conversation.messages]
    history = history[-settings.max_history_messages :] + [{"role": "user", "content": ctx.message}]

    parts: list[str] = []
    result = StreamResult()
    try:
        async for text in registry.stream_chat(ctx.model, history, result):
            parts.append(text)
            yield {"type": "delta", "text": text}
    except LLMError as error:
        logger.warning("chat failed (model=%s): %s", ctx.model.id, error.__cause__ or error)
        yield {"type": "error", "message": error.user_message}
        return
    finally:
        # Also runs when the user presses Stop (client disconnect): keep the partial answer.
        if parts:
            store.add_turn(
                ctx.conversation,
                Message(role="user", content=ctx.message),
                Message(role="assistant", content="".join(parts), model=ctx.model.id),
            )

    yield {
        "type": "done",
        "finish_reason": result.finish_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    }


# ---------- conversations ----------


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations(store: ConversationStore = Depends(get_store)):
    return [
        ConversationSummary(id=c.id, title=c.title, model=c.model, message_count=len(c.messages), updated_at=c.updated_at)
        for c in store.list()
    ]


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str, store: ConversationStore = Depends(get_store)):
    conversation = store.get(conversation_id)
    if conversation is None:
        raise HTTPException(404, "Conversation not found.")
    return conversation


@app.patch("/api/conversations/{conversation_id}", response_model=Conversation)
def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    registry: ModelRegistry = Depends(get_registry),
    store: ConversationStore = Depends(get_store),
):
    """Select the model for this conversation's next turns and/or rename it."""
    conversation = store.get(conversation_id)
    if conversation is None:
        raise HTTPException(404, "Conversation not found.")
    if body.model:
        conversation.model = resolve_model(registry, body.model).id
    if body.title:
        conversation.title = body.title
    return conversation


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, store: ConversationStore = Depends(get_store)):
    if not store.delete(conversation_id):
        raise HTTPException(404, "Conversation not found.")
    return Response(status_code=204)
```

#### `backend/tests/test_api.py`

API tests with a fake model registry.

```python
"""API tests. No GPU or running model needed: the model registry is replaced with a fake."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import ModelsFile
from app.llm import LLMError, ModelRegistry
from app.main import app, get_registry, get_store
from app.store import ConversationStore

MODELS = {
    "default_model": "general",
    "providers": {"local": {"base_url_env": "TEST_LLM_BASE_URL"}},
    "models": [
        {"id": "general", "label": "General", "provider": "local", "provider_model": "gemma4:e4b"},
        {"id": "coder", "label": "Coder", "provider": "local", "provider_model": "nemotron-3-nano:4b"},
    ],
}


class FakeRegistry(ModelRegistry):
    """Real model lookup, fake runtime: replies '[<runtime model>] reply' and records prompts."""

    def __init__(self, fail_with: str | None = None):
        super().__init__(ModelsFile.model_validate(MODELS))
        self.fail_with = fail_with
        self.prompts: list[list[dict]] = []

    async def installed_models(self):
        return {"local": {"gemma4:e4b"}}  # "coder" is not installed

    async def stream_chat(self, model, messages, result):
        self.prompts.append(messages)
        if self.fail_with:
            raise LLMError(self.fail_with)
        for word in [f"[{model.provider_model}] ", "reply"]:
            yield word
        result.finish_reason = "stop"
        result.output_tokens = 2


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("TEST_LLM_BASE_URL", "http://fake/v1")
    registry, store = FakeRegistry(), ConversationStore()
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), registry, store
    app.dependency_overrides.clear()


def events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_models_lists_default_and_availability(setup):
    client, _, _ = setup
    body = client.get("/api/models").json()
    assert body["default_model"] == "general"
    assert [(m["id"], m["available"]) for m in body["models"]] == [("general", True), ("coder", False)]


def test_chat_streams_and_saves_history(setup):
    client, registry, store = setup
    first = events(client.post("/api/chat", json={"message": "hello"}))
    assert [e["type"] for e in first] == ["start", "delta", "delta", "done"]
    assert first[0]["model"] == "general"
    conversation_id = first[0]["conversation_id"]

    # Second turn: previous messages are sent to the model as context
    client.post("/api/chat", json={"message": "again", "conversation_id": conversation_id})
    assert [m["content"] for m in registry.prompts[-1]] == ["hello", "[gemma4:e4b] reply", "again"]
    assert len(store.get(conversation_id).messages) == 4


def test_model_switch_per_request_and_per_conversation(setup):
    client, _, _ = setup
    start = events(client.post("/api/chat", json={"message": "hi", "model": "coder"}))
    assert start[1]["text"] == "[nemotron-3-nano:4b] "
    conversation_id = start[0]["conversation_id"]

    assert client.patch(f"/api/conversations/{conversation_id}", json={"model": "general"}).status_code == 200
    follow_up = events(client.post("/api/chat", json={"message": "hi", "conversation_id": conversation_id}))
    assert follow_up[0]["model"] == "general"


def test_validation_errors(setup):
    client, _, _ = setup
    assert client.post("/api/chat", json={"message": "hi", "model": "gpt-9"}).status_code == 400
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 20_000}).status_code == 400
    assert client.post("/api/chat", json={"message": "hi", "conversation_id": "../etc"}).status_code == 422


def test_runtime_error_is_streamed_and_nothing_saved(setup, monkeypatch):
    client, registry, store = setup
    registry.fail_with = "The local model server is not reachable."
    stream = events(client.post("/api/chat", json={"message": "hi"}))
    assert stream[-1] == {"type": "error", "message": "The local model server is not reachable."}
    assert store.list() == []


def test_conversation_crud(setup):
    client, _, _ = setup
    conversation_id = events(client.post("/api/chat", json={"message": "hello"}))[0]["conversation_id"]
    assert client.get("/api/conversations").json()[0]["id"] == conversation_id
    assert client.get(f"/api/conversations/{conversation_id}").json()["title"] == "hello"
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


def test_models_file_validation():
    bad = {**MODELS, "default_model": "missing"}
    with pytest.raises(ValueError):
        ModelsFile.model_validate(bad)
```

#### `backend/scripts/mock_llm_server.py`

Fake OpenAI-compatible runtime for testing without a GPU.

```python
"""A tiny OpenAI-compatible server for testing without a GPU or a real model.

    python scripts/mock_llm_server.py            # serves http://127.0.0.1:11500/v1

Set OLLAMA_BASE_URL=http://127.0.0.1:11500/v1 in backend/.env to use it.
Replies echo the model name, which makes model-switching easy to verify.
"""

import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

MODELS = os.getenv("MOCK_MODELS", "gemma4:e4b,nemotron-3-nano:4b").split(",")
PORT = int(os.getenv("MOCK_PORT", "11500"))

app = FastAPI()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body["model"]
    if model not in MODELS:
        return JSONResponse({"error": {"message": f'model "{model}" not found, try pulling it first'}}, status_code=404)

    user_text = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
    words = f"[{model}] You said: {user_text}".split(" ")
    base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

    async def events():
        for i, word in enumerate(words):
            await asyncio.sleep(0.03)
            delta = {"content": word + (" " if i < len(words) - 1 else "")}
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        usage = {"prompt_tokens": 10, "completion_tokens": len(words), "total_tokens": 10 + len(words)}
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
```

### Frontend

#### `frontend/package.json`

Frontend dependencies and scripts.

```json
{
  "name": "local-llm-chat-frontend",
  "version": "1.0.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^19.3.0",
    "react-dom": "^19.3.0"
  },
  "devDependencies": {
    "@types/react": "^19.3.0",
    "@types/react-dom": "^19.3.0",
    "@vitejs/plugin-react": "^6.1.2",
    "typescript": "^7.0.2",
    "vite": "^8.3.3"
  }
}
```

#### `frontend/tsconfig.json`

TypeScript settings.

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "types": ["vite/client"],
    "noEmit": true
  },
  "include": ["src"]
}
```

#### `frontend/vite.config.ts`

Dev server port and `/api` proxy, read from `.env`.

```ts
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  // Reads frontend/.env (all variables, not only VITE_*) for dev-server settings.
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    server: {
      host: env.DEV_SERVER_HOST || '127.0.0.1',
      port: Number(env.DEV_SERVER_PORT || 5173),
      strictPort: true,
      // The browser calls /api on the Vite server; Vite forwards it to FastAPI.
      // No CORS issues in development and no API URL hard-coded in the React code.
      proxy: {
        '/api': { target: env.API_PROXY_TARGET || 'http://127.0.0.1:8000', changeOrigin: true },
      },
    },
  };
});
```

#### `frontend/.env.example`

Frontend settings. Copy to `.env`.

```dotenv
# Copy to frontend/.env

# Vite dev server
DEV_SERVER_HOST=127.0.0.1
DEV_SERVER_PORT=5173

# Where the dev server forwards /api requests (the FastAPI backend)
API_PROXY_TARGET=http://127.0.0.1:8000

# API origin used by the browser. Leave empty to call the same origin
# (the Vite proxy in development). Set only if the API is on another origin,
# e.g. http://localhost:8000 — then add this site's origin to CORS_ORIGINS in backend/.env.
VITE_API_BASE_URL=
```

#### `frontend/index.html`

HTML shell.

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link rel="icon" href="data:," />
    <title>Local LLM Chat</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

#### `frontend/src/main.tsx`

Entry point.

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './styles.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

#### `frontend/src/App.tsx`

Page layout, model loading and selection, banners.

```tsx
import { useEffect, useState } from 'react';
import { fetchModels } from './api';
import { ChatInput } from './components/ChatInput';
import { MessageList } from './components/MessageList';
import { ModelSelector } from './components/ModelSelector';
import type { ModelInfo } from './types';
import { useChat } from './useChat';

export function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [model, setModel] = useState('');
  const [loadError, setLoadError] = useState<string | null>(null);
  const chat = useChat();

  const loadModels = () => {
    fetchModels()
      .then((data) => {
        setModels(data.models);
        setModel((current) => current || data.default_model);
        setLoadError(null);
      })
      .catch((err: Error) => setLoadError(`Cannot reach the API: ${err.message}`));
  };
  useEffect(loadModels, []);

  const selected = models.find((m) => m.id === model);
  const busy = chat.status !== 'idle';

  return (
    <div className="app">
      <header>
        <h1>Local LLM Chat</h1>
        <div className="controls">
          <ModelSelector models={models} value={model} onChange={setModel} />
          <button type="button" className="secondary" onClick={chat.newChat}>
            New chat
          </button>
        </div>
      </header>

      <MessageList messages={chat.messages} waiting={chat.status === 'waiting'} models={models} />

      {loadError && (
        <div className="banner error">
          {loadError} <button onClick={loadModels}>Retry</button>
        </div>
      )}
      {chat.error && (
        <div className="banner error">
          {chat.error} <button onClick={() => chat.retry(model)}>Retry</button>
          <button onClick={chat.clearError}>Dismiss</button>
        </div>
      )}
      {selected && !selected.available && !chat.error && (
        <div className="banner warning">"{selected.label}" is not installed in the local runtime yet.</div>
      )}

      <ChatInput busy={busy} onSend={(text) => chat.send(text, model)} onStop={chat.stop} />
    </div>
  );
}
```

#### `frontend/src/useChat.ts`

Chat state: send, stream, stop, retry, new chat.

```ts
import { useCallback, useRef, useState } from 'react';
import { streamChat } from './api';
import type { ChatMessage } from './types';

type Status = 'idle' | 'waiting' | 'streaming';

/** All chat state in one hook: messages, streaming status, errors, stop, retry, new chat. */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<Status>('idle');
  const [error, setError] = useState<string | null>(null);
  const conversationId = useRef<string | null>(null);
  const abortController = useRef<AbortController | null>(null);

  const updateLast = (change: (message: ChatMessage) => ChatMessage) =>
    setMessages((list) => list.map((m, i) => (i === list.length - 1 ? change(m) : m)));

  const send = useCallback(async (text: string, model: string) => {
    setError(null);
    setStatus('waiting');
    const controller = new AbortController();
    abortController.current = controller;
    setMessages((list) => [
      ...list,
      { id: crypto.randomUUID(), role: 'user', content: text },
      { id: crypto.randomUUID(), role: 'assistant', content: '', model },
    ]);

    try {
      await streamChat(
        { message: text, model, conversation_id: conversationId.current },
        (event) => {
          if (event.type === 'start') {
            conversationId.current = event.conversation_id;
            setStatus('streaming');
          } else if (event.type === 'delta') {
            updateLast((m) => ({ ...m, content: m.content + event.text }));
          } else if (event.type === 'done') {
            updateLast((m) => ({ ...m, outputTokens: event.output_tokens }));
          } else if (event.type === 'error') {
            setError(event.message);
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Request failed');
    } finally {
      // Drop an assistant bubble that never received any text (error before first token).
      setMessages((list) => (list.at(-1)?.role === 'assistant' && !list.at(-1)?.content ? list.slice(0, -1) : list));
      setStatus('idle');
    }
  }, []);

  const stop = useCallback(() => abortController.current?.abort(), []);

  /** Re-send the last user message (after an error). */
  const retry = useCallback(
    (model: string) => {
      const lastUser = [...messages].reverse().find((m) => m.role === 'user');
      if (!lastUser) return;
      setMessages((list) => list.slice(0, list.lastIndexOf(lastUser)));
      void send(lastUser.content, model);
    },
    [messages, send],
  );

  const newChat = useCallback(() => {
    abortController.current?.abort();
    conversationId.current = null;
    setMessages([]);
    setError(null);
  }, []);

  return { messages, status, error, send, stop, retry, newChat, clearError: () => setError(null) };
}
```

#### `frontend/src/api.ts`

API calls and the Server-Sent Events reader.

```ts
import type { ChatEvent, ModelsResponse } from './types';

// Empty = same origin (the Vite proxy forwards /api to the backend in development).
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

/** Turns FastAPI error bodies ({"detail": "..."} or a validation list) into one readable message. */
async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join('; ');
  } catch {
    // not JSON
  }
  return `Request failed (HTTP ${response.status})`;
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE}/api/models`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

/**
 * POST /api/chat and call onEvent for every Server-Sent Event.
 * Pass an AbortSignal to stop generation (the backend stops the model too).
 */
export async function streamChat(
  body: { message: string; model?: string; conversation_id?: string | null },
  onEvent: (event: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) throw new Error(await errorMessage(response));

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    // SSE events are separated by a blank line; lines starting with ":" are keep-alive comments.
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? '';
    for (const block of blocks) {
      for (const line of block.split(/\r?\n/)) {
        if (line.startsWith('data:')) onEvent(JSON.parse(line.slice(5).trim()) as ChatEvent);
      }
    }
  }
}
```

#### `frontend/src/types.ts`

API types.

```ts
// Shapes returned by the FastAPI backend.

export interface ModelInfo {
  id: string;
  label: string;
  description: string;
  provider: string;
  is_default: boolean;
  available: boolean;
}

export interface ModelsResponse {
  default_model: string;
  models: ModelInfo[];
}

// Server-Sent Events sent by POST /api/chat
export type ChatEvent =
  | { type: 'start'; conversation_id: string; model: string }
  | { type: 'delta'; text: string }
  | { type: 'done'; finish_reason: string | null; input_tokens: number | null; output_tokens: number | null }
  | { type: 'error'; message: string };

// A message as shown in the UI
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  outputTokens?: number | null;
}
```

#### `frontend/src/components/ModelSelector.tsx`

Model dropdown.

```tsx
import type { ModelInfo } from '../types';

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
}

export function ModelSelector({ models, value, onChange }: Props) {
  return (
    <label className="model-selector">
      Model
      <select value={value} onChange={(e) => onChange(e.target.value)} disabled={models.length === 0}>
        {models.map((m) => (
          <option key={m.id} value={m.id} title={m.description}>
            {m.label}
            {m.available ? '' : ' (not installed)'}
          </option>
        ))}
      </select>
    </label>
  );
}
```

#### `frontend/src/components/MessageList.tsx`

Message list with model name and token count.

```tsx
import { useEffect, useRef } from 'react';
import type { ChatMessage, ModelInfo } from '../types';

interface Props {
  messages: ChatMessage[];
  waiting: boolean;
  models: ModelInfo[];
}

export function MessageList({ messages, waiting, models }: Props) {
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => bottom.current?.scrollIntoView({ block: 'end' }), [messages]);

  if (messages.length === 0) {
    return <div className="messages empty">Ask anything. Answers are generated on this machine.</div>;
  }

  const label = (id?: string) => models.find((m) => m.id === id)?.label ?? id;

  return (
    <div className="messages">
      {messages.map((m, i) => (
        <div key={m.id} className={`message ${m.role}`}>
          {m.role === 'assistant' && !m.content && waiting && i === messages.length - 1 ? (
            <span className="typing">Thinking…</span>
          ) : (
            <div className="text">{m.content}</div>
          )}
          {m.role === 'assistant' && m.content && (
            <div className="meta">
              {label(m.model)}
              {m.outputTokens ? ` · ${m.outputTokens} tokens` : ''}
            </div>
          )}
        </div>
      ))}
      <div ref={bottom} />
    </div>
  );
}
```

#### `frontend/src/components/ChatInput.tsx`

Input box with Send / Stop.

```tsx
import { useState, type FormEvent, type KeyboardEvent } from 'react';

interface Props {
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function ChatInput({ busy, onSend, onStop }: Props) {
  const [text, setText] = useState('');

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (!text.trim() || busy) return;
    onSend(text.trim());
    setText('');
  };

  // Enter sends, Shift+Enter adds a new line
  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form className="chat-input" onSubmit={submit}>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Type a message… (Enter to send, Shift+Enter for a new line)"
        rows={2}
      />
      {busy ? (
        <button type="button" className="stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" disabled={!text.trim()}>
          Send
        </button>
      )}
    </form>
  );
}
```

#### `frontend/src/styles.css`

Styles (light and dark).

```css
:root {
  --bg: #f7f7f5;
  --panel: #ffffff;
  --text: #1d1d1b;
  --muted: #6b6b66;
  --border: #e3e2dd;
  --accent: #2f5d50;
  --user: #e8efe9;
  --error-bg: #fbeceA;
  --error: #a63d32;
  --warn-bg: #fdf5e1;
  --warn: #7a5b10;
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  color: var(--text);
  background: var(--bg);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161716; --panel: #1f201f; --text: #ecebe6; --muted: #9d9c95; --border: #33342f;
    --accent: #7fb7a4; --user: #263630; --error-bg: #3a2220; --error: #f0a094; --warn-bg: #352c16; --warn: #e8cf8f;
  }
}
* { box-sizing: border-box; }
body { margin: 0; }
button, select, textarea { font: inherit; color: inherit; }

.app { display: flex; flex-direction: column; height: 100dvh; max-width: 900px; margin: 0 auto; }
header { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid var(--border); }
header h1 { font-size: 1rem; margin: 0; }
.controls { display: flex; gap: 10px; align-items: center; }
.model-selector { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 0.9em; }
.model-selector select { padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; background: var(--panel); color: var(--text); }
button { cursor: pointer; }
.secondary { padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px; background: transparent; }

.messages { flex: 1; overflow-y: auto; padding: 20px 16px; display: flex; flex-direction: column; gap: 14px; }
.messages.empty { justify-content: center; align-items: center; color: var(--muted); }
.message { max-width: 85%; line-height: 1.55; }
.message.user { align-self: flex-end; background: var(--user); padding: 10px 14px; border-radius: 14px 14px 4px 14px; }
.message.assistant { align-self: flex-start; }
.text { white-space: pre-wrap; overflow-wrap: anywhere; }
.meta { margin-top: 4px; font-size: 0.75rem; color: var(--muted); }
.typing { color: var(--muted); font-style: italic; }

.banner { margin: 0 16px 8px; padding: 10px 14px; border-radius: 8px; display: flex; gap: 12px; align-items: center; }
.banner button { border: 0; background: none; text-decoration: underline; padding: 0; }
.banner.error { background: var(--error-bg); color: var(--error); }
.banner.warning { background: var(--warn-bg); color: var(--warn); }

.chat-input { display: flex; gap: 10px; padding: 12px 16px 18px; }
.chat-input textarea { flex: 1; resize: none; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); }
.chat-input button { padding: 0 18px; border: 0; border-radius: 10px; background: var(--accent); color: #fff; }
.chat-input button:disabled { opacity: 0.5; cursor: not-allowed; }
.chat-input .stop { background: transparent; color: var(--text); border: 1px solid var(--border); }
```

---

## Appendix B — Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ollama` not recognized | Terminal opened before installing | Open a new terminal. On Windows, check `%LOCALAPPDATA%\Programs\Ollama` is on `PATH`. |
| `ollama ps` shows a `CPU/GPU` split; slow replies; high CPU | Model plus context doesn't fit in VRAM | Lower `OLLAMA_CONTEXT_LENGTH`, set `OLLAMA_KV_CACHE_TYPE=q8_0`, close other GPU apps, or use a smaller model / a text-only GGUF via llama.cpp |
| `ollama ps` shows `100% CPU` | GPU not detected | Update the NVIDIA driver (≥ 551.61 on Windows); check `%LOCALAPPDATA%\Ollama\server.log` |
| Ollama ignores new environment variables | Ollama wasn't restarted | Quit from the tray and start again (Windows/macOS); `sudo systemctl restart ollama` (Linux) |
| `python` not found / opens the Microsoft Store (Windows) | Python not on PATH | Reinstall with "Add python.exe to PATH", or use `py -3.13` |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Backend exits with a Pydantic `ValidationError` | Mistake in `models.json` or `.env` | The message names the field; fix it |
| Model shows "(not installed)" | `provider_model` differs from `ollama ls` | `ollama pull <name>`, or correct the name in `models.json` |
| Banner "Cannot reach the API" | Backend not running, or wrong `API_PROXY_TARGET` | Start `python -m app`; check ports 8000 / 5173 |
| Browser CORS error | Calling the API from an origin not in `CORS_ORIGINS` | Use the Vite proxy (leave `VITE_API_BASE_URL` empty) or add the origin |
| Reply stops mid-sentence | `max_tokens` reached | Raise `max_tokens` for that model in `models.json` |
| Model forgets the start of a long chat | Context window or `MAX_HISTORY_MESSAGES` | Raise them (watch VRAM) or start a new chat |
| First reply after switching models is slow | Model loading into VRAM | Expected; it's fast once loaded |
| Port already in use | Another process on 8000 / 5173 / 11434 | Change `PORT` / `DEV_SERVER_PORT` (and `API_PROXY_TARGET` / `CORS_ORIGINS`) |
