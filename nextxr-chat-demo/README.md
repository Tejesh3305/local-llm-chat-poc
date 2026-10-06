# NextXR local chat demo: interface

The chat interface for Deliverable A of the Demo and Pitch Requirements (v1.1). It is one file with no network resources. It talks straight to Ollama from the browser; there is no backend to run.

| File | Purpose |
|---|---|
| `nextxr-chat.html` | The whole app. Copy this one file to the demo laptop. |
| `dev/mock-ollama.mjs` | A stand-in engine with canned replies, for working on the interface without models. Not needed on the demo laptop. |

## Connecting it on the demo laptop

1. Set Ollama up as in section 2.6 of the requirements: the `setx` lines (including `OLLAMA_ORIGINS "*"`), restart Ollama, pull both models.
2. Open `nextxr-chat.html` in Chrome or Edge.
3. The status line under the message box should read **Gemma 4 E2B ready in memory**.
4. Click the sliders icon (top right). The Connection panel should show a tick for the engine and for each model. If a model is only partly on the GPU, the status line shows the percentage.

The page starts loading a model as soon as it is selected, so the load after a switch overlaps with typing the next question.

## What to edit

Everything a model change needs is in the `CONFIG` block at the top of the `<script>` in `nextxr-chat.html`.

| Setting | Default | Change it when |
|---|---|---|
| `engine.url` | `http://127.0.0.1:11434` | Ollama listens somewhere else. Can also be changed in the Connection panel. |
| `engine.api` | `ollama` | The engine is llama.cpp, LM Studio or vLLM: set `openai`. |
| `engine.keepAlive` | empty | You want the page, not `OLLAMA_KEEP_ALIVE`, to decide how long a model stays loaded. |
| `engine.preload` | `true` | You do not want a model loaded the moment it is selected. |
| `models` | Gemma 4 E2B, Llama 3.2 3B | The tags differ from what was pulled, or Llama is swapped for Granite (`granite4.1:3b`). |
| `models[].think` | not sent | A model reasons before answering and the first word is slow: set `think: false` on it. |
| `systemPrompt` | names the model, asks for the user's language | Hindi answers or tone need tuning. |
| `options` | `{ temperature: 0.7 }` | Sampling needs changing. Context size is left to `OLLAMA_CONTEXT_LENGTH`. |

## What the page calls

| When | Request |
|---|---|
| Every 5 s (every 2 s while the engine is down) | `GET /api/tags`, `GET /api/ps` |
| Page opened, model selected, engine just started | `POST /api/generate` with only the model name, which loads it |
| Message sent | `POST /api/chat` with `stream: true` and the whole conversation |

With `engine.api: 'openai'` it calls `GET /v1/models` and `POST /v1/chat/completions` instead, and does not preload.

The figures under each reply:

- **First word** is timed in the page, from Send to the first text. After a switch it includes the model load, and says so.
- **Speed** is `eval_count / eval_duration` as reported by Ollama. In `openai` mode it is the reported output tokens over the time since the first word.

## Messages and what they mean

| Message | Cause |
|---|---|
| Local engine not running. Start Ollama. | Nothing answers at the engine address. Clears by itself once Ollama is up. |
| Ollama is running but refuses requests from this page. | `OLLAMA_ORIGINS` is not set, or Ollama was not restarted after setting it. |
| *Model* is not installed on this machine. | The tag in `CONFIG.models` is not in `ollama ls`. |

## Working on the interface without models

- Open `nextxr-chat.html?preview`. Replies are simulated inside the page and a banner says so. Opening the file without `?preview` always uses the real engine.
- Or run `node dev/mock-ollama.mjs`, which listens on `127.0.0.1:11434` and answers like Ollama with canned text. Options are listed at the top of that file, including `MOCK_NO_CORS=1` and `MOCK_MODELS=` to reproduce the last two messages above.

## Not yet checked

The page has been run in Chrome 154 and Edge 154 against the mock engine only. It has not yet been run against real Ollama or the two real models. On the first real run, check:

- both models answer and the first-word and speed figures appear;
- a Hindi question gets a Hindi answer from each model;
- the switch takes about as long as section 2.5 allows;
- everything still works with Wi-Fi off.
