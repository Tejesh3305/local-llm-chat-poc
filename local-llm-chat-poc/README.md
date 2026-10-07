# NextXR local AI: chat and workflows

A React frontend and a FastAPI backend. The browser talks only to the backend; the backend talks to the local model runtime (Ollama) and does all the calculating and searching.

```text
Browser (React)  ──►  FastAPI backend  ──►  Ollama on this machine
  frontend/             backend/app/          gemma4:e2b-it-qat, llama3.2:3b
```

The page has two views, switched at the top left:

- **Chat**: talk to either model and switch between them mid-conversation. The **Corpus** button makes the model answer from the scenario and the sample documents, and lists its sources.
- **Workflows**: the seven outputs of the illustrative scenario (mobility, operational plan, tactical recommendation, weather, BPM, casualty, cyber).

## Running it

One-time setup, while online:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # then set RELOAD=false for the demo

cd ..\frontend
npm install
npm run build                      # writes frontend/dist, which the backend serves
```

Every time after that, with Ollama running:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m app                      # open http://127.0.0.1:8000
```

Nothing is fetched from the internet once this is installed. Because the browser never calls Ollama directly, `OLLAMA_ORIGINS` does not need to be set.

**Without a GPU or models:** run `python scripts/mock_llm_server.py` in a second terminal and set `OLLAMA_BASE_URL=http://127.0.0.1:11500/v1` in `backend/.env`. Replies are canned and say so.

**While changing the frontend:** run `npm run dev` in `frontend/` and open http://127.0.0.1:5173. It reloads on save and forwards `/api` to the backend.

## Connecting the models

| File | Holds |
|---|---|
| `backend/.env` | Where the runtime listens: `OLLAMA_BASE_URL=http://127.0.0.1:11434/v1` |
| `backend/models.json` | The two models: the name shown, the Ollama tag (`provider_model`), vendor, licence, system prompt, temperature |

The status line under the message box should read **Gemma 4 E2B ready in memory**. The sliders icon (top right) lists each model as installed or not, in memory or not, and the share on the GPU. The page asks the backend to load a model as soon as it is selected, so the load after a switch overlaps with typing the next question.

If a model reasons before answering and the first word is slow, that is a runtime setting; the backend uses the OpenAI-compatible API and sends no thinking option.

## How a workflow runs, and what is the model

| Stage | Where | Label on screen |
|---|---|---|
| Calculation | `backend/app/scenario.py` and `workflows.py`: plain Python over the scenario and the rules | *Calculation* on each step; *Calculated, no model* over the result cards |
| Corpus search | `backend/app/corpus.py`: keyword search (BM25) over the sample documents | *Corpus search* |
| Draft | The active model writes the brief from the calculated facts and the references found | *Model draft*, with the model name and speed |

The calculations never depend on the model. If the runtime is down, or the draft is stopped before its first word, the workflow shows its brief in a fixed format built from the same facts, labelled *Standard format, no model*.

| Output | Calculated | The model drafts |
|---|---|---|
| A Mobility plan | Route choice, convoy speed on snow, bridge limits, vehicles, timings, fuel | Movement brief |
| B Operational plan | Force ratio against the trigger, reserve reaction time, deployment matrix | Commander's summary |
| C Tactical recommendation | Runs A, D and the force model together; ranks actions, each with its reason | Tactical assessment |
| D Weather analysis | Wind chill and cold injury band, exposure limit, avalanche risk by slope, flying window | Weather brief |
| E BPM and diplomacy | Finds the meeting records that apply; hotline deadline; interlocutor level | Hotline message |
| F Casualty planning | Helicopter or road, sorties and times, filled-in 9-line request, strength over 48 hours | Casualty brief |
| G Cyber and communications | Nets lost, next net in order, containment timeline, signal teams | Incident report |

**Deciding what to show as AI.** Run each workflow on both real models. Where a draft is not good enough, set `"enabled": false` for that workflow in `backend/data/prompts.json` and restart the backend. It then has no model step, shows the standard-format brief, and is not labelled as a model draft.

## What to edit

All workflow data is in `backend/data/`. It is read at startup; every workflow is run once then, so a mistake stops the backend with an error instead of surfacing mid-demo.

| File | Holds | Change it when |
|---|---|---|
| `scenario.json` | The situation: weather, routes, units, casualties, nets, reports | The scenario changes. All of it is invented sample data. |
| `rules.json` | The planning rules behind the calculations | Real SOP figures are available. The current values are samples. |
| `corpus.json` | The 14 sample documents, and the query aliases (including Hindi terms) | Real extracts are available. Add `{ id, type, title, text }`. |
| `prompts.json` | The system prompt, each workflow's task, temperature, length limit, languages | Tuning the drafts, or switching a workflow's model step off |

For each draft the model receives the system prompt and one message:

```text
TASK
<briefs.<id>.task> <the language instruction>

FACTS
- <one line per calculated fact>

REFERENCES
[<document id>] <title>
<document text>
```

A whole request is about 750 tokens, so it fits a 4,096-token context with room for the answer. `temperature` is 0.2 to keep small models close to the facts. A draft that reaches `max_tokens` (500) is marked *Cut off at the length limit*; Hindi needs more tokens than English.

Chat with **Corpus** on uses `corpus_chat` as its system prompt, followed by a digest of the scenario and the three best-matching documents.

## API

Interactive docs: http://127.0.0.1:8000/docs

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/models` | The models, whether each is installed and in memory, and each provider's status |
| `POST` | `/api/models/{id}/load` | Load a model into memory ahead of use |
| `POST` | `/api/chat` | Send a message; streams the reply. `grounded: true` answers from the corpus and reports sources |
| `GET` | `/api/workflows` | The workflows with their inputs, the situation, the languages |
| `POST` | `/api/workflows/{id}/compute` | Calculations and references only, for given input values |
| `POST` | `/api/workflows/{id}/run` | Streams the calculations, then the model's draft |
| `GET` | `/api/corpus` | The sample documents |
| `GET` | `/api/health`, `/api/conversations…` | As before |

The speed figures under each reply are measured by the backend: **First word** is from sending the request to the runtime until its first text (so it includes a model load), and **Speed** is output tokens over the time from first text to last.

## What has been checked

- `pytest` in `backend/`: 24 tests covering the calculations, the search, every workflow, the model step and Corpus chat. None needs a model.
- `npm run build` in `frontend/`: type-check and production build.
- End to end in Chrome 154 and Edge 154 against the mock runtime, through the backend: chat, model switching, Stop, Retry, a stopped runtime, a model that is not installed, Corpus chat with sources, all seven workflows, live recalculation, a workflow with its model step off.

Not yet checked, because it needs the real models:

- how good each model's drafts are, in English and in Hindi, and whether they stay within the facts;
- whether Corpus answers quote the right document numbers;
- first-word time and speed on the demo laptop;
- everything with Wi-Fi off on the demo laptop.

`local-llm-chat-guide.md` in the folder above describes the earlier version of this app. Its setup sections still apply; its API list, model list and source appendix predate the workflows.
