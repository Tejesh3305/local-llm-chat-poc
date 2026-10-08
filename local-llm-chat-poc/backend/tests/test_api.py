"""API tests. No GPU or running model needed: the model registry is replaced with a fake."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import workflows
from app.config import ModelsFile
from app.llm import LLMError, ModelRegistry, StreamResult
from app.main import app, base_scenario, corpus, get_registry, get_store, rules
from app.scenario import casualty_model, comms_model, force_model, mobility_model, weather_model
from app.store import ConversationStore

MODELS = {
    "default_model": "general",
    "providers": {"local": {"base_url_env": "TEST_LLM_BASE_URL"}},
    "models": [
        {"id": "general", "label": "General", "provider": "local", "provider_model": "gemma4:e4b", "thinking": False},
        {"id": "coder", "label": "Coder", "provider": "local", "provider_model": "nemotron-3-nano:4b"},
    ],
}


class FakeOllama:
    """Stands in for Ollama's own API: tracks which models are 'in memory'."""

    def __init__(self, loaded: set[str], down: bool = False):
        self.loaded = set(loaded)
        self.down = down

    async def get(self, url, **_):
        if self.down:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"models": [{"name": name, "size": 100, "size_vram": 100} for name in self.loaded]})

    async def post(self, url, json, **_):
        if self.down:
            raise httpx.ConnectError("refused")
        (self.loaded.discard if json.get("keep_alive") == 0 else self.loaded.add)(json["model"])
        return httpx.Response(200, json={"done": True})


class FakeRegistry(ModelRegistry):
    """Real model lookup, fake runtime: replies '[<runtime model>] reply' and records what it was sent."""

    def __init__(self, fail_with: str | None = None):
        super().__init__(ModelsFile.model_validate(MODELS))
        self.fail_with = fail_with
        self.prompts: list[list[dict]] = []
        self.calls: list[dict] = []

    async def installed_models(self):
        return {"local": {"gemma4:e4b"}}  # "coder" is not installed

    def use_ollama(self, loaded: set[str], down: bool = False) -> FakeOllama:
        """Treat the provider as one of type "ollama", with these models in memory."""
        self.native["local"] = "http://fake"
        self._http = FakeOllama(loaded, down)
        return self._http

    async def stream_chat(self, model, messages, result, *, system=None, temperature=None, max_tokens=None, thinking=None):
        self.prompts.append(messages)
        self.calls.append({"system": system, "temperature": temperature, "max_tokens": max_tokens, "thinking": thinking})
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


# ---------- models and chat ----------


def test_models_lists_default_and_availability(setup):
    client, _, _ = setup
    body = client.get("/api/models").json()
    assert body["default_model"] == "general"
    assert [(m["id"], m["available"]) for m in body["models"]] == [("general", True), ("coder", False)]
    assert body["providers"] == {"local": "ok"}
    assert body["models"][0]["loaded"] is None  # this runtime cannot be asked what is in memory


def test_chat_streams_and_saves_history(setup):
    client, registry, store = setup
    first = events(client.post("/api/chat", json={"message": "hello"}))
    assert [e["type"] for e in first] == ["start", "delta", "delta", "done"]
    assert first[0]["model"] == "general"
    assert "sources" not in first[0]
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


def test_models_report_what_is_in_memory(setup):
    client, registry, _ = setup
    registry.use_ollama({"gemma4:e4b"})
    models = client.get("/api/models").json()["models"]
    assert [(m["loaded"], m["gpu_percent"]) for m in models] == [(True, 100), (False, None)]


def test_loading_a_model_unloads_the_other_one(setup):
    client, registry, _ = setup
    # "other-tool:1b" is not in this app's catalogue, so it must be left alone.
    ollama = registry.use_ollama({"gemma4:e4b", "other-tool:1b"})

    assert client.post("/api/models/coder/load").json() == {"loaded": True}
    assert ollama.loaded == {"nemotron-3-nano:4b", "other-tool:1b"}
    assert client.post("/api/models/gpt-9/load").status_code == 400


def test_chat_and_workflows_unload_the_previous_model_first(setup):
    client, registry, _ = setup
    ollama = registry.use_ollama({"gemma4:e4b"})
    stream = events(client.post("/api/chat", json={"message": "hi", "model": "coder"}))
    assert stream[-1]["type"] == "done" and ollama.loaded == set()

    ollama.loaded = {"nemotron-3-nano:4b"}
    stream = events(client.post("/api/workflows/weather/run", json={"model": "general"}))
    assert stream[-1]["type"] == "done" and ollama.loaded == set()


def test_runtime_down_while_switching_models(setup):
    client, registry, _ = setup
    registry.use_ollama(set(), down=True)

    response = client.post("/api/models/coder/load")
    assert response.status_code == 502 and response.json()["detail"] == "Local engine not running. Start the model server."
    assert events(client.post("/api/chat", json={"message": "hi"}))[-1]["type"] == "done"  # the fake reply still streams
    assert not registry.slot.locked()  # a failed request must not block the next one


def test_thinking_can_be_chosen_per_message(setup):
    client, registry, _ = setup
    client.post("/api/chat", json={"message": "hi"})
    client.post("/api/chat", json={"message": "hi", "thinking": True})
    assert [call["thinking"] for call in registry.calls] == [None, True]  # None = use the model's default


class EmptyStream:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class RecordingCompletions:
    """Captures the options sent to the runtime; streams nothing back."""

    def __init__(self):
        self.calls: list[dict] = []

    async def create(self, **options):
        self.calls.append(options)
        return EmptyStream()


@pytest.mark.parametrize(
    ("model_id", "requested", "turns_thinking_off"),
    [
        ("general", None, True),  # model default: thinking off
        ("general", False, True),
        ("general", True, False),  # asked to think: leave the runtime's default (thinking on)
        ("coder", None, False),  # no thinking switch: never send the option
        ("coder", False, False),
    ],
)
def test_thinking_option_sent_to_the_runtime(monkeypatch, model_id, requested, turns_thinking_off):
    monkeypatch.setenv("TEST_LLM_BASE_URL", "http://fake/v1")
    registry = ModelRegistry(ModelsFile.model_validate(MODELS))
    completions = RecordingCompletions()
    registry.clients["local"] = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def run():
        messages = [{"role": "user", "content": "hi"}]
        async for _ in registry.stream_chat(registry.get(model_id), messages, StreamResult(), thinking=requested):
            pass

    asyncio.run(run())
    assert (completions.calls[0].get("reasoning_effort") == "none") == turns_thinking_off


def test_health_reports_where_the_model_servers_run(setup):
    client, _, _ = setup
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["providers"] == {"local": "ok"}
    assert body["network"] == "external" and body["endpoints"] == ["fake"]


def test_validation_errors(setup):
    client, _, _ = setup
    assert client.post("/api/chat", json={"message": "hi", "model": "gpt-9"}).status_code == 400
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 20_000}).status_code == 400
    assert client.post("/api/chat", json={"message": "hi", "conversation_id": "../etc"}).status_code == 422


def test_runtime_error_is_streamed_and_nothing_saved(setup, monkeypatch):
    client, registry, store = setup
    registry.fail_with = "Local engine not running. Start Ollama."
    stream = events(client.post("/api/chat", json={"message": "hi"}))
    assert stream[-1] == {"type": "error", "message": "Local engine not running. Start Ollama."}
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


# ---------- scenario calculations and corpus search (no model involved) ----------


def test_scenario_models_give_the_expected_figures():
    weather = weather_model(base_scenario, rules)
    assert round(weather["wind_chill"]) == -41
    assert weather["cold"]["risk"] == "Very high"
    assert weather["worst"]["name"] == "Ridge Bravo, Sector 4" and weather["worst"]["risk"] == "High"
    assert weather["window"]["text"] == "0900 to 1400 h"

    mobility = mobility_model(base_scenario, rules)
    assert mobility["chosen"]["name"] == "Axis Bravo"
    assert round(mobility["chosen"]["transit_h"], 1) == 6.5
    assert (mobility["carriers"], mobility["snow_vehicles"], mobility["round_trips"]) == (12, 4, 2)
    assert [v["type"] for v in mobility["fleet"] if not v["cleared"]] == ["T-72 tank"]

    force = force_model(base_scenario, rules)
    assert round(force["ratio"], 2) == 1.42 and force["deficit"]
    assert round(force["reaction_min"]) == 45

    casualty = casualty_model(base_scenario, rules)
    assert casualty["by_helo"] and casualty["sorties"] == 1 and casualty["lift_at"] == 10 * 60 + 58
    assert round(casualty["unit"]["after"]) == 92

    comms = comms_model(base_scenario, rules)
    assert comms["active"]["name"] == "HF Net NOVEMBER"
    assert [net["name"] for net in comms["affected"]] == ["VHF Net ALPHA"]


def test_corpus_search_finds_the_right_documents():
    assert corpus.search("Which vehicles can cross Bridge 52B?")[0]["id"] == "ENG-52B"
    assert corpus.search("What do we do if VHF Net ALPHA is compromised?")[0]["id"] == "SOP-CY-02"
    assert corpus.search("ग्रिड 47B पर तैनात जवानों के लिए शीतदंश का खतरा कितना है?")[0]["id"] == "SOP-W-12"  # Hindi, through the aliases
    assert corpus.search("What is the capital of France?") == []
    assert {doc["type"] for doc in corpus.search("patrol overlap Grid 47B", types=["BPM record"])} == {"BPM record"}


def test_inputs_change_the_scenario_and_are_validated():
    changed = workflows.apply_inputs(base_scenario, {"weather.snowfall_cm": 10, "routes.0.status": "open", "comms.compromised": ["LIMA"]})
    assert changed["weather"]["snowfall_cm"] == 10 and changed["routes"][0]["status"] == "open"
    assert base_scenario["weather"]["snowfall_cm"] == 60  # the base scenario is never modified
    assert workflows.apply_inputs(base_scenario, {"weather.temp_c": -500})["weather"]["temp_c"] == -60  # clamped
    for bad in ({"weather.hourly": []}, {"routes.0.status": "maybe"}, {"movement.departure": "25:00"}, {"comms.compromised": ["MARS"]}, {"bpm.grid": ""}):
        with pytest.raises(ValueError):
            workflows.apply_inputs(base_scenario, bad)


# ---------- workflows ----------


def test_workflows_are_listed_with_inputs_and_plan(setup):
    client, _, _ = setup
    body = client.get("/api/workflows").json()
    assert [w["letter"] for w in body["workflows"]] == list("ABCDEFG")
    assert body["corpus_count"] == 14 and body["languages"][0] == "English"
    mobility = body["workflows"][0]
    assert mobility["inputs"][1] == {"path": "weather.snowfall_cm", "label": "Snowfall, next 24 h", "type": "number", "min": 0, "max": 200, "step": 5, "unit": "cm", "options": [], "value": 60}
    assert mobility["plan"][-1] == {"label": "Find references", "kind": "Corpus search"}
    assert body["situation"]["indicators"][0]["value"] == "1.42 : 1"


@pytest.mark.parametrize("workflow_id", [workflow.id for workflow in workflows.WORKFLOWS])
def test_every_workflow_computes_without_a_model(setup, workflow_id):
    client, registry, _ = setup
    result = client.post(f"/api/workflows/{workflow_id}/compute", json={}).json()["result"]
    assert result["steps"] and result["cards"] and result["refs"] and result["standard"]
    assert all(step["kind"] in ("Calculation", "Corpus search") for step in result["steps"])
    assert registry.calls == []


def test_compute_follows_the_inputs(setup):
    client, _, _ = setup
    tiles = lambda inputs: client.post("/api/workflows/mobility/compute", json={"inputs": inputs}).json()["result"]["cards"][0]["tiles"]
    assert [tile["value"] for tile in tiles({})] == ["Axis Bravo", "6.5 h", "+4.2 h", "1200 h"]
    assert [tile["value"] for tile in tiles({"weather.snowfall_cm": 10})][:2] == ["Axis Bravo", "4.9 h"]
    assert tiles({"weather.snowfall_cm": 10, "routes.0.status": "open"})[0]["value"] == "Axis Alpha"
    # the Situation panel is recalculated with the same inputs
    side = client.post("/api/workflows/weather/compute", json={"inputs": {"weather.temp_c": -5, "weather.wind_kmh": 0}}).json()["situation"]
    assert side["indicators"][1] == {"label": "Wind chill", "value": "−5 °C, low", "tone": "good"}

    assert client.post("/api/workflows/mobility/compute", json={"inputs": {"enemy.at": "x"}}).status_code == 400
    assert client.post("/api/workflows/nope/compute", json={}).status_code == 404


def test_run_streams_the_result_then_the_model_draft(setup):
    client, registry, _ = setup
    stream = events(client.post("/api/workflows/weather/run", json={"model": "general", "language": "हिन्दी"}))
    assert [e["type"] for e in stream] == ["result", "start", "delta", "delta", "done"]
    assert stream[0]["result"]["steps"][0]["detail"] == "−41 °C from −27 °C and 25 km/h wind"
    assert stream[1]["model"] == "general"

    sent, call = registry.prompts[-1][0]["content"], registry.calls[-1]
    assert sent.startswith("TASK\nWrite a weather brief") and "Write in Hindi" in sent
    assert "\nFACTS\n- Forecast: night minimum −27 °C" in sent and "\nREFERENCES\n[SOP-W-12]" in sent
    assert call["system"].startswith("You are General, working as a staff assistant")
    assert (call["temperature"], call["max_tokens"]) == (0.2, 500)

    assert client.post("/api/workflows/weather/run", json={"language": "Klingon"}).status_code == 400
    assert client.post("/api/workflows/weather/run", json={"model": "gpt-9"}).status_code == 400


def test_run_keeps_the_result_when_the_model_fails(setup):
    client, registry, _ = setup
    registry.fail_with = "Local engine not running. Start Ollama."
    stream = events(client.post("/api/workflows/bpm/run", json={}))
    assert [e["type"] for e in stream] == ["result", "start", "error"]
    assert stream[0]["result"]["standard"].startswith("**To:** Counterpart formation, by hotline")
    assert [ref["id"] for ref in stream[0]["result"]["refs"]][0] == "BPM 14/2024"


def test_a_workflow_with_its_model_step_off_never_calls_the_model(setup, monkeypatch):
    client, registry, _ = setup
    from app.main import prompts

    monkeypatch.setattr(prompts.briefs["cyber"], "enabled", False)
    stream = events(client.post("/api/workflows/cyber/run", json={}))
    assert [e["type"] for e in stream] == ["result", "done"] and stream[1]["skipped"] is True
    assert registry.calls == []


# ---------- chat with Corpus on ----------


def test_grounded_chat_reports_its_sources(setup):
    client, registry, _ = setup
    stream = events(client.post("/api/chat", json={"message": "Which vehicles can cross Bridge 52B?", "grounded": True}))
    assert stream[0]["sources"] == [{"id": "ENG-52B", "type": "Engineer report", "title": "Bridge 52B load classification"}]
    system = registry.calls[-1]["system"]
    assert system.startswith("You are General, a staff assistant") and "\nEXTRACTS\n[ENG-52B]" in system and "Bridge 52B limit 30 t" in system

    nothing = events(client.post("/api/chat", json={"message": "What is the capital of France?", "grounded": True}))
    assert nothing[0]["sources"] == []
    assert registry.calls[-1]["system"].endswith("No matching document.")


def test_corpus_is_listed(setup):
    client, _, _ = setup
    documents = client.get("/api/corpus").json()["documents"]
    assert len(documents) == 14 and set(documents[0]) == {"id", "type", "title", "text"}
