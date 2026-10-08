"""API tests. No GPU or running model needed: the model registry is replaced with a fake."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import ModelsFile
from app.llm import LLMError, ModelRegistry, StreamResult
from app.main import app, get_registry, get_store
from app.store import ConversationStore

MODELS = {
    "default_model": "general",
    "providers": {"local": {"base_url_env": "TEST_LLM_BASE_URL"}},
    "models": [
        {"id": "general", "label": "General", "provider": "local", "provider_model": "gemma4:e4b", "thinking": False},
        {"id": "coder", "label": "Coder", "provider": "local", "provider_model": "nemotron-3-nano:4b"},
    ],
}


class FakeRuntime:
    """Stands in for OllamaRuntime: tracks which models are 'in memory'."""

    def __init__(self, loaded: set[str], fail_with: Exception | None = None):
        self.loaded = set(loaded)
        self.fail_with = fail_with

    async def running(self) -> set[str]:
        if self.fail_with:
            raise self.fail_with
        return set(self.loaded)

    async def load(self, name: str) -> None:
        self.loaded.add(name)

    async def unload(self, name: str) -> None:
        self.loaded.discard(name)


class FakeRegistry(ModelRegistry):
    """Real model lookup, fake runtime: replies '[<runtime model>] reply' and records prompts."""

    def __init__(self, fail_with: str | None = None):
        super().__init__(ModelsFile.model_validate(MODELS))
        self.fail_with = fail_with
        self.prompts: list[list[dict]] = []
        self.thinking: list[bool | None] = []

    async def installed_models(self):
        return {"local": {"gemma4:e4b"}}  # "coder" is not installed

    async def stream_chat(self, model, messages, result, thinking=None):
        self.prompts.append(messages)
        self.thinking.append(thinking)
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


def test_models_report_what_is_in_memory(setup):
    client, registry, _ = setup
    assert [m["loaded"] for m in client.get("/api/models").json()["models"]] == [False, False]
    registry.runtimes["local"] = FakeRuntime({"gemma4:e4b"})
    assert [m["loaded"] for m in client.get("/api/models").json()["models"]] == [True, False]


def test_loading_a_model_unloads_the_other_one(setup):
    client, registry, _ = setup
    # "other-tool:1b" is not in this app's catalogue, so it must be left alone.
    runtime = registry.runtimes["local"] = FakeRuntime({"gemma4:e4b", "other-tool:1b"})

    assert client.post("/api/models/coder/load").status_code == 204
    assert runtime.loaded == {"nemotron-3-nano:4b", "other-tool:1b"}
    assert client.post("/api/models/gpt-9/load").status_code == 400


def test_chat_unloads_the_previous_model_first(setup):
    client, registry, _ = setup
    runtime = registry.runtimes["local"] = FakeRuntime({"gemma4:e4b"})

    stream = events(client.post("/api/chat", json={"message": "hi", "model": "coder"}))
    assert stream[-1]["type"] == "done"
    assert "gemma4:e4b" not in runtime.loaded


def test_runtime_down_while_switching_models(setup):
    client, registry, store = setup
    registry.runtimes["local"] = FakeRuntime(set(), fail_with=httpx.ConnectError("refused"))

    response = client.post("/api/models/coder/load")
    assert response.status_code == 503
    assert "not reachable" in response.json()["detail"]

    stream = events(client.post("/api/chat", json={"message": "hi"}))
    assert stream[-1]["type"] == "error" and "not reachable" in stream[-1]["message"]
    assert store.list() == []
    assert not registry.slot.locked()  # a failed request must not block the next one


def test_thinking_switch(setup):
    client, registry, _ = setup
    # "general" has a thinking switch (off by default); "coder" has none.
    assert [m["thinking"] for m in client.get("/api/models").json()["models"]] == [False, None]

    client.post("/api/chat", json={"message": "hi"})
    client.post("/api/chat", json={"message": "hi", "thinking": True})
    assert registry.thinking == [None, True]  # None = use the model's default


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
        async for _ in registry.stream_chat(registry.get(model_id), messages, StreamResult(), requested):
            pass

    asyncio.run(run())
    assert (completions.calls[0].get("reasoning_effort") == "none") == turns_thinking_off


def test_done_event_reports_speed(setup):
    client, _, store = setup
    stream = events(client.post("/api/chat", json={"message": "hello"}))
    done = stream[-1]
    assert done["output_tokens"] == 2 and done["first_token_ms"] >= 0 and done["tokens_per_second"] > 0
    saved = store.get(stream[0]["conversation_id"]).messages[-1]
    assert saved.stats.tokens_per_second == done["tokens_per_second"]


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
