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
