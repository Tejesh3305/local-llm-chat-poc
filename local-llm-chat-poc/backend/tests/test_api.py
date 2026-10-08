"""API tests. No GPU or running model needed: the model registry is replaced with a fake."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app import workflows
from app.config import ModelsFile
from app.documents import DocumentLibrary, split_text
from app.llm import LLMError, ModelRegistry, StreamResult
from app.actions import file_request
from app.files import GeneratedFiles
from app.main import app, base_scenario, corpus, get_files, get_library, get_registry, get_store, rules
from app.router import choose_model
from app.scenario import casualty_model, comms_model, force_model, mobility_model, weather_model
from app.schemas import Message
from app.store import ConversationStore

MODELS = {
    "default_model": "general",
    "providers": {"local": {"base_url_env": "TEST_LLM_BASE_URL"}},
    "models": [
        {"id": "general", "label": "General", "provider": "local", "provider_model": "gemma4:e4b", "thinking": False, "tags": ["vision"]},
        {"id": "coder", "label": "Coder", "provider": "local", "provider_model": "nemotron-3-nano:4b", "tags": ["code"]},
    ],
    "embedding": {"provider": "local", "provider_model": "nomic-embed-text"},
}


class FakeOllama:
    """Stands in for Ollama's own API: tracks which models are 'in memory'."""

    def __init__(self, loaded: set[str], down: bool = False):
        self.loaded = set(loaded)
        self.down = down

    async def get(self, url, **_):
        if self.down:
            raise httpx.ConnectError("refused")
        if url.endswith("/api/tags"):  # everything stored on disk
            stored = ["gemma4:e4b", "nomic-embed-text:latest", "other-tool:1b"]
            return httpx.Response(200, json={"models": [{"name": name, "size": 500, "details": {"parameter_size": "4B"}} for name in stored]})
        return httpx.Response(200, json={"models": [{"name": name, "size": 100, "size_vram": 100} for name in self.loaded]})

    async def post(self, url, json, **_):
        if self.down:
            raise httpx.ConnectError("refused")
        if url.endswith("/api/show"):
            return httpx.Response(200, json={"capabilities": ["completion"], "model_info": {"gemma4.context_length": 8192}})
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

    async def embed(self, texts, kind="document"):
        """Fake vectors: one dimension per topic word, so texts about the same topic are similar."""
        topics = ["leave", "salary", "travel", "security"]
        return [[1.0 if topic in text.lower() else 0.01 for topic in topics] for text in texts]


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_LLM_BASE_URL", "http://fake/v1")
    registry, store, library, made = FakeRegistry(), ConversationStore(), DocumentLibrary(), GeneratedFiles(str(tmp_path / "files"))
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_library] = lambda: library
    app.dependency_overrides[get_files] = lambda: made
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
    assert store.summaries() == []


def test_conversation_crud(setup):
    client, _, _ = setup
    conversation_id = events(client.post("/api/chat", json={"message": "hello"}))[0]["conversation_id"]
    assert client.get("/api/conversations").json()[0]["id"] == conversation_id
    assert client.get(f"/api/conversations/{conversation_id}").json()["title"] == "hello"
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 204
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404


# ---------- chat history ----------


def test_conversations_survive_a_restart(tmp_path):
    path = str(tmp_path / "chat.db")
    store = ConversationStore(path)
    conversation = store.new("general")
    store.add_turn(conversation, Message(role="user", content="hello"), Message(role="assistant", content="hi", model="general"))

    reopened = ConversationStore(path)  # a new process would open the same file
    assert [c.title for c in reopened.summaries()] == ["hello"]
    assert [m.content for m in reopened.get(conversation.id).messages] == ["hello", "hi"]


def test_a_saved_chat_keeps_what_the_live_one_showed(setup):
    client, _, _ = setup
    stream = events(client.post("/api/chat", json={"message": "Which vehicles can cross Bridge 52B?", "grounded": True, "model": "auto"}))
    saved = client.get(f"/api/conversations/{stream[0]['conversation_id']}").json()["messages"][1]
    assert saved["grounded"] and saved["sources"][0]["id"] == "ENG-52B"
    assert saved["route_reason"] == "general question" and saved["stats"]["output_tokens"] == 2


def test_rename_is_saved(setup):
    client, _, store = setup
    conversation_id = events(client.post("/api/chat", json={"message": "hello"}))[0]["conversation_id"]
    client.patch(f"/api/conversations/{conversation_id}", json={"title": "Greeting"})
    assert store.get(conversation_id).title == "Greeting"
    assert client.get("/api/conversations").json()[0]["message_count"] == 2


# ---------- automatic model choice ----------


def test_auto_picks_a_model_by_rule():
    general, coder = ModelsFile.model_validate(MODELS).models
    assert choose_model([general, coder], "general", "Fix this Python function") == (coder, "programming question")
    assert choose_model([general, coder], "general", "What is the capital of France?") == (general, "general question")
    assert choose_model([general], "general", "Fix this Python function") == (general, "general question")  # no code model usable


def test_auto_only_uses_installed_models(setup):
    client, _, _ = setup
    # "coder" would be the rule's choice, but it is not installed.
    start = events(client.post("/api/chat", json={"message": "Write a Python script", "model": "auto"}))[0]
    assert (start["model"], start["route_reason"]) == ("general", "general question")
    assert events(client.post("/api/workflows/weather/run", json={"model": "auto"}))[1]["model"] == "general"


# ---------- uploaded documents ----------


def upload(client, name: str, text: str):
    return client.post("/api/documents", files={"file": (name, text.encode(), "text/plain")})


def test_answers_use_the_matching_document(setup):
    client, registry, store = setup
    assert upload(client, "leave.txt", "Annual leave is 30 days per year.").status_code == 201
    assert upload(client, "travel.txt", "Travel must be booked through the transport office.").status_code == 201
    assert [d["filename"] for d in client.get("/api/documents").json()] == ["travel.txt", "leave.txt"]

    stream = events(client.post("/api/chat", json={"message": "How much leave do I get?", "use_documents": True}))
    assert [e["type"] for e in stream] == ["start", "excerpts", "delta", "delta", "done"]
    assert stream[1]["excerpts"][0]["filename"] == "leave.txt"
    sent = registry.prompts[-1][-1]["content"]
    assert "Annual leave is 30 days" in sent and sent.endswith("Question: How much leave do I get?")

    saved = store.get(stream[0]["conversation_id"]).messages
    assert saved[0].content == "How much leave do I get?"  # the question is stored as typed, without the excerpts
    assert saved[1].excerpts[0].filename == "leave.txt"


def test_a_question_can_name_the_documents_to_answer_from(setup):
    client, registry, _ = setup
    upload(client, "leave.txt", "Annual leave is 30 days per year.")
    travel_id = upload(client, "travel.txt", "Travel must be booked through the transport office.").json()["id"]

    # A leave question, but only travel.txt was picked: leave.txt must not be used. Picking documents is enough to turn them on.
    stream = events(client.post("/api/chat", json={"message": "How much leave do I get?", "document_ids": [travel_id]}))
    assert [e["filename"] for e in stream[1]["excerpts"]] == ["travel.txt"]
    assert "Annual leave" not in registry.prompts[-1][-1]["content"]
    saved = client.get(f"/api/conversations/{stream[0]['conversation_id']}").json()["messages"][0]
    assert saved["documents"] == ["travel.txt"]  # a reopened chat shows which documents were picked

    gone = events(client.post("/api/chat", json={"message": "How much leave do I get?", "document_ids": ["0" * 32]}))
    assert [e["type"] for e in gone] == ["start", "delta", "delta", "done"]  # a removed document: answered without excerpts


def test_a_short_picked_document_is_read_whole(setup):
    client, registry, _ = setup
    document_id = upload(client, "orders.txt", "Part one about travel.\n\nPart two about leave.").json()["id"]
    stream = events(client.post("/api/chat", json={"message": "What is this document about?", "document_ids": [document_id]}))
    assert stream[1]["excerpts"][0]["position"] == 0  # the whole document, not excerpts
    sent = registry.prompts[-1][-1]["content"]
    assert "given in full" in sent and "Part one about travel." in sent and "Part two about leave." in sent


# ---------- files asked for in the chat ----------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("crate a word doc of given doc", ("docx", True)),
        ("we want to convert given doc in word format .word", ("docx", True)),
        ("convert Liam_Dao.pdf to word", ("docx", True)),
        ("make a powerpoint of this document", ("pptx", False)),
        ("create a pdf report on winter readiness", ("pdf", False)),
        ("write a summary of the document in word format", ("docx", False)),
        ("make a pdf of your last answer", ("pdf", True)),
        ("summarise the pdf", None),
        ("Explain the word count of Liam_Dao.pdf", None),
        ("what is a word", None),
    ],
)
def test_file_requests_are_recognised(message, expected):
    request = file_request(message)
    assert (request and (request.format, request.as_is)) == expected


def test_chat_converts_a_picked_document_to_word(setup):
    import io
    import zipfile

    client, registry, store = setup
    document_id = upload(client, "Liam_Dao.txt", "Liam Dao profile.\n\nServed in the northern sector for three years.").json()["id"]
    stream = events(client.post("/api/chat", json={"message": "crate a word doc of given doc", "document_ids": [document_id]}))
    assert [e["type"] for e in stream] == ["start", "excerpts", "delta", "file", "done"]
    assert stream[0]["action"] == {"type": "file", "format": "docx", "label": "Converting to a Word document"}
    assert registry.calls == []  # a conversion needs no model
    made = stream[3]["file"]
    assert made["filename"] == "Liam-Dao.docx" and made["format"] == "docx"

    response = client.get(f"/api/files/{made['id']}")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "Served in the northern sector" in archive.read("word/document.xml").decode()
    assert store.get(stream[0]["conversation_id"]).messages[1].files == [made]  # a reopened chat keeps the link

    # A follow-up that only says "the given doc" uses the document picked earlier in the chat
    again = events(client.post("/api/chat", json={"message": "convert the given doc to pdf", "conversation_id": stream[0]["conversation_id"]}))
    assert again[3]["file"]["filename"] == "Liam-Dao.pdf"
    assert client.get(f"/api/files/{again[3]['file']['id']}").content.startswith(b"%PDF")
    assert client.get(f"/api/files/{'0' * 32}").status_code == 404


def test_chat_writes_a_deck_with_the_model(setup):
    client, registry, _ = setup
    upload(client, "orders.txt", "Convoy departs at 0530 h from the transit camp.")
    stream = events(client.post("/api/chat", json={"message": "make a powerpoint of this document"}))   # the only uploaded document
    assert [e["type"] for e in stream] == ["start", "excerpts", "delta", "delta", "file", "done"]
    sent = registry.prompts[-1][0]["content"]
    assert "briefing presentation" in sent and "Convoy departs at 0530 h" in sent and registry.calls[-1]["system"].startswith("You are SAINIK AI")
    assert stream[4]["file"]["format"] == "pptx"

    plain = events(client.post("/api/chat", json={"message": "create a pdf report on winter readiness"}))
    assert [e["type"] for e in plain] == ["start", "delta", "delta", "file", "done"]  # no document involved
    assert registry.prompts[-1][0]["content"].endswith("for this request: create a pdf report on winter readiness")


def test_documents_are_not_used_unless_asked(setup):
    client, registry, _ = setup
    upload(client, "leave.txt", "Annual leave is 30 days per year.")
    stream = events(client.post("/api/chat", json={"message": "How much leave do I get?"}))
    assert [e["type"] for e in stream] == ["start", "delta", "delta", "done"]
    assert registry.prompts[-1][-1]["content"] == "How much leave do I get?"


def test_document_upload_errors_and_delete(setup):
    client, _, _ = setup
    assert upload(client, "virus.exe", "x").status_code == 400
    assert upload(client, "empty.txt", "   ").status_code == 400
    document_id = upload(client, "notes.md", "Security briefing notes.").json()["id"]
    assert client.delete(f"/api/documents/{document_id}").status_code == 204
    assert client.delete(f"/api/documents/{document_id}").status_code == 404
    assert client.get("/api/documents").json() == []


def test_long_text_is_split_into_overlapping_pieces():
    pieces = split_text("This is a sentence about travel. " * 200)
    assert len(pieces) > 5 and all(len(piece) <= 900 for piece in pieces)
    assert pieces[1][:40] in pieces[0] + pieces[1]


# ---------- AI Library ----------


def test_models_describe_themselves_for_the_library(setup):
    client, _, _ = setup
    general, coder = client.get("/api/models").json()["models"]
    assert (general["thinking"], general["tags"]) == (False, ["vision"]) and (coder["thinking"], coder["tags"]) == (None, ["code"])


def test_repository_lists_stored_models_and_their_role(setup):
    client, registry, _ = setup
    assert client.get("/api/repository").json() == []  # this runtime cannot be asked
    registry.use_ollama({"gemma4:e4b"})
    stored = {m["name"]: m for m in client.get("/api/repository").json()}
    assert {name: m["role"] for name, m in stored.items()} == {"gemma4:e4b": "chat", "nomic-embed-text:latest": "embedding", "other-tool:1b": "unused"}
    chat_model = stored["gemma4:e4b"]
    assert chat_model["catalogue_id"] == "general" and chat_model["loaded"] and chat_model["context_length"] == 8192
    assert (chat_model["parameters"], chat_model["capabilities"]) == ("4B", ["completion"])

    registry.use_ollama(set(), down=True)
    assert client.get("/api/repository").status_code == 503


# ---------- AI Studio ----------

SAMPLE = """# Patrol report
Opening paragraph with **bold** words.

## Findings
- First point
- Second point with `code`
1. A numbered point

## Recommendations
Move the reserve forward."""


def test_text_becomes_an_outline():
    from app import studio

    doc = studio.outline("", SAMPLE)
    assert doc.title == "Patrol report"
    assert [s.heading for s in doc.sections] == ["", "Findings", "Recommendations"]
    assert doc.sections[1].blocks == [("bullet", "First point"), ("bullet", "Second point with `code`"), ("bullet", "A numbered point")]
    assert studio.plain("**Bold** and *soft* `code`") == "Bold and soft code"
    deck = studio.slides(doc)
    assert [s["kind"] for s in deck] == ["title", "content", "content", "content", "end"]


@pytest.mark.parametrize(("fmt", "magic", "inside"), [("pdf", b"%PDF", None), ("docx", b"PK", "word/document.xml"), ("pptx", b"PK", "ppt/slides/slide2.xml")])
def test_export_makes_a_real_file(setup, fmt, magic, inside):
    import io
    import zipfile

    client, registry, _ = setup
    response = client.post("/api/studio/export", json={"format": fmt, "title": "", "content": SAMPLE})
    assert response.status_code == 200 and response.content.startswith(magic)
    assert response.headers["content-disposition"] == f'attachment; filename="Patrol-report.{fmt}"'
    if inside:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert "Second point" in archive.read(inside).decode() or fmt == "pptx"
    assert registry.calls == []  # no model needed
    assert client.post("/api/studio/export", json={"format": "odt", "content": "x"}).status_code == 422


def test_outline_preview(setup):
    client, _, _ = setup
    body = client.post("/api/studio/outline", json={"format": "pptx", "content": SAMPLE}).json()
    assert body["title"] == "Patrol report" and body["filename"] == "Patrol-report.pptx" and len(body["slides"]) == 5


def test_demo_image_follows_the_prompt(setup):
    client, _, _ = setup
    first = client.post("/api/studio/image", json={"prompt": "Snow ridge at dawn", "style": "landscape"}).json()
    again = client.post("/api/studio/image", json={"prompt": "Snow ridge at dawn", "style": "landscape"}).json()
    other = client.post("/api/studio/image", json={"prompt": "Desert convoy", "style": "terrain map"}).json()
    assert first["svg"] == again["svg"] and first["svg"] != other["svg"]  # same prompt, same picture
    assert (first["palette"], other["palette"]) == ("snow", "desert")
    assert first["svg"].startswith("<svg") and "NOT A GENERATIVE MODEL" in first["svg"]
    assert client.post("/api/studio/image", json={"prompt": "<script>", "style": "poster"}).json()["svg"].count("<script>") == 0


def test_draft_streams_from_the_model(setup):
    client, registry, _ = setup
    stream = events(client.post("/api/studio/draft", json={"topic": "Winter road clearance", "kind": "briefing", "model": "auto"}))
    assert [e["type"] for e in stream] == ["start", "delta", "delta", "done"] and stream[0]["model"] == "general"
    assert "briefing presentation" in registry.prompts[-1][0]["content"] and registry.calls[-1]["system"].startswith("You are SAINIK AI")


def test_image_analysis_uses_a_model_that_can_read_images(setup):
    client, registry, _ = setup
    pixel = "data:image/png;base64,iVBORw0KGgo="
    stream = events(client.post("/api/studio/image-analysis", json={"image": pixel, "task": "text", "model": "coder"}))
    assert stream[0]["model"] == "general"  # "coder" cannot read images, so the vision model is used
    content = registry.prompts[-1][0]["content"]
    assert content[0]["type"] == "text" and content[1] == {"type": "image_url", "image_url": {"url": pixel}}
    assert client.post("/api/studio/image-analysis", json={"image": "data:text/html;base64,AAAA"}).status_code == 422

    registry.config.models[0].tags = []  # no model can read images: a plain 400, before any streaming
    response = client.post("/api/studio/image-analysis", json={"image": pixel})
    assert response.status_code == 400 and "read images" in response.json()["detail"]


def test_document_analysis_and_insights(setup):
    client, registry, _ = setup
    text = "Annual leave policy. " * 30 + "Effective 12 Jan 2026 for all units."
    document_id = upload(client, "leave.txt", text).json()["id"]
    facts = client.get(f"/api/documents/{document_id}/insights").json()
    assert facts["keywords"][:2] == ["annual", "leave"] and "12 Jan 2026" in facts["dates"] and facts["words"] > 90

    stream = events(client.post("/api/studio/document-analysis", json={"document_id": document_id, "task": "actions"}))
    assert stream[0]["task"] == "Actions and risks" and stream[0]["read_percent"] == 100 and stream[-1]["type"] == "done"
    sent = registry.prompts[-1][0]["content"]
    assert "Effective 12 Jan 2026" in sent and sent.count("Annual leave policy.") == 30  # the pieces are joined without repeats
    assert client.get(f"/api/documents/{'0' * 32}/insights").status_code == 404


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
