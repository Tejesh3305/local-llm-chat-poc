"""FastAPI application: routes for models, chat (streaming), workflows, the corpus, conversations and documents."""

import logging
import re
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles

from . import actions, studio, workflows
from .config import AUTO_MODEL, BACKEND_DIR, ModelConfig, PromptsFile, get_settings, load_data_file, load_models_file
from .corpus import Corpus, extracts, reference
from .documents import SUPPORTED, DocumentError, DocumentLibrary, with_documents, with_excerpts
from .files import GeneratedFiles
from .llm import LLMError, ModelRegistry, ProviderNotConfiguredError, StreamResult, UnknownModelError
from .router import choose_model
from .scenario import Scenario, digest, situation
from .schemas import (
    ChatRequest,
    Conversation,
    ConversationSummary,
    DocumentAnalysisRequest,
    DocumentInfo,
    DraftRequest,
    ExportRequest,
    ImageAnalysisRequest,
    ImageRequest,
    Message,
    ModelInfo,
    ModelsResponse,
    RepositoryModel,
    Source,
    SystemInfo,
    UpdateConversationRequest,
    WorkflowRequest,
)
from .store import ConversationStore

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

registry = ModelRegistry(
    load_models_file(settings.models_config_path, settings.default_model),
    timeout_seconds=settings.request_timeout_seconds,
)
store = ConversationStore(settings.database_path)
library = DocumentLibrary(settings.database_path)
files = GeneratedFiles(settings.files_path)

# Workflow data. All of it is sample data for the exercise scenario; see data/*.json.
base_scenario: Scenario = load_data_file(settings.data_dir, "scenario.json")
rules = load_data_file(settings.data_dir, "rules.json")
_corpus_file = load_data_file(settings.data_dir, "corpus.json")
corpus = Corpus(_corpus_file["documents"], _corpus_file.get("aliases"))
prompts = PromptsFile.model_validate(load_data_file(settings.data_dir, "prompts.json"))

# Run every workflow once now, so a mistake in the data files stops the server with a clear error.
for _workflow in workflows.WORKFLOWS:
    if _workflow.id not in prompts.briefs:
        raise ValueError(f"data/prompts.json has no entry under 'briefs' for workflow '{_workflow.id}'")
    workflows.run(_workflow, base_scenario, rules, corpus)

GROUNDED_HISTORY_MESSAGES = 6  # the scenario and extracts take most of a small model's context
# Picked documents up to this length are given to the model whole, instead of the excerpts closest to the question,
# so "summarise it" or "what is this about" work. It is also what a small model can read in one go.
WHOLE_DOCUMENT_CHARS = 12_000
MAX_CONVERTED_CHARS = 400_000  # a document copied as it is into a file

app = FastAPI(title="Local LLM Chat API", version="1.1.0")
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


def get_library() -> DocumentLibrary:
    return library


def get_files() -> GeneratedFiles:
    return files


def resolve_model(registry: ModelRegistry, model_id: str) -> ModelConfig:
    try:
        return registry.get(model_id)
    except UnknownModelError:
        raise HTTPException(400, f"Unknown model '{model_id}'. See GET /api/models.")
    except ProviderNotConfiguredError as error:
        raise HTTPException(503, f"Provider '{error}' is not configured (missing base URL in .env).")


def scenario_for(inputs: dict[str, Any]) -> Scenario:
    """The scenario with the UI's input values applied."""
    try:
        return workflows.apply_inputs(base_scenario, inputs)
    except ValueError as error:
        raise HTTPException(400, str(error))


def stats(result: StreamResult) -> dict:
    return {
        "finish_reason": result.finish_reason,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "first_token_ms": result.first_token_ms,
        "generation_ms": result.generation_ms,
    }


# ---------- health & models ----------


@app.get("/api/health", response_model=SystemInfo)
async def health(registry: ModelRegistry = Depends(get_registry)):
    installed = await registry.installed_models()
    providers = {pid: ("ok" if models is not None else "unreachable") for pid, models in installed.items() if pid in registry.clients}
    healthy = any(v == "ok" for v in providers.values())
    network, endpoints = registry.network()
    info = SystemInfo(status="ok" if healthy else "degraded", providers=providers, network=network, endpoints=endpoints)
    return JSONResponse(info.model_dump(), status_code=200 if healthy else 503)


@app.get("/api/models", response_model=ModelsResponse)
async def list_models(registry: ModelRegistry = Depends(get_registry)):
    installed = await registry.installed_models()
    loaded = await registry.loaded_models(reachable={pid for pid, names in installed.items() if names is not None})
    models = []
    for m in registry.config.models:
        name, in_memory = registry.runtime_name(m), loaded.get(m.provider)
        models.append(
            ModelInfo(
                id=m.id,
                label=m.label,
                description=m.description,
                vendor=m.vendor,
                licence=m.licence,
                accent=m.accent,
                provider=m.provider,
                provider_name=registry.config.providers[m.provider].name,
                provider_model=m.provider_model,
                local=m.provider in registry.on_this_machine,
                is_default=m.id == registry.default_model_id,
                available=name in (installed.get(m.provider) or set()),
                loaded=None if in_memory is None else name in in_memory,
                gpu_percent=in_memory.get(name) if in_memory else None,
                thinking=m.thinking,
                tags=m.tags,
                use_cases=m.use_cases,
            )
        )
    return ModelsResponse(
        default_model=registry.default_model_id,
        models=models,
        providers={
            pid: "not_configured" if pid not in registry.clients else "ok" if names is not None else "unreachable"
            for pid, names in installed.items()
        },
    )


@app.post("/api/models/{model_id}/load")
async def load_model(model_id: str, registry: ModelRegistry = Depends(get_registry)):
    """Loads a model into memory ahead of the first message, where the runtime supports it.

    The catalogue's other models are unloaded first, so only one is in memory at a time.
    """
    model = resolve_model(registry, model_id)
    try:
        async with registry.slot:  # waits for a generation that is still running
            await registry.activate(model)
            return {"loaded": await registry.load(model)}
    except LLMError as error:
        raise HTTPException(502, error.user_message)


@app.get("/api/repository", response_model=list[RepositoryModel])
async def repository(registry: ModelRegistry = Depends(get_registry)):
    """Every model stored in the local runtimes, including ones this app's catalogue does not use."""
    try:
        stored = await registry.repository()
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("the runtime did not list its models: %s", error)
        raise HTTPException(503, "Local engine not running, so its stored models cannot be listed.")
    embedding = registry.config.embedding
    # Ollama reports names with a tag; "nomic-embed-text" means "nomic-embed-text:latest".
    embedding_name = embedding and (embedding.provider_model if ":" in embedding.provider_model else f"{embedding.provider_model}:latest").lower()
    result = []
    for item in stored:
        entry = next((m for m in registry.config.models if m.provider == item["provider"] and registry.runtime_name(m) == item["name"]), None)
        is_embedding = embedding is not None and embedding.provider == item["provider"] and item["name"] == embedding_name
        role = "chat" if entry else "embedding" if is_embedding else "unused"
        result.append(RepositoryModel(**item, role=role, catalogue_id=entry.id if entry else None))
    return sorted(result, key=lambda m: (m.role == "unused", m.name))


# ---------- chat ----------


@dataclass
class ChatContext:
    conversation: Conversation
    model: ModelConfig
    message: str
    thinking: bool | None = None  # None: use the model's default
    use_documents: bool = False  # answer from the uploaded documents
    document_ids: list[str] | None = None  # only these documents; None = all of them
    route_reason: str | None = None  # set when the Auto choice picked the model
    system: str | None = None  # replaces the model's own system prompt when Corpus is on
    sources: list[dict] | None = None  # the documents found, when Corpus is on
    file: actions.FileRequest | None = None  # the message asks for a PDF, Word or PowerPoint file


async def prepare_chat(
    body: ChatRequest,
    registry: ModelRegistry = Depends(get_registry),
    store: ConversationStore = Depends(get_store),
) -> ChatContext:
    """Runs BEFORE streaming starts, so validation errors become normal 4xx JSON responses."""
    if len(body.message) > settings.max_message_chars:
        raise HTTPException(400, f"Message is longer than {settings.max_message_chars} characters.")

    # An unknown id is treated as a new conversation (e.g. the first turn failed before it was saved).
    conversation = store.get(body.conversation_id) if body.conversation_id else None

    route_reason = None
    if body.model == AUTO_MODEL:
        installed = await registry.installed_models()
        usable = [
            m
            for m in registry.config.models
            # A provider that does not answer is not ruled out here: the chat reports that error itself.
            if m.provider in registry.clients and (installed.get(m.provider) is None or registry.runtime_name(m) in installed[m.provider])
        ]
        if not usable:
            raise HTTPException(503, "None of the models in the catalogue is installed. See GET /api/models.")
        model, route_reason = choose_model(usable, registry.default_model_id, body.message)
    else:
        # Model resolution: request > conversation > default
        model_id = body.model or (conversation.model if conversation else registry.default_model_id)
        model = resolve_model(registry, model_id)

    if conversation is None:
        conversation = store.new(model.id, body.conversation_id)
    context = ChatContext(
        conversation=conversation, model=model, message=body.message,
        thinking=body.thinking, use_documents=body.use_documents or bool(body.document_ids),
        document_ids=body.document_ids or None, route_reason=route_reason,
    )
    if not body.grounded:
        context.file = actions.file_request(body.message)

    if body.grounded:
        scenario = scenario_for(body.inputs)
        documents = corpus.search(body.message, limit=3)
        context.sources = [reference(doc) for doc in documents]
        context.system = (
            prompts.corpus_chat.replace("{model}", model.label)
            + f"\n\nSCENARIO: {scenario['name']}, as of {scenario['as_of']} (exercise data)\n{digest(scenario, rules)}"
            + f"\n\nEXTRACTS\n{extracts(documents) or 'No matching document.'}"
        )
    return context


@app.post("/api/chat", response_class=EventSourceResponse)
async def chat(
    ctx: ChatContext = Depends(prepare_chat),
    registry: ModelRegistry = Depends(get_registry),
    store: ConversationStore = Depends(get_store),
    library: DocumentLibrary = Depends(get_library),
    files: GeneratedFiles = Depends(get_files),
) -> AsyncIterable[dict]:
    """Streams Server-Sent Events: start -> (excerpts) -> delta... -> (file) -> done (or error)."""
    if ctx.file:
        async for event in file_turn(ctx, registry, store, library, files):
            yield event
        return

    start = {"type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id}
    if ctx.route_reason:
        start["route_reason"] = ctx.route_reason
    if ctx.sources is not None:
        start["sources"] = ctx.sources
    yield start

    keep = settings.max_history_messages if ctx.system is None else GROUNDED_HISTORY_MESSAGES
    history = [{"role": m.role, "content": m.content} for m in ctx.conversation.messages][-keep:]

    parts: list[str] = []
    excerpts: list[Source] = []
    result = StreamResult()
    try:
        question = ctx.message
        whole = picked_texts(ctx, library)
        if whole and sum(len(text) for _, _, text in whole) <= WHOLE_DOCUMENT_CHARS:
            # Short enough to read in full: better than the few excerpts closest to the question
            excerpts = [Source(document_id=doc_id, filename=name, position=0, text=text[:400], score=1.0) for doc_id, name, text in whole]
            yield {"type": "excerpts", "excerpts": [excerpt.model_dump() for excerpt in excerpts]}
            question = with_documents(ctx.message, [(name, text) for _, name, text in whole])
        elif ctx.use_documents:
            excerpts = await library.search(ctx.message, registry.embed, settings.document_excerpts, ctx.document_ids)
            if excerpts:
                yield {"type": "excerpts", "excerpts": [excerpt.model_dump() for excerpt in excerpts]}
                question = with_excerpts(ctx.message, excerpts)
        history.append({"role": "user", "content": question})

        async with registry.slot:  # one model in memory, so one generation at a time
            await registry.activate(ctx.model)
            async for text in registry.stream_chat(ctx.model, history, result, system=ctx.system, thinking=ctx.thinking):
                parts.append(text)
                yield {"type": "delta", "text": text}
    except LLMError as error:
        logger.warning("chat failed (model=%s): %s", ctx.model.id, error.__cause__ or error)
        yield {"type": "error", "message": error.user_message}
        return
    finally:
        # Also runs when the user presses Stop (client disconnect): keep the partial answer.
        if parts:
            picked = [doc.filename for doc in library.documents() if doc.id in ctx.document_ids] if ctx.document_ids else []
            store.add_turn(
                ctx.conversation,
                Message(role="user", content=ctx.message, documents=picked),
                Message(
                    role="assistant", content="".join(parts), model=ctx.model.id, route_reason=ctx.route_reason,
                    grounded=ctx.system is not None, sources=ctx.sources, excerpts=excerpts, stats=stats(result),
                ),
            )

    yield {"type": "done", **stats(result)}


def picked_texts(ctx: ChatContext, library: DocumentLibrary) -> list[tuple[str, str, str]]:
    """(id, file name, text) of the documents picked with @ for this message."""
    if not ctx.document_ids:
        return []
    return [(doc.id, doc.filename, library.text(doc.id) or "") for doc in library.documents() if doc.id in ctx.document_ids]


def file_sources(ctx: ChatContext, library: DocumentLibrary) -> list[tuple[str, str, str]]:
    """The documents a file request is about: picked with @, else all with Documents on, else the ones the
    message refers to ("the given doc"): those picked earlier in this chat, or the only uploaded document."""
    if ctx.document_ids:
        return picked_texts(ctx, library)
    documents = library.documents()
    if ctx.use_documents:
        chosen = documents[:3]
    elif ctx.file and ctx.file.mentions_document:
        earlier = next((m.documents for m in reversed(ctx.conversation.messages) if m.role == "user" and m.documents), [])
        chosen = [doc for doc in documents if doc.filename in earlier] or (documents if len(documents) == 1 else [])
    else:
        chosen = []
    return [(doc.id, doc.filename, library.text(doc.id) or "") for doc in chosen]


async def file_turn(
    ctx: ChatContext, registry: ModelRegistry, store: ConversationStore, library: DocumentLibrary, files: GeneratedFiles
) -> AsyncIterable[dict]:
    """A chat message that asks for a PDF, Word or PowerPoint file.

    Converting a document (or an earlier answer) copies its text into the file, with no model involved.
    Anything else, such as a summary or a briefing deck, is written by the model first.
    """
    request = ctx.file
    sources = file_sources(ctx, library)
    earlier = next((m.content for m in reversed(ctx.conversation.messages) if m.role == "assistant"), None)
    if sources:
        name = ", ".join(filename for _, filename, _ in sources)
        text = sources[0][2] if len(sources) == 1 else "\n\n".join(f"## {filename}\n\n{body}" for _, filename, body in sources)
        default_title = actions.stem(sources[0][1])
    elif request.mentions_answer and earlier:
        name, text, default_title = "the previous answer", earlier, "Chat answer"
    else:
        name, text, default_title = None, None, ""
    copy = request.as_is and bool(text)

    label = f"{'Converting to' if copy else 'Writing'} a {request.label}"
    yield {
        "type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id,
        "action": {"type": "file", "format": request.format, "label": label},
    }
    if sources:
        refs = [Source(document_id=doc_id, filename=filename, position=0, text=body[:400], score=1.0) for doc_id, filename, body in sources]
        yield {"type": "excerpts", "excerpts": [ref.model_dump() for ref in refs]}

    result = StreamResult()
    if copy:
        content = text[:MAX_CONVERTED_CHARS]
        words = len(content.split())
        reply = (
            f"Here is **{name}** as a {request.label}: about {words:,} words. "
            "The text is copied as it is; pictures, tables and the original page layout are not carried over."
            if sources else f"Here is {name} as a {request.label}."
        )
        yield {"type": "delta", "text": reply}
    else:
        kind = studio.DRAFT_KINDS["briefing" if request.format == "pptx" else "report"]
        if text:
            ask = (
                f"{ctx.message}\n\nWrite it as {kind}. Base it only on the source below and do not add facts that are not in it.\n\n"
                f"SOURCE ({name})\n{text[:WHOLE_DOCUMENT_CHARS]}"
            )
        else:
            ask = f"Write {kind} for this request: {ctx.message}"
        parts: list[str] = []
        try:
            async with registry.slot:
                await registry.activate(ctx.model)
                async for piece in registry.stream_chat(
                    ctx.model, [{"role": "user", "content": ask}], result, system=studio.DRAFT_SYSTEM, temperature=0.4, max_tokens=1800
                ):
                    parts.append(piece)
                    yield {"type": "delta", "text": piece}
        except LLMError as error:
            logger.warning("file request failed (model=%s): %s", ctx.model.id, error.__cause__ or error)
            yield {"type": "error", "message": error.user_message}
            return
        content = reply = "".join(parts)
        if not content.strip():
            yield {"type": "error", "message": "The model returned no text to put in the file. Please try again."}
            return

    document = studio.outline(default_title if copy else "", content)   # the model's draft brings its own "# Title"
    if document.title == "Untitled document" and default_title:
        document.title = default_title
    made = files.save(studio.RENDERERS[request.format](document), studio.filename(document.title, request.format), request.format)
    yield {"type": "file", "file": made}

    store.add_turn(
        ctx.conversation,
        Message(role="user", content=ctx.message, documents=[filename for _, filename, _ in sources] if ctx.document_ids else []),
        Message(role="assistant", content=reply, model=ctx.model.id, route_reason=ctx.route_reason,
                excerpts=[Source(document_id=d, filename=f, position=0, text=b[:400], score=1.0) for d, f, b in sources],
                stats=stats(result), files=[made]),
    )
    yield {"type": "done", **stats(result)}


@app.get("/api/files/{file_id}")
def download_file(file_id: str, files: GeneratedFiles = Depends(get_files)):
    """A file made in the chat."""
    path = files.get(file_id) if re.fullmatch(r"[a-f0-9]{32}", file_id) else None
    if path is None:
        raise HTTPException(404, "File not found. It may have been removed.")
    fmt = path.suffix.lstrip(".")
    return FileResponse(path, media_type=studio.FORMATS.get(fmt, "application/octet-stream"), filename=path.name)


# ---------- workflows ----------


def find_workflow(workflow_id: str) -> workflows.Workflow:
    workflow = workflows.BY_ID.get(workflow_id)
    if workflow is None:
        raise HTTPException(404, f"Unknown workflow '{workflow_id}'. See GET /api/workflows.")
    return workflow


@app.get("/api/workflows")
def list_workflows():
    """The workflows with their inputs, plus what the page needs around them: scenario, situation, languages."""
    described = []
    for workflow in workflows.WORKFLOWS:
        steps = workflows.run(workflow, base_scenario, rules, corpus).result["steps"]
        described.append({**workflows.describe(workflow, base_scenario, prompts), "plan": [{"label": s["label"], "kind": s["kind"]} for s in steps]})
    return {
        "scenario": {key: base_scenario[key] for key in ("name", "title", "as_of", "questions")},
        "languages": list(prompts.languages),
        "workflows": described,
        "situation": situation(base_scenario, rules),
        "corpus_count": len(corpus.documents),
    }


@app.post("/api/workflows/{workflow_id}/compute")
def compute_workflow(workflow_id: str, body: WorkflowRequest):
    """The calculations and references only, with no model step. Used to recalculate as inputs change."""
    workflow, scenario = find_workflow(workflow_id), scenario_for(body.inputs)
    return {"result": workflows.run(workflow, scenario, rules, corpus).result, "situation": situation(scenario, rules)}


@dataclass
class RunContext:
    workflow: workflows.Workflow
    scenario: Scenario
    computed: workflows.Computed
    model: ModelConfig | None  # None when this workflow's model step is switched off
    language: str


def prepare_run(workflow_id: str, body: WorkflowRequest, registry: ModelRegistry = Depends(get_registry)) -> RunContext:
    """Runs BEFORE streaming starts, so bad requests still get a proper 4xx status."""
    workflow, scenario = find_workflow(workflow_id), scenario_for(body.inputs)
    language = body.language or next(iter(prompts.languages))
    if language not in prompts.languages:
        raise HTTPException(400, f"Unknown language '{language}'. Choose one of: {', '.join(prompts.languages)}.")
    # Auto is a chat choice; a workflow draft goes to the default model.
    model_id = body.model if body.model and body.model != AUTO_MODEL else registry.default_model_id
    model = resolve_model(registry, model_id) if prompts.briefs[workflow.id].enabled else None
    return RunContext(workflow, scenario, workflows.run(workflow, scenario, rules, corpus), model, language)


@app.post("/api/workflows/{workflow_id}/run", response_class=EventSourceResponse)
async def run_workflow(ctx: RunContext = Depends(prepare_run), registry: ModelRegistry = Depends(get_registry)) -> AsyncIterable[dict]:
    """Streams Server-Sent Events: result (calculations and references) -> start -> delta... -> done (or error).

    The result never depends on the model. If the model step fails, the client still has everything else.
    """
    yield {"type": "result", "result": ctx.computed.result, "situation": situation(ctx.scenario, rules)}
    if ctx.model is None:
        yield {"type": "done", "skipped": True}
        return

    yield {"type": "start", "model": ctx.model.id}
    system, user = workflows.draft_messages(prompts, ctx.workflow, ctx.computed, ctx.model.label, ctx.language)
    result = StreamResult()
    try:
        async with registry.slot:  # one model in memory, so one generation at a time
            await registry.activate(ctx.model)
            async for text in registry.stream_chat(
                ctx.model, [{"role": "user", "content": user}], result,
                system=system, temperature=prompts.temperature, max_tokens=prompts.max_tokens,
            ):
                yield {"type": "delta", "text": text}
    except LLMError as error:
        logger.warning("workflow draft failed (workflow=%s, model=%s): %s", ctx.workflow.id, ctx.model.id, error.__cause__ or error)
        yield {"type": "error", "message": error.user_message}
        return
    yield {"type": "done", **stats(result)}


@app.get("/api/corpus")
def list_corpus():
    """The sample documents the search runs over."""
    return {"documents": [{**reference(doc), "text": doc["text"]} for doc in corpus.documents]}


# ---------- conversations ----------


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations(store: ConversationStore = Depends(get_store)):
    return store.summaries()


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
    store.update(conversation)
    return conversation


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, store: ConversationStore = Depends(get_store)):
    if not store.delete(conversation_id):
        raise HTTPException(404, "Conversation not found.")
    return Response(status_code=204)


# ---------- uploaded documents ----------


@app.get("/api/documents", response_model=list[DocumentInfo])
def list_documents(library: DocumentLibrary = Depends(get_library)):
    return library.documents()


@app.post("/api/documents", response_model=DocumentInfo, status_code=201)
async def upload_document(
    file: UploadFile, registry: ModelRegistry = Depends(get_registry), library: DocumentLibrary = Depends(get_library)
):
    """Read the file, split it into pieces and store their embeddings for document search."""
    filename = (file.filename or "document").replace("\\", "/").rsplit("/", 1)[-1][:200]
    if not filename.lower().endswith(SUPPORTED):
        raise HTTPException(400, f"Unsupported file type. Upload one of: {', '.join(SUPPORTED)}")
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"The file is larger than {settings.max_upload_mb} MB.")
    try:
        return await library.add(filename, data, registry.embed)
    except DocumentError as error:
        raise HTTPException(400, str(error))
    except LLMError as error:
        logger.warning("embedding failed for '%s': %s", filename, error.__cause__ or error)
        raise HTTPException(503, error.user_message)


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str, library: DocumentLibrary = Depends(get_library)):
    if not library.delete(document_id):
        raise HTTPException(404, "Document not found.")
    return Response(status_code=204)


# ---------- AI Studio ----------

STUDIO_DOCUMENT_CHARS = 12_000  # what a small model can read in one go, with room for its answer
MAX_IMAGE_CHARS = 8_000_000  # about 6 MB of picture as base64


def studio_model(registry: ModelRegistry, model_id: str | None, need_vision: bool = False) -> ModelConfig:
    """The model a Studio task runs on: the one asked for, or the default (Auto is a chat choice).

    For a picture, a model that can read images is used instead when the chosen one cannot.
    """
    model = resolve_model(registry, model_id if model_id and model_id != AUTO_MODEL else registry.default_model_id)
    if need_vision and "vision" not in model.tags:
        reader = next((m for m in registry.config.models if "vision" in m.tags and m.provider in registry.clients), None)
        if reader is None:
            raise HTTPException(400, 'No model in models.json can read images. Tag one with "vision".')
        model = reader
    return model


@dataclass
class StudioJob:
    """One Studio request for the model, checked before the stream starts so mistakes get a proper 4xx status."""

    model: ModelConfig
    content: str | list[dict]  # the user message; a list when it carries a picture
    system: str
    temperature: float
    max_tokens: int
    extra: dict  # added to the "start" event


async def model_events(registry: ModelRegistry, job: StudioJob) -> AsyncIterable[dict]:
    """Server-Sent Events for one Studio request: start -> delta... -> done (or error)."""
    yield {"type": "start", "model": job.model.id, **job.extra}
    result = StreamResult()
    try:
        async with registry.slot:  # one model in memory, so one generation at a time
            await registry.activate(job.model)
            async for text in registry.stream_chat(
                job.model, [{"role": "user", "content": job.content}], result,
                system=job.system, temperature=job.temperature, max_tokens=job.max_tokens,
            ):
                yield {"type": "delta", "text": text}
    except LLMError as error:
        logger.warning("studio request failed (model=%s): %s", job.model.id, error.__cause__ or error)
        yield {"type": "error", "message": error.user_message}
        return
    yield {"type": "done", **stats(result)}


def prepare_draft(body: DraftRequest, registry: ModelRegistry = Depends(get_registry)) -> StudioJob:
    ask = f"Write {studio.DRAFT_KINDS[body.kind]} about the following.\n\n{body.topic}"
    return StudioJob(studio_model(registry, body.model), ask, studio.DRAFT_SYSTEM, 0.5, 1400, {})


@app.post("/api/studio/draft", response_class=EventSourceResponse)
async def studio_draft(job: StudioJob = Depends(prepare_draft), registry: ModelRegistry = Depends(get_registry)) -> AsyncIterable[dict]:
    """The model writes a report, letter or briefing in Markdown, ready to be exported."""
    async for event in model_events(registry, job):
        yield event


@app.post("/api/studio/export")
def studio_export(body: ExportRequest):
    """The text as a PDF, Word or PowerPoint file. No model involved."""
    document = studio.outline(body.title, body.content)
    data = studio.RENDERERS[body.format](document)
    name = studio.filename(document.title, body.format)
    return Response(data, media_type=studio.FORMATS[body.format], headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/studio/outline")
def studio_outline(body: ExportRequest):
    """How the text will be laid out: sections for PDF and Word, slides for PowerPoint. Used for the preview."""
    document = studio.outline(body.title, body.content)
    return {**document.as_dict(), "slides": studio.slides(document), "filename": studio.filename(document.title, body.format)}


@app.post("/api/studio/image")
def studio_image(body: ImageRequest):
    """A demo picture drawn from the prompt's words. Not a generative model, and labelled as such."""
    svg, details = studio.demo_image(body.prompt, body.style)
    return {"svg": svg, **details}


def prepare_image_analysis(body: ImageAnalysisRequest, registry: ModelRegistry = Depends(get_registry)) -> StudioJob:
    if len(body.image) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "The picture is too large. Use one under 6 MB.")
    label, task = studio.IMAGE_TASKS[body.task]
    content = [{"type": "text", "text": task}, {"type": "image_url", "image_url": {"url": body.image}}]
    system = "You are SAINIK AI, describing images for a staff officer. Say only what you can see; if unsure, say so."
    return StudioJob(studio_model(registry, body.model, need_vision=True), content, system, 0.3, 700, {"task": label})


@app.post("/api/studio/image-analysis", response_class=EventSourceResponse)
async def studio_image_analysis(job: StudioJob = Depends(prepare_image_analysis), registry: ModelRegistry = Depends(get_registry)) -> AsyncIterable[dict]:
    """A model that can read images describes, summarises or reads the text in a picture."""
    async for event in model_events(registry, job):
        yield event


@app.get("/api/documents/{document_id}/insights")
def document_insights(document_id: str, library: DocumentLibrary = Depends(get_library)):
    """Figures about an uploaded document that need no model."""
    text = library.text(document_id)
    if text is None:
        raise HTTPException(404, "Document not found.")
    return studio.insights(text)


def prepare_document_analysis(
    body: DocumentAnalysisRequest, registry: ModelRegistry = Depends(get_registry), library: DocumentLibrary = Depends(get_library)
) -> StudioJob:
    text = library.text(body.document_id)
    if text is None:
        raise HTTPException(404, "Document not found.")
    label, task = studio.ANALYSIS_TASKS[body.task]
    read = min(1.0, STUDIO_DOCUMENT_CHARS / max(1, len(text)))
    ask = f"{task}\n\nDOCUMENT\n{text[:STUDIO_DOCUMENT_CHARS]}"
    return StudioJob(studio_model(registry, body.model), ask, studio.ANALYSIS_SYSTEM, 0.2, 900, {"task": label, "read_percent": round(read * 100)})


@app.post("/api/studio/document-analysis", response_class=EventSourceResponse)
async def studio_document_analysis(job: StudioJob = Depends(prepare_document_analysis), registry: ModelRegistry = Depends(get_registry)) -> AsyncIterable[dict]:
    """The model reads an uploaded document and writes a summary, key points, names and dates, or actions."""
    async for event in model_events(registry, job):
        yield event


# ---------- the built frontend ----------

# After `npm run build` the backend serves the page itself, so one process is enough on the demo machine.
_dist = Path(settings.frontend_dist)
_dist = _dist if _dist.is_absolute() else BACKEND_DIR / _dist
if (_dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
