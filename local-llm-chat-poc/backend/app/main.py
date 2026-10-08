"""FastAPI application: routes for models, chat (streaming), workflows, the corpus, conversations and documents."""

import asyncio
import base64
import logging
import random
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
from .imagegen import ImageGenerator
from .vision import Detector, summary_line, targets_for
from .llm import LLMError, ModelRegistry, ProviderNotConfiguredError, StreamResult, UnknownModelError
from .router import choose_model
from .scenario import Scenario, digest, situation
from .schemas import (
    ChatRequest,
    Conversation,
    ConversationSummary,
    DetectRequest,
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
images = ImageGenerator(str(BACKEND_DIR / settings.image_model_dir))
images.warm_up()
detector = Detector(str(BACKEND_DIR / settings.detection_model_dir))

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


def get_images() -> ImageGenerator:
    return images


def get_detector() -> Detector:
    return detector


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
    picture: actions.PictureRequest | None = None  # the message asks for a picture
    image: str | None = None  # a picture sent with the message (data URL)


async def prepare_chat(
    body: ChatRequest,
    registry: ModelRegistry = Depends(get_registry),
    store: ConversationStore = Depends(get_store),
    library: DocumentLibrary = Depends(get_library),
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
    if body.image:
        # A picture with the message: answered by a model that can read images (and the detector, when asked)
        if len(body.image) > MAX_IMAGE_CHARS:
            raise HTTPException(413, "The picture is too large. Use one under 6 MB.")
        context.image = body.image
        if "vision" not in model.tags:
            reader = next((m for m in registry.config.models if "vision" in m.tags and m.provider in registry.clients), None)
            if reader is None:
                raise HTTPException(400, 'No model in models.json can read images. Tag one with "vision".')
            context.model, context.route_reason = reader, ("picture attached" if body.model == AUTO_MODEL else context.route_reason)
        return context
    if not body.grounded:
        context.file = actions.file_request(body.message)
        context.picture = None if context.file else actions.picture_request(body.message)
        if not (context.file or context.picture) and conversation and actions.REPEAT.match(body.message):
            # "try again" after a picture or file request repeats that request
            previous = next((m for m in reversed(conversation.messages) if m.role == "user"), None)
            if previous:
                context.picture = actions.picture_request(previous.content)
                context.file = None if context.picture else actions.file_request(previous.content)
                if context.file and previous.documents:  # with the documents picked for it
                    context.document_ids = [doc.id for doc in library.documents() if doc.filename in previous.documents] or None

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
    images: ImageGenerator = Depends(get_images),
    detector: Detector = Depends(get_detector),
) -> AsyncIterable[dict]:
    """Streams Server-Sent Events: start -> (excerpts) -> delta... -> (file) -> done (or error)."""
    if ctx.image:
        async for event in vision_turn(ctx, registry, store, files, detector):
            yield event
        return
    if ctx.file:
        async for event in file_turn(ctx, registry, store, library, files):
            yield event
        return
    if ctx.picture:
        async for event in picture_turn(ctx, registry, store, files, images):
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


FILE_HISTORY_MESSAGES = 6  # earlier turns sent with a file request, for context


def earlier_answer(conversation: Conversation, topic: tuple[str, ...]) -> tuple[str, str] | None:
    """The earlier answer a file request means, and the question it answered.

    Answers that were themselves files or pictures are skipped ("convert it into ppt" twice must not turn the
    first deck into the second). When the request names a topic ("the tank information"), the latest answer
    that mentions it is used; otherwise the latest answer.
    """
    pairs = []
    for index, message in enumerate(conversation.messages):
        if message.role == "assistant" and not message.files and message.content.strip():
            question = next((m.content for m in reversed(conversation.messages[:index]) if m.role == "user"), "")
            pairs.append((question, message.content))
    if not pairs:
        return None
    if topic:
        stems = [word.rstrip("s") for word in topic if len(word) > 2]
        scored = [(sum(stem in (q + " " + a).lower() for stem in stems), i) for i, (q, a) in enumerate(pairs)]
        best = max(scored)
        if best[0] > 0:
            return pairs[best[1]]
    return pairs[-1]


def title_from(question: str) -> str:
    """A file title from the question an answer replied to: 'tell me all the type of indian tanks' -> 'Indian tanks'."""
    text = re.sub(r"\s+", " ", question).strip(" ?.!")
    text = re.sub(r"^(please\s+)?(tell me|explain|describe|give me|list|what are|what is|show me)\s+(about\s+)?(all\s+)?(the\s+)?", "", text, flags=re.I)
    text = re.sub(r"^(types?|kinds?)\s+of\s+", "", text, flags=re.I) if len(text.split()) > 3 else text
    return (text[:1].upper() + text[1:70]) if text else "Chat answer"


def strip_preamble(markdown: str) -> str:
    """Drop chatter a model puts before the document itself ("Sure, here is…", "I cannot create a .ppt file…")."""
    lines = markdown.splitlines()
    first = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    return "\n".join(lines[first:]).strip() if first else markdown.strip()


def file_sources(ctx: ChatContext, library: DocumentLibrary) -> list[tuple[str, str, str]]:
    """The documents a file request is about: only those picked with @ (or, for API callers, use_documents).

    Never guessed from words like "it" or "the given doc": the user always knows what the model reads.
    """
    if ctx.document_ids:
        return picked_texts(ctx, library)
    if ctx.use_documents:
        return [(doc.id, doc.filename, library.text(doc.id) or "") for doc in library.documents()[:3]]
    return []


async def file_turn(
    ctx: ChatContext, registry: ModelRegistry, store: ConversationStore, library: DocumentLibrary, files: GeneratedFiles
) -> AsyncIterable[dict]:
    """A chat message that asks for a PDF, Word or PowerPoint file.

    Converting a document (or an earlier answer) copies its text into the file, with no model involved.
    Anything else, such as a summary or a briefing deck, is written by the model first.
    """
    request = ctx.file
    sources = file_sources(ctx, library)
    earlier = earlier_answer(ctx.conversation, request.topic) if request.refers_back or request.mentions_answer else None
    if sources:
        name = ", ".join(filename for _, filename, _ in sources)
        text = sources[0][2] if len(sources) == 1 else "\n\n".join(f"## {filename}\n\n{body}" for _, filename, body in sources)
        default_title = actions.stem(sources[0][1])
    elif request.mentions_document and not request.mentions_answer:
        # "Convert the given doc", but no document was picked: ask, instead of guessing which one
        first = next(iter(library.documents()), None)
        tip = (
            "Which document should I use? Type **@** in the message box and pick it, then ask again, for example: "
            f"*@{first.filename if first else 'report.pdf'} {ctx.message}*. "
            "I only read a document when you pick it with @."
        )
        yield {"type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id}
        yield {"type": "delta", "text": tip}
        store.add_turn(ctx.conversation, Message(role="user", content=ctx.message), Message(role="assistant", content=tip, model=ctx.model.id))
        yield {"type": "done", **stats(StreamResult())}
        return
    elif earlier:
        # "convert it into ppt": an answer given earlier in this chat is the content
        name, text, default_title = "an earlier answer", earlier[1], title_from(earlier[0])
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
            if sources else f"Here is the earlier answer about **{default_title}** as a {request.label}."
        )
        yield {"type": "delta", "text": reply}
    else:
        kind = studio.DRAFT_KINDS["briefing" if request.format == "pptx" else "report"]
        # The model only writes the text; it must not answer that it "cannot create a .ppt file"
        rules = (
            f"The app turns your text into a {request.label} automatically. Write only the content, in Markdown, "
            "starting with '# ' and a title. No introduction, no notes about files or formats, no remarks about what you can or cannot do."
        )
        if text:
            ask = (
                f"{rules}\n\nThe user asked: \"{ctx.message}\". Write {kind}, based only on the source below; "
                f"do not add facts that are not in it.\n\nSOURCE ({name})\n{text[:WHOLE_DOCUMENT_CHARS]}"
            )
            recent = []  # the source is the context; earlier turns (an older deck, say) would only confuse
        else:
            ask = f"{rules}\n\nWrite {kind} for this request: {ctx.message}"
            # The recent turns, so "the second one" or "that tank" in the request still make sense
            recent = [{"role": m.role, "content": m.content} for m in ctx.conversation.messages][-FILE_HISTORY_MESSAGES:]
        parts: list[str] = []
        try:
            async with registry.slot:
                await registry.activate(ctx.model)
                async for piece in registry.stream_chat(
                    ctx.model, recent + [{"role": "user", "content": ask}], result, system=studio.DRAFT_SYSTEM, temperature=0.4, max_tokens=1800
                ):
                    parts.append(piece)
                    yield {"type": "delta", "text": piece}
        except LLMError as error:
            logger.warning("file request failed (model=%s): %s", ctx.model.id, error.__cause__ or error)
            yield {"type": "error", "message": error.user_message}
            return
        content = reply = strip_preamble("".join(parts))
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


async def picture_turn(
    ctx: ChatContext, registry: ModelRegistry, store: ConversationStore, files: GeneratedFiles, images: ImageGenerator
) -> AsyncIterable[dict]:
    """A chat message that asks for a picture: drawn by the image model, or by the demo renderer without one."""
    request = ctx.picture
    engine = images.status()
    real = engine["available"]
    label = f"Drawing with {engine['model']}" if real else "Drawing a demo picture"
    yield {"type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id, "action": {"type": "image", "label": label}}

    name = studio.filename(request.subject[:50] or "picture", "png" if real else "svg")
    result = StreamResult()
    if real:
        seed = random.randint(0, 2**31 - 1)
        try:
            async with registry.slot:  # the GPU is shared: the chat model is unloaded to make room
                await registry.activate(None)
                png, details = await asyncio.to_thread(images.generate, request.subject, request.style, seed, 2)
        except Exception as error:
            logger.exception("image generation in the chat failed")
            yield {"type": "error", "message": f"The image model could not draw the picture: {error}"}
            return
        made = files.save(png, name, "png")
        reply = (
            f"Here is a picture of **{request.subject}**, drawn by {engine['model']} on this machine in {details['seconds']} s "
            f"(Powered by Stability AI). It is a small model, so details may be imperfect; ask again for a different version."
        )
    else:
        svg, _ = studio.demo_image(request.subject, "landscape" if request.style == "photo" else request.style)
        made = files.save(svg.encode(), name, "svg")
        reply = (
            f"No image model is installed, so this is a **demo drawing** for \"{request.subject}\", not a real picture. "
            f"{engine['reason']}"
        )
    yield {"type": "delta", "text": reply}
    yield {"type": "file", "file": made}
    store.add_turn(
        ctx.conversation,
        Message(role="user", content=ctx.message),
        Message(role="assistant", content=reply, model=ctx.model.id, stats=stats(result), files=[made]),
    )
    yield {"type": "done", **stats(result)}


def decode_image(data_url: str) -> tuple[bytes, str]:
    """The bytes and file extension of a picture sent as a data URL."""
    header, data = data_url.split(",", 1)
    return base64.b64decode(data), "jpg" if "jpeg" in header else header.split("/")[1].split(";")[0]


async def run_detection(registry: ModelRegistry, detector: Detector, image: bytes, wanted: str) -> tuple[bytes, list[dict], dict, bool]:
    """Boxes for the objects named in `wanted` (or the default military and everyday list). Holds the model slot:
    the detector shares the GPU, so it does not run while a picture is being drawn."""
    targets, named = targets_for(wanted)
    async with registry.slot:
        boxed, found, info = await asyncio.to_thread(detector.detect, image, targets)
    info["looked_for"] = [label for label, _ in targets] if named else []
    return boxed, found, info, named


async def vision_turn(ctx: ChatContext, registry: ModelRegistry, store: ConversationStore, files: GeneratedFiles, detector: Detector) -> AsyncIterable[dict]:
    """A message with a picture: object detection (boxes), classification, or any question about the picture.

    Detection runs OWLv2 first, then the vision model explains what was found. Everything else goes to the
    vision model with the picture.
    """
    intent = actions.vision_intent(ctx.message)
    engine = detector.status()
    detecting = intent == "detect" and engine["available"]
    label = "Detecting objects with OWLv2" if detecting else "Classifying the picture" if intent == "classify" else "Looking at the picture"
    yield {
        "type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id,
        "route_reason": ctx.route_reason, "action": {"type": "vision", "intent": intent, "label": label},
    }
    raw, extension = decode_image(ctx.image)
    sent = files.save(raw, f"picture.{extension}", "png" if extension == "png" else "jpg")

    made: list[dict] = []
    lead = ""
    if intent == "detect" and not engine["available"]:
        lead = f"*Object detection is not installed, so the boxes cannot be drawn ({engine['reason']}). Here is what the model sees instead.*\n\n"
        task = f"{ctx.message}\n\nList each kind of object you can see in the picture, how many, and where (left, centre, right, top, bottom)."
    elif detecting:
        try:
            boxed, found, info, named = await run_detection(registry, detector, raw, ctx.message)
        except Exception as error:
            logger.exception("object detection failed")
            yield {"type": "error", "message": f"Object detection failed: {error}"}
            return
        made.append(files.save(boxed, "detections.png", "png"))
        yield {"type": "file", "file": made[0]}
        what = ", ".join(info["looked_for"]) if named else "military and everyday objects"
        if found:
            rows = "\n".join(f"| {d['label']} | {round(d['score'] * 100)}% |" for d in found[:15])
            lead = (
                f"**Detected ({engine['model']}, {info['seconds']} s on {info['device']}):** {summary_line(found)}.\n\n"
                f"| Object | Confidence |\n|---|---|\n{rows}\n\n"
            )
            listing = "; ".join(f"{d['label']} ({round(d['score'] * 100)}%, box {[round(v) for v in d['box']]})" for d in found[:25])
            task = (
                f"{ctx.message}\n\nAn object detector looked for {what} and found: {listing}. "
                f"The picture is {info['width']} x {info['height']} pixels. In 2 to 5 short sentences, say what the picture shows, "
                "answer the question using these detections, and point out any detection that looks wrong or anything it missed."
            )
        else:
            lead = f"**No {what} found** by {engine['model']} ({info['seconds']} s). Small, distant or hidden objects can be missed.\n\n"
            task = f"{ctx.message}\n\nAn object detector looked for {what} and found nothing. Say briefly what the picture does show."
    elif intent == "classify":
        task = f"{studio.CLASSIFY_PROMPT}\n\nThe user asked: {ctx.message}"
    else:
        task = ctx.message

    if lead:
        yield {"type": "delta", "text": lead}
    # Earlier turns as plain text; only this message carries the picture
    history = [{"role": m.role, "content": m.content} for m in ctx.conversation.messages][-settings.max_history_messages :]
    history.append({"role": "user", "content": [{"type": "text", "text": task}, {"type": "image_url", "image_url": {"url": ctx.image}}]})
    parts: list[str] = []
    result = StreamResult()
    try:
        async with registry.slot:
            await registry.activate(ctx.model)
            async for piece in registry.stream_chat(ctx.model, history, result, system=studio.VISION_SYSTEM, temperature=0.3, max_tokens=900):
                parts.append(piece)
                yield {"type": "delta", "text": piece}
    except LLMError as error:
        logger.warning("vision chat failed (model=%s): %s", ctx.model.id, error.__cause__ or error)
        if not made:
            yield {"type": "error", "message": error.user_message}
            return
        note = f"\n\n*The model could not add an explanation: {error.user_message}*"
        parts.append(note)
        yield {"type": "delta", "text": note}

    store.add_turn(
        ctx.conversation,
        Message(role="user", content=ctx.message, image_file=sent),
        Message(role="assistant", content=lead + "".join(parts), model=ctx.model.id, route_reason=ctx.route_reason, stats=stats(result), files=made),
    )
    yield {"type": "done", **stats(result)}


@app.get("/api/studio/detect-engine")
def detect_engine(detector: Detector = Depends(get_detector)):
    """Whether object detection is installed, and if not, why not."""
    return detector.status()


@app.post("/api/studio/detect")
async def studio_detect(body: DetectRequest, registry: ModelRegistry = Depends(get_registry), detector: Detector = Depends(get_detector)):
    """Boxes around the objects in a picture: the ones named in `objects`, or military and everyday objects."""
    if len(body.image) > MAX_IMAGE_CHARS:
        raise HTTPException(413, "The picture is too large. Use one under 6 MB.")
    if not detector.status()["available"]:
        raise HTTPException(503, detector.status()["reason"])
    raw, _ = decode_image(body.image)
    try:
        boxed, found, info, named = await run_detection(registry, detector, raw, f"detect {body.objects}" if body.objects.strip() else "")
    except Exception as error:
        logger.exception("object detection failed")
        raise HTTPException(500, f"Object detection failed: {error}")
    return {
        "image": "data:image/png;base64," + base64.b64encode(boxed).decode(),
        "detections": found, "summary": summary_line(found), "model": detector.status()["model"], **info,
    }


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


@app.get("/api/studio/image-engine")
def image_engine(images: ImageGenerator = Depends(get_images)):
    """Whether pictures come from the image model (SD-Turbo) or the demo renderer, and why."""
    return images.status()


@app.post("/api/studio/image")
async def studio_image(body: ImageRequest, registry: ModelRegistry = Depends(get_registry), images: ImageGenerator = Depends(get_images)):
    """A picture from the prompt: drawn by SD-Turbo when it is installed, else by the demo renderer (labelled as such)."""
    if body.engine == "model" or (body.engine == "auto" and images.status()["available"]):
        if not images.status()["available"]:
            raise HTTPException(503, images.status()["reason"])
        seed = body.seed if body.seed is not None else random.randint(0, 2**31 - 1)
        try:
            async with registry.slot:  # the GPU is shared: no chat answer while drawing
                await registry.activate(None)  # unload the chat model to make room
                png, details = await asyncio.to_thread(images.generate, body.prompt, body.style, seed, body.steps)
        except Exception as error:  # out of GPU memory, damaged model files...
            logger.exception("image generation failed")
            raise HTTPException(500, f"The image model could not draw the picture: {error}")
        image = "data:image/png;base64," + base64.b64encode(png).decode()
        return {"kind": "model", "image": image, "model": images.status()["model"], "style": body.style, **details}
    svg, details = studio.demo_image(body.prompt, body.style)
    return {"kind": "demo", "svg": svg, **details}


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
