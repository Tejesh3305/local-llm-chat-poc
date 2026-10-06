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
