"""FastAPI application: routes for models, chat (streaming), workflows, the corpus and conversations."""

import logging
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles

from . import workflows
from .config import BACKEND_DIR, ModelConfig, PromptsFile, get_settings, load_data_file, load_models_file
from .corpus import Corpus, extracts, reference
from .llm import LLMError, ModelRegistry, ProviderNotConfiguredError, StreamResult, UnknownModelError
from .scenario import Scenario, digest, situation
from .schemas import (
    ChatRequest,
    Conversation,
    ConversationSummary,
    Message,
    ModelInfo,
    ModelsResponse,
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
store = ConversationStore()

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


@app.get("/api/health")
async def health(registry: ModelRegistry = Depends(get_registry)):
    installed = await registry.installed_models()
    providers = {pid: ("ok" if models is not None else "unreachable") for pid, models in installed.items() if pid in registry.clients}
    healthy = any(v == "ok" for v in providers.values())
    return JSONResponse({"status": "ok" if healthy else "degraded", "providers": providers}, status_code=200 if healthy else 503)


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
    """Loads a model into memory ahead of the first message, where the runtime supports it."""
    model = resolve_model(registry, model_id)
    try:
        return {"loaded": await registry.load(model)}
    except LLMError as error:
        raise HTTPException(502, error.user_message)


# ---------- chat ----------


@dataclass
class ChatContext:
    conversation: Conversation
    model: ModelConfig
    message: str
    system: str | None = None  # replaces the model's own system prompt when Corpus is on
    sources: list[dict] | None = None  # the documents found, when Corpus is on


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
    context = ChatContext(conversation=conversation, model=model, message=body.message)

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
) -> AsyncIterable[dict]:
    """Streams Server-Sent Events: start -> delta... -> done (or error)."""
    start = {"type": "start", "conversation_id": ctx.conversation.id, "model": ctx.model.id}
    if ctx.sources is not None:
        start["sources"] = ctx.sources
    yield start

    keep = settings.max_history_messages if ctx.system is None else GROUNDED_HISTORY_MESSAGES
    history = [{"role": m.role, "content": m.content} for m in ctx.conversation.messages]
    history = history[-keep:] + [{"role": "user", "content": ctx.message}]

    parts: list[str] = []
    result = StreamResult()
    try:
        async for text in registry.stream_chat(ctx.model, history, result, system=ctx.system):
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

    yield {"type": "done", **stats(result)}


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
    model = resolve_model(registry, body.model or registry.default_model_id) if prompts.briefs[workflow.id].enabled else None
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


# ---------- the built frontend ----------

# After `npm run build` the backend serves the page itself, so one process is enough on the demo machine.
_dist = Path(settings.frontend_dist)
_dist = _dist if _dist.is_absolute() else BACKEND_DIR / _dist
if (_dist / "index.html").is_file():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
