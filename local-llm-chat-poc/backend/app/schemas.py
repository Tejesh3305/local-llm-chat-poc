"""Request and response models (validated by Pydantic, documented in /docs)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    model: str | None = None  # which model wrote an assistant message


class Conversation(BaseModel):
    id: str
    title: str
    model: str
    messages: list[Message]
    created_at: str
    updated_at: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    model: str
    message_count: int
    updated_at: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    # Omit to start a new conversation; the server returns its id in the "start" event.
    # Send that id back on the next turn to continue the conversation.
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    # Public model id from GET /api/models. Omit to use the conversation's model or the default.
    model: str | None = Field(default=None, max_length=64)
    # Corpus on: answer from the scenario and the documents found for this message, and report the sources.
    grounded: bool = False
    # Workflow input values to apply to the scenario first (see GET /api/workflows). Only used when grounded.
    inputs: dict[str, Any] = Field(default_factory=dict)
    # Let the model reason before answering (slower). Omit to use the model's default.
    thinking: bool | None = None


class UpdateConversationRequest(BaseModel):
    model: str | None = Field(default=None, max_length=64)
    title: str | None = Field(default=None, min_length=1, max_length=120)


class ModelInfo(BaseModel):
    id: str
    label: str
    description: str
    vendor: str
    licence: str
    accent: str
    provider: str
    provider_name: str
    provider_model: str  # the runtime's name for it, shown so "not installed" can say what to pull
    local: bool  # the runtime's address is this computer
    is_default: bool
    available: bool  # True when the runtime reports the model as installed
    loaded: bool | None  # in memory now; None when the runtime cannot be asked
    gpu_percent: int | None  # share of the loaded model on the GPU


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]
    providers: dict[str, Literal["ok", "unreachable", "not_configured"]]


class SystemInfo(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, str]  # provider id -> "ok" | "unreachable"
    # Where the model servers run: "this-machine", "local-network" or "external".
    network: Literal["this-machine", "local-network", "external"]
    endpoints: list[str]  # host names of the configured model servers


class WorkflowRequest(BaseModel):
    # Input values by path, as offered by GET /api/workflows. Anything left out keeps its scenario value.
    inputs: dict[str, Any] = Field(default_factory=dict)
    model: str | None = Field(default=None, max_length=64)  # for the model step; default model if omitted
    language: str | None = Field(default=None, max_length=32)  # one of the languages in data/prompts.json
