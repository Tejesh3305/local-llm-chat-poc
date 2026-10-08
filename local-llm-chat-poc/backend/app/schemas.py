"""Request and response models (validated by Pydantic, documented in /docs)."""

from typing import Literal

from pydantic import BaseModel, Field


class Stats(BaseModel):
    output_tokens: int | None = None
    first_token_ms: int | None = None  # time from sending the message to the first word (includes model loading)
    tokens_per_second: float | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    model: str | None = None  # which model wrote an assistant message
    stats: Stats | None = None


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
    # Let the model reason before answering (slower). Omit to use the model's default.
    thinking: bool | None = None


class UpdateConversationRequest(BaseModel):
    model: str | None = Field(default=None, max_length=64)
    title: str | None = Field(default=None, min_length=1, max_length=120)


class ModelInfo(BaseModel):
    id: str
    label: str
    description: str
    provider: str
    is_default: bool
    available: bool  # True when the runtime reports the model as installed
    loaded: bool  # True when the model is in memory right now
    thinking: bool | None  # default for the thinking switch; None when the model has no such switch


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]


class SystemInfo(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, str]  # provider id -> "ok" | "unreachable"
    # Where the model servers run: "this-machine", "local-network" or "external".
    network: Literal["this-machine", "local-network", "external"]
    endpoints: list[str]  # host names of the configured model servers
