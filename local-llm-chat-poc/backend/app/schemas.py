"""Request and response models (validated by Pydantic, documented in /docs)."""

from typing import Literal

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


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]
