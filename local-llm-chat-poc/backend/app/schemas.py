"""Request and response models (validated by Pydantic, documented in /docs)."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """An excerpt of an uploaded document that was given to the model for an answer."""

    document_id: str
    filename: str
    position: int  # which piece of the document (1 = first)
    text: str
    score: float  # similarity to the question, 0..1


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    model: str | None = None  # which model wrote an assistant message
    # The rest is kept so a reopened chat shows what the live one did.
    documents: list[str] = []  # user message: the documents picked with @, by file name
    route_reason: str | None = None  # why the Auto choice picked this model
    grounded: bool = False  # answered with Corpus on
    sources: list[dict] | None = None  # the corpus documents found, when Corpus was on
    excerpts: list[Source] = []  # the excerpts of uploaded documents that were used
    stats: dict | None = None  # the figures of the "done" event
    files: list[dict] = []  # assistant message: files made for the user ({id, filename, format, size_bytes})


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
    # Public model id from GET /api/models, or "auto" to let the server choose.
    # Omit to use the conversation's model or the default.
    model: str | None = Field(default=None, max_length=64)
    # Answer from the uploaded documents: the most relevant excerpts are added to the question.
    use_documents: bool = False
    # Answer only from these uploaded documents (ids from GET /api/documents). Implies use_documents.
    document_ids: list[str] = Field(default_factory=list, max_length=20)
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
    thinking: bool | None  # default for the Think switch; None when the model has no such switch
    tags: list[str]
    use_cases: list[str]


class ModelsResponse(BaseModel):
    default_model: str
    models: list[ModelInfo]
    providers: dict[str, Literal["ok", "unreachable", "not_configured"]]


class DocumentInfo(BaseModel):
    id: str
    filename: str
    size_bytes: int
    chunks: int
    created_at: str


class RepositoryModel(BaseModel):
    """A model stored in the local runtime, whether or not this app's catalogue uses it."""

    name: str
    provider: str
    size_bytes: int | None = None
    parameters: str | None = None  # e.g. "7.5B"
    quantization: str | None = None  # e.g. "Q4_K_M"
    family: str | None = None
    context_length: int | None = None  # the most tokens the model can read at once
    capabilities: list[str] = []
    loaded: bool = False
    memory_bytes: int | None = None  # memory in use while loaded
    role: Literal["chat", "embedding", "unused"] = "unused"  # how this app uses it
    catalogue_id: str | None = None


class SystemInfo(BaseModel):
    status: Literal["ok", "degraded"]
    providers: dict[str, str]  # provider id -> "ok" | "unreachable"
    # Where the model servers run: "this-machine", "local-network" or "external".
    network: Literal["this-machine", "local-network", "external"]
    endpoints: list[str]  # host names of the configured model servers


class DraftRequest(BaseModel):
    """AI Studio: ask the model to write text that is then turned into a file."""

    topic: str = Field(min_length=3, max_length=4000)
    kind: Literal["report", "letter", "briefing"] = "report"
    model: str | None = Field(default=None, max_length=64)


class ExportRequest(BaseModel):
    """AI Studio: Markdown-style text to turn into a PDF, Word or PowerPoint file."""

    format: Literal["pdf", "docx", "pptx"]
    title: str = Field(default="", max_length=200)
    content: str = Field(min_length=1, max_length=60_000)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=2, max_length=500)
    style: Literal["terrain map", "landscape", "poster"] = "landscape"


class ImageAnalysisRequest(BaseModel):
    image: str = Field(pattern=r"^data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$")
    task: Literal["describe", "summary", "text"] = "describe"
    model: str | None = Field(default=None, max_length=64)


class DocumentAnalysisRequest(BaseModel):
    document_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    task: Literal["summary", "key_points", "entities", "actions", "hindi"] = "summary"
    model: str | None = Field(default=None, max_length=64)


class WorkflowRequest(BaseModel):
    # Input values by path, as offered by GET /api/workflows. Anything left out keeps its scenario value.
    inputs: dict[str, Any] = Field(default_factory=dict)
    model: str | None = Field(default=None, max_length=64)  # for the model step; default model if omitted
    language: str | None = Field(default=None, max_length=32)  # one of the languages in data/prompts.json
