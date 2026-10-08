"""Settings (from environment / .env), the model catalogue (models.json) and the workflow data (data/*.json)."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent

# Sent as the model id to let the backend pick a model for each message.
AUTO_MODEL = "auto"

# Load backend/.env into os.environ. Variables already set in the real
# environment win, because load_dotenv does not override by default.
load_dotenv(BACKEND_DIR / ".env")


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    models_config_path: str = "models.json"
    default_model: str | None = None
    data_dir: str = "data"  # scenario.json, rules.json, corpus.json, prompts.json
    frontend_dist: str = "../frontend/dist"  # served at / when it exists (after `npm run build`)
    # Folder for the SQLite file that holds the saved chats and the uploaded documents.
    storage_dir: str = "storage"

    cors_origins: str = "http://localhost:5173"  # comma-separated

    request_timeout_seconds: float = 120
    max_message_chars: int = 16_000
    max_history_messages: int = 40
    max_upload_mb: int = 20
    # How many excerpts of the uploaded documents are sent to the model with a question
    document_excerpts: int = 4

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def database_path(self) -> str:
        folder = Path(self.storage_dir)
        if not folder.is_absolute():
            folder = BACKEND_DIR / folder
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder / "chat.db")


class ProviderConfig(BaseModel):
    # Names of environment variables, never literal URLs or keys.
    base_url_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    name: str = "the model server"  # shown in messages: "Local engine not running. Start Ollama."
    # "ollama" also allows asking the runtime which model is in memory, loading one ahead of use
    # and unloading the others, so only one is in memory at a time.
    # "openai" is any other OpenAI-compatible server, which manages its own memory.
    type: Literal["openai", "ollama"] = "openai"


class ModelConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")  # public id used by the UI/API
    label: str
    description: str = ""
    vendor: str = ""
    licence: str = ""
    accent: Literal["teal", "violet", "amber", "blue"] = "teal"  # colour that marks this model's replies
    provider: str  # key into "providers"
    provider_model: str  # name the runtime expects, e.g. "gemma4:e2b-it-qat"
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    # For models that reason silently before answering: whether they do so by default.
    # Off is much faster for everyday questions; the UI can turn it on per message.
    # Leave unset for models/runtimes without a thinking switch.
    thinking: bool | None = None
    # What the model is good at. "vision" = can read images,
    # "code" = preferred by the Auto choice for programming questions.
    tags: list[Literal["vision", "code"]] = []
    # Shown in the AI Library: what to use this model for.
    use_cases: list[str] = []


class EmbeddingConfig(BaseModel):
    """The model that turns the text of uploaded documents into vectors for document search."""

    provider: str
    provider_model: str  # e.g. "nomic-embed-text"


class ModelsFile(BaseModel):
    default_model: str
    providers: dict[str, ProviderConfig]
    models: list[ModelConfig] = Field(min_length=1)
    embedding: EmbeddingConfig | None = None

    @model_validator(mode="after")
    def check_references(self) -> "ModelsFile":
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        if AUTO_MODEL in ids:
            raise ValueError(f"'{AUTO_MODEL}' is reserved for the automatic model choice")
        for model in self.models:
            if model.provider not in self.providers:
                raise ValueError(f"model '{model.id}' uses unknown provider '{model.provider}'")
        if self.default_model not in ids:
            raise ValueError(f"default_model '{self.default_model}' is not in models")
        if self.embedding and self.embedding.provider not in self.providers:
            raise ValueError(f"embedding uses unknown provider '{self.embedding.provider}'")
        return self


def load_models_file(path: str, default_override: str | None = None) -> ModelsFile:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = BACKEND_DIR / file_path
    models_file = ModelsFile.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
    if default_override:
        if default_override not in {model.id for model in models_file.models}:
            raise ValueError(f"DEFAULT_MODEL '{default_override}' is not defined in {file_path}")
        models_file.default_model = default_override
    return models_file


class BriefConfig(BaseModel):
    enabled: bool = True  # false: the workflow has no model step and shows its standard-format brief
    title: str
    task: str


class PromptsFile(BaseModel):
    """data/prompts.json: what the model is asked to do at the end of a workflow, and in Chat with Corpus on."""

    system: str
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=500, gt=0)
    languages: dict[str, str] = Field(min_length=1)
    briefs: dict[str, BriefConfig]
    corpus_chat: str


def load_data_file(data_dir: str, name: str) -> Any:
    folder = Path(data_dir)
    if not folder.is_absolute():
        folder = BACKEND_DIR / folder
    return json.loads((folder / name).read_text(encoding="utf-8"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
