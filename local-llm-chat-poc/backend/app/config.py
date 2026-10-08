"""Settings (from environment / .env) and the model catalogue (from models.json)."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parent.parent

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

    cors_origins: str = "http://localhost:5173"  # comma-separated

    request_timeout_seconds: float = 120
    max_message_chars: int = 16_000
    max_history_messages: int = 40

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


class ProviderConfig(BaseModel):
    # Names of environment variables, never literal URLs or keys.
    base_url_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]*$")
    # "ollama" lets the app load/unload models so only one is in memory at a time.
    # "openai" is any other OpenAI-compatible server, which manages its own memory.
    runtime: Literal["openai", "ollama"] = "openai"


class ModelConfig(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")  # public id used by the UI/API
    label: str
    description: str = ""
    provider: str  # key into "providers"
    provider_model: str  # name the runtime expects, e.g. "gemma4:e4b"
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    # For models that reason silently before answering: whether they do so by default.
    # Off is much faster for everyday questions; the UI can turn it on per message.
    # Leave unset for models/runtimes without a thinking switch.
    thinking: bool | None = None


class ModelsFile(BaseModel):
    default_model: str
    providers: dict[str, ProviderConfig]
    models: list[ModelConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def check_references(self) -> "ModelsFile":
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        for model in self.models:
            if model.provider not in self.providers:
                raise ValueError(f"model '{model.id}' uses unknown provider '{model.provider}'")
        if self.default_model not in ids:
            raise ValueError(f"default_model '{self.default_model}' is not in models")
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
