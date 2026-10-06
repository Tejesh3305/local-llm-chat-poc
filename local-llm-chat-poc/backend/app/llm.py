"""Model registry: maps public model ids to an OpenAI-compatible client + runtime model name.

Every local runtime used here (Ollama, llama.cpp, LM Studio, vLLM) speaks the
OpenAI Chat Completions API, so one client type covers all of them. Switching
models or runtimes is a change to models.json / .env only.
"""

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

from .config import ModelConfig, ModelsFile

logger = logging.getLogger("app.llm")


class UnknownModelError(Exception):
    pass


class ProviderNotConfiguredError(Exception):
    pass


class LLMError(Exception):
    """An error from the runtime, with a message that is safe to show to users."""

    def __init__(self, user_message: str):
        super().__init__(user_message)
        self.user_message = user_message


@dataclass
class StreamResult:
    """Filled in while streaming; read after the stream ends."""

    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelRegistry:
    def __init__(self, models_file: ModelsFile, timeout_seconds: float = 120):
        self.config = models_file
        self.clients: dict[str, AsyncOpenAI] = {}
        for provider_id, provider in models_file.providers.items():
            base_url = os.getenv(provider.base_url_env, "").strip()
            if not base_url:
                logger.info("provider '%s' skipped: %s is not set", provider_id, provider.base_url_env)
                continue
            api_key = os.getenv(provider.api_key_env, "") if provider.api_key_env else ""
            self.clients[provider_id] = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed",  # local runtimes ignore the key; the SDK requires a value
                timeout=timeout_seconds,
                max_retries=1,
            )

    @property
    def default_model_id(self) -> str:
        return self.config.default_model

    def get(self, model_id: str) -> ModelConfig:
        model = next((m for m in self.config.models if m.id == model_id), None)
        if model is None:
            raise UnknownModelError(model_id)
        if model.provider not in self.clients:
            raise ProviderNotConfiguredError(model.provider)
        return model

    async def installed_models(self) -> dict[str, set[str] | None]:
        """Per provider: the model names the runtime reports, or None if unreachable/not configured."""
        result: dict[str, set[str] | None] = {provider_id: None for provider_id in self.config.providers}
        for provider_id, client in self.clients.items():
            try:
                page = await client.with_options(timeout=3, max_retries=0).models.list()
                result[provider_id] = {model.id for model in page.data}
            except Exception as error:  # runtime down, wrong URL, etc.
                logger.warning("provider '%s' unreachable: %s", provider_id, error)
        return result

    async def stream_chat(
        self, model: ModelConfig, messages: list[dict[str, str]], result: StreamResult
    ) -> AsyncIterator[str]:
        """Yields text chunks. Raises LLMError with a user-friendly message on failure."""
        client = self.clients[model.provider]
        prompt = ([{"role": "system", "content": model.system_prompt}] if model.system_prompt else []) + messages

        options: dict = {}
        if model.temperature is not None:
            options["temperature"] = model.temperature
        if model.max_tokens is not None:
            options["max_tokens"] = model.max_tokens

        try:
            stream = await client.chat.completions.create(
                model=model.provider_model,
                messages=prompt,
                stream=True,
                stream_options={"include_usage": True},
                **options,
            )
            async with stream:  # closes the upstream connection if the client disconnects
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        if choice.delta and choice.delta.content:
                            yield choice.delta.content
                        if choice.finish_reason:
                            result.finish_reason = choice.finish_reason
                    if chunk.usage:
                        result.input_tokens = chunk.usage.prompt_tokens
                        result.output_tokens = chunk.usage.completion_tokens
        except openai.APIConnectionError as error:
            if isinstance(error, openai.APITimeoutError):
                raise LLMError("The model took too long to respond. Try a shorter prompt or a smaller model.") from error
            raise LLMError("The local model server is not reachable. Make sure Ollama (or your runtime) is running.") from error
        except openai.NotFoundError as error:
            raise LLMError(
                f"Model '{model.provider_model}' is not installed in the runtime. Run: ollama pull {model.provider_model}"
            ) from error
        except openai.APIStatusError as error:
            logger.error("runtime returned HTTP %s: %s", error.status_code, error.message)
            raise LLMError("The model failed to generate a response. Please try again.") from error
