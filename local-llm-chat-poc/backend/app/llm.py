"""Model registry: maps public model ids to an OpenAI-compatible client + runtime model name.

Every local runtime used here (Ollama, llama.cpp, LM Studio, vLLM) speaks the
OpenAI Chat Completions API, so one client type covers all of them. Switching
models or runtimes is a change to models.json / .env only.
"""

import asyncio
import ipaddress
import logging
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
import openai
from openai import AsyncOpenAI

from .config import ModelConfig, ModelsFile
from .ollama import OllamaRuntime, tagged

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
    # For the speed figure: time spent writing answer text.
    generation_seconds: float = 0.0


class ModelRegistry:
    def __init__(self, models_file: ModelsFile, timeout_seconds: float = 120):
        self.config = models_file
        self.clients: dict[str, AsyncOpenAI] = {}
        self.runtimes: dict[str, OllamaRuntime] = {}  # providers whose memory this app manages
        self.base_urls: dict[str, str] = {}  # configured providers -> where their server runs
        # Hold this while loading or generating: with one model in memory at a time,
        # a request for another model has to wait for the current one to finish.
        self.slot = asyncio.Lock()
        for provider_id, provider in models_file.providers.items():
            base_url = os.getenv(provider.base_url_env, "").strip()
            if not base_url:
                logger.info("provider '%s' skipped: %s is not set", provider_id, provider.base_url_env)
                continue
            api_key = os.getenv(provider.api_key_env, "") if provider.api_key_env else ""
            self.base_urls[provider_id] = base_url
            self.clients[provider_id] = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed",  # local runtimes ignore the key; the SDK requires a value
                timeout=timeout_seconds,
                max_retries=1,
            )
            if provider.runtime == "ollama":
                self.runtimes[provider_id] = OllamaRuntime(base_url, api_key, timeout_seconds)

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

    def network(self) -> tuple[str, list[str]]:
        """Where the model servers run: ("this-machine" | "local-network" | "external", host names).

        This reads the configured addresses; it does not watch network traffic.
        """
        hosts = sorted({urlsplit(url).hostname or url for url in self.base_urls.values()})
        scopes = set()
        for host in hosts:
            try:
                address = ipaddress.ip_address(host)
                scopes.add("this-machine" if address.is_loopback else "local-network" if address.is_private else "external")
            except ValueError:  # a name, not an IP address
                scopes.add("this-machine" if host == "localhost" else "external")
        scope = "external" if "external" in scopes else "local-network" if "local-network" in scopes else "this-machine"
        return scope, hosts

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

    async def running_models(self) -> dict[str, set[str]]:
        """Per managed provider: the model names currently loaded in memory (empty if unreachable)."""
        result: dict[str, set[str]] = {}
        for provider_id, runtime in self.runtimes.items():
            try:
                result[provider_id] = await runtime.running()
            except (httpx.HTTPError, ValueError) as error:
                logger.warning("provider '%s' did not report running models: %s", provider_id, error)
        return result

    async def activate(self, model: ModelConfig, preload: bool = False) -> None:
        """Make `model` the only catalogue model in memory. Call while holding `slot`.

        Unloads the other models of this catalogue (models loaded by other tools are left
        alone). With preload=True it also loads `model` now, instead of on its first message.
        Raises LLMError with a user-friendly message on failure.
        """
        try:
            for provider_id, runtime in self.runtimes.items():
                running = await runtime.running()
                for other in self.config.models:
                    is_target = other.provider == model.provider and other.provider_model == model.provider_model
                    if other.provider == provider_id and not is_target and tagged(other.provider_model) in running:
                        logger.info("unloading '%s' to make room for '%s'", other.provider_model, model.provider_model)
                        await runtime.unload(other.provider_model)
                if preload and provider_id == model.provider and tagged(model.provider_model) not in running:
                    logger.info("loading '%s'", model.provider_model)
                    await runtime.load(model.provider_model)
        except httpx.TimeoutException as error:
            raise LLMError(f"Loading '{model.label}' took too long. Try again or pick a smaller model.") from error
        except httpx.TransportError as error:
            raise LLMError("The local model server is not reachable. Make sure Ollama (or your runtime) is running.") from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise LLMError(
                    f"Model '{model.provider_model}' is not installed in the runtime. Run: ollama pull {model.provider_model}"
                ) from error
            logger.error("runtime returned HTTP %s: %s", error.response.status_code, error.response.text[:500])
            raise LLMError(f"'{model.label}' could not be loaded. The machine may not have enough free memory.") from error

    async def stream_chat(
        self, model: ModelConfig, messages: list[dict[str, str]], result: StreamResult, thinking: bool | None = None
    ) -> AsyncIterator[str]:
        """Yields text chunks. Raises LLMError with a user-friendly message on failure.

        `thinking` overrides the model's default; it is ignored for models without a thinking switch.
        """
        client = self.clients[model.provider]
        prompt = ([{"role": "system", "content": model.system_prompt}] if model.system_prompt else []) + messages

        options: dict = {}
        if model.temperature is not None:
            options["temperature"] = model.temperature
        if model.max_tokens is not None:
            options["max_tokens"] = model.max_tokens
        if model.thinking is not None and not (model.thinking if thinking is None else thinking):
            # Thinking models reason by default; the reasoning is never shown, so skipping it
            # removes the long silent wait before the first word.
            options["reasoning_effort"] = "none"

        try:
            stream = await client.chat.completions.create(
                model=model.provider_model,
                messages=prompt,
                stream=True,
                stream_options={"include_usage": True},
                **options,
            )
            writing_since: float | None = None
            try:
                async with stream:  # closes the upstream connection if the client disconnects
                    async for chunk in stream:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.delta and choice.delta.content:
                                writing_since = writing_since or time.perf_counter()
                                yield choice.delta.content
                            if choice.finish_reason:
                                result.finish_reason = choice.finish_reason
                        if chunk.usage:
                            result.input_tokens = chunk.usage.prompt_tokens
                            result.output_tokens = chunk.usage.completion_tokens
            finally:
                if writing_since:
                    result.generation_seconds = time.perf_counter() - writing_since
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
