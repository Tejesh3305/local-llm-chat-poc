"""Model registry: maps public model ids to an OpenAI-compatible client + runtime model name.

Every local runtime used here (Ollama, llama.cpp, LM Studio, vLLM) speaks the
OpenAI Chat Completions API, so one client type covers all of them. Switching
models or runtimes is a change to models.json / .env only.

Providers of type "ollama" get three extras through Ollama's own API: asking
which model is in memory, loading a model ahead of the first message, and
unloading the catalogue's other models so only one is in memory at a time.
"""

import asyncio
import ipaddress
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
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
    first_token_ms: int | None = None  # request sent -> first text
    generation_ms: int | None = None  # first text -> end of the reply


def _tag(name: str) -> str:
    """Ollama treats 'llama3.2' and 'llama3.2:latest' as the same model."""
    return (name if ":" in name else f"{name}:latest").lower()


class ModelRegistry:
    def __init__(self, models_file: ModelsFile, timeout_seconds: float = 120):
        self.config = models_file
        self.timeout_seconds = timeout_seconds
        self.clients: dict[str, AsyncOpenAI] = {}
        self.native: dict[str, str] = {}  # provider id -> root of Ollama's own API
        self.on_this_machine: set[str] = set()  # providers whose address is this computer
        self.base_urls: dict[str, str] = {}  # configured providers -> where their server runs
        # Hold this while loading or generating: with one model in memory at a time,
        # a request for another model has to wait for the current one to finish.
        self.slot = asyncio.Lock()
        for provider_id, provider in models_file.providers.items():
            base_url = os.getenv(provider.base_url_env, "").strip()
            if not base_url:
                logger.info("provider '%s' skipped: %s is not set", provider_id, provider.base_url_env)
                continue
            if urlparse(base_url).hostname in ("127.0.0.1", "localhost", "::1"):
                self.on_this_machine.add(provider_id)
            api_key = os.getenv(provider.api_key_env, "") if provider.api_key_env else ""
            self.base_urls[provider_id] = base_url
            self.clients[provider_id] = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "not-needed",  # local runtimes ignore the key; the SDK requires a value
                timeout=timeout_seconds,
                max_retries=0,  # a local engine that refuses a connection is down; retrying only delays saying so
            )
            if provider.type == "ollama":
                self.native[provider_id] = re.sub(r"/v1$", "", base_url.rstrip("/"))
        self._http = httpx.AsyncClient()  # one client for Ollama's own API; creating one per call is slow

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

    def runtime_name(self, model: ModelConfig) -> str:
        """The model's name in the form the runtime reports it, for comparing with installed/loaded lists."""
        return _tag(model.provider_model) if model.provider in self.native else model.provider_model

    def _down(self, model: ModelConfig) -> str:
        return f"Local engine not running. Start {self.config.providers[model.provider].name}."

    def _missing(self, model: ModelConfig) -> str:
        hint = f" Run: ollama pull {model.provider_model}" if model.provider in self.native else ""
        return f"{model.label} is not installed on this machine.{hint}"

    def network(self) -> tuple[str, list[str]]:
        """Where the model servers run: ("this-machine" | "local-network" | "external", host names).

        This reads the configured addresses; it does not watch network traffic.
        """
        hosts = sorted({urlparse(url).hostname or url for url in self.base_urls.values()})
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
                names = {model.id for model in page.data}
                result[provider_id] = {_tag(name) for name in names} if provider_id in self.native else names
            except Exception as error:  # runtime down, wrong URL, etc.
                logger.warning("provider '%s' unreachable: %s", provider_id, error)
        return result

    async def loaded_models(self, reachable: set[str] | None = None) -> dict[str, dict[str, int | None] | None]:
        """Per provider: the models in memory with the share of each on the GPU in percent. None if it cannot be asked.

        Pass `reachable` to skip providers already known to be down.
        """
        result: dict[str, dict[str, int | None] | None] = {provider_id: None for provider_id in self.config.providers}
        for provider_id, root in self.native.items():
            if reachable is not None and provider_id not in reachable:
                continue
            try:
                running = (await self._http.get(f"{root}/api/ps", timeout=3)).json().get("models", [])
                result[provider_id] = {
                    _tag(entry.get("model") or entry["name"]): round(100 * entry.get("size_vram", 0) / entry["size"]) if entry.get("size") else None
                    for entry in running
                }
            except Exception as error:
                logger.debug("provider '%s' did not report loaded models: %s", provider_id, error)
        return result

    async def load(self, model: ModelConfig) -> bool:
        """Loads the model into memory ahead of the first message. False if the runtime has no way to ask for that."""
        root = self.native.get(model.provider)
        if root is None:
            return False
        try:
            response = await self._http.post(f"{root}/api/generate", json={"model": model.provider_model, "stream": False}, timeout=self.timeout_seconds)
        except httpx.HTTPError as error:
            raise LLMError(self._down(model)) from error
        if response.status_code == 404:
            raise LLMError(self._missing(model))
        if response.status_code >= 400:
            logger.error("loading %s failed: HTTP %s %s", model.provider_model, response.status_code, response.text[:200])
            raise LLMError(f"{model.label} could not be loaded into memory.")
        return True

    async def activate(self, model: ModelConfig) -> None:
        """Make `model` the only catalogue model in memory. Call while holding `slot`.

        Unloads the other models of this catalogue; models loaded by other tools are left alone.
        A runtime that cannot be asked is skipped: the request that follows reports it.
        """
        for provider_id, in_memory in (await self.loaded_models()).items():
            for other in self.config.models:
                is_target = other.provider == model.provider and other.provider_model == model.provider_model
                if other.provider != provider_id or is_target or _tag(other.provider_model) not in (in_memory or {}):
                    continue
                logger.info("unloading '%s' to make room for '%s'", other.provider_model, model.provider_model)
                try:  # a generate call with keep_alive 0 and no prompt only unloads the model
                    payload = {"model": other.provider_model, "keep_alive": 0, "stream": False}
                    await self._http.post(f"{self.native[provider_id]}/api/generate", json=payload, timeout=30)
                except httpx.HTTPError as error:
                    logger.warning("could not unload '%s': %s", other.provider_model, error)

    async def stream_chat(
        self,
        model: ModelConfig,
        messages: list[dict[str, str]],
        result: StreamResult,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> AsyncIterator[str]:
        """Yields text chunks. Raises LLMError with a user-friendly message on failure.

        `system`, `temperature` and `max_tokens` replace the model's own settings from models.json when given.
        `thinking` overrides the model's default; it is ignored for models without a thinking switch.
        """
        client = self.clients[model.provider]
        system_prompt = model.system_prompt if system is None else system
        prompt = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + messages

        options: dict = {}
        temperature = model.temperature if temperature is None else temperature
        max_tokens = model.max_tokens if max_tokens is None else max_tokens
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        if model.thinking is not None and not (model.thinking if thinking is None else thinking):
            # Thinking models reason by default; the reasoning is never shown, so skipping it
            # removes the long silent wait before the first word.
            options["reasoning_effort"] = "none"

        started = time.perf_counter()
        first: float | None = None
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
                            if first is None:
                                first = time.perf_counter()
                                result.first_token_ms = round((first - started) * 1000)
                            yield choice.delta.content
                        if choice.finish_reason:
                            result.finish_reason = choice.finish_reason
                    if chunk.usage:
                        result.input_tokens = chunk.usage.prompt_tokens
                        result.output_tokens = chunk.usage.completion_tokens
            if first is not None:
                result.generation_ms = round((time.perf_counter() - first) * 1000)
        except openai.APIConnectionError as error:
            if isinstance(error, openai.APITimeoutError):
                raise LLMError("The model took too long to respond. Try a shorter prompt or a smaller model.") from error
            raise LLMError(self._down(model)) from error
        except openai.NotFoundError as error:
            raise LLMError(self._missing(model)) from error
        except openai.APIStatusError as error:
            logger.error("runtime returned HTTP %s: %s", error.status_code, error.message)
            raise LLMError("The model failed to generate a response. Please try again.") from error
