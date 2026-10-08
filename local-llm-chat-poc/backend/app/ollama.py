"""Ollama's native API, used only for what the OpenAI-compatible API cannot do:
see which models are in memory, and load / unload them.
"""

import httpx


def tagged(name: str) -> str:
    """Ollama reports names with a tag; 'gemma4' means 'gemma4:latest'."""
    return name if ":" in name else f"{name}:latest"


class OllamaRuntime:
    def __init__(self, openai_base_url: str, api_key: str = "", timeout_seconds: float = 120):
        # The native API lives next to the OpenAI-compatible one: http://host:11434/v1 -> http://host:11434
        root = openai_base_url.rstrip("/").removesuffix("/v1")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=root, headers=headers, timeout=timeout_seconds)

    async def running(self) -> set[str]:
        """Names of the models currently loaded in memory."""
        response = await self.http.get("/api/ps", timeout=5)
        response.raise_for_status()
        return {tagged(model["name"]) for model in response.json().get("models", [])}

    async def load(self, name: str) -> None:
        """Returns once the model is in memory (can take a while for a cold start)."""
        await self._generate({"model": name})

    async def unload(self, name: str) -> None:
        await self._generate({"model": name, "keep_alive": 0})

    async def _generate(self, payload: dict) -> None:
        # A generate call without a prompt only changes the model's loaded state.
        response = await self.http.post("/api/generate", json={**payload, "stream": False})
        response.raise_for_status()
