"""Talk to Ollama through its OpenAI-compatible API."""

import time
from collections.abc import AsyncIterator
from typing import Protocol

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

MODEL_CACHE_SECONDS = 30.0


class ChatBackend(Protocol):
    async def list_models(self) -> list[str]: ...

    def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam]
    ) -> AsyncIterator[str]: ...


class OllamaBackend:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._models: list[str] = []
        self._models_fetched_at = 0.0

    async def list_models(self) -> list[str]:
        """Return available model names, or an empty list if Ollama is unreachable."""
        if time.monotonic() - self._models_fetched_at < MODEL_CACHE_SECONDS:
            return self._models
        try:
            page = await self._client.models.list()
        except OpenAIError:
            return []
        # Ollama answers `"data": null` rather than `[]` when no model is pulled.
        self._models = sorted(model.id for model in page.data or [])
        self._models_fetched_at = time.monotonic()
        return self._models

    async def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam]
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
        async for chunk in stream:
            if chunk.choices and (text := chunk.choices[0].delta.content):
                yield text
