"""Talk to Ollama.

Chat goes through the OpenAI-compatible API. Thinking levels are only listed
by Ollama's native `/api/show`, so that one call uses the native API.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx2
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

MODEL_CACHE_SECONDS = 30.0

# Passed as `reasoning_effort` to turn thinking off.
THINKING_OFF = "none"

type ChunkKind = Literal["thinking", "content"]


@dataclass(frozen=True, slots=True)
class ThinkingOptions:
    """Thinking levels a model accepts, e.g. ["none", "low", "medium", "high"]."""

    levels: list[str]
    default: str | None


class ChatBackend(Protocol):
    async def list_models(self) -> list[str]: ...

    async def thinking_options(self, model: str) -> ThinkingOptions | None: ...

    def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam], thinking: str = ""
    ) -> AsyncIterator[tuple[ChunkKind, str]]: ...


class OllamaBackend:
    def __init__(
        self, url: str, api_key: str, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self._native = httpx2.AsyncClient(base_url=url, transport=transport)
        self._client = AsyncOpenAI(
            base_url=f"{url}/v1",
            api_key=api_key,
            http_client=httpx2.AsyncClient(transport=transport) if transport else None,
        )
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

    async def thinking_options(self, model: str) -> ThinkingOptions | None:
        """Return the model's thinking levels, or None if it cannot think."""
        if not model:
            return None
        try:
            response = await self._native.post("/api/show", json={"model": model})
            response.raise_for_status()
        except httpx2.HTTPError:
            return None
        info = response.json()
        if "thinking" not in info.get("capabilities", []):
            return None
        # e.g. {"values": [false, "low", "medium", "high"], "default": "medium"}.
        # `false` means thinking can be turned off. `true` (on/off-only models) is
        # left out, since leaving the level at the default already turns it on.
        spec = info.get("thinking") or {}
        levels = [_level_name(v) for v in spec.get("values", [False]) if v is not True]
        default = spec.get("default")
        return ThinkingOptions(
            levels=levels,
            default=_level_name(default) if isinstance(default, str | bool) else None,
        )

    async def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam], thinking: str = ""
    ) -> AsyncIterator[tuple[ChunkKind, str]]:
        """Yield ("thinking", text) and ("content", text) chunks as they arrive.

        `thinking` is a level from `thinking_options`; empty means the model's default.
        """
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            # Sent raw because level names are model-defined and may fall outside
            # the SDK's `reasoning_effort` literal type.
            extra_body={"reasoning_effort": thinking} if thinking else None,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Ollama streams thinking in a non-standard `reasoning` field.
            if reasoning := getattr(delta, "reasoning", None):
                yield "thinking", reasoning
            if delta.content:
                yield "content", delta.content


def _level_name(value: str | bool) -> str:
    if value is False:
        return THINKING_OFF
    if value is True:
        return "on"
    return value
