"""Talk to Ollama.

Chat goes through the OpenAI-compatible API. Thinking levels and capabilities
such as vision are only listed by Ollama's native `/api/show`, and loaded models
can only be listed and unloaded natively, so those calls use the native API.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx2
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

MODEL_CACHE_SECONDS = 30.0
# Ollama can answer slowly while it loads a model, so unloading waits longer than
# the 5 s default before giving up.
UNLOAD_TIMEOUT_SECONDS = 30.0

# Passed as `reasoning_effort` to turn thinking off.
THINKING_OFF = "none"

type ChunkKind = Literal["thinking", "content"]


@dataclass(frozen=True, slots=True)
class ThinkingOptions:
    """Thinking levels a model accepts, e.g. ["none", "low", "medium", "high"]."""

    levels: list[str]
    default: str | None


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """What a model supports: thinking levels (None if it cannot think) and images."""

    thinking: ThinkingOptions | None = None
    vision: bool = False


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model Ollama holds in memory, with its size in bytes (weights and KV cache)."""

    name: str
    size: int


@dataclass(frozen=True, slots=True)
class UnloadResult:
    """Models unloaded now, and ones that unload once their current reply finishes."""

    unloaded: list[LoadedModel]
    pending: list[LoadedModel]


class UnloadError(Exception):
    """Unloading failed; the message says why, in words fit for the user."""


class ChatBackend(Protocol):
    async def list_models(self) -> list[str]: ...

    async def model_info(self, model: str) -> ModelInfo: ...

    async def unload_models(self) -> UnloadResult: ...

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

    async def model_info(self, model: str) -> ModelInfo:
        """Return what the model supports; nothing if it is unknown or Ollama errors."""
        if not model:
            return ModelInfo()
        try:
            response = await self._native.post("/api/show", json={"model": model})
            response.raise_for_status()
        except httpx2.HTTPError:
            return ModelInfo()
        info = response.json()
        # e.g. ["completion", "vision", "tools", "thinking"]
        capabilities = info.get("capabilities") or []
        return ModelInfo(
            thinking=_thinking_options(info) if "thinking" in capabilities else None,
            vision="vision" in capabilities,
        )

    async def unload_models(self) -> UnloadResult:
        """Unload every model Ollama has in memory.

        A request with `keep_alive: 0` and no prompt unloads a model. An idle model
        is gone by the time that request returns, but one still answering another
        client stays loaded until its reply finishes, so the loaded models are
        listed again afterwards to tell the two apart.
        """
        try:
            loaded = await self._loaded_models()
            for model in loaded:
                unload = await self._native.post(
                    "/api/generate",
                    json={"model": model.name, "keep_alive": 0},
                    timeout=UNLOAD_TIMEOUT_SECONDS,
                )
                unload.raise_for_status()
            still_loaded = {model.name for model in await self._loaded_models()}
        except httpx2.TimeoutException as e:
            raise UnloadError("Ollama did not answer in time; it may be loading a model") from e
        except httpx2.HTTPError as e:
            raise UnloadError("Could not reach Ollama") from e
        return UnloadResult(
            unloaded=[model for model in loaded if model.name not in still_loaded],
            pending=[model for model in loaded if model.name in still_loaded],
        )

    async def _loaded_models(self) -> list[LoadedModel]:
        response = await self._native.get("/api/ps", timeout=UNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        return [
            LoadedModel(name=model["name"], size=model.get("size", 0))
            for model in response.json().get("models") or []
        ]

    async def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam], thinking: str = ""
    ) -> AsyncIterator[tuple[ChunkKind, str]]:
        """Yield ("thinking", text) and ("content", text) chunks as they arrive.

        `thinking` is a level from `model_info`; empty means the model's default.
        Closing this generator early (a stopped reply) closes the connection, which
        makes Ollama stop generating.
        """
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            # Sent raw because level names are model-defined and may fall outside
            # the SDK's `reasoning_effort` literal type.
            extra_body={"reasoning_effort": thinking} if thinking else None,
        )
        async with stream:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Ollama streams thinking in a non-standard `reasoning` field.
                if reasoning := getattr(delta, "reasoning", None):
                    yield "thinking", reasoning
                if delta.content:
                    yield "content", delta.content


def _thinking_options(info: dict[str, Any]) -> ThinkingOptions:
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


def _level_name(value: str | bool) -> str:
    if value is False:
        return THINKING_OFF
    if value is True:
        return "on"
    return value
