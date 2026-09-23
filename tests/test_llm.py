import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx2
import pytest

from simplechat.llm import (
    LoadedModel,
    ModelInfo,
    OllamaBackend,
    ThinkingOptions,
    UnloadError,
    UnloadResult,
)


def backend_with(handler: Callable[[httpx2.Request], httpx2.Response]) -> OllamaBackend:
    return OllamaBackend("http://ollama.test", "unused", transport=httpx2.MockTransport(handler))


def backend_returning(body: dict[str, object]) -> OllamaBackend:
    return backend_with(lambda request: httpx2.Response(200, json=body))


def test_list_models_sorted() -> None:
    backend = backend_returning(
        {
            "object": "list",
            "data": [
                {"id": "qwen3", "object": "model", "created": 0, "owned_by": "library"},
                {"id": "llama3", "object": "model", "created": 0, "owned_by": "library"},
            ],
        }
    )
    assert asyncio.run(backend.list_models()) == ["llama3", "qwen3"]


def test_list_models_when_none_pulled() -> None:
    backend = backend_returning({"object": "list", "data": None})
    assert asyncio.run(backend.list_models()) == []


def test_model_info_from_api_show() -> None:
    backend = backend_returning(
        {
            "capabilities": ["completion", "vision", "tools", "thinking"],
            "thinking": {"values": [False, "low", "medium", "xhigh"], "default": "medium"},
        }
    )
    assert asyncio.run(backend.model_info("qwen")) == ModelInfo(
        thinking=ThinkingOptions(levels=["none", "low", "medium", "xhigh"], default="medium"),
        vision=True,
    )


def test_thinking_options_on_off_only() -> None:
    backend = backend_returning(
        {"capabilities": ["thinking"], "thinking": {"values": [False, True], "default": True}}
    )
    assert asyncio.run(backend.model_info("m")).thinking == ThinkingOptions(
        levels=["none"], default="on"
    )


def test_thinking_options_without_metadata_can_still_turn_off() -> None:
    backend = backend_returning({"capabilities": ["completion", "thinking"]})
    assert asyncio.run(backend.model_info("m")).thinking == ThinkingOptions(
        levels=["none"], default=None
    )


def test_model_info_for_text_only_model() -> None:
    backend = backend_returning({"capabilities": ["completion"]})
    assert asyncio.run(backend.model_info("m")) == ModelInfo(thinking=None, vision=False)


def test_model_info_for_vision_model_without_thinking() -> None:
    backend = backend_returning({"capabilities": ["completion", "vision"]})
    assert asyncio.run(backend.model_info("m")) == ModelInfo(thinking=None, vision=True)


def test_empty_model_info_when_ollama_errors() -> None:
    backend = backend_with(lambda request: httpx2.Response(404, json={"error": "not found"}))
    assert asyncio.run(backend.model_info("m")) == ModelInfo()


def sse(*deltas: dict[str, str], done: bool = True) -> bytes:
    events = [
        "data: "
        + json.dumps(
            {
                "id": "1",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "m",
                "choices": [{"index": 0, "delta": delta}],
            }
        )
        for delta in deltas
    ]
    if done:
        events.append("data: [DONE]")
    return ("\n\n".join(events) + "\n\n").encode()


def test_stream_chat_splits_thinking_and_content() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        body = sse(
            {"role": "assistant", "content": "", "reasoning": "Let me think"},
            {"content": "", "reasoning": "..."},
            {"content": "391"},
        )
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async def collect() -> list[tuple[str, str]]:
        backend = backend_with(handler)
        messages = [{"role": "user", "content": "17 * 23?"}]
        return [chunk async for chunk in backend.stream_chat("m", messages, thinking="low")]

    assert asyncio.run(collect()) == [
        ("thinking", "Let me think"),
        ("thinking", "..."),
        ("content", "391"),
    ]
    assert requests[0]["reasoning_effort"] == "low"


def test_stream_chat_default_thinking_sends_no_effort() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200, content=sse({"content": "hi"}), headers={"content-type": "text/event-stream"}
        )

    async def collect() -> list[tuple[str, str]]:
        backend = backend_with(handler)
        return [chunk async for chunk in backend.stream_chat("m", [])]

    assert asyncio.run(collect()) == [("content", "hi")]
    assert "reasoning_effort" not in requests[0]


class EndlessStream(httpx2.AsyncByteStream):
    """A response body that sends its events, then hangs like a model still generating."""

    def __init__(self, *events: bytes) -> None:
        self.events = events
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            yield event
        await asyncio.Event().wait()  # never set: the reply goes on

    async def aclose(self) -> None:
        self.closed = True


def test_stopping_a_stream_closes_the_connection() -> None:
    # Closing the response is what tells Ollama to stop generating.
    body = EndlessStream(sse({"content": "one"}, done=False))
    backend = backend_with(
        lambda request: httpx2.Response(
            200, stream=body, headers={"content-type": "text/event-stream"}
        )
    )

    async def read_one_then_stop() -> tuple[tuple[str, str], bool]:
        chunks = backend.stream_chat("m", [])
        first = await anext(chunks)
        await chunks.aclose()
        # Checked here, not after asyncio.run(): its shutdown closes leftover
        # generators anyway, which would hide a connection left open.
        return first, body.closed

    assert asyncio.run(read_one_then_stop()) == (("content", "one"), True)


def ollama_with_loaded_models(
    loaded: dict[str, int], busy: set[str], unloads: list[dict[str, object]]
) -> OllamaBackend:
    """A fake Ollama: keep_alive 0 unloads an idle model at once, a busy one only later."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/api/ps":
            # Shaped like a real Ollama 0.34 response, trimmed.
            models = [{"name": name, "size": size} for name, size in loaded.items()]
            return httpx2.Response(200, json={"models": models})
        body = json.loads(request.content)
        unloads.append(body)
        if body["model"] not in busy:
            del loaded[body["model"]]
        return httpx2.Response(200, json={"model": body["model"], "done_reason": "unload"})

    return backend_with(handler)


def test_unload_models_sends_keep_alive_zero_for_each_loaded_model() -> None:
    unloads: list[dict[str, object]] = []
    loaded = {"qwen3": 19_704_971_204, "llama3": 4_000_000_000}
    backend = ollama_with_loaded_models(loaded, busy=set(), unloads=unloads)

    result = asyncio.run(backend.unload_models())
    assert result == UnloadResult(
        unloaded=[LoadedModel("qwen3", 19_704_971_204), LoadedModel("llama3", 4_000_000_000)],
        pending=[],
    )
    assert unloads == [
        {"model": "qwen3", "keep_alive": 0},
        {"model": "llama3", "keep_alive": 0},
    ]


def test_unload_models_reports_models_still_answering() -> None:
    loaded = {"qwen3": 19_704_971_204, "llama3": 4_000_000_000}
    backend = ollama_with_loaded_models(loaded, busy={"llama3"}, unloads=[])

    result = asyncio.run(backend.unload_models())
    assert result == UnloadResult(
        unloaded=[LoadedModel("qwen3", 19_704_971_204)],
        pending=[LoadedModel("llama3", 4_000_000_000)],
    )


def test_unload_models_with_nothing_loaded() -> None:
    backend = backend_returning({"models": []})
    assert asyncio.run(backend.unload_models()) == UnloadResult(unloaded=[], pending=[])


def test_unload_models_raises_when_ollama_errors() -> None:
    backend = backend_with(lambda request: httpx2.Response(500, json={"error": "boom"}))
    with pytest.raises(UnloadError, match="Could not reach Ollama"):
        asyncio.run(backend.unload_models())


def test_unload_models_explains_a_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(UnloadError, match="may be loading a model"):
        asyncio.run(backend_with(handler).unload_models())
