import asyncio
import json
from collections.abc import Callable

import httpx2

from simplechat.llm import OllamaBackend, ThinkingOptions


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


def test_thinking_options_from_api_show() -> None:
    backend = backend_returning(
        {
            "capabilities": ["completion", "thinking"],
            "thinking": {"values": [False, "low", "medium", "xhigh"], "default": "medium"},
        }
    )
    assert asyncio.run(backend.thinking_options("qwen")) == ThinkingOptions(
        levels=["none", "low", "medium", "xhigh"], default="medium"
    )


def test_thinking_options_on_off_only() -> None:
    backend = backend_returning(
        {"capabilities": ["thinking"], "thinking": {"values": [False, True], "default": True}}
    )
    assert asyncio.run(backend.thinking_options("m")) == ThinkingOptions(
        levels=["none"], default="on"
    )


def test_thinking_options_without_metadata_can_still_turn_off() -> None:
    backend = backend_returning({"capabilities": ["completion", "thinking"]})
    assert asyncio.run(backend.thinking_options("m")) == ThinkingOptions(
        levels=["none"], default=None
    )


def test_no_thinking_options_for_non_thinking_model() -> None:
    backend = backend_returning({"capabilities": ["completion"]})
    assert asyncio.run(backend.thinking_options("m")) is None


def test_no_thinking_options_when_ollama_errors() -> None:
    backend = backend_with(lambda request: httpx2.Response(404, json={"error": "not found"}))
    assert asyncio.run(backend.thinking_options("m")) is None


def sse(*deltas: dict[str, str]) -> bytes:
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
    return ("\n\n".join([*events, "data: [DONE]"]) + "\n\n").encode()


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
