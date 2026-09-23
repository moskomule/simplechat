import asyncio

import httpx2

from simplechat.llm import OllamaBackend


def backend_returning(body: dict[str, object]) -> OllamaBackend:
    backend = OllamaBackend(base_url="http://ollama.test/v1", api_key="unused")
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=body))
    backend._client = backend._client.with_options(
        http_client=httpx2.AsyncClient(transport=transport)
    )
    return backend


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
