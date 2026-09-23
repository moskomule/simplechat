import re
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError
from openai.types.chat import ChatCompletionMessageParam

from simplechat.config import Settings
from simplechat.main import create_app
from simplechat.store import Conversation, Store

SETTINGS = Settings(
    ollama_base_url="http://unused",
    ollama_api_key="unused",
    default_model="",
    host="127.0.0.1",
    port=8000,
)


class FakeBackend:
    def __init__(self, chunks: list[str] | None = None, fail: bool = False) -> None:
        self.chunks = chunks or ["Hello", ", <world>", "!\nBye"]
        self.fail = fail
        self.calls: list[tuple[str, list[ChatCompletionMessageParam]]] = []

    async def list_models(self) -> list[str]:
        return ["llama3", "qwen3"]

    async def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam]
    ) -> AsyncIterator[str]:
        self.calls.append((model, messages))
        if self.fail:
            raise APIConnectionError(request=None)  # type: ignore[arg-type]
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def store() -> Store:
    return Store()


@pytest.fixture
def client(backend: FakeBackend, store: Store) -> TestClient:
    return TestClient(create_app(settings=SETTINGS, backend=backend, store=store))


@pytest.fixture
def conversation(client: TestClient, store: Store) -> Conversation:
    client.post("/c")
    return store.recent()[0]


def stream(client: TestClient, conversation: Conversation, message_id: str) -> str:
    return client.get(f"/c/{conversation.id}/stream/{message_id}").text


def test_index_creates_conversation_with_first_model(client: TestClient, store: Store) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    (conversation,) = store.recent()
    assert response.headers["location"] == f"/c/{conversation.id}"
    assert conversation.model == "llama3"


def test_chat_page_renders(client: TestClient, conversation: Conversation) -> None:
    response = client.get(f"/c/{conversation.id}")
    assert response.status_code == 200
    assert '<option value="qwen3">qwen3</option>' in response.text


def test_unknown_conversation_is_404(client: TestClient) -> None:
    assert client.get("/c/nope").status_code == 404


def test_send_and_stream(
    client: TestClient, conversation: Conversation, backend: FakeBackend
) -> None:
    conversation.system_prompt = "Be brief."
    response = client.post(f"/c/{conversation.id}/messages", data={"content": "Hi <b>"})
    assert response.status_code == 200
    assert "Hi &lt;b&gt;" in response.text
    assert 'sse-connect="/c/' in response.text
    assert 'hx-swap-oob="true"' in response.text

    reply = conversation.active_path()[1]
    body = stream(client, conversation, reply.id)

    assert "event: token\ndata: , &lt;world&gt;" in body
    # A newline inside a token becomes two data lines, which the browser joins with "\n".
    assert "data: !\ndata: Bye" in body
    assert "event: done" in body
    assert reply.content == "Hello, <world>!\nBye"
    assert reply.status == "done"
    assert backend.calls == [
        (
            "llama3",
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi <b>"},
            ],
        )
    ]


def test_stream_twice_does_not_regenerate(
    client: TestClient, conversation: Conversation, backend: FakeBackend
) -> None:
    client.post(f"/c/{conversation.id}/messages", data={"content": "Hi"})
    reply = conversation.active_path()[1]
    stream(client, conversation, reply.id)
    body = stream(client, conversation, reply.id)
    assert "event: token" not in body
    assert "event: done" in body
    assert len(backend.calls) == 1


def test_stream_error_is_shown(client: TestClient, store: Store) -> None:
    client.app.state.backend = FakeBackend(fail=True)  # type: ignore[attr-defined]
    client.post("/c")
    conversation = store.recent()[0]
    client.post(f"/c/{conversation.id}/messages", data={"content": "Hi"})
    reply = conversation.active_path()[1]
    body = stream(client, conversation, reply.id)
    assert reply.status == "error"
    assert "Connection error" in body
    assert "Retry" in body


def test_send_while_busy_is_409(client: TestClient, conversation: Conversation) -> None:
    client.post(f"/c/{conversation.id}/messages", data={"content": "Hi"})
    response = client.post(f"/c/{conversation.id}/messages", data={"content": "Again"})
    assert response.status_code == 409


def test_empty_message_is_rejected(client: TestClient, conversation: Conversation) -> None:
    response = client.post(f"/c/{conversation.id}/messages", data={"content": "   "})
    assert response.status_code == 422


def test_edit_regenerate_and_switch(client: TestClient, conversation: Conversation) -> None:
    client.post(f"/c/{conversation.id}/messages", data={"content": "Hi"})
    user, reply = conversation.active_path()
    stream(client, conversation, reply.id)
    base = f"/c/{conversation.id}/messages"

    form = client.get(f"{base}/{user.id}/edit")
    assert "<textarea" in form.text

    response = client.post(f"{base}/{user.id}/edit", data={"content": "Hello"})
    assert response.status_code == 200
    assert "1 / 2" not in response.text
    assert "2 / 2" in response.text
    new_user, new_reply = conversation.active_path()
    assert new_user.content == "Hello"
    stream(client, conversation, new_reply.id)

    response = client.post(f"{base}/{new_reply.id}/regenerate")
    assert response.status_code == 200
    newest_reply = conversation.active_path()[1]
    assert newest_reply.status == "pending"
    stream(client, conversation, newest_reply.id)

    response = client.post(f"{base}/{new_user.id}/switch", data={"step": "-1"})
    assert response.status_code == 200
    assert [m.content for m in conversation.active_path()] == ["Hi", reply.content]
    assert re.search(r">\s*1 / 2\s*<", response.text)


def test_regenerate_user_message_is_400(client: TestClient, conversation: Conversation) -> None:
    client.post(f"/c/{conversation.id}/messages", data={"content": "Hi"})
    user, reply = conversation.active_path()
    stream(client, conversation, reply.id)
    response = client.post(f"/c/{conversation.id}/messages/{user.id}/regenerate")
    assert response.status_code == 400


def test_update_settings(client: TestClient, conversation: Conversation) -> None:
    response = client.post(
        f"/c/{conversation.id}/settings",
        data={"model": "qwen3", "system_prompt": "Answer in Japanese."},
    )
    assert response.status_code == 204
    assert conversation.model == "qwen3"
    assert conversation.system_prompt == "Answer in Japanese."


def test_delete_current_redirects(
    client: TestClient, conversation: Conversation, store: Store
) -> None:
    response = client.delete(f"/c/{conversation.id}", params={"current": conversation.id})
    assert response.headers["HX-Redirect"] == "/"
    assert store.get(conversation.id) is None


def test_delete_other_returns_sidebar(client: TestClient, store: Store) -> None:
    client.post("/c")
    client.post("/c")
    current, other = store.recent()
    response = client.delete(f"/c/{other.id}", params={"current": current.id})
    assert 'id="conversation-list"' in response.text
    assert other.id not in response.text
