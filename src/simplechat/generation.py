"""Generate replies in the background so any number of SSE clients can follow them.

A generation is not tied to the request that watches it: a reload, a network blip
or a second device can attach at any time, replay what was produced so far, and
keep receiving chunks until the reply finishes.
"""

import asyncio
import base64
from collections.abc import AsyncIterator

from openai import OpenAIError
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam

from simplechat.llm import ChatBackend, ChunkKind
from simplechat.store import Conversation, Image, Message


class Generation:
    def __init__(self) -> None:
        self.chunks: list[tuple[ChunkKind, str]] = []
        self.finished = False
        self._changed = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    def add(self, kind: ChunkKind, text: str) -> None:
        self.chunks.append((kind, text))
        self._notify()

    def finish(self) -> None:
        self.finished = True
        self._notify()

    def _notify(self) -> None:
        # Wake everyone waiting on the current event, then start a fresh one.
        self._changed.set()
        self._changed = asyncio.Event()

    async def follow(self) -> AsyncIterator[tuple[ChunkKind, str]]:
        """Yield every chunk from the start, then new ones until the reply finishes."""
        index = 0
        while True:
            # Take the event before reading the chunks, so a chunk added in
            # between still wakes us up.
            changed = self._changed
            new = self.chunks[index:]
            index += len(new)
            for chunk in new:
                yield chunk
            if index == len(self.chunks):
                if self.finished:
                    return
                await changed.wait()


class Generations:
    """Replies currently being generated, keyed by message id."""

    def __init__(self) -> None:
        self._running: dict[str, Generation] = {}

    def get(self, message_id: str) -> Generation | None:
        return self._running.get(message_id)

    @property
    def any_running(self) -> bool:
        return bool(self._running)

    def start(self, backend: ChatBackend, conversation: Conversation, reply: Message) -> None:
        generation = Generation()
        self._running[reply.id] = generation
        # Keep a reference to the task; the event loop only holds a weak one.
        generation.task = asyncio.create_task(self._run(backend, conversation, reply, generation))

    async def _run(
        self,
        backend: ChatBackend,
        conversation: Conversation,
        reply: Message,
        generation: Generation,
    ) -> None:
        reply.status = "streaming"
        try:
            if not conversation.model:
                raise ValueError("No model selected. Is Ollama running?")
            chunks = backend.stream_chat(
                conversation.model,
                build_chat_messages(conversation, reply),
                thinking=conversation.thinking,
            )
            async for kind, text in chunks:
                if kind == "thinking":
                    reply.thinking += text
                else:
                    reply.content += text
                generation.add(kind, text)
            reply.status = "done"
        except (OpenAIError, ValueError) as e:
            reply.status = "error"
            reply.error = str(e)
        finally:
            # Cancelled (e.g. at shutdown): keep what was generated so far.
            if reply.status == "streaming":
                reply.status = "done"
            del self._running[reply.id]
            generation.finish()


def build_chat_messages(
    conversation: Conversation, reply: Message
) -> list[ChatCompletionMessageParam]:
    """The prompt for `reply`: the system prompt plus every earlier message in its branch."""
    messages: list[ChatCompletionMessageParam] = []
    if conversation.system_prompt.strip():
        messages.append({"role": "system", "content": conversation.system_prompt})
    for message in conversation.ancestors(reply.id):
        if message.role == "user":
            messages.append({"role": "user", "content": _user_content(message)})
        else:
            messages.append({"role": "assistant", "content": message.content})
    return messages


def _user_content(message: Message) -> str | list[ChatCompletionContentPartParam]:
    """Plain text, or image parts followed by a text part when there are images."""
    if not message.images:
        return message.content
    parts: list[ChatCompletionContentPartParam] = [
        {"type": "image_url", "image_url": {"url": _data_url(image)}} for image in message.images
    ]
    if message.content:
        parts.append({"type": "text", "text": message.content})
    return parts


def _data_url(image: Image) -> str:
    return f"data:{image.media_type};base64,{base64.b64encode(image.data).decode()}"
