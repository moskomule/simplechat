import asyncio

from openai.types.chat import ChatCompletionMessageParam

from simplechat.generation import Generations
from simplechat.llm import ChatStream
from simplechat.store import Conversation


class RecordingBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(
        self, model: str, messages: list[ChatCompletionMessageParam], thinking: str = ""
    ) -> ChatStream:
        self.calls += 1
        yield "content", "never reached"


def test_stop_before_the_task_runs_still_finishes_the_reply() -> None:
    async def scenario() -> tuple[str, bool, int]:
        conversation = Conversation(model="m")
        _, reply = conversation.send("Hi")
        backend = RecordingBackend()
        generations = Generations()
        generations.start(backend, conversation, reply)  # type: ignore[arg-type]
        # Stopped before the task gets to run: `_run` never starts, so only the
        # done-callback can clean up. Without it the chat would stay busy forever.
        generations.stop(reply.id)
        for _ in range(10):
            await asyncio.sleep(0)
        return reply.status, generations.any_running, backend.calls

    assert asyncio.run(scenario()) == ("done", False, 0)
