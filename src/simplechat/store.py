"""In-memory conversation storage.

Each conversation is a tree of messages. A message's children are alternative
versions of the next turn: editing a user message or regenerating an assistant
reply adds a sibling instead of overwriting. `active_child` picks which version
is shown, and following it from the root gives the active path.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

type Role = Literal["root", "user", "assistant"]
type Status = Literal["pending", "streaming", "done", "error"]

DEFAULT_TITLE = "New chat"
TITLE_LENGTH = 40


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class BusyError(Exception):
    """Raised when changing a conversation while a reply is being generated."""


@dataclass(frozen=True, slots=True)
class Image:
    """An image attached to a user message."""

    data: bytes
    media_type: str


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    images: list[Image] = field(default_factory=list)
    thinking: str = ""
    parent_id: str | None = None
    status: Status = "done"
    error: str | None = None
    id: str = field(default_factory=new_id)
    children: list[str] = field(default_factory=list)
    active_child: int = 0

    @property
    def is_busy(self) -> bool:
        return self.status in ("pending", "streaming")


@dataclass(slots=True)
class Conversation:
    model: str
    system_prompt: str = ""
    # A level from the model's thinking options; empty means the model's default.
    thinking: str = ""
    title: str = DEFAULT_TITLE
    id: str = field(default_factory=new_id)
    updated_at: float = field(default_factory=time.time)
    messages: dict[str, Message] = field(default_factory=dict)
    root_id: str = ""

    def __post_init__(self) -> None:
        root = Message(role="root")
        self.messages[root.id] = root
        self.root_id = root.id

    # --- queries ---

    def active_path(self) -> list[Message]:
        """Messages shown in the UI, from the first turn to the last."""
        path: list[Message] = []
        node = self.messages[self.root_id]
        while node.children:
            node = self.messages[node.children[node.active_child]]
            path.append(node)
        return path

    def ancestors(self, message_id: str) -> list[Message]:
        """Messages before `message_id` in its branch, excluding the root."""
        chain: list[Message] = []
        parent_id = self.messages[message_id].parent_id
        while parent_id is not None and parent_id != self.root_id:
            parent = self.messages[parent_id]
            chain.append(parent)
            parent_id = parent.parent_id
        return chain[::-1]

    def version(self, message_id: str) -> tuple[int, int]:
        """1-based index of the message among its siblings, and the sibling count."""
        message = self.messages[message_id]
        assert message.parent_id is not None
        siblings = self.messages[message.parent_id].children
        return siblings.index(message_id) + 1, len(siblings)

    @property
    def is_busy(self) -> bool:
        return any(m.is_busy for m in self.messages.values())

    # --- mutations ---

    def send(self, content: str, images: Sequence[Image] = ()) -> tuple[Message, Message]:
        """Append a user message and a pending assistant reply to the active path."""
        self._ensure_idle()
        path = self.active_path()
        parent_id = path[-1].id if path else self.root_id
        user = self._add_child(parent_id, "user", content, images=images)
        if self.title == DEFAULT_TITLE:
            self.title = _make_title(content)
        return user, self._add_child(user.id, "assistant", status="pending")

    def edit(self, message_id: str, content: str, images: Sequence[Image] = ()) -> Message:
        """Add an edited version of a user message and return its pending reply."""
        self._ensure_idle()
        original = self._get(message_id, "user")
        assert original.parent_id is not None
        user = self._add_child(original.parent_id, "user", content, images=images)
        return self._add_child(user.id, "assistant", status="pending")

    def regenerate(self, message_id: str) -> Message:
        """Add a new pending version of an assistant reply."""
        self._ensure_idle()
        original = self._get(message_id, "assistant")
        assert original.parent_id is not None
        return self._add_child(original.parent_id, "assistant", status="pending")

    def switch(self, message_id: str, step: int) -> None:
        """Show the previous (step=-1) or next (step=+1) version of a message."""
        self._ensure_idle()
        message = self.messages[message_id]
        assert message.parent_id is not None
        parent = self.messages[message.parent_id]
        index = parent.children.index(message_id) + step
        parent.active_child = max(0, min(index, len(parent.children) - 1))

    def _add_child(
        self,
        parent_id: str,
        role: Role,
        content: str = "",
        status: Status = "done",
        images: Sequence[Image] = (),
    ) -> Message:
        parent = self.messages[parent_id]
        child = Message(
            role=role, content=content, images=list(images), parent_id=parent_id, status=status
        )
        self.messages[child.id] = child
        parent.children.append(child.id)
        parent.active_child = len(parent.children) - 1
        self.updated_at = time.time()
        return child

    def _get(self, message_id: str, role: Role) -> Message:
        message = self.messages[message_id]
        if message.role != role:
            raise ValueError(f"Message {message_id} is not a {role} message")
        return message

    def _ensure_idle(self) -> None:
        if self.is_busy:
            raise BusyError("A reply is still being generated")


def _make_title(content: str) -> str:
    title = " ".join(content.split())
    if len(title) > TITLE_LENGTH:
        title = title[: TITLE_LENGTH - 1].rstrip() + "…"
    return title or DEFAULT_TITLE


class Store:
    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}

    def create(self, model: str) -> Conversation:
        conversation = Conversation(model=model)
        self._conversations[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def delete(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)

    def recent(self) -> list[Conversation]:
        return sorted(self._conversations.values(), key=lambda c: c.updated_at, reverse=True)
