import pytest

from simplechat.store import BusyError, Conversation, Image, Message, Store


def finish(message: Message, content: str) -> None:
    message.content = content
    message.status = "done"


def contents(conversation: Conversation) -> list[str]:
    return [m.content for m in conversation.active_path()]


def test_send_appends_to_active_path() -> None:
    conversation = Conversation(model="m")
    user, reply = conversation.send("hi")
    assert user.role == "user"
    assert reply.role == "assistant"
    assert reply.status == "pending"
    assert conversation.active_path() == [user, reply]
    assert conversation.title == "hi"


def test_send_while_busy_raises() -> None:
    conversation = Conversation(model="m")
    conversation.send("hi")
    with pytest.raises(BusyError):
        conversation.send("again")


def test_ancestors_exclude_root_and_message() -> None:
    conversation = Conversation(model="m")
    user1, reply1 = conversation.send("q1")
    finish(reply1, "a1")
    user2, reply2 = conversation.send("q2")
    assert conversation.ancestors(reply2.id) == [user1, reply1, user2]


def test_edit_creates_sibling_branch() -> None:
    conversation = Conversation(model="m")
    user, reply = conversation.send("q1")
    finish(reply, "a1")
    _, reply2 = conversation.send("q2")
    finish(reply2, "a2")

    new_reply = conversation.edit(user.id, "q1 edited")
    finish(new_reply, "a1'")

    assert contents(conversation) == ["q1 edited", "a1'"]
    new_user = conversation.active_path()[0]
    assert conversation.version(new_user.id) == (2, 2)

    # The old branch, including its follow-up turn, is still reachable.
    conversation.switch(new_user.id, -1)
    assert contents(conversation) == ["q1", "a1", "q2", "a2"]


def test_regenerate_creates_sibling_reply() -> None:
    conversation = Conversation(model="m")
    user, reply = conversation.send("q")
    finish(reply, "a")

    new_reply = conversation.regenerate(reply.id)
    assert new_reply.parent_id == user.id
    assert new_reply.status == "pending"
    assert conversation.version(new_reply.id) == (2, 2)
    assert conversation.active_path() == [user, new_reply]


def test_switch_clamps_to_range() -> None:
    conversation = Conversation(model="m")
    _, reply = conversation.send("q")
    finish(reply, "a")
    conversation.switch(reply.id, 1)
    assert conversation.version(conversation.active_path()[1].id) == (1, 1)


def test_role_checks() -> None:
    conversation = Conversation(model="m")
    user, reply = conversation.send("q")
    finish(reply, "a")
    with pytest.raises(ValueError):
        conversation.edit(reply.id, "x")
    with pytest.raises(ValueError):
        conversation.regenerate(user.id)


def test_long_title_is_truncated() -> None:
    conversation = Conversation(model="m")
    conversation.send("word " * 50)
    assert len(conversation.title) <= 40
    assert conversation.title.endswith("…")


def test_store_recent_orders_by_update() -> None:
    store = Store()
    first = store.create("m")
    second = store.create("m")
    first.send("newer")
    assert store.recent() == [first, second]
    store.delete(first.id)
    assert store.recent() == [second]


def test_image_bytes_counts_shared_images_once() -> None:
    store = Store()
    conversation = store.create("m")
    image = Image(data=b"12345", media_type="image/png")
    user, reply = conversation.send("look", [image])
    finish(reply, "ok")
    # An edit that keeps the image shares it with the original message.
    conversation.edit(user.id, "look again", [image])
    assert store.image_bytes() == 5
    store.delete(conversation.id)
    assert store.image_bytes() == 0
