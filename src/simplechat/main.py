"""A ChatGPT-style chat UI for Ollama."""

import argparse
import socket
from collections.abc import AsyncIterator
from html import escape
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from simplechat.config import Settings
from simplechat.llm import ChatBackend, OllamaBackend
from simplechat.store import BusyError, Conversation, Message, Store

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True
router = APIRouter()


# --- dependencies ---


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_backend(request: Request) -> ChatBackend:
    return request.app.state.backend


type SettingsDep = Annotated[Settings, Depends(get_settings)]
type StoreDep = Annotated[Store, Depends(get_store)]
type BackendDep = Annotated[ChatBackend, Depends(get_backend)]


def get_conversation(cid: str, store: StoreDep) -> Conversation:
    if (conversation := store.get(cid)) is None:
        raise HTTPException(404, "Conversation not found")
    return conversation


type ConversationDep = Annotated[Conversation, Depends(get_conversation)]


def get_message(mid: str, conversation: ConversationDep) -> Message:
    message = conversation.messages.get(mid)
    if message is None or message.role == "root":
        raise HTTPException(404, "Message not found")
    return message


type MessageDep = Annotated[Message, Depends(get_message)]


# --- helpers ---


async def pick_default_model(settings: Settings, backend: ChatBackend) -> str:
    if settings.default_model:
        return settings.default_model
    models = await backend.list_models()
    return models[0] if models else ""


def render_thread(request: Request, conversation: Conversation) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/thread.html", {"conversation": conversation}
    )


def build_chat_messages(
    conversation: Conversation, reply: Message
) -> list[ChatCompletionMessageParam]:
    """The prompt for `reply`: the system prompt plus every earlier message in its branch."""
    messages: list[ChatCompletionMessageParam] = []
    if conversation.system_prompt.strip():
        messages.append({"role": "system", "content": conversation.system_prompt})
    for message in conversation.ancestors(reply.id):
        if message.role == "user":
            messages.append({"role": "user", "content": message.content})
        else:
            messages.append({"role": "assistant", "content": message.content})
    return messages


# --- pages ---


@router.get("/")
async def index(store: StoreDep, settings: SettingsDep, backend: BackendDep) -> RedirectResponse:
    if conversations := store.recent():
        conversation = conversations[0]
    else:
        conversation = store.create(await pick_default_model(settings, backend))
    return RedirectResponse(f"/c/{conversation.id}", status_code=303)


@router.post("/c")
async def create_conversation(
    store: StoreDep, settings: SettingsDep, backend: BackendDep
) -> RedirectResponse:
    conversation = store.create(await pick_default_model(settings, backend))
    return RedirectResponse(f"/c/{conversation.id}", status_code=303)


@router.get("/c/{cid}", response_class=HTMLResponse)
async def show_conversation(
    request: Request, conversation: ConversationDep, store: StoreDep, backend: BackendDep
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "conversation": conversation,
            "conversations": store.recent(),
            "models": await backend.list_models(),
            "thinking_options": await backend.thinking_options(conversation.model),
        },
    )


@router.delete("/c/{cid}")
def delete_conversation(request: Request, cid: str, store: StoreDep, current: str = "") -> Response:
    store.delete(cid)
    if cid == current:
        return Response(headers={"HX-Redirect": "/"})
    return templates.TemplateResponse(
        request,
        "partials/sidebar.html",
        {"conversations": store.recent(), "current_id": current},
    )


@router.post("/c/{cid}/settings", response_class=HTMLResponse)
async def update_settings(
    request: Request,
    conversation: ConversationDep,
    backend: BackendDep,
    model: Annotated[str, Form()] = "",
    thinking: Annotated[str, Form()] = "",
    system_prompt: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Save the settings and return the thinking picker, whose levels depend on the model."""
    if model:
        conversation.model = model
    conversation.system_prompt = system_prompt
    options = await backend.thinking_options(conversation.model)
    # A level picked for the previous model may not exist for this one.
    conversation.thinking = thinking if options and thinking in options.levels else ""
    return templates.TemplateResponse(
        request,
        "partials/thinking_select.html",
        {"conversation": conversation, "thinking_options": options},
    )


# --- messages ---


@router.post("/c/{cid}/messages", response_class=HTMLResponse)
def send_message(
    request: Request,
    conversation: ConversationDep,
    store: StoreDep,
    content: Annotated[str, Form()],
) -> HTMLResponse:
    if not content.strip():
        raise HTTPException(422, "Message is empty")
    try:
        user, reply = conversation.send(content)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    return templates.TemplateResponse(
        request,
        "partials/turn.html",
        {
            "conversation": conversation,
            "new_messages": [user, reply],
            "conversations": store.recent(),
        },
    )


@router.get("/c/{cid}/messages/{mid}", response_class=HTMLResponse)
def show_message(
    request: Request, conversation: ConversationDep, message: MessageDep
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/message.html", {"conversation": conversation, "message": message}
    )


@router.get("/c/{cid}/messages/{mid}/edit", response_class=HTMLResponse)
def edit_form(request: Request, conversation: ConversationDep, message: MessageDep) -> HTMLResponse:
    if message.role != "user":
        raise HTTPException(400, "Only user messages can be edited")
    return templates.TemplateResponse(
        request, "partials/edit_form.html", {"conversation": conversation, "message": message}
    )


@router.post("/c/{cid}/messages/{mid}/edit", response_class=HTMLResponse)
def edit_message(
    request: Request,
    conversation: ConversationDep,
    message: MessageDep,
    content: Annotated[str, Form()],
) -> HTMLResponse:
    if not content.strip():
        raise HTTPException(422, "Message is empty")
    try:
        conversation.edit(message.id, content)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return render_thread(request, conversation)


@router.post("/c/{cid}/messages/{mid}/regenerate", response_class=HTMLResponse)
def regenerate_message(
    request: Request, conversation: ConversationDep, message: MessageDep
) -> HTMLResponse:
    try:
        conversation.regenerate(message.id)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return render_thread(request, conversation)


@router.post("/c/{cid}/messages/{mid}/switch", response_class=HTMLResponse)
def switch_version(
    request: Request,
    conversation: ConversationDep,
    message: MessageDep,
    step: Annotated[int, Form(ge=-1, le=1)],
) -> HTMLResponse:
    try:
        conversation.switch(message.id, step)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    return render_thread(request, conversation)


@router.get("/c/{cid}/stream/{mid}", response_class=EventSourceResponse)
async def stream_reply(
    conversation: ConversationDep, message: MessageDep, backend: BackendDep
) -> AsyncIterator[ServerSentEvent]:
    # A browser reconnecting (or a page reload) must not start a second generation,
    # so only a pending reply is generated. Anything else just gets its final state.
    if message.status == "pending":
        message.status = "streaming"
        try:
            if not conversation.model:
                raise ValueError("No model selected. Is Ollama running?")
            chunks = backend.stream_chat(
                conversation.model,
                build_chat_messages(conversation, message),
                thinking=conversation.thinking,
            )
            async for kind, text in chunks:
                if kind == "thinking":
                    message.thinking += text
                else:
                    message.content += text
                yield ServerSentEvent(event=kind, raw_data=escape(text))
            message.status = "done"
        except (OpenAIError, ValueError) as e:
            message.status = "error"
            message.error = str(e)
        finally:
            # The client went away mid-stream: keep what was generated so far.
            if message.status == "streaming":
                message.status = "done"
    html = templates.get_template("partials/message.html").render(
        conversation=conversation, message=message
    )
    yield ServerSentEvent(event="done", raw_data=html)


# --- app ---


def create_app(
    settings: Settings | None = None,
    backend: ChatBackend | None = None,
    store: Store | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="SimpleChat")
    app.state.settings = settings
    app.state.backend = backend or OllamaBackend(settings.ollama_url, settings.ollama_api_key)
    app.state.store = store or Store()
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    app.include_router(router)
    return app


def lan_address() -> str | None:
    """Best-effort guess of this machine's address on the local network."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # No packets are sent; this only asks the OS which interface it would use.
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return None


def parse_args(settings: Settings, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="simplechat", description=__doc__)
    parser.add_argument(
        "--host",
        default=settings.host,
        help="address to bind; 127.0.0.1 keeps it local (default: $HOST or %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="port to listen on (default: $PORT or %(default)s)",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(Settings.from_env(), argv)
    print(f"SimpleChat: http://localhost:{args.port}", flush=True)
    if args.host == "0.0.0.0" and (address := lan_address()):
        print(f"On your local network: http://{address}:{args.port}", flush=True)
    # A single worker: conversations live in this process's memory.
    uvicorn.run(
        "simplechat.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
    )
