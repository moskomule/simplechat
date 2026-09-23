"""A ChatGPT-style chat UI for Ollama."""

import argparse
import socket
from collections.abc import AsyncIterator
from html import escape
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from simplechat.config import Settings
from simplechat.generation import Generations
from simplechat.llm import ChatBackend, OllamaBackend
from simplechat.store import BusyError, Conversation, Image, Message, Store

BASE_DIR = Path(__file__).parent
# Image types that Ollama's vision models read, and limits per message.
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
MAX_IMAGES = 10
MAX_IMAGE_BYTES = 20 * 2**20
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True
router = APIRouter()

# Every route and store-reading dependency is `async def` on purpose: FastAPI runs
# plain `def` ones in a thread pool, but the store is not thread-safe. On the event
# loop, a check such as "is a reply being generated?" and the mutation after it
# cannot interleave with another request, as long as there is no `await` between them.


# --- dependencies ---


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_backend(request: Request) -> ChatBackend:
    return request.app.state.backend


def get_generations(request: Request) -> Generations:
    return request.app.state.generations


type SettingsDep = Annotated[Settings, Depends(get_settings)]
type StoreDep = Annotated[Store, Depends(get_store)]
type BackendDep = Annotated[ChatBackend, Depends(get_backend)]
type GenerationsDep = Annotated[Generations, Depends(get_generations)]


async def get_conversation(cid: str, store: StoreDep) -> Conversation:
    if (conversation := store.get(cid)) is None:
        raise HTTPException(404, "Conversation not found")
    return conversation


type ConversationDep = Annotated[Conversation, Depends(get_conversation)]


async def get_message(mid: str, conversation: ConversationDep) -> Message:
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


async def read_images(uploads: list[UploadFile]) -> list[Image]:
    """Check and read uploaded images. Call before any busy check, since it awaits."""
    # Skip the empty part a browser sends for a file input with nothing selected.
    uploads = [upload for upload in uploads if upload.filename or upload.size]
    if len(uploads) > MAX_IMAGES:
        raise HTTPException(400, f"At most {MAX_IMAGES} images can be sent at once")
    images: list[Image] = []
    for upload in uploads:
        if upload.content_type not in IMAGE_TYPES:
            raise HTTPException(400, "Images must be PNG, JPEG, WebP or GIF")
        # The size is known once the form is parsed, before reading into memory.
        if upload.size is None or upload.size > MAX_IMAGE_BYTES:
            raise HTTPException(413, f"Images must be {MAX_IMAGE_BYTES // 2**20} MB or smaller")
        images.append(Image(data=await upload.read(), media_type=upload.content_type))
    return images


async def ensure_vision(backend: ChatBackend, conversation: Conversation) -> None:
    if not (await backend.model_info(conversation.model)).vision:
        raise HTTPException(400, f"{conversation.model or 'This model'} cannot read images")


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
    info = await backend.model_info(conversation.model)
    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "conversation": conversation,
            "conversations": store.recent(),
            "models": await backend.list_models(),
            "thinking_options": info.thinking,
            "vision": info.vision,
        },
    )


@router.delete("/c/{cid}")
async def delete_conversation(
    request: Request, cid: str, store: StoreDep, current: str = ""
) -> Response:
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
    """Save the settings and return the thinking picker, whose levels depend on the model.

    The attach button, shown only for vision models, is swapped in out of band.
    """
    if model:
        conversation.model = model
    conversation.system_prompt = system_prompt
    info = await backend.model_info(conversation.model)
    options = info.thinking
    # A level picked for the previous model may not exist for this one.
    conversation.thinking = thinking if options and thinking in options.levels else ""
    return templates.TemplateResponse(
        request,
        "partials/model_controls.html",
        {"conversation": conversation, "thinking_options": options, "vision": info.vision},
    )


# --- messages ---


@router.post("/c/{cid}/messages", response_class=HTMLResponse)
async def send_message(
    request: Request,
    conversation: ConversationDep,
    store: StoreDep,
    backend: BackendDep,
    generations: GenerationsDep,
    content: Annotated[str, Form()] = "",
    images: Annotated[list[UploadFile] | None, File()] = None,
) -> HTMLResponse:
    # Everything that awaits happens before `send`, which checks for a running reply.
    attached = await read_images(images or [])
    if not content.strip() and not attached:
        raise HTTPException(422, "Message is empty")
    if attached:
        await ensure_vision(backend, conversation)
    try:
        user, reply = conversation.send(content, attached)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    generations.start(backend, conversation, reply)
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
async def show_message(
    request: Request, conversation: ConversationDep, message: MessageDep
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/message.html", {"conversation": conversation, "message": message}
    )


@router.get("/c/{cid}/messages/{mid}/edit", response_class=HTMLResponse)
async def edit_form(
    request: Request, conversation: ConversationDep, message: MessageDep
) -> HTMLResponse:
    if message.role != "user":
        raise HTTPException(400, "Only user messages can be edited")
    return templates.TemplateResponse(
        request, "partials/edit_form.html", {"conversation": conversation, "message": message}
    )


@router.get("/c/{cid}/messages/{mid}/images/{index}")
async def show_image(message: MessageDep, index: int) -> Response:
    if not 0 <= index < len(message.images):
        raise HTTPException(404, "Image not found")
    image = message.images[index]
    # A message's images never change: editing creates a new message.
    return Response(
        image.data,
        media_type=image.media_type,
        headers={
            "Cache-Control": "private, max-age=31536000, immutable",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/c/{cid}/messages/{mid}/edit", response_class=HTMLResponse)
async def edit_message(
    request: Request,
    conversation: ConversationDep,
    message: MessageDep,
    backend: BackendDep,
    generations: GenerationsDep,
    content: Annotated[str, Form()] = "",
    keep: Annotated[list[int] | None, Form()] = None,
) -> HTMLResponse:
    """Add an edited version; `keep` lists the indexes of the images still attached."""
    indexes = sorted(set(keep or []))
    if not all(0 <= i < len(message.images) for i in indexes):
        raise HTTPException(400, "Image not found")
    images = [message.images[i] for i in indexes]
    if not content.strip() and not images:
        raise HTTPException(422, "Message is empty")
    if images:
        await ensure_vision(backend, conversation)
    try:
        reply = conversation.edit(message.id, content, images)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    generations.start(backend, conversation, reply)
    return render_thread(request, conversation)


@router.post("/c/{cid}/messages/{mid}/regenerate", response_class=HTMLResponse)
async def regenerate_message(
    request: Request,
    conversation: ConversationDep,
    message: MessageDep,
    backend: BackendDep,
    generations: GenerationsDep,
) -> HTMLResponse:
    try:
        reply = conversation.regenerate(message.id)
    except BusyError as e:
        raise HTTPException(409, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    generations.start(backend, conversation, reply)
    return render_thread(request, conversation)


@router.post("/c/{cid}/messages/{mid}/switch", response_class=HTMLResponse)
async def switch_version(
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
    conversation: ConversationDep, message: MessageDep, generations: GenerationsDep
) -> AsyncIterator[ServerSentEvent]:
    """Follow a reply's generation, then send the finished message as `done`.

    Generation runs in the background, started by the route that created the reply.
    Every client (a reconnect, a reload, a second device) replays the chunks from
    the start and then follows along, so `done` is only sent once the reply is
    really finished. A reply that has already finished gets `done` right away.
    """
    if (generation := generations.get(message.id)) is not None:
        async for kind, text in generation.follow():
            yield ServerSentEvent(event=kind, raw_data=escape(text))
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
    app.state.generations = Generations()
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
