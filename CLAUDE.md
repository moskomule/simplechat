# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
uv run simplechat                       # serve on 0.0.0.0:8000; prints the LAN URL
uv run simplechat --host 127.0.0.1 --port 8765
uv run pytest                           # all tests
uv run pytest tests/test_routes.py::test_send_and_stream   # one test
uv run ruff check --fix
uv run ruff format
```

Needs Python 3.14 (see `.python-version`). A running Ollama is only needed to use the app, not for tests.
Configuration comes from environment variables in `config.py` (`OLLAMA_URL`, `DEFAULT_MODEL`, `HOST`, `PORT`, …). CLI flags override `HOST` and `PORT`.

## Architecture

A server-rendered chat UI: FastAPI returns HTML fragments, htmx swaps them into the page, and replies stream over Server-Sent Events. There is no frontend build step. htmx and its SSE extension are vendored in `static/` so phones on a LAN without internet still work.

### Message tree (`store.py`)

Conversations are in-memory `Conversation` objects, so everything is lost on restart and the server must run as **one process**. Each conversation is a tree of `Message`s under a dummy `root` message:

- A message's `children` are alternative versions of the next turn. `active_child` selects which one is shown. Following `active_child` from the root gives `active_path()`, which is what the UI renders.
- Editing a user message adds a sibling user message plus a new `pending` assistant reply. Regenerating adds a sibling assistant message. Nothing is overwritten, and `switch()` moves between siblings.
- The prompt for a reply is `ancestors(reply_id)`: its branch, not the active path.
- Every mutation raises `BusyError` while any message is `pending` or `streaming`. Routes turn that into a 409, and `app.js` makes the action and send buttons `inert` (blocking keyboard as well as pointer).
- The store is not thread-safe. **Every route and store-reading dependency is `async def`**, so it runs on the event loop and a busy check and the mutation after it can't interleave. Don't add `await` between them, and don't add plain `def` routes, which FastAPI would run in a thread pool.

### Streaming lifecycle (`generation.py`, `main.py`, `partials/message.html`)

1. A route that creates a reply (send, edit or regenerate) calls `Generations.start()`. That runs the model call as a background `asyncio` task, **not tied to any request**: it keeps going if every client disconnects, and it writes into the `Message` as chunks arrive.
2. The route returns HTML containing the busy assistant `<article>`. Only in that state does it carry `hx-ext="sse" sse-connect=".../stream/{mid}"`. A busy message renders **empty**, because the stream replays everything generated so far.
3. `GET /c/{cid}/stream/{mid}` follows the running `Generation`: it replays all chunks from the start, then yields new ones as `thinking` and `content` events with HTML-escaped text. `sse-swap` appends them with `beforeend`. Any number of clients (a reconnect, a reload, a second device) can follow the same generation, and the model is called only once.
4. Only after the generation finishes does the stream send `done`, whose data is the whole re-rendered, finished message. The article is replaced via `sse-swap="done" hx-swap="outerHTML"`, and `sse-close="done"` stops the stream. A reply that already finished gets `done` right away. `done` must never carry a busy message, or the client would reconnect in a loop.
5. The stream endpoint must always answer 200 and end with `done`. The htmx SSE extension retries forever on error responses.

In tests, the `client` fixture keeps one `TestClient` context (and so one event loop) open across requests, so background generations survive between requests. To check streamed chunks, hold the reply with `FakeBackend.released = False` and use `concurrently()`.

Edit, regenerate and switch return the whole `#thread` (`partials/thread.html`), because everything below the changed message can change. Sending returns only the new turn (`partials/turn.html`), plus an out-of-band swap of the sidebar so the title updates.

Jinja is configured with `trim_blocks` and `lstrip_blocks`. The `.content` and `.thinking-content` divs must keep their `{{ … }}` on one line, because CSS uses `white-space: pre-wrap` (there is no Markdown rendering).

### Images

- User messages keep images in memory as `Message.images` (`store.Image`: bytes plus media type). `GET /c/{cid}/messages/{mid}/images/{i}` serves them with a long `immutable` cache header, because a message's images never change (an edit creates a new message). Never inline `data:` URLs in HTML, since the thread is re-rendered often.
- The composer posts `multipart/form-data`. `read_images` in `main.py` checks type, count and size, and must run before the busy check because it awaits. It skips the empty part a browser sends for an empty file input. Images for a model without `vision` get a 400. `ensure_vision` awaits `/api/show`, and the model can change meanwhile, so it refuses (409) if the model changed during the check. Call it last before the mutation.
- Images stay in memory until their chat is deleted, so `MAX_STORED_IMAGE_BYTES` caps the total across all chats (413 when full). `Store.image_bytes()` counts images shared between an original and its edit only once.
- The attach button (`partials/attach_button.html`) shows only for vision models. The settings route swaps it out of band next to the thinking picker (`partials/model_controls.html`).
- `app.js` keeps the selected files in an array and writes them back to the hidden file input with a `DataTransfer`, so the normal htmx submit sends them. The edit form sends a `keep` index per image. Removing an image's element drops that index.
- `build_chat_messages` sends a message with images as OpenAI content parts: `image_url` parts with `data:` URLs, then a text part. Messages without images keep plain string content.

### Ollama backend (`llm.py`)

`ChatBackend` is a Protocol. Routes get it from `app.state` through dependencies, and tests inject `FakeBackend` (`tests/test_routes.py`) or an `httpx2.MockTransport` (`tests/test_llm.py`). `create_app()` is a factory that takes `settings`, `backend` and `store`, and it creates the `Generations` registry. `run()` starts uvicorn with `factory=True`.

Ollama specifics that the code depends on:
- Chat uses the OpenAI-compatible API at `{OLLAMA_URL}/v1`. Thinking levels and capabilities come only from the native `POST /api/show`, which is why the setting is the server root rather than the `/v1` URL. `model_info()` reads both in one call: `capabilities` (e.g. `["completion", "vision", "thinking"]`) gives `ModelInfo.vision`.
- `/api/show` returns `thinking.values` like `[false, "low", "medium", "high"]`. `false` is mapped to `"none"`, which turns thinking off. The level is sent as `extra_body={"reasoning_effort": ...}`, because level names are model-defined, and it's omitted for the model's default.
- Thinking text streams in the non-standard `delta.reasoning` field. It is stored in `Message.thinking` and never sent back as history.
- `/v1/models` returns `"data": null` when no model is pulled.
- "Free memory" (`POST /free-memory` → `unload_models()`) lists loaded models with native `GET /api/ps`, then unloads each with `POST /api/generate {"model": ..., "keep_alive": 0}` (no prompt). That frees weights and KV cache together; Ollama has no way to drop only the KV cache. An idle model is gone from `/api/ps` as soon as that request returns (measured on Ollama 0.34), but one still answering another client stays loaded until its reply ends, so `/api/ps` is read again to report those as pending. These calls use a 30 s timeout, because Ollama can answer slowly while loading a model. The route refuses while any reply is generating, and answers every outcome with a 200 status text, since htmx does not swap error responses. Unloading and starting a generation both await, so they are serialized by `Generations.lock`: every route that calls `Generations.start()` must hold it from its last check through `start()`.

### Library versions

These are newer than many references, so check installed sources before assuming APIs:
- **FastAPI:** SSE is built in (`fastapi.sse.EventSourceResponse` / `ServerSentEvent`, with `raw_data` for non-JSON). PEP 695 `type X = Annotated[..., Depends(...)]` aliases work for dependencies.
- **Starlette `TestClient`:** uses `httpx2`, not `httpx`.
- **openai:** its clients are built on `httpx2` too.
- **htmx:** it's 2.x; 4.x is only a pre-release.
