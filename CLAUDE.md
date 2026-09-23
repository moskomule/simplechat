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
- Every mutation raises `BusyError` while any message is `pending` or `streaming`. Routes turn that into a 409, and CSS (`.chat:has(.msg.pending, .msg.streaming)`) disables the buttons.

### Streaming lifecycle (`main.py` + `partials/message.html`)

1. A route that creates a reply (send, edit or regenerate) returns HTML containing a `pending` assistant `<article>`. Only in that state does it carry `hx-ext="sse" sse-connect=".../stream/{mid}"`.
2. The browser opens `GET /c/{cid}/stream/{mid}`. That route yields `ServerSentEvent`s named `thinking` and `content` with HTML-escaped text; `sse-swap` appends them with `beforeend`. It then sends a final `done` event whose data is the whole re-rendered message. The article is replaced via `sse-swap="done" hx-swap="outerHTML"`, and `sse-close="done"` stops the stream.
3. **Only a `pending` reply starts generation.** A reconnect or page reload hitting the stream URL gets `done` with the stored state right away. If the client disconnects mid-stream, the `finally` block marks the reply `done` with the partial text. Keep these guards when you change streaming.
4. The stream endpoint must always answer 200 and end with `done`. The htmx SSE extension retries forever on error responses.

Edit, regenerate and switch return the whole `#thread` (`partials/thread.html`), because everything below the changed message can change. Sending returns only the new turn (`partials/turn.html`), plus an out-of-band swap of the sidebar so the title updates.

Jinja is configured with `trim_blocks` and `lstrip_blocks`. The `.content` and `.thinking-content` divs must keep their `{{ … }}` on one line, because CSS uses `white-space: pre-wrap` (there is no Markdown rendering).

### Ollama backend (`llm.py`)

`ChatBackend` is a Protocol. Routes get it from `app.state` through dependencies, and tests inject `FakeBackend` (`tests/test_routes.py`) or an `httpx2.MockTransport` (`tests/test_llm.py`). `create_app()` is a factory that takes `settings`, `backend` and `store`. `run()` starts uvicorn with `factory=True`.

Ollama specifics that the code depends on:
- Chat uses the OpenAI-compatible API at `{OLLAMA_URL}/v1`. Thinking levels come only from the native `POST /api/show`, which is why the setting is the server root rather than the `/v1` URL.
- `/api/show` returns `thinking.values` like `[false, "low", "medium", "high"]`. `false` is mapped to `"none"`, which turns thinking off. The level is sent as `extra_body={"reasoning_effort": ...}`, because level names are model-defined, and it's omitted for the model's default.
- Thinking text streams in the non-standard `delta.reasoning` field. It is stored in `Message.thinking` and never sent back as history.
- `/v1/models` returns `"data": null` when no model is pulled.

### Library versions

These are newer than many references, so check installed sources before assuming APIs:
- **FastAPI:** SSE is built in (`fastapi.sse.EventSourceResponse` / `ServerSentEvent`, with `raw_data` for non-JSON). PEP 695 `type X = Annotated[..., Depends(...)]` aliases work for dependencies.
- **Starlette `TestClient`:** uses `httpx2`, not `httpx`.
- **openai:** its clients are built on `httpx2` too.
- **htmx:** it's 2.x; 4.x is only a pre-release.
