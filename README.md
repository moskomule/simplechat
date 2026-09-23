# simplechat

A minimal ChatGPT-style chat UI for [Ollama](https://ollama.com), built with FastAPI and htmx.

- Streaming replies (Server-Sent Events)
- Edit your messages and regenerate replies; every version is kept and you can switch between them with ‹ 1/2 ›
- Model picker and a system prompt for each conversation; the system prompt is saved with its Save button (or Ctrl/Cmd+Enter)
- Thinking level for models that can think (e.g. off, low, medium, high), with the model's thoughts shown in a collapsible section
- Images for vision models: attach them (on phones, from the camera or photo library) or paste them, preview before sending, and drop them when editing a message. PNG, JPEG, WebP or GIF, up to 10 per message and 20 MB each
- Works on phones: open it from any device on the same network
- Collapsible chat list and top bar for more room; each browser remembers the choice
- A "free memory" button in the top bar that unloads every model from Ollama (weights and KV cache), without restarting it. The next message reloads the model, so its first reply is slower

Conversations are kept in memory, so they are lost when the server stops.

## Requirements

- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- Ollama running with at least one model pulled (e.g. `ollama pull qwen3`)

## Usage

```sh
uv run simplechat
```

It prints two addresses:

```
SimpleChat: http://localhost:8000
On your local network: http://192.168.1.23:8000
```

Open the second one on your phone. On macOS, allow incoming connections if the firewall asks.
Ollama itself can stay bound to localhost, because only the server talks to it.

Enter sends a message and Shift+Enter adds a new line. On touch devices, Enter adds a new line and you send with the button.

To use another port or address, pass flags, which take precedence over the environment variables below:

```sh
uv run simplechat --port 8080
uv run simplechat --host 127.0.0.1   # only this machine
```

### Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server. Chat uses its OpenAI-compatible API under `/v1`; thinking levels and vision support come from its native `/api/show`, and "free memory" uses `/api/ps` and `/api/generate` |
| `OLLAMA_API_KEY` | `ollama` | Ignored by Ollama, but required by the OpenAI client |
| `DEFAULT_MODEL` | first model Ollama lists | Model for new conversations |
| `HOST` | `0.0.0.0` | Address to bind; use `127.0.0.1` to keep it local |
| `PORT` | `8000` | Port to listen on |

The server must run as a single process, because conversations live in its memory.

## Development

```sh
uv run pytest
uv run ruff check --fix
uv run ruff format
```

htmx 2.0.11 and htmx-ext-sse 2.2.4 are vendored in `src/simplechat/static/`, so the UI works without internet access.
