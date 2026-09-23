from simplechat.config import Settings
from simplechat.main import parse_args

SETTINGS = Settings(
    ollama_base_url="http://unused",
    ollama_api_key="unused",
    default_model="",
    host="0.0.0.0",
    port=8000,
)


def test_defaults_come_from_settings() -> None:
    args = parse_args(SETTINGS, [])
    assert (args.host, args.port) == ("0.0.0.0", 8000)


def test_flags_override_settings() -> None:
    args = parse_args(SETTINGS, ["--port", "8123", "--host", "127.0.0.1"])
    assert (args.host, args.port) == ("127.0.0.1", 8123)
