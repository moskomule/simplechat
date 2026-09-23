import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str
    ollama_api_key: str
    default_model: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            # Ollama ignores the key, but the OpenAI client requires one.
            ollama_api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            default_model=os.environ.get("DEFAULT_MODEL", ""),
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
        )
