"""Environment-only configuration. Imports never read credentials or open connections."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from openai import OpenAI

DEFAULT_MODEL = "gemma4:31b"


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    base_url: str = "https://ollama.com/v1"
    api_key: str = field(default="", repr=False)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = ""
    postgres_user: str = ""
    postgres_password: str = field(default="", repr=False)
    input_price: float = 0.0
    cached_input_price: float = 0.0
    output_price: float = 0.0


def get_settings(env_file: Path | None = None) -> Settings:
    """Read .env in the working directory; shell/container values take precedence."""
    file_values = dotenv_values(env_file or Path.cwd() / ".env")
    values = {key: value for key, value in file_values.items() if value is not None}
    values.update(os.environ)
    input_price = float(values.get("LLM_INPUT_PRICE_PER_MILLION", "0"))
    settings = Settings(
        model=values.get("LLM_MODEL", DEFAULT_MODEL),
        base_url=values.get("LLM_BASE_URL", "https://ollama.com/v1"),
        api_key=values.get("LLM_API_KEY") or values.get("OLLAMA_API_KEY", ""),
        postgres_host=values.get("POSTGRES_HOST", "localhost"),
        postgres_port=int(values.get("POSTGRES_PORT", "5432")),
        postgres_db=values.get("POSTGRES_DB", ""),
        postgres_user=values.get("POSTGRES_USER", ""),
        postgres_password=values.get("POSTGRES_PASSWORD", ""),
        input_price=input_price,
        cached_input_price=float(
            values.get("LLM_CACHED_INPUT_PRICE_PER_MILLION", str(input_price))
        ),
        output_price=float(values.get("LLM_OUTPUT_PRICE_PER_MILLION", "0")),
    )
    url = urlsplit(settings.base_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("LLM_BASE_URL must be an HTTP(S) URL without embedded credentials.")
    if not 1 <= settings.postgres_port <= 65535:
        raise ValueError("POSTGRES_PORT must be between 1 and 65535.")
    if any(
        not 0 <= price < float("inf")
        for price in (settings.input_price, settings.cached_input_price, settings.output_price)
    ):
        raise ValueError("Token prices must be finite and nonnegative.")
    return settings


def create_llm_client(settings: Settings | None = None) -> OpenAI:
    """Create a client owned by the caller; credentials are never logged."""
    settings = settings or get_settings()
    if not settings.api_key:
        raise ValueError("Set OLLAMA_API_KEY or LLM_API_KEY in your project .env.")
    return OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=60, max_retries=0)
