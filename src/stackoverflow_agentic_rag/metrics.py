from dataclasses import dataclass, field
from datetime import UTC, datetime

from openai.types import CompletionUsage

from .config import Settings, get_settings


@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_time: float
    cost: float | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: int | None = None
    question: str | None = None


def calculate_cost(usage: CompletionUsage, settings: Settings | None = None) -> float:
    """Estimate USD from configured per-million-token rates, not a guessed model price."""
    if usage is None:
        raise ValueError("Response does not contain token usage")

    settings = settings or get_settings()
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0

    cached = min(max(cached, 0), usage.prompt_tokens)
    return (
        (usage.prompt_tokens - cached) * settings.input_price
        + cached * settings.cached_input_price
        + usage.completion_tokens * settings.output_price
    ) / 1_000_000
