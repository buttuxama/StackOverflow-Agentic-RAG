"""Command-line entry point and assistant factory."""

import argparse
import logging

from openai import OpenAIError

from .agent import AgentError, AgenticRAG
from .config import create_llm_client, get_settings


def create_assistant() -> AgenticRAG:
    settings = get_settings()
    return AgenticRAG(create_llm_client(settings), model=settings.model, settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question grounded in Stack Overflow.")
    parser.add_argument("question", help="Programming question to research")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("question must not be empty")
    assistant = None
    try:
        assistant = create_assistant()
        print(assistant.ask(args.question))
    except OpenAIError, AgentError, ValueError:
        logging.getLogger(__name__).error("Request failed. Check configuration and connectivity.")
        return 1
    finally:
        if assistant is not None:
            assistant.llm_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
