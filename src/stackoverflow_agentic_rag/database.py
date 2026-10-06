"""Parameterized PostgreSQL persistence with explicit connection lifetimes."""

from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
from psycopg.rows import dict_row

from .config import get_settings
from .metrics import LLMCallRecord


@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float


def row_to_record(row: dict) -> LLMCallRecord:
    return LLMCallRecord(**row)


def get_db_connection() -> psycopg.Connection:
    """Open a connection; callers own its transaction and lifetime."""
    settings = get_settings()
    if not all((settings.postgres_db, settings.postgres_user, settings.postgres_password)):
        raise psycopg.OperationalError(
            "Set POSTGRES_DB, POSTGRES_USER and POSTGRES_PASSWORD in .env."
        )
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        connect_timeout=10,
    )


def init_conversations_table(drop: bool = False) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS conversations")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    model TEXT NOT NULL,
                    instructions TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    response_time FLOAT NOT NULL,
                    cost FLOAT,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
            cur.execute("""
                ALTER TABLE conversations
                    ALTER COLUMN prompt_tokens DROP NOT NULL,
                    ALTER COLUMN completion_tokens DROP NOT NULL,
                    ALTER COLUMN total_tokens DROP NOT NULL,
                    ALTER COLUMN cost DROP NOT NULL
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS conversations_timestamp_idx
                ON conversations (timestamp DESC)
            """)


def init_feedback_table(drop: bool = False) -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            if drop:
                cur.execute("DROP TABLE IF EXISTS feedback")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)


def save_conversation(record: LLMCallRecord, question: str) -> int:
    timestamp = datetime.now(UTC)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, model, instructions, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    record.model,
                    record.instructions,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
    return conversation_id


def save_feedback(
    conversation_id: int,
    source: str,
    relevance: str | None = None,
    explanation: str | None = None,
    score: int | None = None,
) -> None:
    if source not in {"user", "judge"}:
        raise ValueError("Feedback source must be user or judge.")
    if source == "user" and score not in {-1, 1}:
        raise ValueError("User feedback score must be -1 or 1.")
    if source == "judge" and relevance not in {"RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"}:
        raise ValueError("Invalid relevance verdict.")
    timestamp = datetime.now(UTC)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (conversation_id, source, relevance, explanation, score, timestamp),
            )


def get_conversations(limit: int = 10) -> list[LLMCallRecord]:
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Conversation limit must be between 1 and 1000.")
    with get_db_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT id, question, answer, model,
                       instructions, prompt,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC, id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [row_to_record(row) for row in rows]


def get_stats() -> Stats:
    with get_db_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    COALESCE(AVG(response_time), 0) AS avg_response_time,
                    COALESCE(SUM(cost), 0) AS total_cost,
                    COALESCE(AVG(total_tokens), 0) AS avg_tokens
                FROM conversations
            """)
            row = cur.fetchone()

    return Stats(**row)


def get_relevance_stats() -> dict[str, int]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relevance, COUNT(*)
                FROM feedback
                WHERE source = 'judge' AND relevance IS NOT NULL
                GROUP BY relevance
            """)
            rows = cur.fetchall()
    return dict(rows)


def get_user_feedback_stats() -> tuple[int, int]:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END), 0)
                FROM feedback
                WHERE source = 'user'
            """)
            row = cur.fetchone()
    return row


def main() -> None:
    """Initialize the schema without deleting stored conversations or feedback."""
    init_conversations_table()
    init_feedback_table()
    print("Database initialized")


if __name__ == "__main__":
    main()
