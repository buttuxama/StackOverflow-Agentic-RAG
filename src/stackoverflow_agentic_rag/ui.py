"""Streamlit workspace for answers, feedback, and conversation analytics."""

from dataclasses import asdict
from html import escape

import pandas as pd
import psycopg
import streamlit as st
from openai import OpenAIError
from pydantic import ValidationError

from .agent import AgentError
from .cli import create_assistant
from .config import get_settings
from .database import (
    get_conversations,
    get_relevance_stats,
    get_stats,
    get_user_feedback_stats,
    save_conversation,
    save_feedback,
)
from .judge import evaluate_relevance
from .metrics import LLMCallRecord
from .styles import STYLE


def format_cost(cost: float | None) -> str:
    return f"${cost:.6f}" if cost is not None else "Unavailable"


def show_cost_configuration() -> None:
    settings = get_settings()
    if not any((settings.input_price, settings.cached_input_price, settings.output_price)):
        st.caption(
            "Token prices are set to zero. Configure LLM_INPUT_PRICE_PER_MILLION, "
            "LLM_CACHED_INPUT_PRICE_PER_MILLION, and LLM_OUTPUT_PRICE_PER_MILLION "
            "in .env and restart the app to estimate costs."
        )


def show_call_metrics(record: LLMCallRecord) -> None:
    columns = st.columns(4)
    columns[0].metric("Response time", f"{record.response_time:.2f}s")
    columns[1].metric(
        "Prompt tokens", record.prompt_tokens if record.prompt_tokens is not None else "—"
    )
    columns[2].metric(
        "Completion tokens",
        record.completion_tokens if record.completion_tokens is not None else "—",
    )
    columns[3].metric("Cost", format_cost(record.cost))
    show_cost_configuration()


def ask_question(question: str) -> None:
    assistant = None
    try:
        assistant = create_assistant()
        with st.spinner("Finding an answer…"):
            answer = assistant.ask(question)
        record = assistant.last_call
        result = {
            "question": question,
            "answer": answer,
            "record": record,
            "conversation_id": None,
            "relevance": None,
            "explanation": None,
            "save_failed": False,
            "judge_failed": False,
            "vote": None,
        }
        st.session_state.result = result
        st.session_state.conversation_id = None
        try:
            conversation_id = save_conversation(record, question)
        except psycopg.Error:
            result["save_failed"] = True
            return
        result["conversation_id"] = conversation_id
        st.session_state.conversation_id = conversation_id
        try:
            with st.spinner("Checking relevance…"):
                relevance, explanation = evaluate_relevance(
                    question, answer, client=assistant.llm_client, model=assistant.model
                )
            save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)
        except OpenAIError, ValidationError, ValueError, psycopg.Error:
            result["judge_failed"] = True
        else:
            result["relevance"] = relevance
            result["explanation"] = explanation
    except OpenAIError, AgentError, ValueError:
        st.error("Could not generate an answer. Check LLM configuration and connectivity.")
    finally:
        if assistant is not None:
            assistant.llm_client.close()


def show_feedback(result: dict) -> None:
    conversation_id = result["conversation_id"]
    if conversation_id is None:
        return
    st.subheader("Was this answer helpful?")
    up, down, confirmation = st.columns([1, 1, 2])
    vote = None
    disabled = result["vote"] is not None
    if up.button(
        "Helpful",
        icon=":material/thumb_up:",
        key=f"feedback_up_{conversation_id}",
        width="stretch",
        disabled=disabled,
    ):
        vote = 1
    if down.button(
        "Not helpful",
        icon=":material/thumb_down:",
        key=f"feedback_down_{conversation_id}",
        width="stretch",
        disabled=disabled,
    ):
        vote = -1
    if vote is not None:
        try:
            save_feedback(conversation_id, "user", score=vote)
        except psycopg.Error:
            st.error("Feedback was not saved. Try again.")
        else:
            result["vote"] = vote
            st.rerun()
    if result["vote"] is not None:
        confirmation.caption("Feedback saved")


def show_assistant() -> None:
    st.title("Stack Overflow Assistant", width="stretch")
    with st.form("question_form", border=False):
        question = st.text_area(
            "Question",
            value=st.session_state.get("result", {}).get("question", ""),
            height=170,
            key="question",
            placeholder="How do I remove duplicates from a Python list while preserving order?",
        )
        submitted = st.form_submit_button("Ask", icon=":material/search:", type="primary")
    if submitted:
        if not question or not question.strip():
            st.warning("Enter a question before asking.")
        else:
            ask_question(question.strip())

    result = st.session_state.get("result")
    if not result:
        st.markdown('<div class="empty-answer">No answer yet.</div>', unsafe_allow_html=True)
        return
    st.divider()
    st.subheader("Answer")
    st.markdown(
        f'<div class="result-question">{escape(result["question"])}</div>', unsafe_allow_html=True
    )
    with st.container(key="answer_content"):
        st.markdown(result["answer"])
    show_call_metrics(result["record"])
    if result["save_failed"]:
        st.warning("This answer could not be saved. Check the database connection.")
    if result["judge_failed"]:
        st.warning(
            "Relevance evaluation is unavailable for this answer. You can still give feedback."
        )
    if result["relevance"]:
        verdicts = {
            "RELEVANT": ("Relevant", "relevant"),
            "PARTLY_RELEVANT": ("Partly relevant", "partly"),
            "NON_RELEVANT": ("Not relevant", "none"),
        }
        label, tone = verdicts[result["relevance"]]
        st.subheader("Judge evaluation")
        st.markdown(f'<span class="verdict verdict-{tone}">{label}</span>', unsafe_allow_html=True)
        st.write(result["explanation"])
    st.divider()
    show_feedback(result)


def show_records(records: list[LLMCallRecord]) -> None:
    for record in records:
        question = record.question or record.prompt
        title = question[:100] + ("…" if len(question) > 100 else "")
        with st.expander(title):
            st.caption(record.timestamp.strftime("%d %b %Y, %H:%M"))
            st.write(question)
            st.markdown(record.answer)
            duration, cost, tokens = st.columns(3)
            duration.metric("Response time", f"{record.response_time:.2f}s")
            cost.metric("Cost", format_cost(record.cost))
            tokens.metric(
                "Total tokens", record.total_tokens if record.total_tokens is not None else "—"
            )
            st.caption(f"Model: {record.model}")


def show_dashboard() -> None:
    title, refresh = st.columns([4, 1])
    title.title("Dashboard")
    refresh.button("Refresh", icon=":material/refresh:", width="stretch", key="refresh_dashboard")
    try:
        stats = get_stats()
        relevance = get_relevance_stats()
        thumbs_up, thumbs_down = get_user_feedback_stats()
        records = get_conversations(limit=100)
    except psycopg.Error:
        st.error("Dashboard data could not be loaded. Check the database connection and refresh.")
        return
    columns = st.columns(4)
    columns[0].metric("Conversations", stats.total)
    columns[1].metric("Avg response time", f"{stats.avg_response_time:.2f}s")
    columns[2].metric("Total cost", format_cost(stats.total_cost))
    columns[3].metric("Avg tokens", f"{stats.avg_tokens:.0f}")
    show_cost_configuration()
    st.divider()

    judge, feedback = st.columns([3, 2], gap="large")
    with judge:
        st.subheader("Judge relevance")
        if relevance:
            labels = {
                "RELEVANT": "Relevant",
                "PARTLY_RELEVANT": "Partly relevant",
                "NON_RELEVANT": "Not relevant",
            }
            data = pd.DataFrame(
                [
                    {
                        "Relevance": labels.get(label, label or "Unclassified"),
                        "Conversations": count,
                    }
                    for label, count in relevance.items()
                ]
            )
            st.bar_chart(
                data,
                x="Relevance",
                y="Conversations",
                color="#7562a8",
                height=230,
            )
        else:
            st.info("No judge evaluations recorded.")
    with feedback:
        st.subheader("User feedback")
        positive, negative = st.columns(2)
        positive.metric("Helpful", int(thumbs_up or 0))
        negative.metric("Not helpful", int(thumbs_down or 0))

    if records:
        df = pd.DataFrame([asdict(record) for record in records]).sort_values("timestamp")
        cost, response = st.columns(2, gap="large")
        with cost:
            st.subheader("Cost over time")
            st.line_chart(df, x="timestamp", y="cost", color="#176b53", height=230)
        with response:
            st.subheader("Response time over time")
            st.line_chart(df, x="timestamp", y="response_time", color="#b07134", height=230)
        st.caption("Latest 100 conversations. Summary metrics include all conversations.")
        st.divider()
        st.subheader("Recent conversations")
        show_records(records[:20])
    else:
        st.info("No conversations recorded.")


def show_history() -> None:
    title, refresh = st.columns([4, 1])
    title.title("Conversations")
    refresh.button("Refresh", icon=":material/refresh:", width="stretch", key="refresh_history")
    try:
        records = get_conversations(limit=100)
    except psycopg.Error:
        st.error("Conversations could not be loaded. Check the database connection and refresh.")
        return
    search, model = st.columns([3, 1])
    query = search.text_input(
        "Search conversations", icon=":material/search:", key="history_search"
    )
    selected_model = model.selectbox(
        "Model",
        ["All models", *sorted({record.model for record in records})],
        key="history_model",
    )
    filtered = [
        record
        for record in records
        if (selected_model == "All models" or record.model == selected_model)
        and (
            not query
            or query.casefold() in f"{record.question} {record.answer} {record.prompt}".casefold()
        )
    ]
    st.caption(f"{len(filtered)} of {len(records)} recent conversations")
    if filtered:
        show_records(filtered)
    elif records:
        st.info("No conversations match these filters.")
    else:
        st.info("No conversations recorded.")


def main() -> None:
    """Render the application and dispatch the selected view."""
    st.set_page_config(
        page_title="Stack Overflow Assistant",
        page_icon=":material/forum:",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(
            '<div class="brand"><strong>Stack Overflow</strong>'
            "<span>Assistant workspace</span></div>",
            unsafe_allow_html=True,
        )
        view = st.radio(
            "Navigation",
            ["Assistant", "Dashboard", "Conversations"],
            key="view",
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Model")
        st.write(get_settings().model)
        st.page_link(
            "https://stackoverflow.com", label="Stack Overflow", icon=":material/open_in_new:"
        )

    if view == "Assistant":
        show_assistant()
    elif view == "Dashboard":
        show_dashboard()
    else:
        show_history()
