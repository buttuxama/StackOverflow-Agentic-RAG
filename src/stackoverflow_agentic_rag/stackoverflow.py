"""Function-based Stack Overflow retrieval tools for the RAG agent."""

import math
import re
from copy import deepcopy
from html import unescape
from html.parser import HTMLParser
from time import monotonic
from urllib.parse import urlsplit

import requests

API_URL = "https://api.stackexchange.com/2.3"
CACHE_TTL = 60
CACHE_SIZE = 128
REQUEST_TIMEOUT = 20
SEARCH_STOP_WORDS = frozenset(
    "a an and are as at be by do does for from how i in is it of on or that the "
    "this to using what when with without".split()
)
_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_BACKOFF: dict[str, float] = {}


class StackOverflowError(RuntimeError):
    """A Stack Exchange request failed or returned malformed data."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.in_pre = False

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            self.in_pre = True
            self.parts.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.parts.append("`")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in {"p", "br", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.in_pre = False
            self.parts.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.parts.append("`")
        elif tag in {"p", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected HTML text")
    parser = _TextExtractor()
    parser.feed(value)
    return "".join(parser.parts).strip()


def _request(path: str, params: dict, *, session: requests.Session) -> list[dict]:
    params = {"site": "stackoverflow", "filter": "withbody", **params}
    key = (path, tuple(sorted(params.items())))
    now = monotonic()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < CACHE_TTL:
        return deepcopy(cached[1])
    # Backoff is shared across parameter variants of the same API method.
    method = "question_answers" if path.startswith("questions/") else path.split("/")[0]
    deadline = _BACKOFF.get(method, 0)
    if now < deadline:
        raise StackOverflowError(f"API cooldown: retry in {math.ceil(deadline - now)} seconds.")
    try:
        response = session.get(f"{API_URL}/{path}", params=params, timeout=REQUEST_TIMEOUT)
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Expected an API object")
        _BACKOFF[method] = monotonic() + max(0, int(data.get("backoff", 0)))
        if "error_id" in data:
            if type(data["error_id"]) is not int:
                raise ValueError("Invalid API error ID")
            raise StackOverflowError(f"Stack Exchange API error {data['error_id']}.")
        response.raise_for_status()
        items = data["items"]
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError("Expected API items")
    except requests.RequestException:
        raise StackOverflowError("Stack Overflow request failed; check connectivity.") from None
    except KeyError, TypeError, ValueError:
        raise StackOverflowError("Stack Overflow returned malformed data.") from None
    if key not in _CACHE and len(_CACHE) >= CACHE_SIZE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (monotonic(), deepcopy(items))
    return items


def _integer(item: dict, key: str, *, positive: bool = False) -> int:
    value = item[key]
    if type(value) is not int or (positive and value <= 0):
        raise ValueError(f"Invalid {key}")
    return value


def search_stackoverflow(
    query: str,
    *,
    tags: tuple[str, ...] = (),
    title: str = "",
    limit: int = 12,
    session: requests.Session,
) -> list[dict]:
    """Find answered questions in API relevance order and expose keyword matches."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query cannot be empty")
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Search limit must be between 1 and 100")
    if not isinstance(title, str):
        raise ValueError("Title filter must be a string")
    if any(not isinstance(tag, str) or not tag.strip() or ";" in tag for tag in tags):
        raise ValueError("Pass individual, nonempty tag strings")
    params = {
        "q": query.strip(),
        "sort": "relevance",
        "order": "desc",
        "pagesize": limit,
        "answers": 1,
    }
    if title.strip():
        params["title"] = title.strip()
    if tags:
        params["tagged"] = ";".join(sorted({tag.strip().lower() for tag in tags}))
    items = _request("search/advanced", params, session=session)
    questions = {}
    try:
        for item in items:
            question_id = _integer(item, "question_id", positive=True)
            url = item["link"]
            if not isinstance(url, str):
                raise ValueError("Invalid question URL")
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.netloc != "stackoverflow.com"
                or parsed.path.split("/")[1:3] != ["questions", str(question_id)]
                or any(char.isspace() or char in '<>"\\' for char in url)
            ):
                raise ValueError("Invalid question URL")
            title = item["title"]
            item_tags = item["tags"]
            if not isinstance(title, str) or not isinstance(item_tags, list):
                raise ValueError("Invalid question metadata")
            if any(not isinstance(tag, str) for tag in item_tags):
                raise ValueError("Invalid question tags")
            questions[question_id] = {
                "question_id": question_id,
                "title": unescape(title),
                "url": url,
                "body": _text(item["body"]),
                "tags": item_tags,
                "score": _integer(item, "score"),
                "accepted_answer_id": item.get("accepted_answer_id"),
            }
    except KeyError, TypeError, ValueError:
        raise StackOverflowError("Stack Overflow returned an invalid question.") from None
    # Keywords are a relevance signal, not a semantic relevance guarantee. Keep
    # synonyms/alternative wording eligible; the agent checks actual constraints.
    keywords = set(re.findall(r"\w+(?:[.+#-]\w+)*[+#]*", query.casefold())) - SEARCH_STOP_WORDS
    for question in questions.values():
        title_words = set(re.findall(r"\w+(?:[.+#-]\w+)*[+#]*", question["title"].casefold()))
        body_words = set(re.findall(r"\w+(?:[.+#-]\w+)*[+#]*", question["body"].casefold()))
        question["matched_keywords"] = sorted(keywords & (title_words | body_words))
        question["title_matched_keywords"] = sorted(keywords & title_words)
    return list(questions.values())[:limit]


def get_answers(
    question: dict,
    *,
    limit: int = 2,
    session: requests.Session,
) -> list[dict]:
    """Fetch answers and prefer the accepted answer, then the highest vote scores."""
    question_id = _integer(question, "question_id", positive=True)
    if type(limit) is not int or not 1 <= limit <= 5:
        raise ValueError("Answer limit must be between 1 and 5")
    items = _request(
        f"questions/{question_id}/answers",
        {"sort": "votes", "order": "desc", "pagesize": 5},
        session=session,
    )
    accepted_id = question.get("accepted_answer_id")
    # An accepted answer may fall outside the first page ordered by votes.
    if accepted_id is not None:
        if type(accepted_id) is not int or accepted_id <= 0:
            raise StackOverflowError("Stack Overflow returned an invalid accepted answer ID.")
        if not any(item.get("answer_id") == accepted_id for item in items):
            items += _request(f"answers/{accepted_id}", {}, session=session)
    answers = {}
    try:
        for item in items:
            answer_id = _integer(item, "answer_id", positive=True)
            if _integer(item, "question_id", positive=True) != question_id:
                raise ValueError("Answer belongs to another question")
            owner = item.get("owner") or {}
            accepted = item["is_accepted"]
            if type(accepted) is not bool or not isinstance(owner, dict):
                raise ValueError("Invalid answer metadata")
            author = owner.get("display_name", "Unknown author")
            license_name = item.get("content_license", "")
            if not isinstance(author, str) or not isinstance(license_name, str):
                raise ValueError("Invalid attribution metadata")
            answers[answer_id] = {
                "answer_id": answer_id,
                "question_id": question_id,
                "url": f"https://stackoverflow.com/a/{answer_id}",
                "body": _text(item["body"]),
                "score": _integer(item, "score"),
                "is_accepted": accepted,
                "author": unescape(author),
                "license": license_name,
            }
    except KeyError, TypeError, ValueError:
        raise StackOverflowError("Stack Overflow returned an invalid answer.") from None
    return sorted(answers.values(), key=lambda a: (a["is_accepted"], a["score"]), reverse=True)[
        :limit
    ]
