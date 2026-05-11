"""Run SAMPLE_QUESTIONS.md against a local or public /ask endpoint."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from router import route_question

SAMPLE_PATH = ROOT_DIR / "SAMPLE_QUESTIONS.md"

CATEGORIES = {
    1: "computational",
    2: "actionable",
    3: "computational",
    4: "informative",
    5: "actionable",
    6: "computational",
    7: "trap",
    8: "informative",
    9: "trap",
    10: "computational",
}

LANGUAGES = {3: "it", 4: "it"}


def sample_questions() -> list[str]:
    text = SAMPLE_PATH.read_text(encoding="utf-8")
    return re.findall(r"^\d+\.\s+\*\*(.*?)\*\*", text, flags=re.MULTILINE)


def post_question(base_url: str, question: str) -> tuple[dict[str, Any], int, int]:
    payload = json.dumps({"question": question}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter()
    try:
        with urlopen(request, timeout=35) as response:
            latency_ms = round((time.perf_counter() - started_at) * 1000)
            body = response.read().decode("utf-8")
            return json.loads(body), response.status, latency_ms
    except HTTPError as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        body = exc.read().decode("utf-8", errors="replace")
        return {"answer": body, "sources": [], "verticale": "error"}, exc.code, latency_ms
    except (TimeoutError, URLError) as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        return {"answer": str(exc), "sources": [], "verticale": "error"}, 0, latency_ms


def truncate(text: str, limit: int = 250) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def table_escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    rows: list[list[Any]] = []
    for index, question in enumerate(sample_questions(), start=1):
        predicted = route_question(question).verticale
        result, status, latency_ms = post_question(base_url, question)
        rows.append(
            [
                index,
                CATEGORIES.get(index, "unknown"),
                LANGUAGES.get(index, "en"),
                predicted,
                status,
                result.get("verticale", ""),
                truncate(result.get("answer", "")),
                ", ".join(result.get("sources", [])),
                latency_ms,
            ]
        )

    headers = [
        "#",
        "category",
        "language",
        "predicted_verticale",
        "status",
        "response_verticale",
        "answer (truncated 250 chars)",
        "sources",
        "latency_ms",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(table_escape(item) for item in row) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
