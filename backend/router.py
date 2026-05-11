"""Rule-based verticale routing for Bocconi AI Buddy.

The router is deliberately simple and testable: it builds one keyword
vocabulary per verticale from manifest metadata and markdown frontmatter,
then routes a question by weighted keyword overlap.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal


Verticale = Literal["relocation", "life_on_campus", "study_abroad", "career_readiness"]
VERTICALI: tuple[Verticale, ...] = ("relocation", "life_on_campus", "study_abroad", "career_readiness")

DATA_DIR = Path(__file__).resolve().parent / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+\-/]*")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

STOPWORDS = {
    "about",
    "access",
    "and",
    "are",
    "available",
    "bocconi",
    "com",
    "current",
    "del",
    "della",
    "delle",
    "does",
    "for",
    "from",
    "how",
    "https",
    "into",
    "italy",
    "it",
    "its",
    "milan",
    "milano",
    "must",
    "not",
    "per",
    "program",
    "programs",
    "service",
    "services",
    "student",
    "students",
    "the",
    "this",
    "under",
    "unibocconi",
    "universita",
    "university",
    "what",
    "when",
    "where",
    "which",
    "with",
    "www",
}

DOMAIN_SEEDS: dict[Verticale, tuple[str, ...]] = {
    "relocation": (
        "accommodation apartment atm bus city codice fiscale fiscal code flat health insurance housing linate malpensa "
        "metro national health service neighborhood permit rent residence ssn soggiorno subway tax code tram transit"
    ),
    "life_on_campus": (
        "association biblioteca campus canteen dining event food library membership opening hours room seats sport "
        "wellbeing wellness bocconi sport student association"
    ),
    "study_abroad": (
        "abroad academic recognition bachelor degree grade credits double degree exchange free mover gpa international "
        "mobility mit partner peking selection score visa"
    ),
    "career_readiness": (
        "almalaurea award bess career cv employment graduate survey internship job market merit placement scholarship "
        "stipend tuition waiver"
    ),
}

PHRASE_SEEDS: dict[Verticale, tuple[str, ...]] = {
    "relocation": ("atm transit", "malpensa", "national health service", "ssn", "residence permit"),
    "life_on_campus": ("bocconi sport", "dining areas", "library", "biblioteca"),
    "study_abroad": ("double degree", "exchange program", "selection score", "academic gpa"),
    "career_readiness": ("bess graduate survey", "merit award", "tuition waiver", "placement results"),
}


@dataclass(frozen=True)
class RouterDecision:
    verticale: Verticale
    search_verticale: Verticale | None
    scores: dict[Verticale, float]
    matched_keywords: dict[Verticale, list[str]]

    @property
    def search_all_verticali(self) -> bool:
        return self.search_verticale is None


def normalize(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith(("ing", "ers")):
        return token[:-3]
    if len(token) > 4 and token.endswith(("ed", "es")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(normalize(text)):
        token = stem(token.strip("-_/"))
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def parse_frontmatter_text(path: Path) -> str:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:3000]
    except OSError:
        return ""
    match = FRONTMATTER_RE.match(head)
    return match.group(1) if match else ""


def add_weighted_tokens(vocab: Counter[str], text: str, weight: int) -> None:
    for token in tokenize(text):
        vocab[token] += weight


def build_verticale_vocabs(data_dir: Path = DATA_DIR) -> dict[Verticale, Counter[str]]:
    manifest_path = data_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    vocabs: dict[Verticale, Counter[str]] = {verticale: Counter() for verticale in VERTICALI}
    for entry in manifest.get("files", []):
        verticale = entry.get("verticale")
        if verticale not in vocabs:
            continue
        vocab = vocabs[verticale]
        add_weighted_tokens(vocab, str(entry.get("title", "")), 4)
        add_weighted_tokens(vocab, str(entry.get("path", "")), 3)
        add_weighted_tokens(vocab, str(entry.get("source_url", "")), 1)

        file_path = data_dir / str(entry.get("path", ""))
        add_weighted_tokens(vocab, parse_frontmatter_text(file_path), 2)

    for verticale, seed_text in DOMAIN_SEEDS.items():
        add_weighted_tokens(vocabs[verticale], seed_text, 10)

    return vocabs


@lru_cache(maxsize=1)
def get_default_vocabs() -> dict[Verticale, Counter[str]]:
    return build_verticale_vocabs(DATA_DIR)


def score_question(
    question: str,
    vocabs: dict[Verticale, Counter[str]] | None = None,
) -> tuple[dict[Verticale, float], dict[Verticale, list[str]]]:
    vocabs = vocabs or get_default_vocabs()
    question_tokens = set(tokenize(question))
    normalized_question = normalize(question)
    scores: dict[Verticale, float] = {}
    matched: dict[Verticale, list[str]] = {}

    for verticale in VERTICALI:
        vocab = vocabs[verticale]
        score = 0.0
        hits: list[str] = []
        for token in sorted(question_tokens):
            if token in vocab:
                score += 1.0 + math.log1p(vocab[token])
                hits.append(token)

        for phrase in PHRASE_SEEDS[verticale]:
            if normalize(phrase) in normalized_question:
                score += 12.0
                hits.append(phrase)

        scores[verticale] = round(score, 4)
        matched[verticale] = hits[:12]

    return scores, matched


def route_question(
    question: str,
    vocabs: dict[Verticale, Counter[str]] | None = None,
    min_score: float = 2.0,
    tie_margin: float = 1.0,
) -> RouterDecision:
    scores, matched = score_question(question, vocabs)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_verticale, best_score = ranked[0]
    second_score = ranked[1][1]

    search_verticale: Verticale | None = best_verticale
    if best_score < min_score or best_score - second_score <= tie_margin:
        search_verticale = None

    return RouterDecision(
        verticale=best_verticale,
        search_verticale=search_verticale,
        scores=scores,
        matched_keywords=matched,
    )
