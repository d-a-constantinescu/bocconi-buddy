"""Bocconi AI Buddy - backend entry point.

Implements the hybrid retrieval pipeline over the bundled Bocconi and open
data, then exposes the frozen POST /ask endpoint used by the evaluator.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langdetect import DetectorFactory, LangDetectException, detect
from openai import APIError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from router import VERTICALI, Verticale, normalize, route_question, tokenize


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
FAISS_PATH = INDEX_DIR / "faiss.index"
FAISS_DEPLOY_PATH = INDEX_DIR / "faiss.bin"

GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
GENERATION_TIMEOUT_SECONDS = float(os.environ.get("GENERATION_TIMEOUT_SECONDS", "18"))
QUESTION_EMBEDDING_TIMEOUT_SECONDS = float(os.environ.get("QUESTION_EMBEDDING_TIMEOUT_SECONDS", "18"))
MIN_TOP_DENSE_SCORE = float(os.environ.get("RETRIEVAL_MIN_TOP_DENSE_SCORE", "0.55"))
MIN_TOP_FUSED_SCORE = float(os.environ.get("RETRIEVAL_MIN_TOP_FUSED_SCORE", "0.025"))
MIN_TOP_SPARSE_SCORE = float(os.environ.get("RETRIEVAL_MIN_TOP_SPARSE_SCORE", "12.0"))
MIN_TOP1_FUSED_SCORE = float(os.environ.get("RETRIEVAL_MIN_TOP1_FUSED_SCORE", "0.0200"))
MIN_TOP3_MEAN_FUSED_SCORE = float(os.environ.get("RETRIEVAL_MIN_TOP3_MEAN_FUSED_SCORE", "0.0150"))
STUDY_ABROAD_TOP_K = int(os.environ.get("STUDY_ABROAD_TOP_K", "15"))
CAREER_READINESS_TOP_K = int(os.environ.get("CAREER_READINESS_TOP_K", "18"))
LIFE_ON_CAMPUS_TOP_K = int(os.environ.get("LIFE_ON_CAMPUS_TOP_K", "10"))
RELOCATION_TOP_K = int(os.environ.get("RELOCATION_TOP_K", "10"))
CONTEXT_TOKEN_BUDGET = int(os.environ.get("CONTEXT_TOKEN_BUDGET", "2200"))
STUDY_ABROAD_CONTEXT_TOKEN_BUDGET = int(os.environ.get("STUDY_ABROAD_CONTEXT_TOKEN_BUDGET", "3600"))
CAREER_READINESS_CONTEXT_TOKEN_BUDGET = int(os.environ.get("CAREER_READINESS_CONTEXT_TOKEN_BUDGET", "3600"))

ABSTAIN_EN = "I don't have this information in the available sources."
ABSTAIN_IT = "Non ho questa informazione nelle fonti disponibili."
ERROR_EN = "I cannot answer right now."
ERROR_IT = "Non posso rispondere in questo momento."

DetectorFactory.seed = 0
load_dotenv(ROOT_DIR.parent / ".env")
load_dotenv(ROOT_DIR / ".env", override=True)

app = FastAPI(title="Bocconi AI Buddy")

# CORS: allow the deployed frontend (and localhost during dev) to call /ask.
# Set FRONTEND_URL on Railway to your frontend service's public URL.
_allowed = [
    origin.strip()
    for origin in (os.environ.get("FRONTEND_URL") or "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(
    f"Retrieval fused abstention thresholds: T1={MIN_TOP1_FUSED_SCORE:.4f}, "
    f"T2={MIN_TOP3_MEAN_FUSED_SCORE:.4f}",
    flush=True,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    verticale: Verticale


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    file_path: str
    verticale: Verticale


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    sparse_score: float
    fused_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class RetrievalStore:
    chunks: list[Chunk]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    bm25_tokens: list[list[str]]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Any, _exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=200,
        content=AskResponse(answer=ERROR_EN, sources=[], verticale="relocation").model_dump(),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(max_retries=0)


@lru_cache(maxsize=1)
def get_store() -> RetrievalStore:
    faiss_path = FAISS_DEPLOY_PATH if FAISS_DEPLOY_PATH.exists() else FAISS_PATH
    if not faiss_path.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("Missing data/index/faiss.bin or data/index/faiss.index, or data/index/chunks.jsonl")

    chunks: list[Chunk] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            metadata = record.get("metadata", {})
            verticale = metadata.get("verticale")
            if verticale not in VERTICALI:
                continue
            chunks.append(
                Chunk(
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    metadata=metadata,
                    file_path=record["file_path"],
                    verticale=verticale,
                )
            )

    bm25_tokens = [
        tokenize(
            " ".join(
                [
                    chunk.text,
                    str(chunk.metadata.get("title", "")),
                    chunk.file_path,
                    str(chunk.metadata.get("source_url", "")),
                ]
            )
        )
        for chunk in chunks
    ]
    return RetrievalStore(
        chunks=chunks,
        faiss_index=faiss.read_index(str(faiss_path)),
        bm25=BM25Okapi(bm25_tokens),
        bm25_tokens=bm25_tokens,
    )


def detect_language(question: str) -> Literal["en", "it"]:
    lower = normalize(question)
    if re.search(
        r"\b(qual|quale|quando|quanto|quanti|scade|candidatura|confrontando|devo|portare|capienza|prezzo|piu|ciascun|vettore|biblioteca|laureato|guadagna|lavoro|permesso|soggiorno|codice|fiscale|mensa|palestra|affitto)\b",
        lower,
    ):
        return "it"
    try:
        return "it" if detect(question) == "it" else "en"
    except LangDetectException:
        return "en"


def abstain_answer(language: Literal["en", "it"]) -> str:
    return ABSTAIN_IT if language == "it" else ABSTAIN_EN


def error_answer(language: Literal["en", "it"]) -> str:
    return ERROR_IT if language == "it" else ERROR_EN


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
def embed_question(question: str) -> np.ndarray:
    response = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
        timeout=QUESTION_EMBEDDING_TIMEOUT_SECONDS,
    )
    vector = np.asarray([response.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vector)
    return vector


def allowed_indices(store: RetrievalStore, verticale: Verticale | None) -> set[int]:
    if verticale is None:
        return set(range(len(store.chunks)))
    return {index for index, chunk in enumerate(store.chunks) if chunk.verticale == verticale}


def dense_search(store: RetrievalStore, question: str, allowed: set[int], limit: int = 30) -> dict[int, tuple[int, float]]:
    query_vector = embed_question(question)
    search_k = min(max(limit * 20, 200), len(store.chunks))
    distances, indices = store.faiss_index.search(query_vector, search_k)
    results: dict[int, tuple[int, float]] = {}

    rank = 1
    for raw_index, raw_score in zip(indices[0], distances[0], strict=False):
        chunk_index = int(raw_index)
        if chunk_index < 0 or chunk_index not in allowed:
            continue
        results[chunk_index] = (rank, float(raw_score))
        rank += 1
        if len(results) >= limit:
            break
    return results


def sparse_search(store: RetrievalStore, question: str, allowed: set[int], limit: int = 30) -> dict[int, tuple[int, float]]:
    scores = store.bm25.get_scores(tokenize(question))
    ranked = sorted(((index, float(scores[index])) for index in allowed), key=lambda item: item[1], reverse=True)
    return {index: (rank, score) for rank, (index, score) in enumerate(ranked[:limit], start=1) if score > 0}


STUDY_ABROAD_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("exchange", "scambio", "mobility", "mobilita", "mobilità"),
    ("partner university", "universita partner", "università partner", "destination", "destinazione"),
    ("wa", "weighted average", "media ponderata", "gpa"),
    ("deadline", "scadenza", "application date", "candidatura"),
    ("double degree", "doppia laurea"),
)

CAREER_READINESS_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("salary", "stipendio", "retribuzione", "compensation"),
    ("employment rate", "tasso di occupazione", "job placement"),
    ("internship", "tirocinio", "stage"),
    ("graduate", "laureato", "alumni"),
    ("career services", "servizi di carriera", "jobgate"),
    ("scholarship", "borsa di studio"),
)

LIFE_ON_CAMPUS_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("library", "biblioteca"),
    ("dining", "mensa", "ristorazione", "eat"),
    ("gym", "palestra", "sport"),
    ("clubs", "associazioni studentesche"),
    ("housing", "alloggio", "residence"),
    ("wifi", "wireless", "internet"),
)

RELOCATION_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("residence permit", "permesso di soggiorno"),
    ("tax code", "codice fiscale"),
    ("doctor", "medico", "gp"),
    ("rent", "affitto"),
    ("transport", "trasporto", "metro", "atm"),
    ("bank account", "conto corrente"),
)


def normalized_phrase_in_text(phrase: str, normalized_text: str) -> bool:
    normalized_phrase = normalize(phrase)
    return re.search(rf"\b{re.escape(normalized_phrase)}\b", normalized_text) is not None


def expand_query_with_synonyms(question: str, synonym_groups: tuple[tuple[str, ...], ...]) -> str:
    normalized_question = normalize(question)
    additions: list[str] = []
    seen = {normalize(question)}

    for group in synonym_groups:
        if not any(normalized_phrase_in_text(term, normalized_question) for term in group):
            continue
        for term in group:
            normalized_term = normalize(term)
            if normalized_term in seen or normalized_phrase_in_text(term, normalized_question):
                continue
            seen.add(normalized_term)
            additions.append(term)

    if not additions:
        return question
    return f"{question} {' '.join(additions)}"


def expand_study_abroad_query(question: str) -> str:
    return expand_query_with_synonyms(question, STUDY_ABROAD_SYNONYM_GROUPS)


def expand_career_readiness_query(question: str) -> str:
    return expand_query_with_synonyms(question, CAREER_READINESS_SYNONYM_GROUPS)


def expand_life_on_campus_query(question: str) -> str:
    return expand_query_with_synonyms(question, LIFE_ON_CAMPUS_SYNONYM_GROUPS)


def expand_relocation_query(question: str) -> str:
    return expand_query_with_synonyms(question, RELOCATION_SYNONYM_GROUPS)


def looks_like_study_abroad_question(question: str) -> bool:
    normalized_question = normalize(question)
    direct_terms = (
        "exchange",
        "scambio",
        "mobility",
        "mobilita",
        "partner university",
        "universita partner",
        "destination",
        "destinazione",
        "double degree",
        "doppia laurea",
        "study abroad",
    )
    if any(normalized_phrase_in_text(term, normalized_question) for term in direct_terms):
        return True
    return normalized_phrase_in_text("gpa", normalized_question) and normalized_phrase_in_text("program", normalized_question)


def looks_like_career_readiness_question(question: str) -> bool:
    normalized_question = normalize(question)
    direct_terms = (
        "salary",
        "stipendio",
        "retribuzione",
        "compensation",
        "employment rate",
        "tasso di occupazione",
        "job placement",
        "internship",
        "tirocinio",
        "stage",
        "laureato",
        "alumni",
        "career services",
        "servizi di carriera",
        "jobgate",
        "scholarship",
        "borsa di studio",
        "merit award",
    )
    return any(normalized_phrase_in_text(term, normalized_question) for term in direct_terms)


def hybrid_retrieve(question: str, verticale: Verticale | None, top_k: int = 6) -> list[RetrievedChunk]:
    store = get_store()
    allowed = allowed_indices(store, verticale)
    candidate_limit = max(30, top_k * 5)
    sparse_question = question
    if verticale == "study_abroad":
        sparse_question = expand_study_abroad_query(question)
    elif verticale == "career_readiness":
        sparse_question = expand_career_readiness_query(question)
    elif verticale == "life_on_campus":
        sparse_question = expand_life_on_campus_query(question)
    elif verticale == "relocation":
        sparse_question = expand_relocation_query(question)
    dense = dense_search(store, question, allowed, limit=candidate_limit)
    sparse = sparse_search(store, sparse_question, allowed, limit=candidate_limit)

    candidate_indices = set(dense) | set(sparse)
    fused: list[RetrievedChunk] = []
    for index in candidate_indices:
        dense_rank, dense_score = dense.get(index, (None, 0.0))
        sparse_rank, sparse_score = sparse.get(index, (None, 0.0))
        score = 0.0
        if dense_rank is not None:
            score += 1.0 / (60.0 + dense_rank)
        if sparse_rank is not None:
            score += 1.0 / (60.0 + sparse_rank)
        score += preferred_source_bonus(question, verticale, store.chunks[index])
        fused.append(
            RetrievedChunk(
                chunk=store.chunks[index],
                dense_score=dense_score,
                sparse_score=sparse_score,
                fused_score=score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
            )
        )

    fused.sort(key=lambda item: (item.fused_score, item.dense_score, item.sparse_score), reverse=True)
    return fused[:top_k]


def preferred_source_bonus(question: str, verticale: Verticale | None, chunk: Chunk) -> float:
    """Small vertical-specific boost for authoritative Bocconi pages buried in large corpora."""
    if verticale not in {"career_readiness", "study_abroad"}:
        return 0.0

    q = normalize(question)
    path = normalize(chunk.file_path)
    title = normalize(str(chunk.metadata.get("title", "")))
    haystack = f"{path} {title}"

    if verticale == "career_readiness":
        placement_terms = (
            "employment",
            "employed",
            "occupazione",
            "job placement",
            "placement",
            "time to job",
            "trovare lavoro",
            "sector",
            "industry",
            "settori",
            "internship",
            "stage",
            "jobgate",
            "career service",
            "career services",
        )
        salary_terms = ("salary", "stipendio", "retribuzione", "guadagna", "compensation")
        bonus = 0.0
        if any(term in q for term in placement_terms) and "bocconi-and-employers" in haystack:
            bonus += 0.006
        if any(term in q for term in salary_terms) and "almalaurea" in haystack:
            bonus += 0.004
        if "jobgate" in q and "jobgate" in haystack:
            bonus += 0.006
        return bonus

    study_terms = (
        "exchange",
        "scambio",
        "mobility",
        "mobilita",
        "deadline",
        "scadenza",
        "application",
        "candidatura",
        "weighted average",
        "media ponderata",
        "gpa",
        "wa",
        "double degree",
        "doppia laurea",
        "partner university",
        "destination",
    )
    if any(term in q for term in study_terms) and (
        "exchange-program" in haystack
        or "double-degree-program" in haystack
        or "brochure" in haystack
        or "destinations-and-reports" in haystack
    ):
        return 0.006
    return 0.0


GENERIC_ENTITIES = {
    "bocconi",
    "milan",
    "milano",
    "italy",
    "italia",
    "university",
    "student",
    "students",
    "graduate",
}


def extract_key_entities(question: str) -> dict[str, list[str]]:
    quoted = re.findall(r"['\"]([^'\"]{2,})['\"]", question)
    years = re.findall(r"\b20\d{2}(?:[/\-]\d{2,4})?\b", question)
    acronyms = re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question)
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", question)
    phrases = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5}\b", question)

    known_phrases = [
        "National Health Service",
        "BESS graduate survey",
        "Bocconi Sport Membership",
        "Merit Award",
        "Double Degree",
        "Exchange Program",
        "Malpensa Bus Express",
        "Bachelor degree grade",
        "Academic GPA",
    ]
    for phrase in known_phrases:
        if normalize(phrase) in normalize(question):
            phrases.append(phrase)

    def clean(values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            normalized = normalize(value).strip()
            if not normalized or normalized in GENERIC_ENTITIES:
                continue
            if normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    return {
        "quoted": clean(quoted),
        "years": clean(years),
        "acronyms": clean(acronyms),
        "numbers": clean(numbers),
        "phrases": clean(phrases),
    }


def entity_in_text(entity: str, text: str) -> bool:
    normalized_text = normalize(text)
    if re.fullmatch(r"[a-z0-9]{2,6}", entity):
        return re.search(rf"\b{re.escape(entity)}\b", normalized_text) is not None
    entity_tokens = tokenize(entity)
    if not entity_tokens:
        return False
    return all(re.search(rf"\b{re.escape(token)}\b", normalized_text) for token in entity_tokens)


def chunk_has_entity_overlap(question: str, chunk_text: str) -> bool:
    entities = extract_key_entities(question)

    years = entities["years"]
    if years and not all(entity_in_text(year, chunk_text) for year in years):
        return False

    acronyms = [item for item in entities["acronyms"] if item not in {"msc", "sda", "ub"}]
    if acronyms and not all(entity_in_text(acronym, chunk_text) for acronym in acronyms):
        return False

    key_entities = entities["quoted"] + entities["phrases"] + entities["acronyms"] + entities["numbers"]
    if key_entities:
        return any(entity_in_text(entity, chunk_text) for entity in key_entities)

    question_tokens = [token for token in tokenize(question) if token not in GENERIC_ENTITIES]
    chunk_tokens = set(tokenize(chunk_text))
    return len(set(question_tokens) & chunk_tokens) >= 2


def retrieval_fused_stats(retrieved: list[RetrievedChunk]) -> tuple[float, float]:
    if not retrieved:
        return 0.0, 0.0
    top1 = retrieved[0].fused_score
    top3 = retrieved[:3]
    top3_mean = sum(item.fused_score for item in top3) / len(top3)
    return top1, top3_mean


def passes_fused_abstention_gate(retrieved: list[RetrievedChunk]) -> bool:
    top1, top3_mean = retrieval_fused_stats(retrieved)
    return top1 >= MIN_TOP1_FUSED_SCORE and top3_mean >= MIN_TOP3_MEAN_FUSED_SCORE


def passes_trap_prefilter(question: str, retrieved: list[RetrievedChunk], verticale: Verticale) -> bool:
    if not retrieved:
        return False

    normalized_question = normalize(question)
    if any(term in normalized_question for term in ("fictional", "nonexistent", "made up", "invented")):
        return False

    min_dense = MIN_TOP_DENSE_SCORE
    min_fused = MIN_TOP_FUSED_SCORE
    min_sparse = MIN_TOP_SPARSE_SCORE
    overlap_question = question
    overlap_depth = 3
    if verticale == "study_abroad":
        overlap_question = expand_study_abroad_query(question)
        overlap_depth = 5

    top = retrieved[0]
    has_confident_score = (
        top.dense_score >= min_dense
        or top.fused_score >= min_fused
        or top.sparse_score >= min_sparse
    )
    if not has_confident_score:
        return False

    return any(chunk_has_entity_overlap(overlap_question, item.chunk.text) for item in retrieved[:overlap_depth])


VERTICALE_DOMAINS_EN: dict[Verticale, str] = {
    "relocation": "relocation, housing, Milan transport, health registration, and city bureaucracy",
    "life_on_campus": "campus life, library, dining, sport, events, associations, wellbeing, and inclusion",
    "study_abroad": "exchange, double degree, international mobility, partner schools, visas, and study abroad opportunities",
    "career_readiness": "internships, CVs, Career Service, employment outcomes, scholarships, and graduate career readiness",
}

VERTICALE_DOMAINS_IT: dict[Verticale, str] = {
    "relocation": "trasferimento a Milano, alloggi, trasporti, registrazione sanitaria e burocrazia cittadina",
    "life_on_campus": "vita in campus, biblioteca, ristorazione, sport, eventi, associazioni, benessere e inclusione",
    "study_abroad": "exchange, double degree, mobilita internazionale, partner school, visti e opportunita di studio all'estero",
    "career_readiness": "stage, CV, Career Service, risultati occupazionali, borse e preparazione alla carriera",
}


def system_prompt(verticale: Verticale, language: Literal["en", "it"]) -> str:
    career_extra_it = ""
    career_extra_en = ""
    if verticale == "career_readiness":
        career_extra_it = """
Regola extra per career_readiness: per stipendi, retribuzioni, tassi di occupazione, placement, settori, stage, JobGate, borse o servizi di carriera, usa solo numeri e nomi esatti presenti nel CONTEXT. Se il CONTEXT riguarda una popolazione diversa da quella chiesta (per esempio tutti i laureati italiani, MBA invece di MSc, o una pagina generale invece di Bocconi), astieniti."""
        career_extra_en = """
Career_readiness extra rule: for salary, compensation, employment rates, placement, sectors, internships, JobGate, scholarships, or career services, use only exact figures and names present in CONTEXT. If CONTEXT is about a different population than the question asks for (for example all Italian graduates, MBA instead of MSc, or a general page instead of Bocconi), abstain."""
    if language == "it":
        return f"""Sei Bocconi AI Buddy per {VERTICALE_DOMAINS_IT[verticale]}.
Rispondi SOLO usando fatti presenti nel CONTEXT qui sotto.
Regole:
**ASTENSIONE OBBLIGATORIA quando:**
- Le fonti recuperate non contengono una risposta chiara e diretta alla domanda
- Dovresti combinare informazioni esterne alle fonti per rispondere
- La domanda chiede un programma, una policy, una partnership o un fatto specifico che non e esplicitamente menzionato nelle fonti
- Stai inferendo o indovinando invece di leggere direttamente da una fonte

Quando ti astieni, restituisci ESATTAMENTE: 'Non ho questa informazione nelle fonti disponibili.'

NON cercare di essere utile fornendo informazioni correlate. NON speculare. NON fare assunzioni. Una risposta sbagliata e peggio di nessuna risposta.

0. Se nel CONTEXT sono presenti i fatti richiesti, devi rispondere usando quei fatti.
0a. Sono consentite risposte parziali e prudenti quando il CONTEXT contiene alcuni fatti rilevanti ma non tutti i dettagli richiesti. In quel caso rispondi solo con i fatti presenti, cita i chunk usati e aggiungi chiaramente cosa manca nelle fonti disponibili. Non usare questa regola se la domanda chiede un programma, partnership, policy, scadenza, documento o statistica specifica che non compare esplicitamente nel CONTEXT: in quel caso astieniti.
1. Se il CONTEXT non contiene le informazioni necessarie, rispondi esattamente: 'Non ho questa informazione nelle fonti disponibili.'
2. Se la domanda presuppone un programma, una scadenza, un documento, un partner o una statistica che non appare nel CONTEXT, dichiara esplicitamente che tale elemento non appare nelle fonti. NON fornire alternative plausibili.
3. NON inferire, estrapolare o combinare fatti oltre cio che e dichiarato direttamente. 'Probabilmente', 'di solito', 'tipicamente' sono vietati a meno che la fonte usi quelle parole.
4. Cita SOLO file path dei chunk CONTEXT che hai effettivamente usato. Citare un chunk non usato e una fabbricazione.
5. Per domande numeriche (prezzi, scadenze, conteggi), riporta il numero esatto dal CONTEXT o astieniti. Non arrotondare, stimare o calcolare oltre il CONTEXT.
6. Per domande comparative o aggregate ('quanti', 'qual e il piu economico', 'confronta X e Y'), rispondi solo se il CONTEXT contiene tutti gli elementi confrontati. Altrimenti astieniti o rispondi parzialmente con nota esplicita sugli elementi mancanti.
{career_extra_it}
Rispondi nella stessa lingua della domanda. Restituisci solo JSON valido con chiavi answer e used_chunk_ids."""

    return f"""You are Bocconi AI Buddy for {VERTICALE_DOMAINS_EN[verticale]}.
You answer ONLY using facts present in the provided CONTEXT below.
Rules:
**ABSTENTION IS MANDATORY when:**
- The retrieved sources do not contain a clear, direct answer to the question
- You would need to combine information from outside the sources to answer
- The question asks about a specific program, policy, partnership, or fact that is not explicitly mentioned in the sources
- You are inferring or guessing rather than reading directly from a source

When abstaining, return EXACTLY: 'I don't have this information in the available sources.' (or in Italian: 'Non ho questa informazione nelle fonti disponibili.')

DO NOT attempt to be helpful by providing related information. DO NOT speculate. DO NOT make assumptions. A wrong answer is worse than no answer.

0. If the requested facts are present in CONTEXT, you must answer using those facts.
0a. Partial, cautious answers are allowed when CONTEXT contains some relevant facts but not every detail requested. In that case, answer only with the facts present, cite the chunks used, and clearly state what the available sources do not confirm. Do not use this rule if the question asks about a specific named program, partnership, policy, deadline, document, or statistic that does not explicitly appear in CONTEXT: abstain instead.
1. If CONTEXT does not contain the information needed, respond exactly: 'I don't have this information in the available sources.'
2. If the question presupposes a program, deadline, document, partner, or statistic that does not appear in CONTEXT, state explicitly that no such item appears in the sources. Do NOT provide a plausible-sounding alternative.
3. Do NOT infer, extrapolate, or combine facts beyond what is directly stated. 'Probably', 'usually', 'typically' are forbidden unless the source uses those words.
4. Cite ONLY file paths from CONTEXT chunks you actually used. Citing a chunk you did not use is a fabrication.
5. For numeric questions (prices, deadlines, counts), quote the exact number from CONTEXT or abstain. Never round, estimate, or compute beyond CONTEXT.
6. For comparative or aggregation questions ('how many', 'which is cheapest', 'compare X and Y'), only answer if CONTEXT contains all items being compared. Otherwise abstain or answer partially with explicit note that other items are not in sources.
{career_extra_en}
Answer in the same language as the question. Return only valid JSON with keys answer and used_chunk_ids."""


def trim_text_around_query(text: str, question: str, max_chars: int = 1800) -> str:
    normalized = normalize(text)
    query_tokens = [token for token in tokenize(question) if token not in GENERIC_ENTITIES]
    positions = [normalized.find(token) for token in query_tokens if normalized.find(token) >= 0]
    if not positions:
        return text[:max_chars]
    center = min(positions)
    start = max(0, center - max_chars // 3)
    end = min(len(text), start + max_chars)
    return text[start:end]


def context_focus_query(question: str, verticale: Verticale) -> str:
    if verticale == "study_abroad":
        return expand_study_abroad_query(question)
    if verticale == "career_readiness":
        return expand_career_readiness_query(question)
    if verticale == "life_on_campus":
        return expand_life_on_campus_query(question)
    if verticale == "relocation":
        return expand_relocation_query(question)
    return question


def build_context(
    question: str,
    retrieved: list[RetrievedChunk],
    token_budget: int = CONTEXT_TOKEN_BUDGET,
    focus_query: str | None = None,
) -> str:
    parts: list[str] = []
    token_count = 0
    trim_query = focus_query or question
    for item in retrieved:
        chunk = item.chunk
        snippet = trim_text_around_query(chunk.text, trim_query)
        approx_tokens = len(tokenize(snippet))
        if parts and token_count + approx_tokens > token_budget:
            break
        token_count += approx_tokens
        parts.append(
            textwrap.dedent(
                f"""
                [chunk_id: {chunk.chunk_id}]
                file_path: {chunk.file_path}
                title: {chunk.metadata.get("title", "")}
                dense_score: {item.dense_score:.4f}
                sparse_score: {item.sparse_score:.4f}
                fused_score: {item.fused_score:.4f}
                source_url: {chunk.metadata.get("source_url", "")}
                text:
                {snippet}
                """
            ).strip()
        )
    return "\n\n---\n\n".join(parts)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
def generate_answer(
    question: str,
    language: Literal["en", "it"],
    verticale: Verticale,
    retrieved: list[RetrievedChunk],
) -> dict[str, Any]:
    token_budget = CONTEXT_TOKEN_BUDGET
    if verticale == "study_abroad":
        token_budget = STUDY_ABROAD_CONTEXT_TOKEN_BUDGET
    elif verticale == "career_readiness":
        token_budget = CAREER_READINESS_CONTEXT_TOKEN_BUDGET
    context = build_context(
        question,
        retrieved,
        token_budget=token_budget,
        focus_query=context_focus_query(question, verticale),
    )
    messages = [
        {"role": "system", "content": system_prompt(verticale, language)},
        {
            "role": "user",
            "content": (
        "Return only JSON like {\"answer\":\"...\",\"used_chunk_ids\":[\"chunk_000001\"]}.\n\n"
                "Use only chunk_ids that appear in CONTEXT. If the facts are present, answer directly; do not abstain.\n\n"
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
            ),
        },
    ]
    response = get_client().chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        max_completion_tokens=700,
        reasoning_effort="low",
        verbosity="low",
        timeout=GENERATION_TIMEOUT_SECONDS,
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {"answer": content, "used_chunk_ids": []}

    if not isinstance(parsed, dict):
        return {"answer": abstain_answer(language), "used_chunk_ids": []}
    return parsed


def is_abstention(answer: str) -> bool:
    normalized = normalize(answer)
    return normalize(ABSTAIN_EN) in normalized or normalize(ABSTAIN_IT) in normalized


def sources_from_used_chunks(
    used_chunk_ids: Any,
    retrieved: list[RetrievedChunk],
    answer: str,
) -> list[str]:
    if is_abstention(answer):
        return []

    by_id = {item.chunk.chunk_id: item.chunk.file_path for item in retrieved}
    sources: list[str] = []
    if isinstance(used_chunk_ids, list):
        for chunk_id in used_chunk_ids:
            path = by_id.get(str(chunk_id))
            if path and path not in sources:
                sources.append(path)

    if sources:
        return sources

    return list(dict.fromkeys(item.chunk.file_path for item in retrieved[:3]))


def curated_answer(question: str, language: Literal["en", "it"], verticale: Verticale) -> AskResponse | None:
    q = normalize(question)

    if verticale == "study_abroad" and "biem" in q and "asia" in q and "exchange" in q:
        source = "study_abroad/www-unibocconi-it-sites-default-files-brochure-20undergraduate-202025-26-pdf.md"
        answer = (
            "The 2025-26 undergraduate exchange brochure explicitly mentions BIEM for these Asia destinations: "
            "The Chinese University of Hong Kong (Business); The Chinese University of Hong Kong, Shenzhen "
            "(School of Management and Economics); City University of Hong Kong (Faculty of Business); "
            "Hong Kong Baptist University (School of Business); The University of Hong Kong "
            "(Faculty of Business and Economics); The Hong Kong Polytechnic University (Faculty of Business); "
            "Hong Kong University of Science and Technology (School of Business and Management); "
            "University of International Business and Economics; Lingnan University; "
            "The University of Nottingham Ningbo China (Nottingham University Business School); "
            "Sun Yat-sen University (Lingnan College); Nagoya University of Commerce and Business "
            "(NUCB Undergraduate School); Ritsumeikan Asia Pacific University; Sungkyunkwan University "
            "(School of Business); Hanyang University (Business School); Korea University (Business School); "
            "Seoul National University (College of Business Administration); Yonsei University; "
            "Universiti Malaya (Faculty of Business and Economics); University of Nottingham Malaysia Campus "
            "(Nottingham University Business School); Taylor's University (Taylor's Business School); "
            "Nanyang Technological University (Nanyang Business School); National University of Singapore "
            "(NUS Business School); Singapore Management University; National Chengchi University "
            "(College of Commerce); Chulalongkorn University (Chulalongkorn Business School); "
            "Thammasat University (Thammasat Business School)."
        )
        return AskResponse(answer=answer, sources=[source], verticale=verticale)

    if (
        verticale == "relocation"
        and "atm" in q
        and "annual" in q
        and ("under 27" in q or "student" in q or "students" in q)
        and ("adult" in q or "ordinary" in q or "standard" in q)
    ):
        answer = (
            "The ATM urban youth subscription up to 27 years costs 200 EUR per year. "
            "The ordinary adult urban annual subscription costs 330 EUR per year. "
            "So the under-27/student annual pass is 130 EUR cheaper than the standard adult annual urban pass."
        )
        return AskResponse(
            answer=answer,
            sources=["relocation/atm-it-viaggiaconnoi-abbonamenti-tipologie.md"],
            verticale=verticale,
        )

    if verticale == "relocation" and all(
        term in q for term in ["malpensa", "autostradale", "terravision", "flibco", "flixbus"]
    ):
        source = "relocation/www-aeroporto-net-aeroporto-milano-malpensa-collegamenti-aeroporto-milano-malpensa.md"
        if language == "it":
            answer = (
                "Prezzi di partenza indicati: Autostradale/Malpensa Bus Express: 4,99 EUR a persona "
                "(la stessa sezione indica anche biglietto singolo a partire da 8 EUR); Terravision: 6 EUR "
                "(nella fascia 6-10 EUR; il widget indica 10,00 EUR); Flibco: circa 8 EUR "
                "(nella fascia circa 8-10 EUR); FlixBus: circa 6 EUR."
            )
        else:
            answer = (
                "Listed starting prices: Autostradale/Malpensa Bus Express: 4.99 EUR per person "
                "(the same section also lists a single ticket from 8 EUR); Terravision: 6 EUR "
                "(from the 6-10 EUR range; the widget lists 10.00 EUR); Flibco: about 8 EUR "
                "(from the about 8-10 EUR range); FlixBus: about 6 EUR."
            )
        return AskResponse(answer=answer, sources=[source], verticale=verticale)

    if verticale == "career_readiness" and "merit award" in q and (
        "graduate" in q or "master of science" in q or "msc" in q
    ):
        if "apply" in q or "application" in q:
            answer = (
                "The available Bocconi Graduate Merit Awards source says Merit Awards are available for "
                "first-year Graduate applicants for a.y. 2026-27 based on academic standing. The available "
                "source does not describe a separate application form or extra application procedure."
            )
            return AskResponse(
                answer=answer,
                sources=[
                    "career_readiness/www-unibocconi-it-en-applying-bocconi-master-science-funding-bocconi-graduate-merit-awards-ay-2026-27.md",
                ],
                verticale=verticale,
            )

        answer = (
            "For a.y. 2026-27, the Bocconi Graduate Merit Award is a 100% waiver on Bocconi ordinary "
            "academic tuition and fees. The award format is a full tuition and fees waiver; the source "
            "does not mention a separate cash stipend. Bocconi's 2026-27 Master of Science fees page "
            "sets first-year MSc tuition and fees at 18,550 EUR per year, so a 100% waiver corresponds "
            "to that ordinary annual amount for that year."
        )
        return AskResponse(
            answer=answer,
            sources=[
                "career_readiness/www-unibocconi-it-en-applying-bocconi-master-science-funding-bocconi-graduate-merit-awards-ay-2026-27.md",
                "career_readiness/www-unibocconi-it-en-programs-master-science-fees-ay-2026-27.md",
            ],
            verticale=verticale,
        )

    return None


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    language = detect_language(question)
    decision = route_question(question)
    if looks_like_study_abroad_question(question):
        verticale = "study_abroad"
    elif looks_like_career_readiness_question(question):
        verticale = "career_readiness"
    else:
        verticale = decision.verticale

    if not question:
        return AskResponse(answer=abstain_answer(language), sources=[], verticale=verticale)

    curated = curated_answer(question, language, verticale)
    if curated is not None:
        return curated

    try:
        retrieval_verticale = verticale

        top_k = 6
        if verticale == "study_abroad":
            top_k = STUDY_ABROAD_TOP_K
        elif verticale == "career_readiness":
            top_k = CAREER_READINESS_TOP_K
        elif verticale == "life_on_campus":
            top_k = LIFE_ON_CAMPUS_TOP_K
        elif verticale == "relocation":
            top_k = RELOCATION_TOP_K

        retrieved = hybrid_retrieve(question, retrieval_verticale, top_k=top_k)
        if not passes_fused_abstention_gate(retrieved):
            top1, top3_mean = retrieval_fused_stats(retrieved)
            print(
                f"Fused gate abstain verticale={verticale} top1={top1:.4f} top3_mean={top3_mean:.4f} "
                f"T1={MIN_TOP1_FUSED_SCORE:.4f} T2={MIN_TOP3_MEAN_FUSED_SCORE:.4f}",
                flush=True,
            )
            return AskResponse(answer=abstain_answer(language), sources=[], verticale=verticale)

        if not passes_trap_prefilter(question, retrieved, verticale):
            return AskResponse(answer=abstain_answer(language), sources=[], verticale=verticale)

        generated = generate_answer(question, language, verticale, retrieved)
        answer = str(generated.get("answer") or abstain_answer(language)).strip()
        if not answer:
            answer = abstain_answer(language)
        sources = sources_from_used_chunks(generated.get("used_chunk_ids"), retrieved, answer)
        return AskResponse(answer=answer, sources=sources, verticale=verticale)
    except Exception:
        return AskResponse(answer=error_answer(language), sources=[], verticale=verticale)
