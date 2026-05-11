"""Build the offline retrieval index for Bocconi AI Buddy.

This script is intentionally one-shot and cost-incurring: it embeds the
bundled markdown corpus with OpenAI, writes a FAISS index to data/index/,
and saves a chunks.jsonl sidecar that the runtime can load without ever
embedding the full corpus again.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = DATA_DIR / "manifest.json"
EMBEDDING_CACHE_DIR = INDEX_DIR / "embedding_batches"

VERTICALI = ("relocation", "life_on_campus", "study_abroad", "career_readiness")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_PRICE_PER_1M_TOKENS = float(os.environ.get("EMBEDDING_PRICE_PER_1M_TOKENS", "0.13"))
CHUNK_TARGET_TOKENS = int(os.environ.get("CHUNK_TARGET_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "80"))
MAX_EMBEDDING_TOKENS = int(os.environ.get("MAX_EMBEDDING_TOKENS", "7000"))
BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "100"))


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_token_count(text: str) -> int:
    """Fast rough token count good enough for markdown chunk sizing and cost logs."""
    return len(TOKEN_RE.findall(text))


def split_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    if not markdown.startswith("---"):
        return {}, markdown

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", markdown, flags=re.DOTALL)
    if not match:
        return {}, markdown

    raw = match.group(1)
    body = markdown[match.end() :]
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        return {}, body
    return loaded, body


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def markdown_blocks(text: str) -> list[str]:
    """Split markdown into blocks without breaking tables or list items."""
    lines = normalize_text(text).splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current
        if current:
            block = "\n".join(current).strip()
            if block:
                blocks.append(block)
            current = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            current.append(line)
            in_fence = not in_fence
            continue

        if in_fence:
            current.append(line)
            continue

        if not stripped:
            flush()
            continue

        if stripped.startswith("#"):
            flush()
            current.append(line)
            flush()
            continue

        current.append(line)

    flush()
    return blocks


def split_text_by_words(text: str, target_tokens: int) -> list[str]:
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = approx_token_count(word)
        if current and current_tokens + word_tokens > target_tokens:
            pieces.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(word)
        current_tokens += word_tokens
    if current:
        pieces.append(" ".join(current))
    return pieces


def split_table_block(block: str, target_tokens: int) -> list[str]:
    lines = block.splitlines()
    if len(lines) < 3:
        return [block]

    has_header_rule = "|" in lines[1] and re.search(r"-{2,}", lines[1])
    header = lines[:2] if has_header_rule else []
    rows = lines[2:] if has_header_rule else lines
    chunks: list[str] = []
    current = header.copy()
    current_tokens = approx_token_count("\n".join(current))

    for row in rows:
        row_tokens = approx_token_count(row)
        if len(current) > len(header) and current_tokens + row_tokens > target_tokens:
            chunks.append("\n".join(current).strip())
            current = header.copy()
            current_tokens = approx_token_count("\n".join(current))

        if row_tokens > target_tokens:
            if len(current) > len(header):
                chunks.append("\n".join(current).strip())
                current = header.copy()
                current_tokens = approx_token_count("\n".join(current))
            chunks.extend(split_text_by_words(row, target_tokens))
            continue

        current.append(row)
        current_tokens += row_tokens

    if len(current) > len(header):
        chunks.append("\n".join(current).strip())
    return chunks or [block]


def split_list_block(block: str, target_tokens: int) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for line in block.splitlines():
        if re.match(r"^\s*(-|\*|\d+\.)\s+", line) and current:
            items.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        items.append("\n".join(current).strip())

    chunks: list[str] = []
    current_items: list[str] = []
    current_tokens = 0
    for item in items:
        item_tokens = approx_token_count(item)
        if current_items and current_tokens + item_tokens > target_tokens:
            chunks.append("\n".join(current_items).strip())
            current_items = []
            current_tokens = 0

        if item_tokens > target_tokens:
            if current_items:
                chunks.append("\n".join(current_items).strip())
                current_items = []
                current_tokens = 0
            chunks.extend(split_text_by_words(item, target_tokens))
            continue

        current_items.append(item)
        current_tokens += item_tokens

    if current_items:
        chunks.append("\n".join(current_items).strip())
    return chunks or [block]


def split_large_block(block: str, target_tokens: int) -> list[str]:
    """Split unusually large markdown blocks without cutting table rows or list items."""
    lines = block.splitlines()
    is_table = len(lines) > 1 and sum(1 for line in lines if "|" in line) >= 2
    is_list = all(re.match(r"^\s*(-|\*|\d+\.)\s+", line) for line in lines if line.strip())
    if approx_token_count(block) <= target_tokens:
        return [block]
    if is_table:
        return split_table_block(block, target_tokens)
    if is_list:
        return split_list_block(block, target_tokens)

    sentences = re.split(r"(?<=[.!?])\s+", block)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = approx_token_count(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            pieces.append(" ".join(current).strip())
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        pieces.append(" ".join(current).strip())
    return pieces or [block]


def make_chunk_text(title: str, body: str) -> str:
    if title:
        return f"Title: {title}\n\n{body.strip()}"
    return body.strip()


def chunk_markdown(body: str, title: str) -> list[dict[str, Any]]:
    blocks: list[str] = []
    for block in markdown_blocks(body):
        blocks.extend(split_large_block(block, CHUNK_TARGET_TOKENS))

    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_tokens = 0

    def emit() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = "\n\n".join(current).strip()
        chunk_text = make_chunk_text(title, text)
        chunks.append({"text": chunk_text, "token_count": approx_token_count(chunk_text)})

        overlap: list[str] = []
        overlap_tokens = 0
        for block in reversed(current):
            block_tokens = approx_token_count(block)
            if overlap and overlap_tokens + block_tokens > CHUNK_OVERLAP_TOKENS:
                break
            overlap.insert(0, block)
            overlap_tokens += block_tokens
            if overlap_tokens >= CHUNK_OVERLAP_TOKENS:
                break
        current = overlap
        current_tokens = overlap_tokens

    for block in blocks:
        block_tokens = approx_token_count(block)
        if current and current_tokens + block_tokens > CHUNK_TARGET_TOKENS:
            emit()
        current.append(block)
        current_tokens += block_tokens

    if current:
        text = "\n\n".join(current).strip()
        chunk_text = make_chunk_text(title, text)
        if not chunks or chunks[-1]["text"] != chunk_text:
            chunks.append({"text": chunk_text, "token_count": approx_token_count(chunk_text)})

    return [chunk for chunk in chunks if chunk["text"].strip()]


def load_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {entry["path"]: entry for entry in manifest.get("files", [])}


def read_markdown_files() -> list[dict[str, Any]]:
    manifest = load_manifest()
    records: list[dict[str, Any]] = []

    for verticale in VERTICALI:
        folder = DATA_DIR / verticale
        if not folder.exists():
            raise FileNotFoundError(f"Missing verticale folder: {folder}")

        for path in sorted(folder.glob("*.md")):
            rel_path = path.relative_to(DATA_DIR).as_posix()
            raw = path.read_text(encoding="utf-8", errors="replace")
            frontmatter, body = split_frontmatter(raw)
            manifest_meta = manifest.get(rel_path, {})
            metadata = {
                **manifest_meta,
                **frontmatter,
                "verticale": frontmatter.get("verticale") or manifest_meta.get("verticale") or verticale,
                "language": frontmatter.get("language") or manifest_meta.get("language") or "unknown",
                "source_url": frontmatter.get("source_url") or manifest_meta.get("source_url") or "",
                "title": frontmatter.get("title") or manifest_meta.get("title") or path.stem,
                "file_path": rel_path,
            }

            for ordinal, chunk in enumerate(chunk_markdown(body, str(metadata["title"]))):
                records.append(
                    {
                        "chunk_id": f"chunk_{len(records):06d}",
                        "text": chunk["text"],
                        "token_count": chunk["token_count"],
                        "metadata": metadata,
                        "file_path": rel_path,
                        "chunk_ordinal": ordinal,
                    }
                )

    return records


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def write_chunks_jsonl(records: list[dict[str, Any]]) -> None:
    chunks_path = INDEX_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_stats(records: list[dict[str, Any]], embedding_dim: int, elapsed_seconds: float) -> None:
    by_verticale = Counter(record["metadata"]["verticale"] for record in records)
    token_count = sum(record["token_count"] for record in records)
    estimated_cost = token_count / 1_000_000 * EMBEDDING_PRICE_PER_1M_TOKENS
    stats = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": embedding_dim,
        "chunk_target_tokens": CHUNK_TARGET_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "total_chunks": len(records),
        "chunks_per_verticale": dict(sorted(by_verticale.items())),
        "estimated_total_tokens": token_count,
        "estimated_embedding_cost_usd": round(estimated_cost, 4),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "built_at_unix": int(time.time()),
    }
    (INDEX_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nFinal index stats")
    print("-----------------")
    for verticale in VERTICALI:
        print(f"{verticale}: {by_verticale.get(verticale, 0)} chunks")
    print(f"Total chunks: {len(records)}")
    print(f"Estimated total tokens embedded: {token_count:,}")
    print(f"Estimated embedding cost: ${estimated_cost:.4f}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Elapsed wall-clock time: {elapsed_seconds:.1f}s")


def validate_records(records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("No chunks were produced.")

    by_verticale = Counter(record["metadata"]["verticale"] for record in records)
    missing = [verticale for verticale in VERTICALI if by_verticale.get(verticale, 0) == 0]
    if missing:
        raise RuntimeError(f"These verticali produced zero chunks: {', '.join(missing)}")

    oversized = [
        (record["chunk_id"], record["file_path"], approx_token_count(record["text"]))
        for record in records
        if approx_token_count(record["text"]) > MAX_EMBEDDING_TOKENS
    ]
    if oversized:
        sample = "; ".join(f"{chunk_id} {path} tokens={tokens}" for chunk_id, path, tokens in oversized[:5])
        raise RuntimeError(f"{len(oversized)} chunks exceed the embedding safety limit. {sample}")


def batch_cache_paths(batch_number: int) -> tuple[Path, Path]:
    stem = f"batch_{batch_number:04d}"
    return EMBEDDING_CACHE_DIR / f"{stem}.npy", EMBEDDING_CACHE_DIR / f"{stem}.json"


def load_cached_batch(batch_number: int, expected_ids: list[str]) -> np.ndarray | None:
    vectors_path, ids_path = batch_cache_paths(batch_number)
    if not vectors_path.exists() or not ids_path.exists():
        return None
    try:
        saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))["chunk_ids"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    if saved_ids != expected_ids:
        return None
    return np.load(vectors_path)


def save_cached_batch(batch_number: int, chunk_ids: list[str], vectors: list[list[float]]) -> np.ndarray:
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    array = np.asarray(vectors, dtype="float32")
    vectors_path, ids_path = batch_cache_paths(batch_number)
    np.save(vectors_path, array)
    ids_path.write_text(json.dumps({"chunk_ids": chunk_ids}), encoding="utf-8")
    return array


def main() -> int:
    load_dotenv(ROOT_DIR.parent / ".env")
    load_dotenv(ROOT_DIR / ".env", override=True)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Add it to .env before running.", file=sys.stderr)
        return 1

    started_at = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if (INDEX_DIR / "faiss.index").exists() and os.environ.get("FORCE_REBUILD") != "1":
        print("ERROR: data/index/faiss.index already exists. Set FORCE_REBUILD=1 to rebuild.", file=sys.stderr)
        return 1

    print(f"Reading markdown corpus from {DATA_DIR}")
    records = read_markdown_files()
    validate_records(records)

    token_count = sum(record["token_count"] for record in records)
    estimated_cost = token_count / 1_000_000 * EMBEDDING_PRICE_PER_1M_TOKENS
    print(f"Prepared {len(records)} chunks from {len({record['file_path'] for record in records})} files.")
    print(f"Estimated tokens to embed: {token_count:,}")
    print(f"Estimated embedding cost at ${EMBEDDING_PRICE_PER_1M_TOKENS}/1M tokens: ${estimated_cost:.4f}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    client = OpenAI()
    vector_batches: list[np.ndarray] = []
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in range(0, len(records), BATCH_SIZE):
        batch_number = batch_start // BATCH_SIZE + 1
        batch = records[batch_start : batch_start + BATCH_SIZE]
        chunk_ids = [record["chunk_id"] for record in batch]
        cached = load_cached_batch(batch_number, chunk_ids)
        if cached is not None:
            print(f"Using cached batch {batch_number}/{total_batches} ({len(batch)} chunks)...", flush=True)
            vector_batches.append(cached)
            continue

        print(f"Embedding batch {batch_number}/{total_batches} ({len(batch)} chunks)...", flush=True)
        embeddings = embed_batch(client, [record["text"] for record in batch])
        vector_batches.append(save_cached_batch(batch_number, chunk_ids, embeddings))

    matrix = np.vstack(vector_batches).astype("float32")

    if matrix.shape[0] != len(records):
        raise RuntimeError(f"Vector count mismatch: {matrix.shape[0]} vectors for {len(records)} records")

    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    write_chunks_jsonl(records)
    write_stats(records, matrix.shape[1], time.time() - started_at)

    print(f"\nWrote FAISS index: {(INDEX_DIR / 'faiss.index').as_posix()}")
    print(f"Wrote chunk sidecar: {(INDEX_DIR / 'chunks.jsonl').as_posix()}")
    print(f"Wrote stats: {(INDEX_DIR / 'stats.json').as_posix()}")
    if EMBEDDING_CACHE_DIR.exists():
        shutil.rmtree(EMBEDDING_CACHE_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
