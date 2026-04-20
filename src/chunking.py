from __future__ import annotations

import math
import re
from typing import Callable


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on sentence boundaries; keep punctuation with the sentence.
        # This is intentionally simple (lab-grade) and deterministic for tests.
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", text.strip())
            if s.strip()
        ]
        if not sentences:
            return []

        # Optional: Add overlap to improve context preservation
        overlap = 1 if self.max_sentences_per_chunk > 1 else 0
        step = self.max_sentences_per_chunk - overlap
        
        chunks: list[str] = []
        for i in range(0, len(sentences), step):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
            if i + self.max_sentences_per_chunk >= len(sentences):
                break
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        separators = self.separators if self.separators else [""]
        chunks = self._split(text, separators)
        return [c.strip() for c in chunks if c and c.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        current_text = current_text or ""
        if not current_text.strip():
            return []
        if len(current_text) <= self.chunk_size:
            return [current_text.strip()]

        if not remaining_separators:
            # Graceful fallback: last resort fixed-size splitting.
            return FixedSizeChunker(chunk_size=self.chunk_size, overlap=0).chunk(current_text)

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        if sep == "":
            return FixedSizeChunker(chunk_size=self.chunk_size, overlap=0).chunk(current_text)

        parts = current_text.split(sep)

        # Rebuild chunks while trying to stay under chunk_size.
        chunks: list[str] = []
        buffer = ""

        def flush() -> None:
            nonlocal buffer
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = ""

        for part in parts:
            candidate = part if not buffer else buffer + sep + part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue

            # If buffer already has something, flush it first.
            if buffer:
                flush()
                candidate = part

            # If still too large, recurse on this part with the next separators.
            if len(candidate) > self.chunk_size:
                chunks.extend(self._split(candidate, rest))
                buffer = ""
            else:
                buffer = candidate

        flush()
        return chunks


class SemanticChunker:
    """
    Semantic chunking (simple heuristic).

    Steps:
      - Split text into sentences.
      - Embed each sentence.
      - Append next sentence into current chunk if similarity is high enough
        (and size/sentence limits are not exceeded).
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]] | None = None,
        similarity_threshold: float = 0.35,
        max_chars_per_chunk: int = 200,
        max_sentences_per_chunk: int = 6,
    ) -> None:
        if embedding_fn is None:
            from .embeddings import _mock_embed

            embedding_fn = _mock_embed

        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.max_chars_per_chunk = max(1, max_chars_per_chunk)
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def _split_sentences(self, text: str) -> list[str]:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])(?:\s+|\n+)", (text or "").strip())
            if s.strip()
        ]

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        def embed_passage(t: str) -> list[float]:
            try:
                return self.embedding_fn(t, "passage")  # type: ignore[misc]
            except TypeError:
                return self.embedding_fn(t)

        sentence_embs = [embed_passage(s) for s in sentences]

        chunks: list[str] = []
        current_sentences: list[str] = [sentences[0]]
        current_emb = sentence_embs[0]

        for sent, emb in zip(sentences[1:], sentence_embs[1:]):
            candidate_sentences = current_sentences + [sent]
            candidate_text = " ".join(candidate_sentences).strip()

            if len(candidate_text) > self.max_chars_per_chunk or len(candidate_sentences) > self.max_sentences_per_chunk:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = [sent]
                current_emb = emb
                continue

            sim = compute_similarity(current_emb, emb)
            if sim >= self.similarity_threshold:
                current_sentences = candidate_sentences
                current_emb = [(a + b) / 2.0 for a, b in zip(current_emb, emb)]
            else:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences = [sent]
                current_emb = emb

        tail = " ".join(current_sentences).strip()
        if tail:
            chunks.append(tail)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies: dict[str, list[str]] = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, min(50, chunk_size // 4))).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        def stats(chunks: list[str]) -> dict:
            count = len(chunks)
            avg_length = (sum(len(c) for c in chunks) / count) if count else 0.0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {name: stats(chunks) for name, chunks in strategies.items()}
