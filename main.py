from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    # Optional dependency for loading .env in local demos.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover

    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


# Windows terminals can default to a legacy code page (cp1252/cp1258),
# which breaks printing Vietnamese text. Force UTF-8 when supported.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker, SentenceChunker, compute_similarity
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


DEFAULT_DATA_DIR = Path("data") / "raw_data (1)"


def get_default_demo_files() -> list[str]:
    """Prefer user-provided documents under data/raw_data (1)."""
    if DEFAULT_DATA_DIR.exists() and DEFAULT_DATA_DIR.is_dir():
        md_files = sorted(DEFAULT_DATA_DIR.rglob("*.md"))
        txt_files = sorted(DEFAULT_DATA_DIR.rglob("*.txt"))
        files = [str(p) for p in (md_files + txt_files)]
        if files:
            return files

    return [
        "data/python_intro.txt",
        "data/vector_store_notes.md",
        "data/rag_system_design.md",
        "data/customer_support_playbook.txt",
        "data/chunking_experiment_report.md",
        "data/vi_retrieval_notes.md",
    ]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in allowed_extensions:
            continue
        if not path.exists() or not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower(), "lang": "vi"},
            )
        )

    return documents


def _make_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def _embed_query(embedder, text: str) -> list[float]:
    try:
        return embedder(text, "query")
    except TypeError:
        return embedder(text)


def _embed_passage(embedder, text: str) -> list[float]:
    try:
        return embedder(text, "passage")
    except TypeError:
        return embedder(text)


def _chunk_documents(docs: list[Document], strategy: str, embedder) -> list[Document]:
    strategy = strategy.strip().lower()

    if strategy == "fixed_size":
        chunker = FixedSizeChunker(chunk_size=int(os.getenv("CHUNK_SIZE", "200") or "200"), overlap=int(os.getenv("CHUNK_OVERLAP", "50") or "50"))
    elif strategy == "by_sentences":
        chunker = SentenceChunker(max_sentences_per_chunk=int(os.getenv("MAX_SENTENCES_PER_CHUNK", "3") or "3"))
    elif strategy == "recursive":
        chunker = RecursiveChunker(chunk_size=int(os.getenv("CHUNK_SIZE", "200") or "200"))
    elif strategy == "semantic":
        chunker = SemanticChunker(
            embedding_fn=embedder,
            similarity_threshold=float(os.getenv("SEMANTIC_SIM_THRESHOLD", "0.35") or "0.35"),
            max_chars_per_chunk=int(os.getenv("SEMANTIC_MAX_CHARS", "200") or "200"),
            max_sentences_per_chunk=int(os.getenv("SEMANTIC_MAX_SENTENCES", "6") or "6"),
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    chunked: list[Document] = []
    for doc in docs:
        rapper = (doc.id or "").replace("_", " ").strip().lower()
        lang = (doc.metadata or {}).get("lang", "vi")
        chunks = chunker.chunk(doc.content)
        for idx, ch in enumerate(chunks):
            chunked.append(
                Document(
                    id=f"{doc.id}::chunk{idx}",
                    content=ch,
                    metadata={
                        **(doc.metadata or {}),
                        "doc_id": doc.id,
                        "chunk_index": idx,
                        "chunk_strategy": strategy,
                        "rapper": rapper,
                        "lang": lang,
                        "doc_type": "bio",
                        "domain": "vietnamese_rapper_bio",
                    },
                )
            )
    return chunked


def _infer_filter_from_query(query: str, available_rappers: set[str]) -> dict | None:
    q = (query or "").lower()
    for rapper in sorted(available_rappers, key=len, reverse=True):
        if rapper and rapper in q:
            return {"rapper": rapper}
    return None


def _is_relevant(retrieved_text: str, gold_answer: str, embedder, threshold: float) -> bool:
    a = _embed_passage(embedder, retrieved_text or "")
    b = _embed_passage(embedder, gold_answer or "")
    return compute_similarity(a, b) >= threshold


def _benchmark_score(top3_relevant: int, top1_relevant: bool, answer_sim: float | None, answer_threshold: float) -> int:
    if top3_relevant <= 0:
        return 0
    if top1_relevant and (answer_sim is None or answer_sim >= answer_threshold):
        return 2
    return 1


def _run_benchmark_for_strategy(*, strategy: str, docs: list[Document], benchmark_items: list[dict], embedder, call_llm: bool) -> dict:
    relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.20") or "0.20")
    answer_threshold = float(os.getenv("ANSWER_SIM_THRESHOLD", "0.25") or "0.25")

    chunk_docs = _chunk_documents(docs, strategy=strategy, embedder=embedder)
    store = EmbeddingStore(collection_name=f"bench_{strategy}", embedding_fn=embedder)
    store.add_documents(chunk_docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)

    def split_sentences(s: str) -> list[str]:
        return [x.strip() for x in re.split(r"(?<=[.!?])(?:\s+|\n+)", (s or "").strip()) if x.strip()]

    def coherence_score(chunks: list[str]) -> float:
        pairs = 0
        total = 0.0
        for ch in chunks:
            sents = split_sentences(ch)
            if len(sents) < 2:
                continue
            embs = [_embed_passage(embedder, t) for t in sents]
            for a, b in zip(embs, embs[1:]):
                total += compute_similarity(a, b)
                pairs += 1
        return (total / pairs) if pairs else 0.0

    available_rappers = {((d.metadata or {}).get("rapper") or "") for d in chunk_docs}
    available_rappers = {r for r in available_rappers if isinstance(r, str) and r}

    rows = []
    meta_rows = []
    for item in benchmark_items:
        q = (item or {}).get("question", "")
        gold = (item or {}).get("answer", "")
        qid = (item or {}).get("id", "?")
        if not isinstance(q, str) or not q.strip():
            continue

        results = store.search(q, top_k=3)
        relevances = [bool(_is_relevant(r.get("content", ""), gold, embedder, relevance_threshold)) for r in results]
        top3_relevant = sum(relevances)
        top1_relevant = bool(relevances[0]) if relevances else False

        answer_sim = None
        if call_llm:
            agent_answer = agent.answer(q, top_k=3)
            answer_sim = compute_similarity(_embed_passage(embedder, agent_answer), _embed_passage(embedder, gold))

        score = _benchmark_score(top3_relevant, top1_relevant, answer_sim, answer_threshold)

        rows.append(
            {
                "id": qid,
                "question": q,
                "top3_relevant": top3_relevant,
                "top1_relevant": top1_relevant,
                "benchmark_score": score,
                "scores": [float(r.get("score", 0.0)) for r in results],
                "sources": [r.get("metadata", {}).get("source") for r in results],
            }
        )

        md_filter = _infer_filter_from_query(q, available_rappers)
        if md_filter is not None:
            filtered = store.search_with_filter(q, top_k=3, metadata_filter=md_filter)
            f_rels = [bool(_is_relevant(r.get("content", ""), gold, embedder, relevance_threshold)) for r in filtered]
            meta_rows.append(
                {"id": qid, "filter": md_filter, "unfiltered_top3_relevant": top3_relevant, "filtered_top3_relevant": sum(f_rels)}
            )

    total_queries = len(rows)
    avg_top3_relevant = (sum(r["top3_relevant"] for r in rows) / total_queries) if total_queries else 0.0
    total_benchmark_score = sum(r["benchmark_score"] for r in rows)
    chunk_texts = [d.content for d in chunk_docs]

    return {
        "strategy": strategy,
        "chunk_count": len(chunk_docs),
        "avg_chunk_len": (sum(len(d.content) for d in chunk_docs) / len(chunk_docs)) if chunk_docs else 0.0,
        "coherence_score": coherence_score(chunk_texts),
        "retrieval_precision_top3": avg_top3_relevant / 3.0 if total_queries else 0.0,
        "retrieval_avg_top3_relevant": avg_top3_relevant,
        "retrieval_benchmark_score_total": total_benchmark_score,
        "retrieval_benchmark_score_max": total_queries * 2,
        "per_query": rows,
        "metadata_utility": meta_rows,
    }


def _write_evaluation_markdown(path: Path, results: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Results (auto-generated)")
    lines.append("")
    lines.append("Generated by `main.py --benchmark ...`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Strategy | Chunk count | Avg chunk len | Coherence score | Avg top-3 relevant | Precision (top-3) | Benchmark score |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| `{r['strategy']}` | {r['chunk_count']} | {r['avg_chunk_len']:.1f} | {r['coherence_score']:.3f} | {r['retrieval_avg_top3_relevant']:.2f} | {r['retrieval_precision_top3']:.2f} | {r['retrieval_benchmark_score_total']}/{r['retrieval_benchmark_score_max']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def demo_llm(prompt: str) -> str:
    provider = (os.getenv("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
    if provider == "openai":
        base_url = (os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")
        api_key = os.getenv("OPENAI_API_KEY", "") or ""
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
        timeout_s = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120") or "120")
        if not api_key.strip():
            return "Thiếu OPENAI_API_KEY trong môi trường/.env."
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Bạn là trợ lý trả lời dựa trên context được cung cấp."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.2") or "0.2"),
        }
        req = urllib.request.Request(
            url=f"{base_url}/chat/completions",
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content", "")
                return content.strip() if isinstance(content, str) else str(content)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
            return f"Không gọi được OpenAI-compatible API để sinh câu trả lời. (chi tiết lỗi: {e})"
    return "LLM_PROVIDER chưa được cấu hình (hãy dùng LLM_PROVIDER=openai hoặc ollama)."


def run_manual_demo(*, question: str | None, strategy: str) -> int:
    files = get_default_demo_files()
    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        return 1

    embedder = _make_embedder()
    chunk_docs = _chunk_documents(docs, strategy=strategy, embedder=embedder)
    store = EmbeddingStore(collection_name="manual_store", embedding_fn=embedder)
    store.add_documents(chunk_docs)

    query = question or "Tóm tắt nội dung chính của bộ tài liệu."
    print(f"Query: {query}")
    results = store.search(query, top_k=3)
    for i, r in enumerate(results, start=1):
        print(f"{i}. score={r['score']:.3f} source={r['metadata'].get('source')}")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 7 manual demo (RAG + vector store).")
    parser.add_argument("question", nargs="*", help="Question to ask (free text).")
    parser.add_argument("--benchmark", type=str, default=None, help="Path to benchmark JSON (list of {id, question, answer}).")
    parser.add_argument("--all-strategies", action="store_true", help="Run benchmark across all chunking strategies.")
    parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=["fixed_size", "by_sentences", "recursive", "semantic"],
        help="Chunking strategy when not running --all-strategies.",
    )
    parser.add_argument("--eval-out", type=str, default="report/EVAL_ALL.md", help="Where to write evaluation markdown.")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM answering; compute retrieval-only metrics.")
    args = parser.parse_args()

    if args.benchmark:
        bench_path = Path(args.benchmark)
        if not bench_path.exists():
            print(f"Benchmark file not found: {bench_path}")
            return 1
        items = json.loads(bench_path.read_text(encoding="utf-8"))
        files = get_default_demo_files()
        docs = load_documents_from_files(files)
        if not docs:
            print("No valid docs.")
            return 1
        embedder = _make_embedder()
        call_llm = not args.no_llm
        if args.all_strategies:
            strategies = ["fixed_size", "by_sentences", "recursive", "semantic"]
            results = [_run_benchmark_for_strategy(strategy=s, docs=docs, benchmark_items=items, embedder=embedder, call_llm=call_llm) for s in strategies]
        else:
            results = [_run_benchmark_for_strategy(strategy=args.strategy, docs=docs, benchmark_items=items, embedder=embedder, call_llm=call_llm)]
        out_path = Path(args.eval_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_evaluation_markdown(out_path, results)
        print(f"Wrote evaluation markdown: {out_path}")
        return 0

    question = " ".join(args.question).strip() if args.question else None
    return run_manual_demo(question=question, strategy=args.strategy)


if __name__ == "__main__":
    raise SystemExit(main())
