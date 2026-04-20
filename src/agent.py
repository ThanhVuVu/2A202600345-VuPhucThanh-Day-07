from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        results = self.store.search(question, top_k=top_k)
        context_blocks = []
        for i, r in enumerate(results, start=1):
            context_blocks.append(f"[{i}] {r.get('content', '')}".strip())

        context = "\n".join(context_blocks).strip()
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the context.\n\n"
            f"Context:\n{context if context else '(no context)'}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        answer = self.llm_fn(prompt)
        return answer if isinstance(answer, str) else str(answer)
