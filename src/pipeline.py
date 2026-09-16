from dataclasses import dataclass

from src.llm import generate_answer
from src.retrieval import HybridRetriever, RetrievedChunk


@dataclass
class RAGResult:
    question: str
    answer: str
    contexts: list[RetrievedChunk]


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def answer(self, question: str, top_k: int | None = None, model: str | None = None) -> RAGResult:
        contexts = self.retriever.search(question, top_k=top_k)
        answer = generate_answer(question, [c.content for c in contexts], model=model)
        return RAGResult(question=question, answer=answer, contexts=contexts)
