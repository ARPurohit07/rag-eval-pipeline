from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import RAGPipeline

app = FastAPI(title="RAG Pipeline API")
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    model: str | None = None


class ContextItem(BaseModel):
    content: str
    source: str
    title: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[ContextItem]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = get_pipeline().answer(req.question, top_k=req.top_k, model=req.model)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        contexts=[
            ContextItem(content=c.content, source=c.source, title=c.title, score=c.score)
            for c in result.contexts
        ],
    )
