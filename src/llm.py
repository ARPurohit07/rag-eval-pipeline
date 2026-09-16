from ollama import Client

from src.config import settings

_client = Client(host=settings.ollama_host)

SYSTEM_PROMPT = (
    "You are a technical assistant that answers questions using ONLY the "
    "provided context passages. Follow these rules strictly:\n"
    "1. Base every claim in your answer on the provided context - do not use "
    "outside knowledge.\n"
    "2. If the context does not contain enough information to answer, say so "
    "explicitly instead of guessing.\n"
    "3. Be concise and directly answer the question.\n"
)


def generate_answer(question: str, context_chunks: list[str], model: str | None = None) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Passage {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    user_prompt = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context passages above."
    )
    response = _client.chat(
        model=model or settings.ollama_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]
