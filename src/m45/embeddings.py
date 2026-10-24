from httpx import TransportError
from langchain_ollama import OllamaEmbeddings
from ollama import Client, ResponseError

from m45.config import Settings

EMBEDDING_DIMENSIONS = 1024
EMBEDDING_CONTEXT_LENGTH = 32768


def prepare_embeddings(settings: Settings) -> tuple[OllamaEmbeddings, dict[str, object]]:
    base_url = str(settings.embedding_base_url)
    models = Client(host=base_url, timeout=30.0).list().models
    installed = next((model for model in models if model.model == settings.embedding_model), None)
    if installed is None or not installed.digest:
        raise ValueError("embedding model must match an installed model with a digest")

    model = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=base_url,
        dimensions=EMBEDDING_DIMENSIONS,
        num_ctx=EMBEDDING_CONTEXT_LENGTH,
        client_kwargs={"timeout": 120.0},
    )
    embedding_config: dict[str, object] = {
        "provider": "ollama",
        "model": settings.embedding_model,
        "digest": installed.digest,
        "dimensions": EMBEDDING_DIMENSIONS,
        "num_ctx": EMBEDDING_CONTEXT_LENGTH,
        "document_format": "passage-content-v1",
    }
    return model, embedding_config


SEARCH_INSTRUCTION = (
    "Given a question about the user's life, retrieve relevant "
    "conversation excerpts and daily log passages."
)


def embed_search_query(model: OllamaEmbeddings, query: str) -> list[float]:
    """Embed a search query using the current Qwen model's instruction format."""
    return model.embed_query(f"Instruct: {SEARCH_INSTRUCTION}\nQuery:{query}")


def is_transient_embedding_error(error: Exception) -> bool:
    if isinstance(error, ResponseError):
        return error.status_code == 429 or 500 <= error.status_code < 600
    return isinstance(error, (ConnectionError, TransportError))
