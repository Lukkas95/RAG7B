from typing import Optional, List

from sentence_transformers import SentenceTransformer

_model: Optional[SentenceTransformer] = None

MODEL_NAME = "all-MiniLM-L6-v2"


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
