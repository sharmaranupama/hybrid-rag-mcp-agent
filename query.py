from qdrant_client import QdrantClient, models
from ollama import Client as OllamaClient
from config import (
    EMBED_MODEL, COLLECTION_NAME, RETRIEVAL_LIMIT, QDRANT_URL, OLLAMA_HOST,
    DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, HYBRID_SEARCH_LIMIT, llm_generate,
)

q_client      = QdrantClient(url=QDRANT_URL)
ollama_client = OllamaClient(host=OLLAMA_HOST)

_sparse_encoder = None

def _get_sparse_encoder():
    global _sparse_encoder
    if _sparse_encoder is None:
        from fastembed import SparseTextEmbedding
        _sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_encoder


def _collection_has_named_vectors() -> bool:
    try:
        info = q_client.get_collection(COLLECTION_NAME)
        return isinstance(info.config.params.vectors, dict)
    except Exception:
        return False


def get_dense_embedding(text: str):
    return ollama_client.embed(model=EMBED_MODEL, input=text).embeddings[0]


def get_sparse_embedding(text: str) -> models.SparseVector:
    result = list(_get_sparse_encoder().query_embed(text))[0]
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def _build_filter(year: int | None) -> models.Filter | None:
    if year is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="year", match=models.MatchValue(value=year))]
    )


def hybrid_search(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    year: int | None = None,
) -> list[models.ScoredPoint]:
    dense_vec    = get_dense_embedding(question)
    query_filter = _build_filter(year)

    if not _collection_has_named_vectors():
        kwargs = dict(collection_name=COLLECTION_NAME, query=dense_vec, limit=limit, with_payload=True)
        if query_filter:
            kwargs["query_filter"] = query_filter
        return q_client.query_points(**kwargs).points

    sparse_vec = get_sparse_embedding(question)
    prefetch = [
        models.Prefetch(query=dense_vec,  using=DENSE_VECTOR_NAME,  limit=HYBRID_SEARCH_LIMIT, filter=query_filter),
        models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, limit=HYBRID_SEARCH_LIMIT, filter=query_filter),
    ]
    return q_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    ).points


def fetch_context(question: str, limit: int = RETRIEVAL_LIMIT, year: int | None = None) -> str:
    chunks = fetch_context_with_scores(question, limit, year)
    return "\n".join(c["text"] for c in chunks) if chunks else ""


def fetch_context_with_scores(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    year: int | None = None,
) -> list[dict]:
    def _to_dict(p: models.ScoredPoint) -> dict:
        return {
            "text":     p.payload.get("original_text", ""),
            "score":    p.score if p.score is not None else 0.0,
            "year":     p.payload.get("year", 0),
            "topic":    p.payload.get("topic", ""),
            "category": p.payload.get("category", ""),
            "point_id": p.id,
        }

    if year is not None:
        return [_to_dict(p) for p in hybrid_search(question, limit, year)]

    global_points = hybrid_search(question, limit, year=None)
    years_found   = list({p.payload.get("year", 0) for p in global_points if p.payload.get("year", 0) != 0})

    all_points: dict[int, models.ScoredPoint] = {p.id: p for p in global_points}
    for yr in years_found:
        for p in hybrid_search(question, limit=2, year=yr):
            if p.id not in all_points or p.score > all_points[p.id].score:
                all_points[p.id] = p

    merged = sorted(all_points.values(), key=lambda p: p.score or 0.0, reverse=True)[:limit]
    return [_to_dict(p) for p in merged]


def ask_rag(question: str, limit: int = RETRIEVAL_LIMIT) -> str:
    context = fetch_context(question, limit)
    if not context:
        return "I couldn't find any relevant company information to answer that."

    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Instructions:\n"
        f"1. Answer using ONLY the context provided — do not add outside knowledge.\n"
        f"2. Present ALL relevant details found in the context clearly and confidently.\n"
        f"3. If the context covers multiple years or plans, include all of them.\n"
        f"4. Only say information is unavailable if the context truly has nothing relevant.\n"
        f"5. Keep the answer concise and well-organized.\n\n"
        f"Answer:"
    )
    return llm_generate(prompt)
