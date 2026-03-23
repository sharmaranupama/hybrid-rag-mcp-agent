"""
query.py — Retrieve relevant chunks from Qdrant and answer questions with RAG.

Search strategy:
  - Hybrid: dense semantic vectors + BM25 keyword vectors, fused via RRF (Reciprocal Rank Fusion).
  - Falls back to dense-only if the collection was created without named/sparse vectors.
  - Multi-year balancing: when no year filter is given, ensures no single year dominates results.
"""

from qdrant_client import QdrantClient, models
from ollama import Client as OllamaClient
from config import (
    EMBED_MODEL, COLLECTION_NAME, RETRIEVAL_LIMIT, QDRANT_URL, OLLAMA_HOST,
    DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, HYBRID_SEARCH_LIMIT, llm_generate,
)

q_client      = QdrantClient(url=QDRANT_URL)
ollama_client = OllamaClient(host=OLLAMA_HOST)

# Lazy-loaded to avoid slow startup; downloaded on first use
_sparse_encoder = None

def _get_sparse_encoder():
    global _sparse_encoder
    if _sparse_encoder is None:
        from fastembed import SparseTextEmbedding
        _sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_encoder


def _collection_has_named_vectors() -> bool:
    """Return True if the collection uses named vectors (hybrid schema)."""
    try:
        info = q_client.get_collection(COLLECTION_NAME)
        return isinstance(info.config.params.vectors, dict)
    except Exception:
        return False


# ── Embedding helpers ─────────────────────────────────────────────────────────

def get_dense_embedding(text: str):
    """Convert text to a dense float vector via Ollama."""
    return ollama_client.embed(model=EMBED_MODEL, input=text).embeddings[0]


def get_sparse_embedding(text: str) -> models.SparseVector:
    """Convert text to a sparse BM25 vector via FastEmbed."""
    result = list(_get_sparse_encoder().query_embed(text))[0]
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def _build_filter(year: int | None) -> models.Filter | None:
    """Build a Qdrant payload filter for an exact year match, or None for no filter."""
    if year is None:
        return None
    return models.Filter(
        must=[models.FieldCondition(key="year", match=models.MatchValue(value=year))]
    )


# ── Core search ───────────────────────────────────────────────────────────────

def hybrid_search(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    year: int | None = None,
) -> list[models.ScoredPoint]:
    """
    Run hybrid search (dense + BM25) with RRF fusion.

    If the collection doesn't have named vectors (old schema), falls back
    to a plain dense-only query so old data still works.
    """
    dense_vec    = get_dense_embedding(question)
    query_filter = _build_filter(year)

    # --- Dense-only fallback for old collections ---
    if not _collection_has_named_vectors():
        kwargs = dict(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            limit=limit,
            with_payload=True,
        )
        if query_filter:
            kwargs["query_filter"] = query_filter
        return q_client.query_points(**kwargs).points

    # --- Hybrid path: prefetch dense + BM25, then fuse with RRF ---
    sparse_vec = get_sparse_embedding(question)

    # Each sub-query fetches HYBRID_SEARCH_LIMIT candidates independently
    prefetch = [
        models.Prefetch(
            query=dense_vec,
            using=DENSE_VECTOR_NAME,
            limit=HYBRID_SEARCH_LIMIT,
            filter=query_filter,
        ),
        models.Prefetch(
            query=sparse_vec,
            using=SPARSE_VECTOR_NAME,
            limit=HYBRID_SEARCH_LIMIT,
            filter=query_filter,
        ),
    ]

    # RRF fusion: re-ranks candidates from both sub-queries by combined rank position
    results = q_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return results.points


# ── Context retrieval ─────────────────────────────────────────────────────────

def fetch_context(question: str, limit: int = RETRIEVAL_LIMIT, year: int | None = None) -> str:
    """Return a plain text block of the top-ranked chunks for use in a prompt."""
    chunks = fetch_context_with_scores(question, limit, year)
    return "\n".join(c["text"] for c in chunks) if chunks else ""


def fetch_context_with_scores(
    question: str,
    limit: int = RETRIEVAL_LIMIT,
    year: int | None = None,
) -> list[dict]:
    """
    Return the top chunks as dicts, each with text, score, year, topic, category.

    When no year filter is given, applies multi-year balancing:
      1. Do a global search (no year filter) to get the overall top results.
      2. Find all distinct years in those results.
      3. Run a small per-year search to rescue good chunks that got buried globally.
      4. Merge, deduplicate (keeping the highest score per chunk), and return top `limit`.

    This prevents a year with many data points from crowding out the others.
    """
    def _to_dict(p: models.ScoredPoint) -> dict:
        return {
            "text":     p.payload.get("original_text", ""),
            "score":    p.score if p.score is not None else 0.0,
            "year":     p.payload.get("year", 0),
            "topic":    p.payload.get("topic", ""),
            "category": p.payload.get("category", ""),
            "point_id": p.id,
        }

    # If a specific year was requested, just run a single filtered search
    if year is not None:
        return [_to_dict(p) for p in hybrid_search(question, limit, year)]

    # Step 1: global search
    global_points = hybrid_search(question, limit, year=None)

    # Step 2: collect distinct years from the global results
    years_found = list({
        p.payload.get("year", 0) for p in global_points if p.payload.get("year", 0) != 0
    })

    # Step 3: per-year search — collect into a dict keyed by point ID to deduplicate
    all_points: dict[int, models.ScoredPoint] = {p.id: p for p in global_points}

    for yr in years_found:
        for p in hybrid_search(question, limit=2, year=yr):
            # Keep the version of each chunk with the highest score
            if p.id not in all_points or p.score > all_points[p.id].score:
                all_points[p.id] = p

    # Step 4: sort by score and return the top `limit`
    merged = sorted(all_points.values(), key=lambda p: p.score or 0.0, reverse=True)[:limit]
    return [_to_dict(p) for p in merged]


# ── RAG answer ────────────────────────────────────────────────────────────────

def ask_rag(question: str, limit: int = RETRIEVAL_LIMIT) -> str:
    """
    Full RAG pipeline: retrieve context, build a prompt, generate an answer.
    The LLM is instructed to use ONLY the retrieved context.
    """
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
