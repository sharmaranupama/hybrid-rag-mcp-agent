"""
ingest.py — Load data.csv into Qdrant with hybrid (dense + BM25) vectors.

What this does:
  1. Drops and recreates the Qdrant collection from scratch.
  2. Adds int8 scalar quantization to cut memory usage ~4x.
  3. Creates payload indexes on year/topic/category for fast filtered queries.
  4. Embeds each row with both a dense vector (nomic-embed-text via Ollama)
     and a sparse BM25 vector (FastEmbed), then upserts them in batches.
"""

import pandas as pd
from ollama import Client as OllamaClient
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier,
    PointStruct, PayloadSchemaType,
    ScalarQuantization, ScalarQuantizationConfig, ScalarType,
)
from config import (
    EMBED_MODEL, VECTOR_SIZE, COLLECTION_NAME,
    INGEST_BATCH_SIZE, QDRANT_URL, OLLAMA_HOST,
    DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ENABLE_QUANTIZATION,
)

ollama_client = OllamaClient(host=OLLAMA_HOST)
q_client      = QdrantClient(url=QDRANT_URL)

# Lazy-loaded so startup is fast and the model only downloads when first used
_sparse_encoder = None

def _get_sparse_encoder():
    """Load the BM25 sparse encoder on first call."""
    global _sparse_encoder
    if _sparse_encoder is None:
        from fastembed import SparseTextEmbedding
        _sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_encoder


def ingest_csv(file_path: str, progress_callback=None):
    """
    Read a CSV and index every row into Qdrant as a hybrid vector point.

    Args:
        file_path:         Path to the CSV file.
        progress_callback: Optional callable(fraction 0–1, message str) for UI updates.
    """
    df = pd.read_csv(file_path)
    sparse_encoder = _get_sparse_encoder()

    # Always start fresh — delete the old collection if it exists
    if q_client.collection_exists(COLLECTION_NAME):
        q_client.delete_collection(COLLECTION_NAME)

    # --- Create collection ---------------------------------------------------
    # int8 quantization: stores vectors as 8-bit integers instead of 32-bit floats
    quant_config = None
    if ENABLE_QUANTIZATION:
        quant_config = ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                quantile=0.99,   # Clip the top 1% of values before quantizing
                always_ram=True, # Keep quantized vectors in RAM for fast access
            )
        )

    q_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # Dense semantic vectors (one float per dimension)
            DENSE_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            # Sparse BM25 keyword vectors; Qdrant computes IDF automatically
            SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
        },
        quantization_config=quant_config,
    )

    # --- Payload indexes (speed up filtered searches) ------------------------
    for field, schema in [
        ("year",     PayloadSchemaType.INTEGER),
        ("topic",    PayloadSchemaType.KEYWORD),
        ("category", PayloadSchemaType.KEYWORD),
    ]:
        if field in df.columns:
            q_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema,
            )
            print(f"  Index created: {field} ({schema})")

    print(f"Ingesting {len(df)} rows with hybrid vectors (dense + BM25)...")

    # --- Batch upsert --------------------------------------------------------
    for i in range(0, len(df), INGEST_BATCH_SIZE):
        batch = df.iloc[i:i + INGEST_BATCH_SIZE]

        # Flatten each row into "col1: val1 | col2: val2 | ..." for embedding
        texts = [
            " | ".join(f"{col}: {val}" for col, val in row.items())
            for _, row in batch.iterrows()
        ]

        # Dense embeddings from Ollama
        dense_embeddings  = ollama_client.embed(model=EMBED_MODEL, input=texts).embeddings
        # Sparse BM25 embeddings from FastEmbed
        sparse_embeddings = list(sparse_encoder.embed(texts))

        points = []
        for j, (dense_vec, sparse_vec, (_, row)) in enumerate(
            zip(dense_embeddings, sparse_embeddings, batch.iterrows())
        ):
            points.append(PointStruct(
                id=i + j,
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: {
                        "indices": sparse_vec.indices.tolist(),
                        "values":  sparse_vec.values.tolist(),
                    },
                },
                payload={
                    "original_text": texts[j],
                    "year":     int(row["year"]) if "year" in row and pd.notna(row.get("year")) else 0,
                    "topic":    str(row.get("topic", "")),
                    "category": str(row.get("category", "")),
                },
            ))

        q_client.upsert(collection_name=COLLECTION_NAME, points=points)
        done = min(i + len(points), len(df))
        print(f"  Indexed rows {i}–{done}")
        if progress_callback:
            progress_callback(done / len(df), f"Indexed {done}/{len(df)} rows")

    # Summary
    info = q_client.get_collection(COLLECTION_NAME)
    print(f"\nDone. '{COLLECTION_NAME}': {info.points_count} points")


if __name__ == "__main__":
    try:
        ingest_csv("data.csv")
    finally:
        q_client.close()
