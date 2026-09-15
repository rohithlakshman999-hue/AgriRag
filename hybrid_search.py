import pickle
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# --------------------------------------------------
# FILES
# --------------------------------------------------

CHUNKS_PATH = Path("data/processed/chunks.pkl")
FAISS_PATH = Path("data/processed/faiss_bge.index")


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

with open(CHUNKS_PATH, "rb") as f:

    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

print("Embedding model loaded.")


# --------------------------------------------------
# CREATE DOCUMENT EMBEDDINGS
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in chunks
]

print("\nCreating document embeddings...")

document_embeddings = embedding_model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

document_embeddings = document_embeddings.astype(
    "float32"
)

dimension = document_embeddings.shape[1]

print(f"Embedding dimension: {dimension}")


# --------------------------------------------------
# FAISS
# --------------------------------------------------

index = faiss.IndexFlatIP(dimension)

index.add(document_embeddings)

print(f"FAISS vectors: {index.ntotal}")

faiss.write_index(
    index,
    str(FAISS_PATH)
)

print(f"FAISS saved to: {FAISS_PATH}")


# --------------------------------------------------
# BM25
# --------------------------------------------------

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index created.")


# --------------------------------------------------
# BM25 SEARCH
# --------------------------------------------------

def bm25_search(query, top_k=30):

    tokens = query.lower().split()

    scores = bm25.get_scores(tokens)

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in ranked_indices:

        results.append({
            "index": int(idx),
            "bm25_score": float(scores[idx]),
            "document": chunks[idx]["document"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"]
        })

    return results


# --------------------------------------------------
# DENSE SEARCH
# --------------------------------------------------

def dense_search(query, top_k=30):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "index": int(idx),
            "dense_score": float(score),
            "document": chunks[idx]["document"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"]
        })

    return results


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60
):

    fused_scores = {}

    metadata = {}

    # Dense ranking
    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        idx = result["index"]

        fused_scores[idx] = (
            fused_scores.get(idx, 0)
            + 1 / (k + rank)
        )

        metadata[idx] = result


    # BM25 ranking
    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        idx = result["index"]

        fused_scores[idx] = (
            fused_scores.get(idx, 0)
            + 1 / (k + rank)
        )

        if idx not in metadata:

            metadata[idx] = result


    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked:

        result = metadata[idx].copy()

        result["fusion_score"] = score

        results.append(result)

    return results


# --------------------------------------------------
# HYBRID SEARCH
# --------------------------------------------------

def hybrid_search(
    query,
    top_k=20
):

    dense_results = dense_search(
        query,
        top_k=30
    )

    bm25_results = bm25_search(
        query,
        top_k=30
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )

    return fused_results[:top_k]


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    )

    results = hybrid_search(
        query,
        top_k=10
    )

    print("\n" + "=" * 70)
    print("HYBRID SEARCH RESULTS")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {rank}"
        )

        print("-" * 70)

        print(
            f"Document: {result['document']}"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"Fusion score: "
            f"{result['fusion_score']:.5f}"
        )

        print("\nText:")

        print(
            result["text"][:1200]
        )