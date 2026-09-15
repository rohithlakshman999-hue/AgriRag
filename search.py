import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


CHUNKS_FILE = "data/processed/chunks.pkl"
FAISS_FILE = "data/processed/faiss.index"
BM25_FILE = "data/processed/bm25.pkl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# -----------------------------
# Load saved data
# -----------------------------

print("Loading chunks...")

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks")


print("Loading FAISS index...")

faiss_index = faiss.read_index(FAISS_FILE)


print("Loading BM25 index...")

with open(BM25_FILE, "rb") as f:
    bm25 = pickle.load(f)


print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Semantic search
# -----------------------------

def semantic_search(query, top_k=10):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx == -1:
            continue

        results.append({
            "index": int(idx),
            "score": float(scores[0][rank]),
            "rank": rank + 1
        })

    return results


# -----------------------------
# BM25 search
# -----------------------------

def bm25_search(query, top_k=10):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices):

        results.append({
            "index": int(idx),
            "score": float(scores[idx]),
            "rank": rank + 1
        })

    return results


# -----------------------------
# Reciprocal Rank Fusion
# -----------------------------

def rrf_fusion(
    semantic_results,
    bm25_results,
    k=60
):

    scores = {}

    # Semantic results
    for result in semantic_results:

        idx = result["index"]

        scores[idx] = scores.get(idx, 0) + (
            1 / (k + result["rank"])
        )

    # BM25 results
    for result in bm25_results:

        idx = result["index"]

        scores[idx] = scores.get(idx, 0) + (
            1 / (k + result["rank"])
        )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# -----------------------------
# Search
# -----------------------------

def search(query, top_k=5):

    semantic_results = semantic_search(
        query,
        top_k=20
    )

    bm25_results = bm25_search(
        query,
        top_k=20
    )

    fused_results = rrf_fusion(
        semantic_results,
        bm25_results
    )

    results = []

    for idx, score in fused_results[:top_k]:

        chunk = chunks[idx]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "section": chunk["section"],
            "subsection": chunk["subsection"],
            "text": chunk["text"],
            "score": score
        })

    return results


# -----------------------------
# Interactive testing
# -----------------------------

if __name__ == "__main__":

    print("\nHybrid RAG Search")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Question: ")

        if query.lower() == "exit":
            break

        results = search(query)

        print("\n" + "=" * 70)

        for i, result in enumerate(results, 1):

            print(f"\nResult {i}")
            print(f"Chunk      : {result['chunk_id']}")
            print(f"Page       : {result['page']}")
            print(f"Section    : {result['section']}")
            print(f"Subsection : {result['subsection']}")
            print(f"RRF Score  : {result['score']:.5f}")
            print(f"Text       : {result['text'][:600]}")

        print("\n" + "=" * 70)