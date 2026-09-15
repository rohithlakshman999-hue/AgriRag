# ============================================================
# KrishiJal AI - Search Engine
# ============================================================
#
# Retrieval pipeline:
#
# Farmer Question
#       ↓
# Semantic Search (FAISS)
#       +
# Keyword Search (BM25)
#       ↓
# Reciprocal Rank Fusion
#       ↓
# Cross-Encoder Reranking
#       ↓
# Final Relevant Evidence
#
# ============================================================

import pickle

import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

from rank_bm25 import BM25Okapi


# ============================================================
# FILES
# ============================================================

CHUNKS_FILE = (
    "data/processed/chunks.pkl"
)

FAISS_FILE = (
    "data/processed/faiss.index"
)

BM25_FILE = (
    "data/processed/bm25.pkl"
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-base-en-v1.5"
)

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-v2-m3"
)


# ============================================================
# SETTINGS
# ============================================================

SEMANTIC_TOP_K = 20

BM25_TOP_K = 20

RRF_TOP_K = 10

FINAL_TOP_K = 3

RRF_K = 60


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 70)
print("KRISHIJAL AI SEARCH ENGINE")
print("=" * 70)

print("\nLoading chunks...")

with open(
    CHUNKS_FILE,
    "rb"
) as f:

    chunks = pickle.load(f)


if not chunks:

    raise RuntimeError(
        "chunks.pkl is empty."
    )


print(
    f"Loaded {len(chunks)} chunks."
)


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS index...")

faiss_index = faiss.read_index(
    FAISS_FILE
)


print(
    f"FAISS vectors: "
    f"{faiss_index.ntotal}"
)


# ============================================================
# VALIDATE FAISS
# ============================================================

if faiss_index.ntotal != len(chunks):

    raise RuntimeError(
        "\nFAISS/chunks mismatch!\n"
        f"FAISS: {faiss_index.ntotal}\n"
        f"Chunks: {len(chunks)}\n\n"
        "Run:\n"
        "python ingest.py\n"
        "python build_index.py"
    )


# ============================================================
# LOAD BM25
# ============================================================

print("\nLoading BM25 index...")

with open(
    BM25_FILE,
    "rb"
) as f:

    bm25 = pickle.load(f)


print(
    "BM25 loaded."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


print(
    f"Embedding model loaded: "
    f"{EMBEDDING_MODEL_NAME}"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("\nLoading cross-encoder reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL_NAME,
    max_length=384
)


print(
    f"Reranker loaded: "
    f"{RERANKER_MODEL_NAME}"
)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    top_k=SEMANTIC_TOP_K
):
    """
    Semantic similarity search using:
        BGE embeddings + FAISS
    """

    top_k = min(
        top_k,
        len(chunks)
    )


    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )


    results = []


    for rank, idx in enumerate(
        indices[0]
    ):

        if idx < 0:

            continue


        idx = int(idx)


        results.append(
            {
                "index": idx,

                "rank": rank + 1,

                "semantic_score": float(
                    scores[0][rank]
                )
            }
        )


    return results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    query,
    top_k=BM25_TOP_K
):
    """
    Keyword search using BM25.
    """

    top_k = min(
        top_k,
        len(chunks)
    )


    query_tokens = (
        query.lower()
        .split()
    )


    if not query_tokens:

        return []


    scores = bm25.get_scores(
        query_tokens
    )


    top_indices = np.argsort(
        scores
    )[::-1][:top_k]


    results = []


    for rank, idx in enumerate(
        top_indices
    ):

        idx = int(idx)


        results.append(
            {
                "index": idx,

                "rank": rank + 1,

                "bm25_score": float(
                    scores[idx]
                )
            }
        )


    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def rrf_fusion(
    semantic_results,
    bm25_results,
    k=RRF_K
):
    """
    Combine semantic and keyword rankings.

    Documents appearing in both searches are
    naturally promoted.
    """

    scores = {}


    # --------------------------------------------------------
    # Semantic results
    # --------------------------------------------------------

    for result in semantic_results:

        idx = result["index"]

        scores[idx] = (
            scores.get(
                idx,
                0.0
            )
            +
            1.0 / (
                k + result["rank"]
            )
        )


    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for result in bm25_results:

        idx = result["index"]

        scores[idx] = (
            scores.get(
                idx,
                0.0
            )
            +
            1.0 / (
                k + result["rank"]
            )
        )


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    return ranked


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank(
    query,
    fused_results,
    top_k=RRF_TOP_K
):
    """
    Re-rank the best hybrid candidates using
    a cross-encoder.
    """

    candidates = fused_results[
        :top_k
    ]


    if not candidates:

        return []


    pairs = []


    for idx, _ in candidates:

        pairs.append(
            (
                query,
                chunks[idx]["text"]
            )
        )


    scores = reranker.predict(
        pairs
    )


    results = []


    for (
        (idx, fusion_score),
        rerank_score
    ) in zip(
        candidates,
        scores
    ):

        chunk = chunks[idx]


        results.append(
            {
                "index": int(idx),

                "chunk_id": chunk.get(
                    "chunk_id",
                    idx
                ),

                "document": chunk.get(
                    "document",
                    "Unknown document"
                ),

                "page": chunk.get(
                    "page",
                    "Unknown"
                ),

                "section": chunk.get(
                    "section"
                ),

                "subsection": chunk.get(
                    "subsection"
                ),

                "text": chunk.get(
                    "text",
                    ""
                ),

                "fusion_score": float(
                    fusion_score
                ),

                "reranker_score": float(
                    rerank_score
                )
            }
        )


    # --------------------------------------------------------
    # Sort by reranker
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )


    return results[
        :FINAL_TOP_K
    ]


# ============================================================
# COMPLETE SEARCH
# ============================================================

def search(
    query,
    top_k=FINAL_TOP_K
):
    """
    Complete KrishiJal retrieval pipeline.
    """

    query = str(
        query
    ).strip()


    if not query:

        return []


    print(
        "\n" + "=" * 70
    )

    print(
        "SEARCH QUERY"
    )

    print(
        "=" * 70
    )

    print(
        query
    )


    # ========================================================
    # SEMANTIC SEARCH
    # ========================================================

    semantic_results = semantic_search(
        query,
        SEMANTIC_TOP_K
    )


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    bm25_results = bm25_search(
        query,
        BM25_TOP_K
    )


    # ========================================================
    # RRF
    # ========================================================

    fused_results = rrf_fusion(
        semantic_results,
        bm25_results
    )


    print(
        f"\nHybrid candidates: "
        f"{min(len(fused_results), RRF_TOP_K)}"
    )


    # ========================================================
    # RERANK
    # ========================================================

    results = rerank(
        query,
        fused_results,
        RRF_TOP_K
    )


    # respect requested top_k
    results = results[
        :top_k
    ]


    # ========================================================
    # DEBUG
    # ========================================================

    print(
        "\nTOP RESULTS"
    )

    print(
        "=" * 70
    )


    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {i}"
        )

        print(
            f"Document : "
            f"{result['document']}"
        )

        print(
            f"Page     : "
            f"{result['page']}"
        )

        print(
            f"Section  : "
            f"{result.get('section') or 'General content'}"
        )

        print(
            f"RRF      : "
            f"{result['fusion_score']:.5f}"
        )

        print(
            f"Reranker : "
            f"{result['reranker_score']:.5f}"
        )

        print(
            "\nText:"
        )

        print(
            result["text"][:800]
        )


    print(
        "\n" + "=" * 70
    )


    return results


# ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nKrishiJal Search Test"
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        query = input(
            "\nQuestion: "
        ).strip()


        if query.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "\nExiting."
            )

            break


        if not query:

            continue


        try:

            results = search(
                query,
                top_k=5
            )


            if not results:

                print(
                    "\nNo results found."
                )

        except Exception as exc:

            print(
                "\nSearch error:"
            )

            print(
                repr(exc)
            )