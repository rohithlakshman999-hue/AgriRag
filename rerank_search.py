# ============================================================
# KrishiJal AI - Hybrid Retrieval + Cross-Encoder Reranking
# ============================================================
#
# Pipeline:
#
# Farmer Question
#       ↓
# Context-aware query
#       ↓
# Semantic Search (FAISS)
#       +
# Keyword Search (BM25)
#       ↓
# RRF Fusion
#       ↓
# Cross-Encoder Reranking
#       ↓
# Section Boost
#       ↓
# Final Evidence
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

from section_router import route_query


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

EMBEDDING_MODEL = (
    "BAAI/bge-base-en-v1.5"
)

RERANKER_MODEL = (
    "BAAI/bge-reranker-v2-m3"
)


# ============================================================
# SETTINGS
# ============================================================

CANDIDATE_K = 15

FINAL_K = 3

RRF_K = 60

SECTION_BOOST = 0.10

# IMPORTANT:
# Do not use an aggressive 0.50 threshold for the
# cross-encoder score.
#
# We only discard extremely weak evidence.
MIN_FINAL_SCORE = -1.0


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 70)
print("KRISHIJAL AI - RERANK SEARCH")
print("=" * 70)

print("\nLoading chunks...")

with open(
    CHUNKS_FILE,
    "rb"
) as f:

    chunks = pickle.load(f)


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
    f"FAISS contains "
    f"{faiss_index.ntotal} vectors."
)


# ============================================================
# SAFETY CHECK
# ============================================================

if faiss_index.ntotal != len(chunks):

    raise RuntimeError(
        "\nINDEX MISMATCH!\n"
        f"chunks.pkl : {len(chunks)} chunks\n"
        f"faiss.index : "
        f"{faiss_index.ntotal} vectors\n\n"
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

print("BM25 loaded.")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    f"Embedding model loaded: "
    f"{EMBEDDING_MODEL}"
)


# ============================================================
# LOAD RERANKER
# ============================================================

print("\nLoading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    max_length=384
)

print(
    f"Reranker loaded: "
    f"{RERANKER_MODEL}"
)


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
    query,
    top_k=CANDIDATE_K
):

    top_k = min(
        top_k,
        faiss_index.ntotal
    )


    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )


    # --------------------------------------------------------
    # FAISS search
    # --------------------------------------------------------

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


        if idx >= len(chunks):

            continue


        results.append(
            {
                "index": idx,

                "rank": rank + 1,

                "dense_score": float(
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
    top_k=CANDIDATE_K
):

    top_k = min(
        top_k,
        len(chunks)
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    query_tokens = (
        query.lower()
        .split()
    )


    if not query_tokens:

        return []


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

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


        if idx < 0 or idx >= len(chunks):

            continue


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
# RRF FUSION
# ============================================================

def rrf_fusion(
    dense_results,
    bm25_results,
    k=RRF_K
):
    """
    Reciprocal Rank Fusion.

    Documents appearing in both semantic and keyword
    retrieval receive stronger fused rankings.
    """

    fused_scores = {}


    # ========================================================
    # DENSE
    # ========================================================

    for result in dense_results:

        idx = result["index"]

        fused_scores[idx] = (
            fused_scores.get(
                idx,
                0.0
            )
            +
            1.0 / (
                k + result["rank"]
            )
        )


    # ========================================================
    # BM25
    # ========================================================

    for result in bm25_results:

        idx = result["index"]

        fused_scores[idx] = (
            fused_scores.get(
                idx,
                0.0
            )
            +
            1.0 / (
                k + result["rank"]
            )
        )


    # ========================================================
    # SORT
    # ========================================================

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    return ranked


# ============================================================
# GET CHUNK
# ============================================================

def get_chunk_result(
    idx
):

    chunk = chunks[idx]


    return {
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
        )
    }


# ============================================================
# CROSS-ENCODER RERANK
# ============================================================

def rerank_candidates(
    query,
    candidates
):

    if not candidates:

        return []


    pairs = []


    for candidate in candidates:

        idx = candidate["index"]

        pairs.append(
            [
                query,
                chunks[idx]["text"]
            ]
        )


    scores = reranker.predict(
        pairs
    )


    results = []


    for candidate, score in zip(
        candidates,
        scores
    ):

        result = dict(
            candidate
        )


        result["reranker_score"] = float(
            score
        )


        result["final_score"] = float(
            score
        )


        results.append(
            result
        )


    return results


# ============================================================
# MAIN SEARCH
# ============================================================

def search(
    query,
    candidate_k=CANDIDATE_K,
    final_k=FINAL_K
):

    query = str(
        query
    ).strip()


    if not query:

        return []


    print(
        "\n" + "=" * 70
    )

    print(
        "KRISHIJAL SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuery:"
    )

    print(
        query
    )


    # ========================================================
    # SECTION ROUTING
    # ========================================================

    routed_section, route_score = route_query(
        query
    )


    print(
        f"\nRouted section : "
        f"{routed_section}"
    )

    print(
        f"Route score    : "
        f"{route_score}"
    )


    # ========================================================
    # SEMANTIC RETRIEVAL
    # ========================================================

    dense_results = dense_search(
        query,
        top_k=candidate_k
    )


    # ========================================================
    # KEYWORD RETRIEVAL
    # ========================================================

    bm25_results = bm25_search(
        query,
        top_k=candidate_k
    )


    print(
        f"\nDense results  : "
        f"{len(dense_results)}"
    )

    print(
        f"BM25 results   : "
        f"{len(bm25_results)}"
    )


    # ========================================================
    # RRF
    # ========================================================

    fused = rrf_fusion(
        dense_results,
        bm25_results
    )


    fused = fused[
        :candidate_k
    ]


    if not fused:

        print(
            "\nNo hybrid candidates."
        )

        return []


    # ========================================================
    # BUILD CANDIDATE OBJECTS
    # ========================================================

    candidates = []


    for idx, fusion_score in fused:

        result = get_chunk_result(
            idx
        )


        result["fusion_score"] = float(
            fusion_score
        )


        candidates.append(
            result
        )


    print(
        f"\nHybrid candidates: "
        f"{len(candidates)}"
    )


    # ========================================================
    # CROSS-ENCODER
    # ========================================================

    print(
        "\nReranking candidates..."
    )


    reranked = rerank_candidates(
        query,
        candidates
    )


    # ========================================================
    # SECTION BOOST
    # ========================================================

    ranked = []


    for result in reranked:

        section = result.get(
            "section"
        )


        section_boost = 0.0


        if (
            routed_section
            and section
        ):

            # Exact match
            if section == routed_section:

                section_boost = SECTION_BOOST


            # Case-insensitive match
            elif (
                str(section).lower()
                ==
                str(
                    routed_section
                ).lower()
            ):

                section_boost = SECTION_BOOST


        result["section_boost"] = (
            section_boost
        )


        result["final_score"] = (
            result["reranker_score"]
            +
            section_boost
        )


        ranked.append(
            result
        )


    # ========================================================
    # SORT FINAL RESULTS
    # ========================================================

    ranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )


    # ========================================================
    # FILTER ONLY EXTREMELY BAD RESULTS
    # ========================================================

    filtered = [

        result

        for result in ranked

        if result["final_score"]
        >= MIN_FINAL_SCORE

    ]


    # ========================================================
    # FINAL TOP K
    # ========================================================

    final_results = filtered[
        :final_k
    ]


    # ========================================================
    # DEBUG OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL RERANKED RESULTS"
    )

    print(
        "=" * 70
    )


    for i, result in enumerate(
        final_results,
        start=1
    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            "-" * 70
        )

        print(
            f"Document       : "
            f"{result['document']}"
        )

        print(
            f"Page           : "
            f"{result['page']}"
        )

        print(
            f"Section        : "
            f"{result.get('section') or 'General content'}"
        )

        print(
            f"Subsection     : "
            f"{result.get('subsection') or 'None'}"
        )

        print(
            f"Fusion score   : "
            f"{result['fusion_score']:.5f}"
        )

        print(
            f"Reranker score : "
            f"{result['reranker_score']:.5f}"
        )

        print(
            f"Section boost  : "
            f"{result['section_boost']:.5f}"
        )

        print(
            f"Final score    : "
            f"{result['final_score']:.5f}"
        )


    print(
        "=" * 70
    )


    return final_results


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nKrishiJal Hybrid + Reranker Test"
    )

    print(
        "Type 'exit' to stop."
    )


    while True:

        try:

            query = input(
                "\nEnter your question: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break


        if query.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "Exiting..."
            )

            break


        if not query:

            print(
                "Please enter a question."
            )

            continue


        try:

            results = search(
                query,
                candidate_k=CANDIDATE_K,
                final_k=FINAL_K
            )


            if not results:

                print(
                    "\nNo relevant evidence found."
                )

            else:

                print(
                    "\nFinal evidence:"
                )


                for i, result in enumerate(
                    results,
                    start=1
                ):

                    print(
                        f"\n{i}. "
                        f"{result['document']} "
                        f"• Page {result['page']}"
                    )

                    print(
                        result["text"][:800]
                    )


        except Exception as exc:

            print(
                "\nERROR:"
            )

            print(
                repr(exc)
            )