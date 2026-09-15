import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from section_router import route_query


# =========================================================
# FILE PATHS
# =========================================================

CHUNKS_FILE = "data/processed/chunks.pkl"
FAISS_FILE = "data/processed/faiss.index"
BM25_FILE = "data/processed/bm25.pkl"


# =========================================================
# MODELS
# =========================================================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# =========================================================
# SETTINGS
# =========================================================

CANDIDATE_K = 20
FINAL_K = 5

# This is only a ranking preference.
SECTION_BOOST = 0.20

# Minimum score used to remove obviously weak results.
MIN_FINAL_SCORE = 0.50


# =========================================================
# LOAD CHUNKS
# =========================================================

print("Loading chunks...")

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks.")


# =========================================================
# LOAD FAISS INDEX
# =========================================================

print("\nLoading FAISS index...")

faiss_index = faiss.read_index(FAISS_FILE)

print(
    f"FAISS contains {faiss_index.ntotal} vectors."
)


# =========================================================
# SAFETY CHECK
# =========================================================

if faiss_index.ntotal != len(chunks):

    raise RuntimeError(
        "\nINDEX MISMATCH!\n"
        f"chunks.pkl : {len(chunks)} chunks\n"
        f"faiss.index : {faiss_index.ntotal} vectors\n\n"
        "Rebuild the index using:\n"
        "python ingest.py\n"
        "python build_index.py"
    )


# =========================================================
# LOAD BM25
# =========================================================

print("\nLoading BM25 index...")

with open(BM25_FILE, "rb") as f:
    bm25 = pickle.load(f)

print("BM25 loaded.")


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# =========================================================
# LOAD RERANKER
# =========================================================

print("\nLoading reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    max_length=512
)

print("Reranker loaded.")


# =========================================================
# DENSE SEARCH
# =========================================================

def dense_search(query, top_k=CANDIDATE_K):

    # Never ask FAISS for more vectors than it contains.
    top_k = min(
        top_k,
        faiss_index.ntotal
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
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

    for rank, idx in enumerate(indices[0]):

        if idx < 0:
            continue

        if idx >= len(chunks):
            continue

        results.append({
            "index": int(idx),
            "rank": rank + 1,
            "score": float(scores[0][rank])
        })

    return results


# =========================================================
# BM25 SEARCH
# =========================================================

def bm25_search(query, top_k=CANDIDATE_K):

    top_k = min(
        top_k,
        len(chunks)
    )

    query_tokens = query.lower().split()

    scores = bm25.get_scores(
        query_tokens
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices):

        idx = int(idx)

        if idx < 0 or idx >= len(chunks):
            continue

        results.append({
            "index": idx,
            "rank": rank + 1,
            "score": float(scores[idx])
        })

    return results


# =========================================================
# RECIPROCAL RANK FUSION
# =========================================================

def rrf_fusion(
    dense_results,
    bm25_results,
    k=60
):

    fused = {}

    # Dense retrieval contribution
    for result in dense_results:

        idx = result["index"]

        fused[idx] = (
            fused.get(idx, 0.0)
            + 1.0 / (k + result["rank"])
        )

    # BM25 contribution
    for result in bm25_results:

        idx = result["index"]

        fused[idx] = (
            fused.get(idx, 0.0)
            + 1.0 / (k + result["rank"])
        )

    return sorted(
        fused.items(),
        key=lambda x: x[1],
        reverse=True
    )


# =========================================================
# MAIN SEARCH
# =========================================================

def search(
    query,
    candidate_k=CANDIDATE_K,
    final_k=FINAL_K
):

    print("\n")
    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(query)


    # =====================================================
    # 1. QUERY ROUTING
    # =====================================================

    routed_section, route_score = route_query(
        query
    )

    print(
        f"\nRouted section : {routed_section}"
    )

    print(
        f"Route score    : {route_score}"
    )


    # =====================================================
    # 2. DENSE SEARCH
    # =====================================================

    dense_results = dense_search(
        query,
        top_k=candidate_k
    )


    # =====================================================
    # 3. BM25 SEARCH
    # =====================================================

    bm25_results = bm25_search(
        query,
        top_k=candidate_k
    )


    # =====================================================
    # 4. RRF FUSION
    # =====================================================

    fused_results = rrf_fusion(
        dense_results,
        bm25_results
    )

    candidates = fused_results[:candidate_k]

    print(
        f"\nHybrid candidates: {len(candidates)}"
    )

    if not candidates:
        return []


    # =====================================================
    # 5. PREPARE RERANKING PAIRS
    # =====================================================

    pairs = []

    for idx, fusion_score in candidates:

        text = chunks[idx]["text"]

        pairs.append([
            query,
            text
        ])


    print(
        "\nReranking candidates..."
    )


    # =====================================================
    # 6. CROSS-ENCODER RERANKING
    # =====================================================

    rerank_scores = reranker.predict(
        pairs
    )


    # =====================================================
    # 7. BUILD RESULT OBJECTS
    # =====================================================

    ranked = []

    for (
        (idx, fusion_score),
        rerank_score
    ) in zip(
        candidates,
        rerank_scores
    ):

        chunk = chunks[idx]

        section = chunk.get(
            "section",
            None
        )

        subsection = chunk.get(
            "subsection",
            None
        )


        # =================================================
        # SECTION BOOST
        # =================================================

        section_boost = 0.0

        if (
            routed_section is not None
            and section is not None
            and section == routed_section
        ):

            section_boost = SECTION_BOOST


        original_score = float(
            rerank_score
        )

        final_score = (
            original_score
            + section_boost
        )


        ranked.append({

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
                0
            ),

            "section": section,

            "subsection": subsection,

            "text": chunk.get(
                "text",
                ""
            ),

            "fusion_score": float(
                fusion_score
            ),

            "reranker_score": original_score,

            "section_boost": float(
                section_boost
            ),

            "final_score": float(
                final_score
            )
        })


    # =====================================================
    # 8. SORT BY FINAL SCORE
    # =====================================================

    ranked.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )


    # =====================================================
    # 9. SECTION-AWARE PRIORITY
    # =====================================================

    if routed_section is not None:

        section_results = []

        other_results = []

        for result in ranked:

            if result["section"] == routed_section:

                section_results.append(
                    result
                )

            else:

                other_results.append(
                    result
                )


        # Keep score ordering inside each group.
        section_results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        other_results.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )


        # Only prioritize the routed section.
        ranked = (
            section_results
            + other_results
        )


    # =====================================================
    # 10. REMOVE WEAK RESULTS
    # =====================================================

    filtered = [

        result

        for result in ranked

        if result["final_score"]
        >= MIN_FINAL_SCORE

    ]


    # =====================================================
    # 11. FALLBACK
    # =====================================================

    if not filtered and ranked:

        print(
            "\nNo result passed the "
            "relevance threshold."
        )

        # Keep strongest result so the LLM can still
        # make the final knowledge-gap decision.
        filtered = [
            ranked[0]
        ]


    # =====================================================
    # 12. RETURN FINAL RESULTS
    # =====================================================

    return filtered[:final_k]


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(results):

    print("\n")
    print("=" * 70)
    print("FINAL RERANKED RESULTS")
    print("=" * 70)


    if not results:

        print(
            "\nNo relevant evidence found."
        )

        return


    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nRESULT {i}"
        )

        print(
            "-" * 70
        )


        print(
            f"Chunk ID       : "
            f"{result['chunk_id']}"
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
            f"{result['section']}"
        )


        print(
            f"Subsection     : "
            f"{result['subsection']}"
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
            "\nText:"
        )


        print(
            result["text"][:1200]
        )


# =========================================================
# INTERACTIVE MODE
# =========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RERANKED HYBRID RAG SEARCH")
    print("=" * 70)


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


        # -------------------------------------------------
        # Exit
        # -------------------------------------------------

        if query.lower() in [
            "exit",
            "quit"
        ]:

            print(
                "Exiting..."
            )

            break


        # -------------------------------------------------
        # Empty question
        # -------------------------------------------------

        if not query:

            print(
                "Please enter a question."
            )

            continue


        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------

        try:

            results = search(
                query,
                candidate_k=CANDIDATE_K,
                final_k=FINAL_K
            )

            display_results(
                results
            )


        except Exception as e:

            print(
                "\nERROR during search:"
            )

            print(e)

            print(
                "\nCheck that these files exist:\n"
                "data/processed/chunks.pkl\n"
                "data/processed/faiss.index\n"
                "data/processed/bm25.pkl"
            )