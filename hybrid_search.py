# ============================================================
# KrishiJal AI - Hybrid Retrieval
# Semantic Search + BM25 + Reciprocal Rank Fusion
# ============================================================

import pickle
import re
from pathlib import Path

import faiss
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# FILE PATHS
# ============================================================

CHUNKS_PATH = Path(
    "data/processed/chunks.pkl"
)

FAISS_PATH = Path(
    "data/processed/faiss.index"
)

BM25_PATH = Path(
    "data/processed/bm25.pkl"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "BAAI/bge-base-en-v1.5"
)


# ============================================================
# RETRIEVAL SETTINGS
# ============================================================

DENSE_TOP_K = 30

BM25_TOP_K = 30

DEFAULT_TOP_K = 10

RRF_K = 60


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Simple BM25 tokenizer.

    Keeps useful terms such as:
        water-saving
        fruit-set
        water-use
    """

    text = text.lower()

    return re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        text
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 70)
print("KRISHIJAL AI - HYBRID SEARCH")
print("=" * 70)

print("\nLoading chunks...")

if not CHUNKS_PATH.exists():

    raise FileNotFoundError(
        f"Missing file: {CHUNKS_PATH}\n"
        "Run ingest.py first."
    )


with open(
    CHUNKS_PATH,
    "rb"
) as f:

    chunks = pickle.load(f)


if not chunks:

    raise RuntimeError(
        "chunks.pkl contains no chunks."
    )


print(
    f"Loaded {len(chunks)} chunks."
)


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS index...")

if not FAISS_PATH.exists():

    raise FileNotFoundError(
        f"Missing file: {FAISS_PATH}\n"
        "Run build_index.py first."
    )


faiss_index = faiss.read_index(
    str(FAISS_PATH)
)


print(
    f"FAISS vectors: "
    f"{faiss_index.ntotal}"
)


if faiss_index.ntotal != len(chunks):

    raise RuntimeError(
        "\nFAISS/chunks mismatch.\n"
        f"FAISS vectors: {faiss_index.ntotal}\n"
        f"Chunks: {len(chunks)}\n\n"
        "Run:\n"
        "python build_index.py"
    )


# ============================================================
# LOAD BM25
# ============================================================

print("\nLoading BM25 index...")

if not BM25_PATH.exists():

    raise FileNotFoundError(
        f"Missing file: {BM25_PATH}\n"
        "Run build_index.py first."
    )


with open(
    BM25_PATH,
    "rb"
) as f:

    bm25 = pickle.load(f)


print("BM25 loaded.")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_NAME
)

print(
    f"Embedding model loaded: "
    f"{MODEL_NAME}"
)


# ============================================================
# BUILD SEARCH TEXT
# ============================================================

def build_search_text(chunk):
    """
    Recreate the same text representation used
    during build_index.py.

    Metadata + content are included.
    """

    document = str(
        chunk.get(
            "document",
            ""
        )
    )

    page = str(
        chunk.get(
            "page",
            ""
        )
    )

    section = str(
        chunk.get(
            "section",
            ""
        )
        or ""
    )

    subsection = str(
        chunk.get(
            "subsection",
            ""
        )
        or ""
    )

    text = str(
        chunk.get(
            "text",
            ""
        )
    )

    return (
        f"Document: {document}\n"
        f"Page: {page}\n"
        f"Section: {section}\n"
        f"Subsection: {subsection}\n"
        f"Content: {text}"
    )


# ============================================================
# DENSE SEARCH
# ============================================================

def dense_search(
    query,
    top_k=DENSE_TOP_K
):
    """
    Semantic search using BGE embeddings + FAISS.
    """

    query = query.strip()

    if not query:

        return []


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
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue


        idx = int(idx)

        chunk = chunks[idx]


        results.append(
            {
                "index": idx,

                "dense_score": float(
                    score
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
    Keyword-based search using BM25.
    """

    query = query.strip()

    if not query:

        return []


    query_tokens = tokenize(
        query
    )


    if not query_tokens:

        return []


    scores = bm25.get_scores(
        query_tokens
    )


    ranked_indices = np.argsort(
        scores
    )[::-1][:top_k]


    results = []


    for idx in ranked_indices:

        idx = int(idx)

        chunk = chunks[idx]


        results.append(
            {
                "index": idx,

                "bm25_score": float(
                    scores[idx]
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
        )


    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=RRF_K
):
    """
    Combine semantic and keyword rankings.

    A document receives:
        1 / (k + rank)

    from each retrieval method.
    """

    fused_scores = {}

    metadata = {}


    # ========================================================
    # DENSE RESULTS
    # ========================================================

    for rank, result in enumerate(
        dense_results,
        start=1
    ):

        idx = result[
            "index"
        ]


        fused_scores[idx] = (
            fused_scores.get(
                idx,
                0.0
            )
            +
            1.0 / (k + rank)
        )


        metadata[idx] = result


    # ========================================================
    # BM25 RESULTS
    # ========================================================

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        idx = result[
            "index"
        ]


        fused_scores[idx] = (
            fused_scores.get(
                idx,
                0.0
            )
            +
            1.0 / (k + rank)
        )


        if idx not in metadata:

            metadata[idx] = result


    # ========================================================
    # SORT
    # ========================================================

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # ========================================================
    # CREATE RESULTS
    # ========================================================

    results = []


    for idx, fusion_score in ranked:

        result = dict(
            metadata[idx]
        )


        result["fusion_score"] = float(
            fusion_score
        )


        results.append(
            result
        )


    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query,
    top_k=DEFAULT_TOP_K
):
    """
    Full hybrid search:

    Query
      ↓
    Dense semantic search
      +
    BM25 keyword search
      ↓
    RRF fusion
      ↓
    Top K
    """

    if not query.strip():

        return []


    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    dense_results = dense_search(
        query,
        top_k=DENSE_TOP_K
    )


    # --------------------------------------------------------
    # Keyword search
    # --------------------------------------------------------

    bm25_results = bm25_search(
        query,
        top_k=BM25_TOP_K
    )


    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results
    )


    return fused_results[
        :top_k
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    ).strip()


    results = hybrid_search(
        query,
        top_k=10
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "HYBRID SEARCH RESULTS"
    )

    print(
        "=" * 70
    )


    if not results:

        print(
            "\nNo results found."
        )


    else:

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\nRESULT {rank}"
            )

            print(
                "-" * 70
            )

            print(
                f"Document      : "
                f"{result['document']}"
            )

            print(
                f"Page          : "
                f"{result['page']}"
            )

            print(
                f"Section       : "
                f"{result.get('section') or 'General content'}"
            )

            print(
                f"Dense score   : "
                f"{result.get('dense_score', 0):.4f}"
            )

            print(
                f"BM25 score    : "
                f"{result.get('bm25_score', 0):.4f}"
            )

            print(
                f"Fusion score  : "
                f"{result.get('fusion_score', 0):.5f}"
            )

            print(
                "\nText:"
            )

            print(
                result["text"][:1200]
            )