# ============================================================
# KrishiJal AI - Semantic Embedding Search
# ============================================================

import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# FILE PATHS
# ============================================================

CHUNKS_PATH = Path(
    "data/processed/chunks.pkl"
)

INDEX_PATH = Path(
    "data/processed/faiss.index"
)


# ============================================================
# MODEL
# IMPORTANT:
# This MUST match build_index.py
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 70)
print("KRISHIJAL AI - SEMANTIC EMBEDDING SEARCH")
print("=" * 70)

print("\nLoading chunks...")

if not CHUNKS_PATH.exists():

    raise FileNotFoundError(
        f"Could not find {CHUNKS_PATH}. "
        "Run ingest.py first."
    )


with open(
    CHUNKS_PATH,
    "rb"
) as f:

    chunks = pickle.load(f)


print(
    f"Loaded {len(chunks)} chunks."
)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS index...")

if not INDEX_PATH.exists():

    raise FileNotFoundError(
        f"Could not find {INDEX_PATH}. "
        "Run build_index.py first."
    )


index = faiss.read_index(
    str(INDEX_PATH)
)


print(
    f"FAISS contains {index.ntotal} vectors."
)


# ============================================================
# VALIDATE INDEX
# ============================================================

if index.ntotal != len(chunks):

    raise RuntimeError(
        "\nFAISS/chunks mismatch.\n"
        f"FAISS vectors : {index.ntotal}\n"
        f"Chunks        : {len(chunks)}\n\n"
        "Run:\n"
        "python build_index.py"
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)

print(
    f"Embedding model loaded: {MODEL_NAME}"
)


# ============================================================
# BUILD SEARCH TEXT
# ============================================================

def build_search_text(chunk):
    """
    Recreate the same representation used during
    index creation.

    Metadata helps semantic search understand the
    document/page/section context.
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
# SEMANTIC SEARCH
# ============================================================

def search(
    query,
    top_k=5
):

    query = query.strip()

    if not query:

        return []


    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = model.encode(
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

    scores, indices = index.search(
        query_embedding,
        top_k
    )


    # --------------------------------------------------------
    # Prepare results
    # --------------------------------------------------------

    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:

            continue


        chunk = chunks[
            int(idx)
        ]


        results.append(
            {
                "chunk_id": int(idx),

                "score": float(score),

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
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your question: "
    ).strip()


    results = search(
        query,
        top_k=5
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "TOP SEMANTIC RESULTS"
    )

    print(
        "=" * 70
    )


    if not results:

        print(
            "\nNo results found."
        )


    else:

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
                f"Document : "
                f"{result['document']}"
            )

            print(
                f"Page     : "
                f"{result['page']}"
            )

            print(
                f"Section  : "
                f"{result['section'] or 'General content'}"
            )

            print(
                f"Score    : "
                f"{result['score']:.4f}"
            )

            print(
                "\nText:"
            )

            print(
                result["text"][:1500]
            )