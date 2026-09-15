# ============================================================
# KrishiJal AI - Build FAISS + BM25 Index
# ============================================================

import pickle
import re

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# FILE PATHS
# ============================================================

CHUNKS_FILE = "data/processed/chunks.pkl"
FAISS_FILE = "data/processed/faiss.index"
BM25_FILE = "data/processed/bm25.pkl"


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):
    """
    Convert text into simple lowercase word tokens.

    Keeps useful words such as:
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

print("=" * 65)
print("KRISHIJAL AI - INDEX BUILD")
print("=" * 65)

print("\nLoading chunks...")

try:

    with open(
        CHUNKS_FILE,
        "rb"
    ) as f:

        chunks = pickle.load(f)

except FileNotFoundError:

    raise FileNotFoundError(
        f"Could not find {CHUNKS_FILE}. "
        "Run ingest.py first."
    )


if not chunks:

    raise RuntimeError(
        "No chunks were found in chunks.pkl."
    )


print(
    f"Total chunks: {len(chunks)}"
)


# ============================================================
# PREPARE SEARCH TEXT
# ============================================================

print("\nPreparing searchable text...")

search_texts = []


for chunk in chunks:

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
        ) or ""
    )

    subsection = str(
        chunk.get(
            "subsection",
            ""
        ) or ""
    )

    text = str(
        chunk.get(
            "text",
            ""
        )
    )


    # --------------------------------------------------------
    # Include metadata before actual content
    # --------------------------------------------------------

    searchable_text = (
        f"Document: {document}\n"
        f"Page: {page}\n"
        f"Section: {section}\n"
        f"Subsection: {subsection}\n"
        f"Content: {text}"
    )


    search_texts.append(
        searchable_text
    )


print(
    f"Prepared {len(search_texts)} searchable records."
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
# CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

embeddings = model.encode(

    search_texts,

    normalize_embeddings=True,

    show_progress_bar=True,

    batch_size=32,

    convert_to_numpy=True
)


embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# ============================================================
# EMBEDDING VALIDATION
# ============================================================

if len(embeddings) != len(chunks):

    raise RuntimeError(
        "Embedding count does not match chunk count."
    )


if embeddings.ndim != 2:

    raise RuntimeError(
        "Embeddings have an unexpected shape."
    )


dimension = embeddings.shape[1]


print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# BUILD FAISS
# ============================================================

print("\nBuilding FAISS index...")

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)


# ============================================================
# VALIDATE FAISS
# ============================================================

if index.ntotal != len(chunks):

    raise RuntimeError(
        f"FAISS/chunk mismatch: "
        f"{index.ntotal} vectors vs "
        f"{len(chunks)} chunks."
    )


# ============================================================
# SAVE FAISS
# ============================================================

faiss.write_index(
    index,
    FAISS_FILE
)

print(
    f"FAISS index saved: {FAISS_FILE}"
)

print(
    f"FAISS vectors: {index.ntotal}"
)


# ============================================================
# BUILD BM25
# ============================================================

print("\nBuilding BM25 index...")

tokenized_texts = [

    tokenize(text)

    for text in search_texts

]


bm25 = BM25Okapi(
    tokenized_texts
)


# ============================================================
# SAVE BM25
# ============================================================

with open(
    BM25_FILE,
    "wb"
) as f:

    pickle.dump(
        bm25,
        f
    )


print(
    f"BM25 index saved: {BM25_FILE}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\nRunning final validation...")

saved_index = faiss.read_index(
    FAISS_FILE
)


if saved_index.ntotal != len(chunks):

    raise RuntimeError(
        "Saved FAISS index validation failed."
    )


with open(
    BM25_FILE,
    "rb"
) as f:

    saved_bm25 = pickle.load(
        f
    )


if len(saved_bm25.doc_freqs) != len(chunks):

    raise RuntimeError(
        "Saved BM25 index validation failed."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)
print("INDEX BUILD COMPLETE")
print("=" * 65)

print(
    f"Chunks       : {len(chunks)}"
)

print(
    f"Embeddings   : {len(embeddings)}"
)

print(
    f"Dimension    : {dimension}"
)

print(
    f"FAISS        : {FAISS_FILE}"
)

print(
    f"BM25         : {BM25_FILE}"
)

print(
    "Validation   : PASSED"
)

print("=" * 65)