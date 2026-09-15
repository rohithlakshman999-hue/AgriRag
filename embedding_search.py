import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

CHUNKS_PATH = Path("data/processed/chunks.pkl")
INDEX_PATH = Path("data/processed/faiss.index")


# --------------------------------------------------
# LOAD CHUNKS
# --------------------------------------------------

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks.")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded.")


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# FAISS expects float32
embeddings = embeddings.astype("float32")

print(f"Embedding shape: {embeddings.shape}")


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"FAISS index contains {index.ntotal} vectors.")


# --------------------------------------------------
# SAVE INDEX
# --------------------------------------------------

faiss.write_index(index, str(INDEX_PATH))

print(f"FAISS index saved to: {INDEX_PATH}")


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def search(query, top_k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "score": float(distance),
            "document": chunks[idx]["document"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"]
        })

    return results


# --------------------------------------------------
# TEST SEARCH
# --------------------------------------------------

if __name__ == "__main__":

    query = input("\nEnter your question: ")

    results = search(query)

    print("\n" + "=" * 70)
    print("TOP RESULTS")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        print(f"\nRESULT {i}")
        print("-" * 70)

        print(f"Document : {result['document']}")
        print(f"Page     : {result['page']}")
        print(f"Score    : {result['score']:.4f}")

        print("\nText:")
        print(result["text"][:1500])