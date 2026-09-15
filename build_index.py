import pickle
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


CHUNKS_FILE = "data/processed/chunks.pkl"

FAISS_FILE = "data/processed/faiss.index"
BM25_FILE = "data/processed/bm25.pkl"


MODEL_NAME = "BAAI/bge-base-en-v1.5"


print("Loading chunks...")

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

texts = [c["text"] for c in chunks]

print(f"Total chunks: {len(texts)}")

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32
)

embeddings = np.array(embeddings).astype("float32")


print("\nBuilding FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

faiss.write_index(index, FAISS_FILE)


print("FAISS index saved.")


print("\nBuilding BM25 index...")

tokenized_texts = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_texts)

with open(BM25_FILE, "wb") as f:
    pickle.dump(bm25, f)


print("BM25 index saved.")

print("\n" + "=" * 60)
print("INDEX BUILD COMPLETE")
print("=" * 60)
print(f"Chunks : {len(chunks)}")
print(f"FAISS  : {FAISS_FILE}")
print(f"BM25   : {BM25_FILE}")