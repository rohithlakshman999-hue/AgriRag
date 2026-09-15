import pickle

with open("data/processed/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

for c in chunks:
    if c["page"] in [8, 9]:
        print("=" * 70)
        print("Chunk ID   :", c["chunk_id"])
        print("Page       :", c["page"])
        print("Section    :", c["section"])
        print("Subsection :", c["subsection"])
        print("Text       :", c["text"][:500])