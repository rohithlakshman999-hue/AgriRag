# ============================================================
# KrishiJal AI - Check Section Metadata
# ============================================================

import pickle


# ============================================================
# FILE
# ============================================================

CHUNKS_FILE = "data/processed/chunks.pkl"


# ============================================================
# LOAD CHUNKS
# ============================================================

print("=" * 75)
print("KRISHIJAL AI - SECTION METADATA CHECK")
print("=" * 75)

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
# CHECK SELECTED PAGES
# ============================================================

pages_to_check = [8, 9]

found = 0


for chunk in chunks:

    if chunk.get("page") in pages_to_check:

        found += 1

        print("\n" + "=" * 75)

        print(
            f"Chunk ID       : "
            f"{chunk.get('chunk_id')}"
        )

        print(
            f"Document       : "
            f"{chunk.get('document')}"
        )

        print(
            f"Page           : "
            f"{chunk.get('page')}"
        )

        print(
            f"Section        : "
            f"{chunk.get('section') or 'None'}"
        )

        print(
            f"Subsection     : "
            f"{chunk.get('subsection') or 'None'}"
        )

        print(
            "\nText:"
        )

        print(
            chunk.get("text", "")[:1000]
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)

print(
    f"Chunks found on pages {pages_to_check}: "
    f"{found}"
)


# ============================================================
# SECTION SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SECTION SUMMARY")
print("=" * 75)


section_counts = {}


for chunk in chunks:

    section = (
        chunk.get("section")
        or "None"
    )

    section_counts[section] = (
        section_counts.get(section, 0)
        + 1
    )


for section, count in sorted(
    section_counts.items(),
    key=lambda x: str(x[0])
):

    print(
        f"{section}: {count} chunks"
    )


# ============================================================
# SUBSECTION SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("SUBSECTION SUMMARY")
print("=" * 75)


subsection_counts = {}


for chunk in chunks:

    subsection = (
        chunk.get("subsection")
        or "None"
    )

    subsection_counts[subsection] = (
        subsection_counts.get(
            subsection,
            0
        )
        + 1
    )


for subsection, count in sorted(
    subsection_counts.items(),
    key=lambda x: str(x[0])
):

    if subsection != "None":

        print(
            f"{subsection}: {count} chunks"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("SECTION CHECK COMPLETE")
print("=" * 75)