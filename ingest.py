import os
import re
import pickle
import pymupdf

RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/chunks.pkl"

CHUNK_SIZE = 350
CHUNK_OVERLAP = 70


# Known main section headings in this PDF
MAIN_SECTIONS = {
    "1": "1. Introduction",
    "2": "2. Objectives",
    "3": "3. Coverage of Farmers",
    "4": "4. Coverage of Crops",
    "5": "5. Coverage of Risks and Exclusions",
}


def normalize_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_sections(text):
    """
    Find real main-section markers such as:
    3. Coverage of Farmers
    4. Coverage of Crops
    5. Coverage of Risks and Exclusions
    """

    positions = []

    patterns = [
        (r"\b3\s+3\.\s+Coverage of Farmers\b", "3. Coverage of Farmers"),
        (r"\b4\.\s+Coverage of Crops\b", "4. Coverage of Crops"),
        (r"\b5\.\s+Coverage of Risks and Exclusions\b",
         "5. Coverage of Risks and Exclusions"),
    ]

    for pattern, section_name in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            positions.append({
                "start": match.start(),
                "name": section_name
            })

    positions.sort(key=lambda x: x["start"])

    return positions


def detect_subsections(text):
    """
    Detect subsection numbers such as:
    3.1
    3.1.1
    3.1.1.1
    5.1
    5.1.1
    """

    pattern = r"\b(\d+\.\d+(?:\.\d+){0,2})\b"

    return list(re.finditer(pattern, text))


def split_page(text, page_number, document):
    text = normalize_text(text)

    section_positions = detect_sections(text)

    results = []

    if not section_positions:
        return [{
            "document": document,
            "page": page_number,
            "section": None,
            "subsection": None,
            "text": text
        }]

    for i, section in enumerate(section_positions):

        start = section["start"]

        if i + 1 < len(section_positions):
            end = section_positions[i + 1]["start"]
        else:
            end = len(text)

        section_text = text[start:end].strip()

        results.append({
            "document": document,
            "page": page_number,
            "section": section["name"],
            "subsection": None,
            "text": section_text
        })

    return results

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def main():

    os.makedirs("data/processed", exist_ok=True)

    all_chunks = []
    chunk_id = 0

    pdf_files = [
        f for f in os.listdir(RAW_DIR)
        if f.lower().endswith(".pdf")
    ]

    print(f"Found PDFs: {len(pdf_files)}")

    for pdf_file in pdf_files:

        pdf_path = os.path.join(RAW_DIR, pdf_file)

        print(f"\nProcessing: {pdf_file}")

        doc = pymupdf.open(pdf_path)

        for page_index, page in enumerate(doc):

            page_number = page_index + 1

            text = page.get_text("text")

            sections = split_page(
                text,
                page_number,
                pdf_file
            )

            for section in sections:

                pieces = chunk_text(section["text"])

                for piece in pieces:

                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "document": section["document"],
                        "page": section["page"],
                        "section": section["section"],
                        "subsection": section["subsection"],
                        "text": piece
                    })

                    chunk_id += 1

        doc.close()

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print("\n" + "=" * 60)
    print(f"Total PDFs   : {len(pdf_files)}")
    print(f"Total chunks : {len(all_chunks)}")
    print(f"Saved to     : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()