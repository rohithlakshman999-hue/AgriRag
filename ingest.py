# ============================================================
# KrishiJal AI - PDF Ingestion Pipeline
# ============================================================
#
# Extracts:
#   - document
#   - page
#   - section
#   - subsection
#   - clean text
#
# Then creates overlapping chunks for RAG retrieval.
# ============================================================

import os
import pickle
import re

import pymupdf


# ============================================================
# PATHS
# ============================================================

RAW_DIR = "data/raw"

OUTPUT_FILE = "data/processed/chunks.pkl"


# ============================================================
# CHUNK SETTINGS
# ============================================================

CHUNK_SIZE = 350

CHUNK_OVERLAP = 70


# ============================================================
# TEXT CLEANING
# ============================================================

def normalize_text(text):
    """
    Clean extracted PDF text while preserving readable content.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces around lines
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def clean_chunk_text(text):
    """
    Final cleanup for chunk text.
    """

    if not text:
        return ""

    text = normalize_text(text)

    # Convert repeated whitespace/newlines into spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SECTION DETECTION
# ============================================================

# Common patterns found in government/agriculture PDFs.
#
# Examples:
#   1. Introduction
#   2. Objectives
#   3. Implementation
#   10.0 State Level Committee
#   1.1 Background
#   2.3 Micro Irrigation
#
# We deliberately don't hard-code PMFBY sections.
# ============================================================

MAIN_SECTION_PATTERNS = [

    re.compile(
        r"^\s*(\d+)\.\s+([A-Z][^\n]{2,150})$"
    ),

    re.compile(
        r"^\s*(\d+)\.0\s+([A-Z][^\n]{2,150})$"
    ),

]


SUBSECTION_PATTERNS = [

    re.compile(
        r"^\s*(\d+\.\d+)\s+(.+)$"
    ),

    re.compile(
        r"^\s*(\d+\.\d+\.\d+)\s+(.+)$"
    ),

    re.compile(
        r"^\s*(\d+\.\d+\.\d+\.\d+)\s+(.+)$"
    ),

]


def looks_like_heading(text):
    """
    Determine whether a line is likely to be a heading.

    This avoids treating normal numbered sentences,
    tables, years, percentages, etc. as headings.
    """

    text = text.strip()

    if not text:
        return False

    # Very long lines are unlikely to be headings
    if len(text) > 180:
        return False

    words = text.split()

    if len(words) > 25:
        return False

    # Ignore obvious table-like lines
    if text.count("|") >= 2:
        return False

    # Ignore lines ending with punctuation
    if text.endswith(
        (".", ",", ";", ":", "?", "!")
    ):
        return False

    return True


def detect_heading(line):
    """
    Detect main section or subsection heading.

    Returns:
        {
            "type": "section" / "subsection",
            "number": ...,
            "name": ...
        }

    or None.
    """

    line = line.strip()

    if not line:
        return None

    if not looks_like_heading(line):
        return None


    # ========================================================
    # Main section
    # ========================================================

    for pattern in MAIN_SECTION_PATTERNS:

        match = pattern.match(line)

        if match:

            number = match.group(1)

            title = match.group(2).strip()

            return {
                "type": "section",
                "number": number,
                "name": f"{number}. {title}"
            }


    # ========================================================
    # Subsection
    # ========================================================

    for pattern in SUBSECTION_PATTERNS:

        match = pattern.match(line)

        if match:

            number = match.group(1)

            title = match.group(2).strip()

            return {
                "type": "subsection",
                "number": number,
                "name": f"{number} {title}"
            }


    return None


# ============================================================
# PAGE SEGMENTATION
# ============================================================

def split_page_into_segments(
    text,
    page_number,
    document,
    current_section=None,
    current_subsection=None
):
    """
    Split one PDF page into semantic sections/subsections.

    Context is carried forward from previous pages.
    """

    text = normalize_text(text)

    if not text:
        return [], current_section, current_subsection


    lines = text.split("\n")


    segments = []

    current_lines = []

    active_section = current_section

    active_subsection = current_subsection


    def flush_segment():

        nonlocal current_lines

        if not current_lines:
            return

        segment_text = clean_chunk_text(
            " ".join(current_lines)
        )

        if segment_text:

            segments.append(
                {
                    "document": document,

                    "page": page_number,

                    "section": active_section,

                    "subsection": active_subsection,

                    "text": segment_text
                }
            )

        current_lines = []


    for line in lines:

        line = line.strip()

        if not line:
            continue


        heading = detect_heading(
            line
        )


        if heading:

            # ----------------------------------------------
            # New main section
            # ----------------------------------------------

            if heading["type"] == "section":

                flush_segment()

                active_section = (
                    heading["name"]
                )

                active_subsection = None

                continue


            # ----------------------------------------------
            # New subsection
            # ----------------------------------------------

            if heading["type"] == "subsection":

                # Only treat subsection as a real
                # subsection when it belongs to the
                # currently active section.

                flush_segment()

                active_subsection = (
                    heading["name"]
                )

                continue


        current_lines.append(
            line
        )


    flush_segment()


    # --------------------------------------------------------
    # If the page produced no segments, keep whole page
    # --------------------------------------------------------

    if not segments:

        cleaned = clean_chunk_text(
            text
        )

        if cleaned:

            segments.append(
                {
                    "document": document,

                    "page": page_number,

                    "section": active_section,

                    "subsection": active_subsection,

                    "text": cleaned
                }
            )


    return (
        segments,
        active_section,
        active_subsection
    )


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split text into overlapping word chunks.
    """

    text = clean_chunk_text(
        text
    )

    if not text:
        return []

    words = text.split()


    if len(words) <= chunk_size:

        return [text]


    chunks = []

    start = 0


    while start < len(words):

        end = min(
            start + chunk_size,
            len(words)
        )


        chunk = " ".join(
            words[start:end]
        )


        if chunk.strip():

            chunks.append(
                chunk.strip()
            )


        if end >= len(words):

            break


        start = end - overlap


    return chunks


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path,
    document
):
    """
    Extract and chunk a single PDF.
    """

    print(
        f"\nProcessing: {document}"
    )


    doc = pymupdf.open(
        pdf_path
    )


    document_chunks = []


    current_section = None

    current_subsection = None


    for page_index, page in enumerate(
        doc
    ):

        page_number = page_index + 1


        try:

            raw_text = page.get_text(
                "text"
            )

        except Exception as exc:

            print(
                f"Warning: Could not extract "
                f"page {page_number}: {exc}"
            )

            continue


        if not raw_text.strip():

            continue


        segments, current_section, current_subsection = (
            split_page_into_segments(
                raw_text,
                page_number,
                document,
                current_section,
                current_subsection
            )
        )


        for segment in segments:

            pieces = chunk_text(
                segment["text"]
            )


            for piece in pieces:

                document_chunks.append(
                    {
                        "document": document,

                        "page": segment["page"],

                        "section": segment[
                            "section"
                        ],

                        "subsection": segment[
                            "subsection"
                        ],

                        "text": piece
                    }
                )


    doc.close()


    return document_chunks


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    print(
        "=" * 70
    )

    print(
        "KRISHIJAL AI - PDF INGESTION"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # FIND PDF FILES
    # ========================================================

    if not os.path.exists(
        RAW_DIR
    ):

        raise FileNotFoundError(
            f"Raw directory not found: {RAW_DIR}"
        )


    pdf_files = sorted(
        [
            file
            for file in os.listdir(RAW_DIR)
            if file.lower().endswith(".pdf")
        ]
    )


    print(
        f"\nFound PDFs: {len(pdf_files)}"
    )


    if not pdf_files:

        raise RuntimeError(
            "No PDF files found in data/raw."
        )


    # ========================================================
    # PROCESS ALL PDFs
    # ========================================================

    all_chunks = []


    for pdf_file in pdf_files:

        pdf_path = os.path.join(
            RAW_DIR,
            pdf_file
        )


        try:

            chunks = process_pdf(
                pdf_path,
                pdf_file
            )


            print(
                f"  Chunks created: "
                f"{len(chunks)}"
            )


            all_chunks.extend(
                chunks
            )


        except Exception as exc:

            print(
                f"\nERROR processing "
                f"{pdf_file}:"
            )

            print(
                exc
            )


    # ========================================================
    # ASSIGN CHUNK IDS
    # ========================================================

    final_chunks = []


    for chunk_id, chunk in enumerate(
        all_chunks
    ):

        chunk["chunk_id"] = chunk_id

        final_chunks.append(
            chunk
        )


    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_FILE,
        "wb"
    ) as f:

        pickle.dump(
            final_chunks,
            f
        )


    # ========================================================
    # STATISTICS
    # ========================================================

    documents = sorted(
        set(
            chunk["document"]
            for chunk in final_chunks
        )
    )


    sections = sorted(
        set(
            chunk["section"]
            for chunk in final_chunks
            if chunk.get("section")
        )
    )


    chunks_with_sections = sum(
        1
        for chunk in final_chunks
        if chunk.get("section")
    )


    chunks_with_subsections = sum(
        1
        for chunk in final_chunks
        if chunk.get("subsection")
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Total PDFs             : "
        f"{len(pdf_files)}"
    )

    print(
        f"Total chunks           : "
        f"{len(final_chunks)}"
    )

    print(
        f"Chunks with sections   : "
        f"{chunks_with_sections}"
    )

    print(
        f"Chunks with subsections: "
        f"{chunks_with_subsections}"
    )

    print(
        f"Saved to               : "
        f"{OUTPUT_FILE}"
    )


    print(
        "\nDocuments:"
    )

    for document in documents:

        count = sum(
            1
            for chunk in final_chunks
            if chunk["document"] == document
        )

        print(
            f"  - {document}: "
            f"{count} chunks"
        )


    print(
        "\nDetected sections:"
    )

    for section in sections:

        print(
            f"  - {section}"
        )


    print(
        "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()