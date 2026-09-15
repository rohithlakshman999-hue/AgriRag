# ============================================================
# KrishiJal AI / AgriRAG
# Evidence-Grounded Agricultural Answer Generation
# ============================================================

import os

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()


# ------------------------------------------------------------
# Helper: safely read Streamlit Cloud secrets
# ------------------------------------------------------------

def get_secret(name, default=None):
    """
    Read a value from Streamlit Cloud Secrets.

    Falls back to environment variables for local execution.
    """

    try:
        value = st.secrets.get(name)

        if value is not None and str(value).strip():
            return str(value).strip()

    except Exception:
        pass

    value = os.getenv(name)

    if value is not None and str(value).strip():
        return str(value).strip()

    return default


HF_TOKEN = get_secret(
    "HF_TOKEN"
)

MODEL = get_secret(
    "HF_MODEL",
    "openai/gpt-oss-120b"
)


# ============================================================
# 2. KNOWLEDGE GAP MESSAGE
# ============================================================

KNOWLEDGE_GAP_MESSAGE = (
    "I don't have enough relevant information in the "
    "provided documents to answer this."
)


# ============================================================
# 3. HUGGING FACE CLIENT
# ============================================================

client = None

if HF_TOKEN:

    try:

        client = InferenceClient(
            api_key=HF_TOKEN
        )

    except Exception as exc:

        print(
            "Failed to initialize Hugging Face client:",
            repr(exc)
        )

        client = None


# ============================================================
# 4. KNOWLEDGE-GAP DETECTION
# ============================================================

def is_knowledge_gap(answer):
    """
    Detect whether the generated response is a complete
    knowledge-gap response.

    A partial answer is NOT considered a complete gap.

    Example:

    "The exact interval is not documented, but the documents
    recommend drip irrigation..."

    This should return False.
    """

    if not answer:
        return True

    text = str(answer).lower().strip()

    complete_gap_phrases = [

        (
            "i don't have enough relevant information in the "
            "provided documents to answer this."
        ),

        "i don't have enough relevant information",

        (
            "i don't have enough information in the provided "
            "documents to answer this."
        ),

        (
            "i don't have enough information in the provided "
            "documents"
        ),

        (
            "the provided documents do not contain enough "
            "information"
        ),

        (
            "the provided documents do not contain relevant "
            "information"
        ),

        "no relevant information is provided",

        (
            "there is no relevant information in the provided "
            "documents"
        ),

        "insufficient evidence in the available documents",

        "insufficient evidence",

    ]

    for phrase in complete_gap_phrases:

        if phrase in text:

            return True

    return False


# ============================================================
# 5. EVIDENCE PREPARATION
# ============================================================

def prepare_evidence(results):
    """
    Convert retrieved results into a clean evidence block
    for the language model.
    """

    if not results:
        return ""

    evidence_blocks = []

    for i, result in enumerate(
        results,
        start=1
    ):

        if not isinstance(result, dict):
            continue

        document = result.get(
            "document",
            "Unknown document"
        )

        page = result.get(
            "page",
            "Unknown"
        )

        section = result.get(
            "section"
        ) or "General content"

        subsection = result.get(
            "subsection"
        )

        text = result.get(
            "text",
            ""
        )

        if text is None:
            text = ""

        text = str(text).strip()

        if not text:
            continue

        location = (
            f"Document: {document}\n"
            f"Page: {page}\n"
            f"Section: {section}"
        )

        if subsection:
            location += (
                f"\nSubsection: {subsection}"
            )

        evidence_blocks.append(
            f"""
EVIDENCE {i}

{location}

CONTENT:
{text}
"""
        )

    return "\n".join(
        evidence_blocks
    )


# ============================================================
# 6. LANGUAGE INSTRUCTIONS
# ============================================================

def get_language_instruction(language):
    """
    Return the response-language instruction.
    """

    if language == "Tamil":

        return """
Respond in simple, natural Tamil that farmers can
understand easily.

Use important agricultural English terms in parentheses
where useful.

Examples:

துளி பாசனம் (Drip irrigation)

மண் ஈரப்பதம் (Soil moisture)

நீர்ப்பாசன அட்டவணை (Irrigation scheduling)

நீர் பயன்பாட்டு திறன் (Water-use efficiency)
"""

    return """
Respond in simple, clear English suitable for farmers.

Avoid unnecessary technical terminology.

Prefer short paragraphs and clear practical wording.
"""


# ============================================================
# 7. SYSTEM PROMPT
# ============================================================

def build_system_prompt(language):

    language_instruction = (
        get_language_instruction(
            language
        )
    )

    return f"""
You are AgriRAG, an evidence-grounded agricultural
irrigation and crop water-management assistant.

{language_instruction}

============================================================
CORE RULE
============================================================

Use ONLY the supplied agricultural evidence.

Do NOT use outside agricultural knowledge.

Do NOT invent facts.

============================================================
WHAT THE ANSWER SHOULD DO
============================================================

Understand exactly what the farmer is asking.

Provide the most useful answer that is directly supported
by the supplied evidence.

When available, consider:

- crop
- growth stage
- irrigation method
- water availability
- farmer intent

Prefer evidence that is:

1. Specific to the crop.
2. Specific to the growth stage.
3. Relevant to the water condition.
4. Directly related to the question.

Ignore unrelated information.

============================================================
STRICT FACTUAL RULES
============================================================

Never invent:

- irrigation intervals
- number of days
- water quantities
- litres
- doses
- timings
- frequencies
- percentages
- crop recommendations

If the farmer asks for an exact value and the evidence
does not contain that value, clearly state:

"The documents do not specify the exact value."

Then continue answering other parts that ARE supported.

============================================================
PARTIAL KNOWLEDGE GAPS
============================================================

There are two different situations.

Situation A:
The supplied evidence does not meaningfully answer
the question.

Then say:

"I don't have enough relevant information in the provided
documents to answer this."

Situation B:
The evidence answers part of the question but does not
contain one requested detail.

In this case:

1. Answer the supported part.
2. Clearly identify the missing detail.
3. Do NOT reject the entire question.

============================================================
MULTI-PART QUESTIONS
============================================================

If the farmer asks multiple things, answer every supported
part.

Example:

"What should I use, what should I avoid, and when should
I prioritize irrigation?"

Use:

**What to use**
- ...

**What to avoid**
- ...

**When to prioritize**
- ...

Do not ignore supported parts.

============================================================
ANSWER STYLE
============================================================

The answer must be:

- Direct
- Concise
- Farmer-friendly
- Practical
- Specific
- Evidence-grounded

Do not repeat the entire evidence.

Do not discuss the internal AI system.

Do not mention:

- RAG
- FAISS
- BM25
- embeddings
- reranking
- vector database
- prompts
- models
- retrieval

Do not create fake citations.

Return ONLY the farmer-facing answer.
"""


# ============================================================
# 8. USER PROMPT
# ============================================================

def build_user_prompt(
    question,
    evidence
):
    """
    Build the user message containing the farmer question
    and retrieved evidence.
    """

    return f"""
FARMER QUESTION
===============

{question}


SUPPLIED AGRICULTURAL EVIDENCE
==============================

{evidence}


TASK
====

Answer the farmer's question using ONLY the supplied
evidence.

Requirements:

1. Understand the exact request.
2. Answer every supported part.
3. Prefer crop-specific and stage-specific evidence.
4. If an exact requested value is missing, say it is
   not documented.
5. Continue answering any other supported parts.
6. Never invent information.
7. Keep the answer concise and practical.
8. Do not discuss the internal AI system.

Return only the final farmer-facing answer.
"""


# ============================================================
# 9. GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    language="English"
):
    """
    Generate an evidence-grounded farmer-facing answer.
    """

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        return KNOWLEDGE_GAP_MESSAGE


    # --------------------------------------------------------
    # Check Hugging Face configuration
    # --------------------------------------------------------

    if client is None:

        return (
            "The language model is not configured.\n\n"
            "Please add HF_TOKEN to Streamlit Secrets "
            "before using AgriRAG."
        )


    # --------------------------------------------------------
    # Prepare evidence
    # --------------------------------------------------------

    evidence = prepare_evidence(
        results
    )

    if not evidence.strip():

        return KNOWLEDGE_GAP_MESSAGE


    # --------------------------------------------------------
    # Prompts
    # --------------------------------------------------------

    system_prompt = build_system_prompt(
        language
    )

    user_prompt = build_user_prompt(
        question,
        evidence
    )


    # --------------------------------------------------------
    # Call Hugging Face
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            temperature=0.1,

            max_tokens=500,
        )


        # ----------------------------------------------------
        # Extract response
        # ----------------------------------------------------

        answer = (
            response
            .choices[0]
            .message
            .content
        )


        if not answer:

            return KNOWLEDGE_GAP_MESSAGE


        answer = str(
            answer
        ).strip()


        # ----------------------------------------------------
        # Remove accidental surrounding quotes
        # ----------------------------------------------------

        if (
            len(answer) >= 2
            and answer.startswith('"')
            and answer.endswith('"')
        ):

            answer = answer[
                1:-1
            ].strip()


        return answer


    except Exception as exc:

        print(
            "LLM generation error:",
            repr(exc)
        )

        return (
            "The answer could not be generated because "
            "the language model is currently unavailable."
        )


# ============================================================
# 10. VERIFIED SOURCE EXTRACTION
# ============================================================

def get_verified_sources(results):
    """
    Return unique document/page/section combinations.

    This information is used by the Streamlit UI.
    """

    if not results:
        return []

    sources = []

    seen = set()

    for result in results:

        if not isinstance(result, dict):
            continue

        document = result.get(
            "document",
            "Unknown document"
        )

        page = result.get(
            "page",
            "Unknown"
        )

        section = result.get(
            "section"
        ) or "General content"

        subsection = result.get(
            "subsection"
        )

        key = (
            document,
            page,
            section,
            subsection
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        sources.append(
            {
                "document": document,
                "page": page,
                "section": section,
                "subsection": subsection,
            }
        )

    return sources