# ============================================================
# KrishiJal AI - Evidence Grounded Answer Generation
# ============================================================

import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "openai/gpt-oss-120b"


# ============================================================
# KNOWLEDGE GAP MESSAGE
# ============================================================

KNOWLEDGE_GAP_MESSAGE = (
    "I don't have enough relevant information in the "
    "provided documents to answer this."
)


# ============================================================
# HUGGING FACE CLIENT
# ============================================================

if HF_TOKEN:

    client = InferenceClient(
        api_key=HF_TOKEN
    )

else:

    client = None


# ============================================================
# KNOWLEDGE GAP DETECTION
# ============================================================

def is_knowledge_gap(answer):
    """
    Detect a complete knowledge gap.

    A partial answer such as:
    "The exact interval is not documented, but the
    documents recommend drip irrigation..."

    is NOT treated as a complete knowledge gap.
    """

    if not answer:

        return True

    text = answer.lower().strip()


    complete_gap_phrases = [

        "i don't have enough relevant information in the "
        "provided documents to answer this.",

        "i don't have enough relevant information",

        "i don't have enough information in the provided "
        "documents to answer this.",

        "i don't have enough information in the provided "
        "documents",

        "the provided documents do not contain enough "
        "information",

        "the provided documents do not contain relevant "
        "information",

        "no relevant information is provided",

        "there is no relevant information in the provided "
        "documents"
    ]


    for phrase in complete_gap_phrases:

        if phrase in text:

            return True


    return False


# ============================================================
# EVIDENCE PREPARATION
# ============================================================

def prepare_evidence(results):
    """
    Convert retrieved chunks into clean evidence for the LLM.
    """

    if not results:

        return ""


    evidence_blocks = []


    for i, result in enumerate(
        results,
        start=1
    ):

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
        ).strip()


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
# LANGUAGE INSTRUCTION
# ============================================================

def get_language_instruction(language):

    if language == "Tamil":

        return """
Respond in simple, natural Tamil that farmers can
understand easily.

Use important English agricultural terms in parentheses
where useful.

For example:
துளி பாசனம் (Drip irrigation)
மண் ஈரப்பதம் (Soil moisture)
"""

    return """
Respond in simple, clear English suitable for farmers.
Avoid unnecessary technical terminology.
"""


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(language):

    language_instruction = get_language_instruction(
        language
    )


    return f"""
You are KrishiJal AI, an evidence-grounded
agricultural irrigation and crop water-management
assistant.

{language_instruction}

You MUST use ONLY the supplied evidence.

============================================================
PRIMARY OBJECTIVE
============================================================

Understand exactly what the farmer is asking and give
the most useful answer supported by the documents.

The answer should be specific to:

- crop
- growth stage
- irrigation method
- water availability
- farmer intent

when those details are available.

============================================================
EVIDENCE PRIORITY
============================================================

When several documents are supplied:

1. Prefer evidence that is specific to the farmer's crop.

2. Prefer evidence that matches the farmer's growth stage.

3. Prefer evidence that matches the farmer's water condition.

4. Prefer evidence that directly answers the farmer's
   question.

5. Use generic agricultural guidance only when it directly
   helps answer the question.

6. Do NOT add unrelated information merely because another
   document contains the word "irrigation" or "water".

============================================================
STRICT FACTUAL RULES
============================================================

1. Use ONLY supplied evidence.

2. Never use outside agricultural knowledge.

3. Never invent facts.

4. Never invent:
   - irrigation intervals
   - number of days
   - water quantities
   - doses
   - timings
   - frequencies
   - percentages
   - crop recommendations

5. If a requested exact value is absent, explicitly say
   that the exact value is not documented.

6. When an exact value is missing, still answer any other
   parts of the question that ARE supported.

7. Do not turn a partial knowledge gap into a complete
   knowledge gap.

============================================================
MULTI-PART QUESTIONS
============================================================

Many farmer questions contain multiple requests.

Example:

"What irrigation method should I use, what should I avoid,
and when should I prioritize watering?"

The response MUST address each supported part separately.

Use:

**What to use**
- ...

**What to avoid**
- ...

**When to prioritize**
- ...

Do not omit a supported part.

============================================================
EXACT-VALUE QUESTIONS
============================================================

If the farmer asks:

"What is the exact irrigation interval in days?"

and the evidence does not provide the exact number:

Say:

"The documents do not specify an exact irrigation
interval in days."

Then provide any related information that IS documented.

Do NOT invent a number.

============================================================
IMPORTANT DISTINCTION
============================================================

These two situations are different.

Situation A:

The evidence is unrelated.

Then respond:

"I don't have enough relevant information in the provided
documents to answer this."

Situation B:

The evidence is related, but one requested detail is missing.

Then answer the supported information and clearly say
which requested detail is not documented.

NEVER use the complete knowledge-gap response for
Situation B.

============================================================
ANSWER STYLE
============================================================

Your answer must be:

- Direct
- Concise
- Farmer-friendly
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
- vector databases
- prompts
- models
- retrieval

Do not create fake citations.

Return ONLY the farmer-facing answer.
"""


# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(
    question,
    evidence
):

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

Follow these rules:

1. Understand the exact request.

2. Answer every part that is supported.

3. If an exact requested value is missing, explicitly
   state that it is not documented.

4. Continue answering the other supported parts.

5. Prefer crop-specific and stage-specific evidence.

6. Ignore unrelated evidence.

7. Never invent information.

8. Keep the response concise and practical.

Return only the final farmer-facing answer.
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    language="English"
):

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not results:

        return KNOWLEDGE_GAP_MESSAGE


    # --------------------------------------------------------
    # HF token check
    # --------------------------------------------------------

    if client is None:

        return (
            "The language model is not configured. "
            "Please set HF_TOKEN before starting KrishiJal AI."
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
    # Build prompts
    # --------------------------------------------------------

    system_prompt = build_system_prompt(
        language
    )


    user_prompt = build_user_prompt(
        question,
        evidence
    )


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.1,

            max_tokens=450
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        if not answer:

            return KNOWLEDGE_GAP_MESSAGE


        answer = answer.strip()


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
# VERIFIED SOURCES
# ============================================================

def get_verified_sources(results):
    """
    Return unique document/page/section combinations.
    """

    if not results:

        return []


    sources = []

    seen = set()


    for result in results:

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
                "subsection": subsection
            }
        )


    return sources