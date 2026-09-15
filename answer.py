import os

from huggingface_hub import InferenceClient

from rerank_search import search


# =========================================================
# HUGGING FACE CONFIGURATION
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "openai/gpt-oss-120b"


if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set.")
    print('Run: $env:HF_TOKEN="YOUR_TOKEN"')
    raise SystemExit


client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# =========================================================
# GENERATE GROUNDED ANSWER
# =========================================================

def generate_answer(question, results):

    evidence = ""

    for i, result in enumerate(results, 1):

        evidence += f"""
SOURCE {i}
Document: {result["document"]}
Page: {result["page"]}
Section: {result["section"]}
Subsection: {result["subsection"]}

Content:
{result["text"]}

----------------------------------------
"""

    system_prompt = """
You are a document-grounded knowledge assistant.

Answer ONLY using the supplied evidence.

Rules:
- Do not use outside knowledge.
- Do not invent facts.
- Do not guess.
- Give a clear and concise answer.
- Use only information supported by the evidence.
- Do not create citation numbers.
- Do not create references such as [1], [2], [Source 1], or 【1†...】.
- Do not mention information that is not supported by the evidence.
- Clearly distinguish covered items from exclusions when relevant.
- If the supplied evidence does not contain enough information to answer
  the question, say exactly:

"I don't have enough information in the provided documents to answer this."
"""

    user_prompt = f"""
QUESTION:
{question}

EVIDENCE:
{evidence}

Answer the question using ONLY the evidence above.
"""

    completion = client.chat.completions.create(
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
        max_tokens=800,
        temperature=0.1
    )

    return completion.choices[0].message.content


# =========================================================
# PRINT VERIFIED SOURCES
# =========================================================

def print_sources(results):

    print("\n" + "=" * 70)
    print("VERIFIED SOURCES")
    print("=" * 70)

    seen = set()
    source_number = 1

    for result in results:

        key = (
            result["document"],
            result["page"],
            result["section"]
        )

        if key in seen:
            continue

        seen.add(key)

        print(
            f"[{source_number}] "
            f"Page {result['page']} | "
            f"{result['section']} | "
            f"{result['document']}"
        )

        source_number += 1


# =========================================================
# INTERACTIVE APPLICATION
# =========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DOCUMENT-GROUNDED RAG ASSISTANT")
    print("=" * 70)

    while True:

        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() in ["exit", "quit"]:
            print("\nExiting...")
            break

        if not question:
            continue

        print("\nRetrieving evidence...")

        results = search(
            question,
            candidate_k=20,
            final_k=5
        )

        print("\nGenerating grounded answer...")

        try:

            answer = generate_answer(
                question,
                results
            )

        except Exception as e:

            print("\nERROR while generating answer:")
            print(e)
            continue

        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)

        print(answer)

        # -------------------------------------------------
        # Only show sources when the assistant actually
        # found evidence worth displaying.
        # -------------------------------------------------

        if results:

            print_sources(results)

        else:

            print("\nNo verified sources found.")