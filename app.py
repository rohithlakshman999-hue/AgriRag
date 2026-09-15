import streamlit as st

from answer import generate_answer
from rerank_search import search


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="KrishiJal AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .hero {
        padding: 10px 0 25px 0;
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 18px;
        margin-top: 6px;
        color: #9ca3af;
    }

    /* Context card */
    .context-card {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 18px;
        background: rgba(255,255,255,0.03);
    }

    /* Source card */
    .source-card {
        border: 1px solid rgba(100,200,140,0.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0;
        background: rgba(100,200,140,0.06);
    }

    .source-title {
        font-weight: 700;
        font-size: 15px;
    }

    .source-meta {
        color: #aeb7c2;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Feature cards */
    .feature-card {
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        background: rgba(255,255,255,0.025);
    }

    .feature-title {
        font-size: 15px;
        font-weight: 700;
    }

    .feature-text {
        color: #9ca3af;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Status badge */
    .status {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 20px;
        background: rgba(80,200,120,0.12);
        color: #70d990;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="status">● AI AGRICULTURAL KNOWLEDGE ASSISTANT</div>

        <div class="hero-title">
            🌱 KrishiJal AI
        </div>

        <div class="hero-subtitle">
            Evidence-grounded irrigation and crop water management assistant
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌾 Farmer Profile")

    crop = st.selectbox(
        "Crop",
        [
            "Not specified",
            "Rice",
            "Wheat",
            "Maize",
            "Sugarcane",
            "Cotton",
            "Groundnut",
            "Vegetables",
            "Other"
        ]
    )

    growth_stage = st.selectbox(
        "Growth Stage",
        [
            "Not specified",
            "Seedling",
            "Vegetative",
            "Flowering",
            "Grain filling",
            "Maturity"
        ]
    )

    irrigation_method = st.selectbox(
        "Irrigation Method",
        [
            "Not specified",
            "Flood irrigation",
            "Drip irrigation",
            "Sprinkler irrigation",
            "Furrow irrigation",
            "Other"
        ]
    )

    water_availability = st.selectbox(
        "Water Availability",
        [
            "Not specified",
            "Abundant",
            "Moderate",
            "Limited"
        ]
    )

    language = st.selectbox(
        "Language",
        [
            "English"
        ]
    )

    st.divider()

    st.markdown("### 🧠 AI Pipeline")

    st.write("✅ Semantic Retrieval")
    st.write("✅ BM25 Retrieval")
    st.write("✅ Hybrid RRF")
    st.write("✅ BGE Reranking")
    st.write("✅ Grounded LLM")
    st.write("✅ Source Verification")

    st.divider()

    st.caption(
        "Answers are generated only from the indexed agricultural documents."
    )


# =========================================================
# FARMER CONTEXT
# =========================================================

context = f"""
Crop: {crop}
Growth Stage: {growth_stage}
Irrigation Method: {irrigation_method}
Water Availability: {water_availability}
Language: {language}
"""


# =========================================================
# FEATURE CARDS
# =========================================================

st.markdown("### 💡 Ask KrishiJal")

col1, col2, col3 = st.columns(3)

selected_question = None


with col1:

    if st.button(
        "💧 Irrigation Scheduling",
        use_container_width=True
    ):

        selected_question = (
            "What are the principles of irrigation scheduling?"
        )


with col2:

    if st.button(
        "🌾 Crop Water Management",
        use_container_width=True
    ):

        selected_question = (
            f"What water management practices are recommended "
            f"for {crop}?"
        )


with col3:

    if st.button(
        "♻️ Water Saving",
        use_container_width=True
    ):

        selected_question = (
            "What practices can improve irrigation water-use efficiency?"
        )


# =========================================================
# CHAT INPUT
# =========================================================

chat_question = st.chat_input(
    "Ask an irrigation or crop water-management question..."
)

if chat_question:

    selected_question = chat_question


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# PROCESS QUESTION
# =========================================================

if selected_question:

    full_question = f"""
FARMER CONTEXT

{context}

QUESTION

{selected_question}
"""


    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": selected_question
        }
    )


    with st.chat_message("user"):

        st.markdown(
            selected_question
        )


        with st.expander(
            "🌾 Farmer context"
        ):

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Crop:** {crop}"
                )

                st.write(
                    f"**Growth stage:** {growth_stage}"
                )

            with col2:
                st.write(
                    f"**Irrigation:** {irrigation_method}"
                )

                st.write(
                    f"**Water:** {water_availability}"
                )


    # -----------------------------------------------------
    # ASSISTANT
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🔎 Retrieving agricultural evidence..."
            ):

                results = search(
                    full_question,
                    candidate_k=20,
                    final_k=5
                )


            # -------------------------------------------------
            # NO EVIDENCE
            # -------------------------------------------------

            if not results:

                answer = (
                    "I don't have enough information "
                    "in the provided documents to answer this."
                )

            else:

                with st.spinner(
                    "🤖 Generating grounded answer..."
                ):

                    answer = generate_answer(
                        full_question,
                        results
                    )


            # -------------------------------------------------
            # ANSWER
            # -------------------------------------------------

            st.markdown(
                answer
            )


            # -------------------------------------------------
            # KNOWLEDGE GAP DETECTION
            # -------------------------------------------------

            knowledge_gap = (
                "I don't have enough information"
                in answer
            )


            # -------------------------------------------------
            # VERIFIED SOURCES
            # -------------------------------------------------

            if results and not knowledge_gap:

                st.markdown(
                    "### 📚 Verified Sources"
                )

                seen = set()

                displayed_sources = 0

                for result in results:

                    key = (
                        result["document"],
                        result["page"],
                        result["section"]
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    displayed_sources += 1

                    document = result["document"]

                    page = result["page"]

                    section = result["section"]

                    subsection = result["subsection"]


                    if not section:

                        section = "General content"


                    subsection_text = ""

                    if subsection:

                        subsection_text = (
                            f" • {subsection}"
                        )


                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-title">
                                📄 {document}
                            </div>

                            <div class="source-meta">
                                Page {page}
                                • {section}
                                {subsection_text}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                if displayed_sources == 0:

                    st.caption(
                        "No verified source was available for this answer."
                    )


            # -------------------------------------------------
            # RETRIEVED EVIDENCE
            # -------------------------------------------------

            if results:

                with st.expander(
                    "🔎 View retrieved evidence"
                ):

                    for i, result in enumerate(
                        results,
                        start=1
                    ):

                        st.markdown(
                            f"**Evidence {i}**"
                        )

                        st.caption(
                            f"Page {result['page']} | "
                            f"{result['section'] or 'General content'} | "
                            f"Score: "
                            f"{result.get('final_score', 0):.3f}"
                        )

                        st.write(
                            result["text"]
                        )

                        st.divider()


            # -------------------------------------------------
            # SAVE ASSISTANT MESSAGE
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )


        except Exception as e:

            st.error(
                f"Something went wrong: {e}"
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777; font-size:13px;">
        🌱 KrishiJal AI &nbsp;•&nbsp;
        Retrieval-Augmented Agricultural Intelligence &nbsp;•&nbsp;
        Evidence-grounded responses
    </div>
    """,
    unsafe_allow_html=True
)