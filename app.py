# ============================================================
# AgriRAG - Farmer-Friendly Agricultural RAG Assistant
# Streamlit Application
# ============================================================

import os
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriRAG",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. LOAD STREAMLIT SECRETS BEFORE OTHER PROJECT IMPORTS
# ============================================================
#
# IMPORTANT:
# answer.py reads HF_TOKEN when it is imported.
# Therefore, load Streamlit Cloud secrets into environment
# variables BEFORE importing answer.py.
#
# This works for both:
#   - Local .env
#   - Streamlit Cloud Secrets
# ============================================================

try:
    if "HF_TOKEN" in st.secrets:
        os.environ["HF_TOKEN"] = str(
            st.secrets["HF_TOKEN"]
        )

    if "HF_MODEL" in st.secrets:
        os.environ["HF_MODEL"] = str(
            st.secrets["HF_MODEL"]
        )

    if "EMBEDDING_MODEL" in st.secrets:
        os.environ["EMBEDDING_MODEL"] = str(
            st.secrets["EMBEDDING_MODEL"]
        )

except Exception:
    # Local execution may not have st.secrets configured.
    pass


# ============================================================
# 3. IMPORT PROJECT MODULES
# ============================================================

from query_context import build_context_query

from answer import (
    generate_answer,
    get_verified_sources,
    is_knowledge_gap,
    KNOWLEDGE_GAP_MESSAGE,
)

from search import search


# ============================================================
# 4. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 5rem;
    }

    body {
        font-family: sans-serif;
    }


    /* =====================================================
       BRAND
       ===================================================== */

    .brand-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.15rem;
    }

    .brand-subtitle {
        font-size: 1.05rem;
        color: #7d8793;
        margin-bottom: 1.2rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }


    /* =====================================================
       METRICS
       ===================================================== */

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(128, 128, 128, 0.03);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.76rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 600;
    }


    /* =====================================================
       CHAT
       ===================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;
        color: #7d8793;
        font-size: 0.78rem;
        padding-top: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    '<div class="brand-title">🌾 AgriRAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand-subtitle">'
    'Evidence-Grounded Irrigation Knowledge Assistant'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "🌱 Ask about irrigation, crop water management, "
    "water-saving practices, irrigation methods, and crop stages. "
    "AgriRAG answers from the indexed agricultural documents."
)


# ============================================================
# 7. SIDEBAR - FARMER PROFILE
# ============================================================

with st.sidebar:

    st.header("🌾 Farmer Profile")

    st.caption(
        "Select what you know. You can also simply mention "
        "these details in your question."
    )


    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    crop = st.selectbox(
        "🌱 Crop",
        [
            "Not specified",
            "Rice",
            "Tomato",
            "Wheat",
            "Maize",
            "Sugarcane",
            "Cotton",
            "Groundnut",
            "Vegetables",
            "Other",
        ],
    )


    # --------------------------------------------------------
    # Growth stage
    # --------------------------------------------------------

    growth_stage = st.selectbox(
        "🌿 Growth Stage",
        [
            "Not specified",
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruit-set",
            "Grain filling",
            "Maturity",
        ],
    )


    # --------------------------------------------------------
    # Current irrigation
    # --------------------------------------------------------

    irrigation_method = st.selectbox(
        "💧 Current Irrigation",
        [
            "Not specified",
            "Flood irrigation",
            "Drip irrigation",
            "Sprinkler irrigation",
            "Furrow irrigation",
            "Other",
        ],
    )


    # --------------------------------------------------------
    # Water availability
    # --------------------------------------------------------

    water_availability = st.selectbox(
        "🚰 Water Availability",
        [
            "Not specified",
            "Abundant",
            "Moderate",
            "Low",
            "Severely limited",
        ],
    )


    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language = st.selectbox(
        "🗣️ Response Language",
        [
            "English",
            "Tamil",
        ],
    )


    st.divider()


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.subheader("🧠 How AgriRAG Works")

    st.write("1️⃣ Understand farmer context")
    st.write("2️⃣ Semantic search")
    st.write("3️⃣ Keyword search")
    st.write("4️⃣ RRF fusion")
    st.write("5️⃣ Evidence reranking")
    st.write("6️⃣ Grounded answer")


    st.divider()

    st.caption(
        "AgriRAG does not invent unsupported "
        "agricultural information."
    )


# ============================================================
# 8. SIDEBAR CONTEXT
# ============================================================

sidebar_context = {
    "crop": crop,
    "growth_stage": growth_stage,
    "irrigation_method": irrigation_method,
    "water_availability": water_availability,
}


# ============================================================
# 9. QUICK QUESTIONS
# ============================================================

st.subheader("💡 Quick Questions")

q1, q2, q3 = st.columns(3)


with q1:

    if st.button(
        "💧 Irrigation Scheduling",
        use_container_width=True,
    ):

        st.session_state.pending_question = (
            "What are the principles of irrigation scheduling?"
        )


with q2:

    if st.button(
        "🌱 Crop Water Management",
        use_container_width=True,
    ):

        if crop == "Not specified":

            st.session_state.pending_question = (
                "What crop water management practices "
                "are discussed in the available documents?"
            )

        else:

            st.session_state.pending_question = (
                f"What crop water management practices "
                f"are discussed for {crop}?"
            )


with q3:

    if st.button(
        "♻️ Save Water",
        use_container_width=True,
    ):

        st.session_state.pending_question = (
            "What practices can improve irrigation "
            "water-use efficiency?"
        )


# ============================================================
# 10. CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 11. CHAT INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask your farming question here..."
)


if typed_question:

    st.session_state.pending_question = (
        typed_question
    )


# ============================================================
# 12. GET QUESTION
# ============================================================

question = (
    st.session_state.pending_question
)


# ============================================================
# 13. PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Prevent duplicate processing
    # --------------------------------------------------------

    st.session_state.pending_question = None


    # --------------------------------------------------------
    # Automatic context extraction
    # --------------------------------------------------------

    detected_context, retrieval_query = (
        build_context_query(
            question,
            sidebar_context,
        )
    )


    # --------------------------------------------------------
    # Store user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    # --------------------------------------------------------
    # Show user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            # =================================================
            # CONTEXT
            # =================================================

            st.subheader(
                "🧠 What AgriRAG understood"
            )

            st.caption(
                "Automatically detected from your question "
                "and combined with your farmer profile."
            )


            # -------------------------------------------------
            # Context row 1
            # -------------------------------------------------

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "🌾 Crop",
                    detected_context.get(
                        "crop"
                    )
                    or "Not specified",
                )


            with c2:

                st.metric(
                    "🌿 Growth Stage",
                    detected_context.get(
                        "growth_stage"
                    )
                    or "Not specified",
                )


            with c3:

                st.metric(
                    "🎯 Intent",
                    detected_context.get(
                        "intent"
                    )
                    or "General",
                )


            # -------------------------------------------------
            # Context row 2
            # -------------------------------------------------

            c4, c5 = st.columns(2)

            with c4:

                st.metric(
                    "💧 Irrigation Method",
                    detected_context.get(
                        "irrigation_method"
                    )
                    or "Not specified",
                )


            with c5:

                st.metric(
                    "🚰 Water Availability",
                    detected_context.get(
                        "water_availability"
                    )
                    or "Not specified",
                )


            st.divider()


            # =================================================
            # SEARCH
            # =================================================

            with st.spinner(
                "🔎 Finding the most relevant evidence..."
            ):

                results = search(
                    retrieval_query,
                    top_k=3,
                )


            # =================================================
            # NO EVIDENCE
            # =================================================

            if not results:

                st.warning(
                    "⚠️ I couldn't find sufficiently relevant "
                    "information in the indexed documents."
                )

                answer = (
                    KNOWLEDGE_GAP_MESSAGE
                )


                st.write(
                    answer
                )


            else:

                # =================================================
                # GENERATION
                # =================================================

                with st.spinner(
                    "🤖 Preparing your answer..."
                ):

                    answer = generate_answer(
                        retrieval_query,
                        results,
                        language=language,
                    )


                # =================================================
                # KNOWLEDGE GAP
                # =================================================

                knowledge_gap = (
                    is_knowledge_gap(
                        answer
                    )
                )


                # =================================================
                # ANSWER
                # =================================================

                st.subheader(
                    "🌱 AgriRAG Answer"
                )


                if knowledge_gap:

                    st.warning(
                        "⚠️ The available documents do not "
                        "fully document the requested information."
                    )


                st.markdown(
                    answer
                )


                # =================================================
                # VERIFIED SOURCES
                # =================================================

                sources = (
                    get_verified_sources(
                        results
                    )
                )


                if sources:

                    st.divider()

                    st.subheader(
                        "📚 Sources Used"
                    )

                    st.caption(
                        "Documents that provided the evidence "
                        "for this response."
                    )


                    for source in sources:

                        with st.container(
                            border=True
                        ):

                            st.markdown(
                                f"📄 **{source['document']}**"
                            )

                            metadata = (
                                f"Page {source['page']} "
                                f"• "
                                f"{source.get('section') or 'General content'}"
                            )


                            if source.get(
                                "subsection"
                            ):

                                metadata += (
                                    f" • "
                                    f"{source['subsection']}"
                                )


                            st.caption(
                                metadata
                            )


                # =================================================
                # SUPPORTING EVIDENCE
                # =================================================

                st.divider()

                with st.expander(
                    "🔎 View supporting evidence"
                ):

                    for i, result in enumerate(
                        results,
                        start=1,
                    ):

                        st.markdown(
                            f"### Evidence {i}"
                        )

                        st.caption(
                            f"{result.get('document', 'Unknown document')} "
                            f"• Page {result.get('page', 'Unknown')} "
                            f"• "
                            f"{result.get('section') or 'General content'}"
                        )

                        st.write(
                            result.get(
                                "text",
                                "",
                            )
                        )

                        if i < len(results):

                            st.divider()


                # =================================================
                # TECHNICAL DETAILS
                # =================================================

                with st.expander(
                    "⚙️ View detected context & search details"
                ):

                    st.markdown(
                        "#### Detected Context"
                    )

                    st.json(
                        detected_context
                    )


                    st.markdown(
                        "#### Retrieval Query"
                    )

                    st.code(
                        retrieval_query,
                        language="text",
                    )


                    st.write(
                        f"Retrieved passages: "
                        f"{len(results)}"
                    )


                    for i, result in enumerate(
                        results,
                        start=1,
                    ):

                        st.write(
                            f"{i}. "
                            f"{result.get('document', 'Unknown')} "
                            f"(Page {result.get('page', 'Unknown')})"
                        )


            # =================================================
            # STORE ASSISTANT RESPONSE
            # =================================================

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


        except Exception as exc:

            st.error(
                "AgriRAG encountered an error."
            )

            st.exception(
                exc
            )


# ============================================================
# 14. EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.divider()

    st.subheader(
        "👋 How can AgriRAG help?"
    )

    st.write(
        "Try asking a question such as:"
    )

    st.markdown(
        """
        **🌱 “My tomato crop is flowering and water availability
        is low. What irrigation method should I use?”**

        **💧 “How can I improve irrigation water-use efficiency?”**

        **📅 “What are the principles of irrigation scheduling?”**

        **🚰 “How can farmers save water using micro-irrigation?”**
        """
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🌾 <b>AgriRAG</b>
        &nbsp;•&nbsp;
        Evidence-grounded Agricultural Intelligence
        &nbsp;•&nbsp;
        Retrieval-Augmented Generation
    </div>
    """,
    unsafe_allow_html=True,
)