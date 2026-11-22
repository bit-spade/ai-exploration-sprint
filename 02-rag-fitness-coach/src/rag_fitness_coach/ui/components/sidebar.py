"""
rag_fitness_coach.ui.components.sidebar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Streamlit sidebar component for configuration, pillar filters, and knowledge base inspection.
"""

from __future__ import annotations

import streamlit as st
from rag_fitness_coach.config import settings
from rag_fitness_coach.ingestion.loader import SOURCE_REGISTRY
from rag_fitness_coach.vectorstore.client import get_collection_info


def render_sidebar() -> dict:
    """
    Renders the sidebar controls and returns the selected UI parameters.
    """
    with st.sidebar:
        st.title("🏋️ RAG Fitness Coach")
        st.caption("Grounded in evidence-based sports science & nutrition")

        st.markdown("---")
        st.subheader("⚙️ Session & Model Settings")

        # Pillar filter
        pillar_options = {
            "All Pillars (Unrestricted)": None,
            "🏋️ Exercise Library & Hypertrophy": "exercise_library",
            "🩺 Health Issues & Rehabilitation": "health_issues",
            "🥗 Sports Nutrition & Macros": "nutrition",
        }
        selected_pillar_label = st.selectbox(
            "Focus Area / Pillar Filter:",
            options=list(pillar_options.keys()),
            index=0,
            help="Filters vector retrieval strictly to a selected sports science pillar.",
        )
        selected_pillar = pillar_options[selected_pillar_label]

        # Candidate settings
        retriever_k = st.slider(
            "Qdrant Candidates (Top-K)",
            min_value=2,
            max_value=12,
            value=settings.retriever_top_k,
            help="Number of initial candidates fetched using dense cosine similarity.",
        )
        reranker_k = st.slider(
            "Cross-Encoder Keep (Top-N)",
            min_value=1,
            max_value=6,
            value=settings.reranker_top_k,
            help="Number of candidates kept after ms-marco cross-encoder reranking.",
        )

        # Clear session button
        if st.button("🗑️ Reset Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("---")
        st.subheader("📊 Knowledge Base Status")
        try:
            info = get_collection_info()
            st.success(f"**Qdrant:** {info['name']} ({info['vectors_count']} vectors)")
        except Exception:
            st.warning("Local Qdrant collection offline")

        st.caption(f"**Embeddings:** {settings.embedding_model}")
        st.caption(f"**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2")

        with st.expander("📚 Curated Source Documents", expanded=False):
            for doc in SOURCE_REGISTRY:
                st.markdown(f"- **{doc['title']}** (`{doc['pillar']}`)")

    return {
        "pillar": selected_pillar,
        "retriever_top_k": retriever_k,
        "reranker_top_k": reranker_k,
    }
