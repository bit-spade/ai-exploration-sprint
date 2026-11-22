"""
rag_fitness_coach.ui.app
~~~~~~~~~~~~~~~~~~~~~~~~
Streamlit Chat Application entry point for RAG Fitness Coach.
Features:
  - Conversational message loop
  - Live streaming tokens and source document accordions
  - Fast local execution via direct chain invocation or FastAPI backend
"""

from __future__ import annotations

import streamlit as st

from rag_fitness_coach.chain.rag_chain import FitnessRAGChain
from rag_fitness_coach.ui.components.chat import render_message
from rag_fitness_coach.ui.components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="RAG Fitness Coach",
    page_icon="🏋️",
    layout="wide",
)


def main() -> None:
    # Render sidebar controls
    ui_config = render_sidebar()

    # Main content header
    st.title("🏋️ Evidence-Based Fitness Coach")
    st.markdown(
        "Ask questions about **hypertrophy, exercise technique, rehabilitation, or sports nutrition**. "
        "Answers are strictly grounded in our curated knowledge base with inline citations."
    )

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your evidence-based Fitness Coach. Ask me anything regarding resistance training, "
                    "rehabilitation exercises (like low back pain), or nutrition requirements!"
                ),
                "sources": None,
            }
        ]

    # Render previous messages
    for msg in st.session_state["messages"]:
        render_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
        )

    # User input prompt
    if prompt := st.chat_input("Ask a fitness, health or nutrition question..."):
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        render_message("user", prompt)

        # Build chain with user parameters
        chain = FitnessRAGChain(
            pillar_filter=ui_config["pillar"],
            retriever_top_k=ui_config["retriever_top_k"],
            reranker_top_k=ui_config["reranker_top_k"],
        )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence & reranking passages..."):
                response = chain.ask(question=prompt, session_id="streamlit_user")

            st.markdown(response["answer"])

            sources = response.get("sources", [])
            if sources:
                with st.expander(f"📚 Cited Sources ({len(sources)} documents)", expanded=False):
                    for src in sources:
                        idx = src.get("index", "?")
                        title = src.get("title", "Document")
                        pillar = src.get("pillar", "general")
                        v_score = src.get("vector_score")
                        r_score = src.get("rerank_score")
                        content_snippet = src.get("content", "").strip()
                        source_url = src.get("source", "")

                        score_str = []
                        if v_score is not None:
                            score_str.append(f"Cosine: {v_score:.3f}")
                        if r_score is not None:
                            score_str.append(f"Rerank: {r_score:.3f}")
                        score_badge = f" `[{', '.join(score_str)}]`" if score_str else ""

                        st.markdown(f"**[Doc {idx}] {title}** ({pillar}){score_badge}")
                        if source_url:
                            st.caption(f"Source: [{source_url}]({source_url})")
                        st.text(content_snippet[:350] + ("..." if len(content_snippet) > 350 else ""))
                        st.divider()

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
