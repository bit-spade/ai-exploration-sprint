"""
rag_fitness_coach.ui.components.chat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chat message and citation bubble rendering for Streamlit.
"""

from __future__ import annotations

from typing import Any, Dict, List
import streamlit as st


def render_message(role: str, content: str, sources: List[Dict[str, Any]] | None = None) -> None:
    """
    Renders a chat message bubble with optional collapsible citations.
    """
    with st.chat_message(role):
        st.markdown(content)

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
