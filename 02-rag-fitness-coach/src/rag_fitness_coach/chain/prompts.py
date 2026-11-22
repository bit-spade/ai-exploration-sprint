"""
rag_fitness_coach.chain.prompts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prompt templates and citation formatting rules for the RAG Coach.
Enforces:
  - Grounded answers based ONLY on retrieved context
  - Clear citation tags in the response: [Doc 1], [Doc 2], etc.
  - Strict refusal when context is insufficient or out of domain
  - Safe medical disclaimers for health/rehabilitation questions
"""

from __future__ import annotations

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are an elite, evidence-based AI Fitness Coach and Sports Science Specialist.
Your mission is to provide accurate, helpful, and safe fitness, rehabilitation, and nutrition advice.

STRICT GROUNDING & CITATION RULES:
1. Base your answer SOLELY on the provided Context documents below.
2. Every major claim, number, recommendation, or principle MUST cite its source document using inline bracket notation, e.g., "[Doc 1]", "[Doc 2]".
3. If the provided Context does NOT contain enough scientific or factual information to answer the question, state honestly:
   "I do not have sufficient evidence in my knowledge base to answer this question accurately." Do NOT speculate or invent facts.
4. If a question touches on acute pain, injuries, or severe medical conditions, include a brief disclaimer encouraging consultation with a qualified physical therapist or medical physician.
5. Keep your tone encouraging, professional, and clear. Format key points with bullet points or numbered steps where appropriate.

Context Documents:
{context}
"""

def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Format retrieved & reranked Documents into an annotated numbered context string.
    Example:
      [Doc 1] Title: Principles of Hypertrophy | Source: ...
      Content: ...
    """
    if not docs:
        return "No relevant context documents found."

    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Reference")
        source = doc.metadata.get("source", "")
        pillar = doc.metadata.get("pillar", "")
        content = doc.page_content.strip()

        header = f"[Doc {i}] Title: {title}"
        if pillar:
            header += f" | Pillar: {pillar}"
        if source:
            header += f" | Source: {source}"

        formatted_parts.append(f"{header}\nContent:\n{content}")

    return "\n\n" + ("\n" + "-" * 50 + "\n").join(formatted_parts) + "\n"


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Builds the conversational ChatPromptTemplate with message history.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
