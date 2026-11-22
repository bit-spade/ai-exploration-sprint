"""
rag_fitness_coach.chain.memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conversation history memory for sliding window dialogue.
Maintains recent turns so follow-up queries retain multi-turn context.
"""

from __future__ import annotations

from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from rag_fitness_coach.config import settings


class ConversationMemoryManager:
    """
    Manages in-memory sliding window message history.
    """

    def __init__(self, window_k: int | None = None):
        self.k = window_k or settings.memory_window_k
        self.messages: List[BaseMessage] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))
        self._trim()

    def add_ai_message(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))
        self._trim()

    def get_messages(self) -> List[BaseMessage]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def _trim(self) -> None:
        # Keep at most 2 * k messages (k human-AI turns)
        max_msgs = self.k * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]


_MEMORY_STORE: dict[str, ConversationMemoryManager] = {}


def get_session_memory(session_id: str = "default") -> ConversationMemoryManager:
    """
    Retrieve or create a memory manager for a given session ID.
    """
    if session_id not in _MEMORY_STORE:
        _MEMORY_STORE[session_id] = ConversationMemoryManager()
    return _MEMORY_STORE[session_id]
