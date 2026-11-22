"""Unit and smoke tests for the Streamlit UI components."""

import pytest
from rag_fitness_coach.ui.components.chat import render_message
from rag_fitness_coach.ui.components.sidebar import render_sidebar
from rag_fitness_coach.ui import app


def test_ui_modules_loadable():
    assert callable(render_message)
    assert callable(render_sidebar)
    assert callable(app.main)
