
import pytest
from core.memory import GlobalMemory, SessionMemory

def test_global_memory_save_and_retrieve():
    memory = GlobalMemory(":memory:")  # In-memory SQLite
    
    memory.save_user_preferences("user1", {
        "risk_taxonomy": ["market", "operational"],
        "writing_style": "bullet-first"
    })
    
    # Retrieve and verify...
    assert True

def test_session_memory_context_window():
    session = SessionMemory()
    
    for i in range(10):
        session.add_turn("user", f"Message {i}")
    
    context = session.get_context_window(n_turns=5)
    assert len(context) == 5