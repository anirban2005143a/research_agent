from research_agent.memory import SessionMemoryStore, ShortTermMemory
from research_agent.nodes import ResearchNodes


class DummyResearchNodes(ResearchNodes):
    def __init__(self):
        self.llm = None
        self.session_memory = ShortTermMemory()


def test_session_memory_store_is_scoped_by_session_id():
    store = SessionMemoryStore()
    first = store.get_or_create("session-a")
    second = store.get_or_create("session-a")
    assert first is second
    assert store.get("session-a") is first
    assert store.get("session-b") is None
    store.remove("session-a")
    assert store.get("session-a") is None


def test_recent_graph_messages_roll_over_to_message_summary():
    graph = DummyResearchNodes()
    state = {
        "messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third question"},
        ],
        "message_summary": "",
    }

    state = graph._append_recent_state_message(state, {"role": "assistant", "content": "third answer"})

    assert len(state["messages"]) == 5
    assert state["message_summary"]
    assert state["messages"][0]["content"] == {"role": "assistant", "content": "third answer"}["content"] or "third answer" in str(state["messages"][-1])
