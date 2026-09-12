from datetime import datetime
import uuid
from pathlib import Path

import streamlit as st

from research_agent.graph import ResearchGraph
from research_agent.memory import ShortTermMemory
from research_agent.rag import HybridRAG, load_uploaded_file

st.set_page_config(page_title="Research Agent", page_icon="R", layout="wide")
st.title("Research Agent")
st.caption("Evidence-focused investigation with optional document grounding")

if "memory" not in st.session_state:
    st.session_state.memory = ShortTermMemory(recent_limit=5)
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "rag" not in st.session_state:
    st.session_state.rag = HybridRAG(session_id=st.session_state.session_id)
if "graph" not in st.session_state:
    st.session_state.graph = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.session_state.session_id

with st.sidebar:
    st.header("Research workspace")
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("Start new session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.memory = ShortTermMemory(recent_limit=5)
        st.session_state.rag = HybridRAG(session_id=st.session_state.session_id)
        st.session_state.graph = None
        st.session_state.pending_question = ""
        st.session_state.pending_query = ""
        st.session_state.thread_id = st.session_state.session_id
        st.rerun()
    uploads = st.file_uploader(
        "Upload source documents",
        type=["pdf", "docx", "txt", "md", "markdown", "csv", "json", "html", "htm", "xml", "py", "xlsx", "pptx"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded documents", use_container_width=True):
        count = 0
        failures = []
        for uploaded in uploads or []:
            try:
                stored_path = st.session_state.rag.save_uploaded_file(
                    uploaded.name, uploaded.getvalue()
                )
                count += st.session_state.rag.add_documents(
                    load_uploaded_file(str(stored_path)),
                    uploaded.name,
                    file_metadata={
                        "mime_type": uploaded.type,
                        "file_size_bytes": uploaded.size,
                        "uploaded_at": str(datetime.now()),
                        "stored_path": str(stored_path),
                    },
                )
            except Exception as exc:
                failures.append(f"{uploaded.name}: {exc}")
        st.success(f"Indexed {count} chunks.")
        for failure in failures:
            st.error(failure)
    st.info("Documents are optional. Web research still runs when no files are uploaded.")
    stored_files = st.session_state.rag.list_files()
    if stored_files:
        st.caption("Stored files")
        for item in stored_files:
            st.write(f"- {item['source']} ({item['chunk_count']} chunks)")

for message in st.session_state.memory.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a research question")
if question:
    st.session_state.memory.add("user", question)
    with st.chat_message("user"):
        st.markdown(question)
    with st.status("Thinking", expanded=True) as progress:
        shown_progress = set()

        def show_progress(message: str) -> None:
            if message in shown_progress:
                return
            shown_progress.add(message)
            progress.write(message)

        try:
            if st.session_state.graph is None:
                st.session_state.graph = ResearchGraph(
                    rag=st.session_state.rag,
                    progress_callback=show_progress,
                )
            result = st.session_state.graph.invoke(
                question,
                messages=st.session_state.memory.messages,
                thread_id=st.session_state.thread_id,
                memory_context=st.session_state.memory.context(),
                progress_callback=show_progress,
            )
            if result.get("needs_hitl") and not result.get("hitl_answer"):
                st.session_state.pending_question = result["hitl_question"]
                st.session_state.pending_query = question
            answer = result["final_answer"]
        except Exception as exc:
            answer = f"Unable to start the research agent: {exc}"
            progress.update(label="Research failed", state="error")
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.memory.add("assistant", answer)

if st.session_state.pending_question:
    st.warning(st.session_state.pending_question)
    clarification = st.text_input("Clarify the research scope", key="clarification")
    if st.button("Continue research") and clarification:
        st.session_state.pending_question = ""
        with st.status("Thinking", expanded=True) as progress:
            shown_progress = set()

            def show_progress(message: str) -> None:
                if message in shown_progress:
                    return
                shown_progress.add(message)
                progress.write(message)

            result = st.session_state.graph.invoke(
                st.session_state.pending_query,
                hitl_answer=clarification,
                thread_id=st.session_state.thread_id,
                memory_context=st.session_state.memory.context(),
                progress_callback=show_progress,
            )
            st.session_state.memory.add("assistant", result["final_answer"])
            st.session_state.pending_query = ""
            st.rerun()
