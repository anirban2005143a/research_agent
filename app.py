import hashlib
import uuid
from pathlib import Path

import streamlit as st

from research_agent.graph import ResearchGraph
from research_agent.memory import SESSION_MEMORY_STORE
from research_agent.rag_system import DocumentHandler, HybridRAG

st.set_page_config(page_title="Research Agent", page_icon="R", layout="wide")
st.title("Research Agent")
st.caption("Evidence-focused investigation with optional document grounding")


def render_sources(sources: list[str]) -> None:
    """Render sources as separate UI metadata, never as answer text."""
    unique_sources = list(dict.fromkeys(str(source).strip() for source in sources if str(source).strip()))
    if not unique_sources:
        return
    st.markdown("**Sources**")
    for source in unique_sources:
        st.markdown(f"- {source}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "memory" not in st.session_state:
    st.session_state.memory = SESSION_MEMORY_STORE.get_or_create(st.session_state.session_id)
if "rag" not in st.session_state:
    st.session_state.document_handler = DocumentHandler(
        Path("documents") / st.session_state.session_id,
    )
    st.session_state.rag = HybridRAG(
        session_id=st.session_state.session_id,
        document_handler=st.session_state.document_handler,
    )
if "graph" not in st.session_state:
    st.session_state.graph = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
if "thread_id" not in st.session_state:
    st.session_state.thread_id = st.session_state.session_id
if "indexed_uploads" not in st.session_state:
    st.session_state.indexed_uploads = set()

with st.sidebar:
    st.header("Research workspace")
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("Start new session", use_container_width=True):
        previous_session_id = st.session_state.get("session_id")
        if previous_session_id:
            SESSION_MEMORY_STORE.remove(previous_session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.memory = SESSION_MEMORY_STORE.get_or_create(st.session_state.session_id)
        st.session_state.document_handler = DocumentHandler(
            Path("documents") / st.session_state.session_id,
        )
        st.session_state.rag = HybridRAG(
            session_id=st.session_state.session_id,
            document_handler=st.session_state.document_handler,
        )
        st.session_state.graph = None
        st.session_state.pending_question = ""
        st.session_state.pending_query = ""
        st.session_state.thread_id = st.session_state.session_id
        st.session_state.indexed_uploads = set()
        st.rerun()
    uploads = st.file_uploader(
        "Upload source documents",
        type=["pdf", "docx", "txt", "md", "markdown", "csv", "json", "html", "htm", "xml", "py", "xlsx", "pptx"],
        accept_multiple_files=True,
    )
    new_uploads = []
    for uploaded in uploads or []:
        content = uploaded.getvalue()
        upload_key = hashlib.sha256(
            f"{uploaded.name}:{hashlib.sha256(content).hexdigest()}".encode()
        ).hexdigest()
        if upload_key not in st.session_state.indexed_uploads:
            new_uploads.append((uploaded, content, upload_key))
    if new_uploads:
        indexed_chunks = 0
        failures = []
        with st.status("Processing uploaded documents...", expanded=True) as upload_status:
            for uploaded, content, upload_key in new_uploads:
                try:
                    upload_status.write(f"Saving and parsing `{uploaded.name}`...")

                    def show_chunk_progress(done: int, total: int, source: str) -> None:
                        upload_status.progress(
                            done / total,
                            text=f"Embedding and storing chunk {done}/{total}: {source}",
                        )

                    stored_path = st.session_state.document_handler.save_upload(
                        uploaded.name, content
                    )
                    indexed_chunks += st.session_state.rag.store_document(
                        stored_path,
                        progress_callback=show_chunk_progress,
                    )
                    st.session_state.indexed_uploads.add(upload_key)
                    upload_status.write(f"Embedded and stored `{uploaded.name}`.")
                except Exception as exc:
                    failures.append(f"{uploaded.name}: {exc}")
            if failures:
                upload_status.update(
                    label="Some documents could not be indexed.", state="error"
                )
            else:
                upload_status.update(
                    label=f"Documents indexed: {indexed_chunks} chunks.", state="complete"
                )
        for failure in failures:
            st.error(failure)
    st.info("Documents are optional. Web research still runs when no files are uploaded.")
    stored_files = st.session_state.document_handler.list_files()
    if stored_files:
        st.caption("Stored files")
        for file_name in stored_files:
            st.write(f"- {file_name}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

question = st.chat_input("Ask a research question")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    if st.session_state.graph is None:
        st.session_state.graph = ResearchGraph(
            rag=st.session_state.rag,
            document_handler=st.session_state.document_handler,
            session_memory=st.session_state.memory,
            session_id=st.session_state.session_id,
        )
    progress_holder = [None]
    shown_progress = set()
    sources = []

    def show_progress(message: str) -> None:
        if message in shown_progress:
            return
        shown_progress.add(message)
        if progress_holder[0] is None:
            progress_holder[0] = st.status("Research progress", expanded=True)
        progress_holder[0].write(message)

    with st.spinner(""):
        try:
            result = st.session_state.graph.invoke(
                question,
                thread_id=st.session_state.thread_id,
            )
            if result.get("needs_hitl"):
                st.session_state.pending_question = result["hitl_question"]
                st.session_state.pending_query = question
                answer = ""
            else:
                st.session_state.pending_question = ""
                st.session_state.pending_query = ""
                answer = result["final_response"]
                sources = result.get("sources", [])
        except Exception as exc:
            answer = f"Unable to start the research agent: {exc}"
            sources = []
    if progress_holder[0] is not None:
        progress_holder[0].update(state="complete")
    if answer:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
            render_sources(sources)

if st.session_state.pending_question:
    st.warning(st.session_state.pending_question)
    clarification = st.text_input("Clarify the research scope", key="clarification")
    continue_research = st.button("Continue research")
    skip_clarification = st.button("Skip clarification")
    if (continue_research and clarification) or skip_clarification:
        st.session_state.pending_question = ""
        with st.status("Research progress", expanded=True) as progress:
            shown_progress = set()

            def show_progress(message: str) -> None:
                if message in shown_progress:
                    return
                shown_progress.add(message)
                progress.write(message)

            result = st.session_state.graph.invoke(
                st.session_state.pending_query,
                hitl_answer=clarification if continue_research else "",
                resume_hitl=True,
                thread_id=st.session_state.thread_id,
            )
            st.session_state.pending_query = ""
            if not result.get("needs_hitl") and result.get("final_response"):
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["final_response"],
                        "sources": result.get("sources", []),
                    }
                )
            st.rerun()
