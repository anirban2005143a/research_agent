# Research Agent

Research Agent is a Python-only, research-focused assistant. It investigates questions with multiple external sources, optionally grounds answers in uploaded documents, checks the draft for quality, and returns a structured research brief. It is deliberately not a general-purpose chatbot: requests that are not research, evidence, comparison, investigation, or analysis are answered with a research-scope message.

## Goals

- Use the Hugging Face hosted chat interface with Llama 3.1 8B Instruct.
- Keep embeddings local with `BAAI/bge-m3` and persist vectors in Chroma.
- Isolate every conversation in a UUID session with separate documents and Chroma storage.
- Use LangGraph for explicit stateful orchestration and LangGraph `ToolNode` execution.
- Search the web, Wikipedia, and ArXiv when the question requires external evidence.
- Support optional PDF, DOCX, TXT, Markdown, CSV, JSON, HTML, XML, Python, Excel, and PowerPoint document collections.
- Improve answer quality with parsing, source grounding, critique, and revision.
- Preserve short-term conversation context while compressing older turns and extracting user preferences.
- Pause for human clarification when a research request is underspecified.
- Show safe high-level progress in Streamlit and detailed node, tool-query, and LLM timing logs in the terminal.

## Architecture

```text
Streamlit UI
	|
	v
ResearchGraph (LangGraph StateGraph)
	|
	+--> scope gate ------> research-only response
	|
	+--> HITL clarification (for underspecified questions)
	|
	+--> planner ----------> focused research questions
	|
	+--> tool-call builder -> LangGraph ToolNode
	|                           +--> DuckDuckGo web search
	|                           +--> Wikipedia
	|                           +--> ArXiv
	|
	+--> optional Chroma RAG
	|       +--> recursive chunking
	|       +--> local BGE-M3 embeddings
	|       +--> dense retrieval + lexical overlap fusion
	|       +--> score ranking
	|
	+--> grounded draft --> parsed quality review --> revision loop
```

Each Streamlit session receives a UUID. Original uploads are preserved under `documents/<session_id>/`, the document registry is stored there, and the LangChain Chroma vector store is persisted under `.chroma/<session_id>/`. Starting a new session creates a new namespace and does not delete older session data.

The graph keeps the research state in `ResearchState` and persists it per thread with LangGraph `MemorySaver`. Tool results use LangChain's standard tool schema and are executed by LangGraph `ToolNode`, allowing the model to select tools through conditional routing. Hugging Face responses are plain text, so planning and critique use `PydanticOutputParser` with `OutputFixingParser` as a recovery layer.

The terminal log includes node start/end times, planner count and subquestions, selected tool names and arguments, tool latency, LLM response latency, response size, source counts, and critique scores. The UI displays only high-level progress states, not hidden chain-of-thought.

## Project structure

```text
research-agent/
|-- app.py                         Streamlit application and HITL controls
|-- requirements.txt               Python dependencies
|-- .env.example                   Safe configuration template
|-- .env                           Local configuration with placeholder token
|-- documents/                     Original uploads grouped by session UUID
|-- .chroma/                       LangChain Chroma data grouped by session UUID
|-- README.md
|-- research_agent/
	|-- config.py                  Environment-backed Settings object
	|-- llm.py                     Hugging Face chat model and call throttling
	|-- graph.py                   LangGraph workflow and ToolNode orchestration
	|-- state.py                   Typed graph state
	|-- tools.py                   Web, Wikipedia, ArXiv, and quality tools
	|-- rag.py                     Loaders, normalization, deduplication, Chroma, hybrid ranking, file registry
	|-- parsers.py                 Pydantic schemas and output-repair parsers
	|-- prompts/                   Planner, tool-selection, RAG, drafting, critique, and revision prompts
	|-- memory.py                  Recent turns, preference capture, and compression
	|-- scope.py                   Research-only policy gate
```

## Configuration

`.env` and `.env.example` intentionally use the same keys:

```dotenv
HUGGINGFACEHUB_API_TOKEN=your_huggingface_read_token
LLM_MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
EMBEDDING_MODEL_ID=BAAI/bge-m3
EMBEDDING_CACHE_DIR=.models
EMBEDDING_LOCAL_FILES_ONLY=false
HF_PROVIDER=auto
LLM_MAX_NEW_TOKENS=1024
LLM_TEMPERATURE=0.1
LLM_CALL_DELAY_SECONDS=20
CHROMA_DIR=.chroma
DOCUMENTS_DIR=documents
RAG_CHUNK_SIZE=900
RAG_CHUNK_OVERLAP=140
RAG_TOP_K=8
MAX_RESEARCH_ITERATIONS=2
```

Replace only the placeholder token in `.env`. Do not commit a real token. `LLM_CALL_DELAY_SECONDS` is applied with `time.sleep()` immediately before each explicit model call in the graph, keeping throttling outside the model object.

The chat client accepts `HUGGINGFACEHUB_API_TOKEN1` through `HUGGINGFACEHUB_API_TOKEN5` or `HF_TOKEN1` through `HF_TOKEN5`. Configured values are deduplicated and selected round-robin for separate Hugging Face chat calls. Embeddings remain local and do not consume these tokens. If a token is exposed, revoke it and create a replacement before using the application again.

## Setup and run

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`. `BAAI/bge-m3` runs locally through `HuggingFaceEmbeddings` on CPU and is cached in `.models`; the first run may download model weights, but embedding inference is local and does not use the hosted Hugging Face chat endpoint. Set `EMBEDDING_LOCAL_FILES_ONLY=true` after the model is cached to prevent any later model download attempts. A Hugging Face read token is required only for the hosted Llama chat endpoint.

## Research workflow

1. The scope gate rejects normal conversation before model or search work.
2. Short or vague research questions generate a human clarification request.
3. The planner creates focused search questions using a repairable Pydantic schema.
4. The model selects the necessary tools from typed descriptions, including local RAG and specific-file reading, and `ToolNode` executes them.
5. Uploaded documents are normalized, split with overlap, deduplicated, embedded locally, persisted in Chroma, and retrieved with dense plus lexical reciprocal-rank fusion.
6. The analyst drafts a formatted response with inline citations, a Sources section, and explicit missing-evidence statements.
7. A quality reviewer checks unsupported claims, balance, citations, and directness.
8. The graph revises once when the reviewer requests more work, bounded by `MAX_RESEARCH_ITERATIONS`.

Wikipedia and ArXiv are explicit typed tools. The planner creates a focused subquestion, the model places it in the tool call's `query` argument, and the terminal prints that exact argument before the LangChain provider executes the search.

## Production hardening roadmap

The current structure is suitable as a strong local prototype. Before deploying to multiple users, replace in-memory `MemorySaver` with a durable checkpointer, add per-user Chroma collections, authentication, request and tool timeouts, retry/backoff policies, source-content caching, structured observability, document deletion/versioning, and a test suite with mocked model and tool responses.
