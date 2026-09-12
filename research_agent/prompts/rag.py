RAG_QUERY_TEMPLATE = """Rewrite the research request into a focused retrieval query for uploaded and stored documents. Preserve named entities, dates, methods, and technical terms. Return one concise query only.

Research request: {query}
Plan context: {plan}"""

RAG_CONTEXT_LABEL = """The following evidence came from the user's uploaded or stored documents. Treat it as a source, cite its file name and page/section metadata when available, and do not assume it answers claims outside its content."""
