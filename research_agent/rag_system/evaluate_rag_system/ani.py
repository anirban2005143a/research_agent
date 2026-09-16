from ..dense_retriever import DenseRetriever
import json

session_id = "paper_eval_session"
retriever = DenseRetriever(session_id)

count = retriever.vector_store._collection.count()
print(f"Number of chunks: {count}")

results = retriever.vector_store.similarity_search_with_relevance_scores(query="Why does GPU-CFR compile a fixed game tree into a static dataflow representation?", k=4)

for i, (doc, score) in enumerate(results, 1):
    print(f"--- Result {i} (Score: {score:.4f}) ---")
    print(f"Content:\n{doc.page_content.strip()}")
    print(f"Metadata:\n{json.dumps(doc.metadata, indent=2)}")
    print("-" * 40 + "\n")