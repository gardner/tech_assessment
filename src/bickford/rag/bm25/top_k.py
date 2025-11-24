# Naive RAG pipeline uses dense embeddings, a sentence splitter, and a simple retriever

from typing import Any, Dict
from pathlib import Path

from bickford.config import CACHE_PATH
from bickford.rag.bm25.query import get_cached_retriever
CACHE_PATH_BM25 = Path(CACHE_PATH / "bm25").resolve()


def call_api(
    prompt: str, options: Dict[str, Any], context: Dict[str, Any]
) -> Dict[str, str]:
    """
    Promptfoo provider that returns top-k retrieved file names as a simple string.
    This allows promptfoo's built-in 'contains' assertion to check retrieval accuracy.
    """
    vars_ = context.get("vars", {}) or {}
    label = context.get("prompt", {}).get("label", "")

    # Defensive: if somehow this provider is used as a grader (llm-rubric),
    # bail out with a stub so we don't try to RAG over the grading prompt.
    if label == "llm-rubric":
        return {
            "output": "Error: RAG provider was used as a grader; llm-rubric should use a different provider."
        }

    # Prefer the structured input variable from tests,
    # fall back to the rendered prompt if not available.
    question = (
        vars_.get("input")
        or vars_.get("query")
        or prompt
    )

    retriever = get_cached_retriever()
    nodes_with_scores = retriever.retrieve(question)


    # Filter out nodes with empty or invalid content before reranking
    nodes = []
    for nws in nodes_with_scores:
        if nws.node and nws.node.get_content() and nws.node.get_content().strip():
            nodes.append(nws)

    if not nodes:
        return "No valid nodes retrieved. Try a different query.", []


    # Extract file names from retrieved nodes (top-5)
    retrieved_files = []
    for nws in nodes:
        node = nws.node
        meta = node.metadata or {}
        fname = meta.get("file_name", "unknown")
        retrieved_files.append(fname)

    # Return as a newline-separated string for easy contains checking
    # This format works well with promptfoo's 'contains' assertion
    output = "\n".join(retrieved_files) if retrieved_files else ""

    return {
        "output": output
    }


