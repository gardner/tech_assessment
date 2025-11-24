# Naive RAG pipeline uses dense embeddings, a sentence splitter, and a simple retriever

import os
import time
from llama_index.core import VectorStoreIndex
from typing import Any, Dict


from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
import chromadb
from chromadb.config import Settings as ChromaSettings
from bickford.rag.naive.query import get_cached_retriever

from bickford.config import CHROMA_PATH, CHROMA_COLLECTION, llm


def get_retriever():
    start_time = time.time()
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=TextEmbeddingsInference(
            base_url=os.getenv("TEXT_EMBEDDINGS_INFERENCE_BASE_URL", "http://localhost:8000"),
            model_name="Qwen/Qwen3-Embedding-0.6B",
            timeout=60,
            embed_batch_size=10,
            # See https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json#L3
            query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        ),

    )

    retriever = index.as_retriever(similarity_top_k=5)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print(f"Retriever initialized in {latency_ms:.2f}ms")
    return retriever

def generate_response(retriever, query) -> tuple[str, list[str]]:
    """
    Generate a response and return both the answer and the list of retrieved file names.
    Returns: (answer_text, list_of_retrieved_file_names)
    """
    start_time = time.time()
    nodes_with_scores = retriever.retrieve(query)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print(f"Retrieval latency: {latency_ms:.2f}ms")

    if not nodes_with_scores:
        return "No hits. Try a different query.", []

    # Extract file names from retrieved nodes (top-5)
    retrieved_files = []
    context_parts = []
    for i, nws in enumerate(nodes_with_scores, start=1):
        node = nws.node
        meta = node.metadata or {}
        title = meta.get("title", "Untitled")
        fname = meta.get("file_name", "unknown")
        retrieved_files.append(fname)
        context_parts.append(
            f"[{i}] score={nws.score:.3f}, file={fname}, title={title}\n"
            f"{node.get_content()}\n"
        )

    context_text = "\n---\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions based only on the context.

Context:
{context_text}

Question: {query}

Answer clearly and concisely. If the answer is not in the context, say you don't know. If there are multiple answers in the context then it's OK to provide two different answers for the user to read.
Always include the values for `file` as inline citations like so: [file=Source-File-Name1.md] and [file=Source-File-Name2.md].
    """.strip()

    resp = llm.complete(prompt)
    answer_text = resp.text if hasattr(resp, "text") else str(resp)
    return answer_text, retrieved_files

def main():
    retriever = get_retriever()
    print("Ready. Type a question, or press Enter to quit.\n")

    while True:
        try:
            print("How many chickens can I have in my backyard?")
            query = input("❓ Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            print("Exiting.")
            break

        answer, _ = generate_response(retriever, query)
        print(answer)
        print("\n")

# For promptfoo
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

    # Extract file names from retrieved nodes (top-5)
    retrieved_files = []
    for nws in nodes_with_scores:
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


if __name__ == "__main__":
    main()