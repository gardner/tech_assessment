# Naive RAG pipeline uses dense embeddings, a sentence splitter, and a simple retriever

import os
import time
from pathlib import Path
from llama_index.core import VectorStoreIndex
from typing import Any, Dict


from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import QueryBundle
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from bickford.config import CHROMA_PATH, CACHE_PATH, CHROMA_COLLECTION_BM25, llm
import nest_asyncio

# from llama_index.postprocessor.cohere_rerank import CohereRerank
CACHE_PATH_BM25 = Path(CACHE_PATH / "bm25").resolve()

nest_asyncio.apply()


def get_retriever():
    start_time = time.time()
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION_BM25)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Load docstore (required for BM25 retriever)
    docstore_path = CACHE_PATH_BM25 / "docstore"
    if docstore_path.exists():
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:
        raise FileNotFoundError(f"Docstore not found at {docstore_path}. Please run ingest.py first.")

    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store
    )

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

    # Verify docstore has nodes before creating BM25 retriever
    node_count = len(docstore.docs)
    if node_count == 0:
        raise ValueError(f"Docstore is empty at {docstore_path}. Please run ingest.py first.")

    retriever = QueryFusionRetriever(
        [
            index.as_retriever(similarity_top_k=5),
            BM25Retriever.from_defaults(
                docstore=docstore, similarity_top_k=5
            ),
        ],
        similarity_top_k=5,
        num_queries=1,
        use_async=True,
        verbose=True,
        mode=FUSION_MODES.RECIPROCAL_RANK
    )

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
    query_bundle = QueryBundle(query)
    nodes_with_scores = retriever.retrieve(query_bundle)

    # Filter out nodes with empty or invalid content before reranking
    valid_nodes = []
    for nws in nodes_with_scores:
        if nws.node and nws.node.get_content() and nws.node.get_content().strip():
            valid_nodes.append(nws)

    if not valid_nodes:
        return "No valid nodes retrieved. Try a different query.", []

    # This code causes the retrieval to take over 1 second.
    # try:
    #     postprocessor = CohereRerank(
    #         api_key=os.getenv("COHERE_API_KEY"),
    #         model="rerank-english-v3.0",
    #         top_n=5,
    #     )

    #     nodes = postprocessor.postprocess_nodes(
    #         nodes=valid_nodes, query_bundle=query_bundle
    #     )
    # except (AssertionError, Exception) as e:
    #     # If reranking fails, use the original nodes (already filtered)
    #     print(f"Warning: Reranking failed ({type(e).__name__}: {e}), using original retrieval results.")
    #     raise e
    #     nodes = valid_nodes
    nodes = valid_nodes

    for i, nws in enumerate(nodes, start=1):
        print(f"{i}: {nws.node}")
        print('--')

    if not nodes:
        return "No hits. Try a different query.", []

    # Extract file names from retrieved nodes (top-5)
    retrieved_files = []
    context_parts = []
    for i, nws in enumerate(nodes, start=1):
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
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print(f"Retrieval latency: {latency_ms:.2f}ms")

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

# For promptfoo
_retriever = None  # simple module-level cache so we don't re-init Chroma every call

def get_cached_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever

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

if __name__ == "__main__":
    main()