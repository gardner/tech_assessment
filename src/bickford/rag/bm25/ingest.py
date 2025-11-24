# This RAG pipeline adds BM25 retrieval and metadata extraction.
# The promptfoo config compares this pipeline with the naive one.

import os
import multiprocessing
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage import StorageContext
import chromadb
from chromadb.config import Settings as ChromaSettings

from bickford.config import DATA_DIR, CHROMA_PATH, CACHE_PATH, CHROMA_COLLECTION_BM25
from dotenv import load_dotenv
load_dotenv()
CACHE_PATH_BM25 = Path(CACHE_PATH / "bm25").resolve()


import nest_asyncio
nest_asyncio.apply()

def build_pipeline():

    transformations = [
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(summaries=["prev", "self"]),
        KeywordExtractor(keywords=10),
        TextEmbeddingsInference(
            base_url=os.getenv("TEXT_EMBEDDINGS_INFERENCE_BASE_URL", "http://localhost:8000"),
            model_name="Qwen/Qwen3-Embedding-0.6B",
            timeout=60,
            embed_batch_size=10,
            # See https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json#L3
            query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        ),
    ]

    pipeline = IngestionPipeline(transformations=transformations)
    if Path(CACHE_PATH_BM25 / "pipeline").exists():
        pipeline.load(CACHE_PATH_BM25 / "pipeline")

    return pipeline


def main():
    nodes = None
    docstore = None
    need_to_rebuild = False

    # 1. Load raw documents
    print(f"Loading documents from {DATA_DIR}...")
    documents = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        required_exts=[".md"]
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print(f"No documents found in the {DATA_DIR} folder.")
        return

    # 2. Check if docstore exists and load it
    docstore_path = CACHE_PATH_BM25 / "docstore"
    if docstore_path.exists():
        print("Loading existing docstore...")
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
        # docs is a property (dict), not a method
        nodes = list(docstore.docs.values())
        print(f"Loaded {len(nodes)} nodes from docstore.")
    else:
        need_to_rebuild = True
        docstore = SimpleDocumentStore()

    # 3. Set up Chroma vector store
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION_BM25)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Check if ChromaDB collection already has vectors
    collection_count = collection.count()
    has_existing_vectors = collection_count > 0

    # 4. Process documents if needed
    if need_to_rebuild or not has_existing_vectors:
        print("Running ingestion pipeline...")
        pipeline = build_pipeline()
        parser = SentenceSplitter(chunk_size=1024, chunk_overlap=204)
        # Parse documents into nodes first
        chunked_nodes = parser.get_nodes_from_documents(documents)
        print(f"Parsed {len(documents)} documents into {len(chunked_nodes)} chunks.")

        # Run pipeline to add embeddings and metadata
        # Note: pipeline.run() accepts nodes when they're already chunked
        # nodes = pipeline.run(nodes=chunked_nodes, num_workers=multiprocessing.cpu_count()) # This causes rate limit errors with the Azure OpenAI Service
        nodes = pipeline.run(nodes=chunked_nodes, num_workers=8)
        pipeline.persist(CACHE_PATH_BM25 / "pipeline")

        # Add nodes to docstore and persist
        docstore.add_documents(nodes)
        docstore.persist(docstore_path)
        print(f"Pipeline produced {len(nodes)} nodes.")

        # 5. Build VectorStoreIndex with pre-embedded nodes
        # VectorStoreIndex will detect that nodes already have embeddings and won't re-embed them
        # The embed_model is still needed for query-time embedding of queries
        storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
            embed_model=TextEmbeddingsInference(
                base_url=os.getenv("TEXT_EMBEDDINGS_INFERENCE_BASE_URL", "http://localhost:8000"),
                model_name="Qwen/Qwen3-Embedding-0.6B",
                timeout=60,
                embed_batch_size=10,
                # See https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/main/config_sentence_transformers.json#L3
                query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            ),
        )
    else:
        # Load existing index from ChromaDB
        print(f"Loading existing index from ChromaDB ({collection_count} vectors)...")
        storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
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

    print("Building BM25 retriever...")
    # Build the keyword index from docstore nodes
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=5
    )
    bm25_retriever.persist(CACHE_PATH_BM25 / "retriever")

    # Get counts
    node_count = len(docstore.docs)
    vector_count = collection.count()
    print(f"Indexed {node_count} nodes ({vector_count} vectors in ChromaDB)")

if __name__ == "__main__":
    main()