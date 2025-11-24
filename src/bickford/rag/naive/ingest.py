# Naive RAG pipeline uses dense embeddings, a sentence splitter, and a simple retriever

import os
import click
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage import StorageContext
import chromadb
from chromadb.config import Settings as ChromaSettings
from pathlib import Path

from bickford.config import DATA_DIR, CHROMA_PATH, CHROMA_COLLECTION, CACHE_PATH
from dotenv import load_dotenv
load_dotenv()


def build_pipeline():

    transformations = [
        SentenceSplitter(chunk_size=1024, chunk_overlap=204),
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
    if Path(CACHE_PATH / "pipeline").exists():
        pipeline.load(CACHE_PATH / "pipeline")

    return pipeline


@click.command()
@click.argument("input_dir", type=Path, required=False, default=DATA_DIR)
def main(input_dir: Path):
    # 1. Load raw documents
    print(f"Loading documents from {input_dir}...")
    documents = SimpleDirectoryReader(
        input_dir=input_dir,
        required_exts=[".md"]
    ).load_data()
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print(f"No documents found in the {input_dir} folder.")
        return

    # 2. Build ingestion pipeline
    url = os.getenv("TEXT_EMBEDDINGS_INFERENCE_BASE_URL", "http://localhost:8000")
    print(f"Running ingestion pipeline... {url}")
    pipeline = build_pipeline()
    nodes = pipeline.run(documents=documents)
    pipeline.persist(CACHE_PATH / "pipeline")

    print(f"Pipeline produced {len(nodes)} nodes.")

    # 3. Set up Chroma vector store
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Build a VectorStoreIndex from pre-embedded nodes
    index = VectorStoreIndex(
        nodes,
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

    # Get counts
    node_count = len(nodes)
    vector_count = collection.count()
    print(f"Indexed {node_count} nodes ({vector_count} vectors in ChromaDB)")

if __name__ == "__main__":
    main()