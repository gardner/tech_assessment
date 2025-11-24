import os
import logging
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


# Suppress PDF parsing warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
# logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.ERROR)

# Paths relative to project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PATH = PROJECT_ROOT / ".chromadb"
CACHE_PATH = PROJECT_ROOT / ".cache"

CHROMA_COLLECTION = "naive"
CHROMA_COLLECTION_BM25 = "bm25"
model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

# LLM for *answering* questions
Settings.llm = OpenAILike(
    model=model,
    api_base=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    context_window=128000,
    is_chat_model=True,
    is_function_calling_model=False,
)
llm = Settings.llm
