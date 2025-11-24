import os
from smolagents import OpenAIModel
import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("travel")

model = OpenAIModel(
    model_id=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
    api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
)

