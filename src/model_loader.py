import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from src.config import (
    EMBEDDING_CACHE_PATH,
    EMBEDDING_MODEL_NAME,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
)
# Load environment variables from project root .env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# --- Create and return the Groq LLM client (reads GROQ_API_KEY from env) ---
def initialise_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from config."""

    # Get API key from environment variables
    api_key: str | None = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Make sure it is set in the environment."
        )

    return Groq(
        api_key=api_key,
        model=LLM_MODEL,
        max_tokens=LLM_MAX_NEW_TOKENS,
        # # Groq OpenAI-compatible endpoint
        base_url="https://api.groq.com/openai/v1",
    )

def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model"""

    # Create the cache directory if it doesn't exist
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )
