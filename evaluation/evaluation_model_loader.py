import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_config import (
    EVALUATION_LLM_MODEL,
    EVALUATION_EMBEDDING_MODEL_NAME,
    EVALUATION_EMBEDDING_CACHE_PATH,
)

# Load .env from project root (rag_project/.env)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def initialise_evaluation_llm() -> Groq:
    """Initialises the Groq LLM with core parameters from evaluation config."""
    api_key: str | None = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Set it in rag_project/.env")

    return Groq(
        api_key=api_key,
        model=EVALUATION_LLM_MODEL,
        base_url="https://api.groq.com/openai/v1",
    )


def load_ragas_models() -> tuple[LlamaIndexLLMWrapper, HuggingFaceEmbeddings]:
    """Loads the LLM and embedding models required for RAGAS evaluation."""
    print("--- 🧠 Loading RAGAS LLM and Embeddings ---")

    llm_for_evaluation: Groq = initialise_evaluation_llm()
    ragas_llm = LlamaIndexLLMWrapper(llm=llm_for_evaluation)

    EVALUATION_EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    ragas_embeddings = HuggingFaceEmbeddings(
        model=EVALUATION_EMBEDDING_MODEL_NAME,
        cache_folder=EVALUATION_EMBEDDING_CACHE_PATH.as_posix(),
    )

    return ragas_llm, ragas_embeddings