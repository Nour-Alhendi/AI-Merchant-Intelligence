from pathlib import Path
from ragas.metrics.base import Metric
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
)


# --- LLM CONFIG ---
EVALUATION_LLM_MODEL: str = "llama-3.1-8b-instant"


# --- EMBEDDING CONFIG ---
EVALUATION_EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"


# --- PATHS ---
EVALUATION_ROOT_PATH: Path = Path(__file__).parent
EVALUATION_RESULTS_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_results"

EVALUATION_VECTOR_STORES_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_vector_stores"
EXPERIMENTAL_VECTOR_STORES_PATH: Path = EVALUATION_VECTOR_STORES_PATH

EVALUATION_EMBEDDING_CACHE_PATH: Path = EVALUATION_ROOT_PATH / "evaluation_embedding_models"

QA_DATASET_CACHE_PATH: Path = EVALUATION_ROOT_PATH / "qa_dataset_cache"


# --- RAGAS METRICS ---
EVALUATION_METRICS: list[Metric] = [
    ContextPrecision(),
    ContextRecall(),
    Faithfulness(),
    AnswerCorrectness(),
]


# --- RATE LIMIT SAFETY ---
SLEEP_PER_EVALUATION: int = 20
SLEEP_PER_QUESTION: int = 2


# --- CHUNKING STRATEGY EXPERIMENTS ---
CHUNKING_STRATEGY_CONFIGS: list[dict[str, int]] = [
    {"size": 512, "overlap": 50},
    {"size": 1024, "overlap": 200},
]


# --- RERANKER CONFIGS (Stage 3) ---
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANKER_CONFIGS: list[dict[str, int]] = [
    {"retriever_k": 10, "reranker_n": 2},
    {"retriever_k": 10, "reranker_n": 5},
    {"retriever_k": 20, "reranker_n": 5},
]