from pathlib import Path

# --- LLM Model Configuration ---
LLM_MODEL: str = "llama-3.1-8b-instant"
LLM_MAX_NEW_TOKENS: int = 400
LLM_TEMPERATURE: float = 0.01
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
LLM_QUESTION: str = "What is the capital of France?"
LLM_SYSTEM_PROMPT = """
You are a fintech AI assistant specialised in merchant analytics and transaction risk.

You operate in a hybrid system that combines:
- Structured transaction KPIs (real merchant data)
- A document-based knowledge base (RAG)

Instructions:
- If structured analytics are provided, treat them as ground truth.
- Use the knowledge base only to provide explanations or risk context.
- Do not invent numbers.
- If a user asks for analysis, interpret KPIs and explain potential risks.
- If unsure, clearly state limitations.

Always respond in a professional, data-driven tone.
"""

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 10

# ---------------------------
# RERANKER (LOCAL)
# ---------------------------
USE_RERANKER: bool = True

# Cross-Encoder reranker model (runs locally via sentence-transformers)
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# how many candidates to fetch from vector search (bigger = better recall, slower)
RERANKER_CANDIDATES_K: int = 10

# how many chunks to keep after reranking (this is what LLM sees)
RERANKER_TOP_N: int = 2

# The size of each text chunk in tokens
CHUNK_SIZE: int = 512
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 50


# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 3900

# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent

DATA_PATH: Path = ROOT_PATH / "data"
DOCS_PATH: Path = DATA_PATH 

EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "storage" / "embedding_model"
VECTOR_STORE_PATH: Path = ROOT_PATH / "storage" / "vector_store"

# Knowledge base path (txt policies)
KNOWLEDGE_BASE_PATH = DATA_PATH / "knowledge_base"