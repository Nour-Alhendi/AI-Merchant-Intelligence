from src.config import KNOWLEDGE_BASE_PATH
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# Reranker: improves retrieval quality by re-ordering retrieved chunks
from llama_index.core.postprocessor import SentenceTransformerRerank

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_PATH,
    LLM_SYSTEM_PROMPT,
    SIMILARITY_TOP_K,
    VECTOR_STORE_PATH,
    CHAT_MEMORY_TOKEN_LIMIT,
)
from src.model_loader import (
    get_embedding_model,
    initialise_llm,
)

from src.router import route_question


# --- Create a new VectorStoreIndex from local docs + knowledge base and persist it ---
def _create_new_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    print("Creating new vector store from PDFs and TXT knowledge base...")

    documents: list[Document] = []

    # Load PDFs/TXTs from data/ (optional)
    if DOCS_PATH.exists():
        documents += SimpleDirectoryReader(
            input_dir=DOCS_PATH.as_posix(),
            required_exts=[".pdf", ".txt"],
            recursive=True,
        ).load_data()

    # Load TXT policies from data/knowledge_base
    if KNOWLEDGE_BASE_PATH.exists():
        documents += SimpleDirectoryReader(
            input_dir=KNOWLEDGE_BASE_PATH.as_posix(),
            required_exts=[".txt"],
            recursive=True,
        ).load_data()

    if not documents:
        raise ValueError("No documents found (PDFs or TXT).")

    # Chunk documents into smaller nodes for retrieval
    splitter: SentenceSplitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # Build the index
    index: VectorStoreIndex = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        embed_model=embed_model,
    )

    # Persist to disk for fast reloads
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH.as_posix())

    print("Vector store created and saved.")
    return index


# --- Load an existing vector store from disk if present; otherwise create a new one ---
def get_vector_store(embed_model: HuggingFaceEmbedding) -> VectorStoreIndex:
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

    if any(VECTOR_STORE_PATH.iterdir()):
        print("Loading existing vector store from disk...")
        storage_context: StorageContext = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH.as_posix()
        )
        return load_index_from_storage(storage_context, embed_model=embed_model)

    return _create_new_vector_store(embed_model)

# --- Create a chat engine (RAG) with memory + reranker ---
def get_chat_engine(llm: Groq, embed_model: HuggingFaceEmbedding) -> BaseChatEngine:
    vector_index: VectorStoreIndex = get_vector_store(embed_model)

     # Chat memory (keeps recent conversation context)
    memory: ChatMemoryBuffer = ChatMemoryBuffer.from_defaults(
        token_limit=CHAT_MEMORY_TOKEN_LIMIT
    )
    # Two-stage retrieval:
    # 1) retrieve more candidates
    # 2) rerank and keep top_n for the LLM
    retriever_k = 10
    reranker_n = 3

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=reranker_n,
    )

    chat_engine: BaseChatEngine = vector_index.as_chat_engine(
        llm=llm,
        memory=memory,
        system_prompt=LLM_SYSTEM_PROMPT,
        similarity_top_k=retriever_k,        # broad retrieval
        node_postprocessors=[reranker],      # rerank
    )

    return chat_engine

# --- Local REPL for quick manual testing (optional) ---
def main_chat_loop() -> None:
    llm: Groq = initialise_llm()
    embed_model: HuggingFaceEmbedding = get_embedding_model()

    chat_engine: BaseChatEngine = get_chat_engine(llm=llm, embed_model=embed_model)

    print("--- RAG Chatbot Initialised (with Reranker) ---")
    chat_engine.chat_repl()


# --- Router-driven orchestrator: chooses structured vs RAG vs hybrid for each question --- 
class HybridEngine:
    def __init__(self, analytics_engine, rag_engine, router_llm):
        self.analytics = analytics_engine
        self.rag = rag_engine
        self.router_llm = router_llm

    def answer(self, question: str) -> dict:
        decision = route_question(self.router_llm, question)

        route = decision["route"]
        merchant_id = decision.get("merchant_id")

        # 1) Global structured
        if route == "GLOBAL_STRUCTURED":
            q = question.lower()
            if "chargeback" in q:
                stats = self.analytics.highest_chargeback_merchant()
                return {
                    "route": "structured_global",
                    "answer": (
                        f"Merchant **{stats['merchant_id']}** has the highest chargeback rate.\n\n"
                        f"- Total transactions: {stats['total_transactions']}\n"
                        f"- Chargebacks: {stats['chargebacks']}\n"
                        f"- Chargeback rate: {stats['chargeback_rate']*100:.2f}%\n\n"
                        f"This is above the **2%** threshold → **elevated risk**."
                    )
                }

            return {"route": "structured_global", "answer": "Global structured intent detected, but not implemented for this metric yet."}

        # 2) Merchant structured
        if route == "MERCHANT_STRUCTURED":
            if not merchant_id:
                return {"route": "merchant_structured", "answer": "Please provide a merchant_id (e.g., m_00062)."}
            return {
                "route": "merchant_structured",
                "answer": self.analytics.build_structured_context(merchant_id)
            }

        # 3) Hybrid (structured context + policy explanation via RAG)
        if route == "HYBRID":
            structured = ""
            if merchant_id:
                structured = self.analytics.build_structured_context(merchant_id)

            rag_answer = self.rag.answer(question, extra_context=structured)
            return {"route": "hybrid", "answer": rag_answer}

        # 4) RAG-only
        rag_answer = self.rag.answer(question)
        return {"route": "rag", "answer": rag_answer}