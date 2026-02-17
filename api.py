# api.py
# FastAPI service layer for Hybrid RAG + Structured Analytics system

from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import logging
import time

# --- Import project modules ---
from src.engine import get_chat_engine
from src.model_loader import initialise_llm, get_embedding_model
from src.analytics import AnalyticsEngine
from src.router import HybridRouter


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Merchant Intelligence API",
    description="Hybrid Structured + RAG Financial AI Service",
    version="1.0"
)


# Request Schema
class QuestionRequest(BaseModel):
    question: str



# Global Components (loaded once at startup)
chat_engine = None
analytics_engine = None
router = None



# --- Startup Event (Load models once) ---
@app.on_event("startup")
def startup_event():
    global chat_engine, analytics_engine, router

    logging.info("Starting AI system...")

    # Load LLM + Embeddings
    llm = initialise_llm()
    embed_model = get_embedding_model()

    # Load RAG engine
    chat_engine = get_chat_engine(llm=llm, embed_model=embed_model)

    # Load Structured Analytics
    merchants_path = Path("data/merchants.csv")
    transactions_path = Path("data/transactions.csv")
    
    analytics_engine = AnalyticsEngine(merchants_path, transactions_path)

    # Hybrid Router
    router = HybridRouter(analytics_engine, chat_engine, llm)

    logging.info("System successfully initialized.")



# --- Health Check Endpoint ---
@app.get("/")
def health_check():
    return {"status": "API running successfully"}



# --- Main AI Endpoint ---
@app.post("/ask")
def ask_question(request: QuestionRequest):

    start_time = time.time()

    logging.info(f"Incoming question: {request.question}")

    try:
        result = router.answer(request.question)

        duration = round(time.time() - start_time, 2)

        logging.info(f"Routing type: {result['route']}")
        logging.info(f"Response time: {duration} seconds")

        return {
            "answer": result["answer"],
            "route": result["route"],
            "response_time": duration
        }

    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return {"error": "Internal server error"}