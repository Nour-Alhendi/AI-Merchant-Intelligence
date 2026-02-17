from datasets import Dataset
import pandas as pd

from llama_index.core.indices import VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# ✅ Stage 3 imports (school guide)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

from evaluation.evaluation_config import (
    CHUNKING_STRATEGY_CONFIGS,
    RERANKER_MODEL_NAME,
    RERANKER_CONFIGS,
)
from evaluation.evaluation_helper_functions import (
    generate_qa_dataset,
    get_evaluation_data,
    get_or_build_index,
    save_results,
    evaluate_with_rate_limit,
)
from evaluation.evaluation_model_loader import load_ragas_models

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SIMILARITY_TOP_K,
)
from src.model_loader import get_embedding_model, initialise_llm


def evaluate_baseline() -> None:
    print("--- 🚀 Stage 1: Evaluating Baseline Configuration ---")

    llm_to_test: Groq = initialise_llm()
    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()

    index: VectorStoreIndex = get_or_build_index(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model=embed_model_to_test,
    )

    query_engine: BaseQueryEngine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        llm=llm_to_test,
    )

    qa_dataset: Dataset = generate_qa_dataset(
        query_engine=query_engine,
        questions=questions,
        ground_truths=ground_truths,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        similarity_top_k=SIMILARITY_TOP_K,
    )

    ragas_llm, ragas_embeddings = load_ragas_models()

    results_df: pd.DataFrame = evaluate_with_rate_limit(
        qa_dataset=qa_dataset,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
    )

    results_df["chunk_size"] = CHUNK_SIZE
    results_df["chunk_overlap"] = CHUNK_OVERLAP
    results_df["similarity_top_k"] = SIMILARITY_TOP_K

    save_results(results_df, "baseline_evaluation")
    print("--- ✅ Baseline Evaluation Complete ---")


def evaluate_chunking_strategies() -> None:
    print("\n--- 🚀 Stage 2: Evaluating Chunking Strategies ---")

    llm_to_test: Groq = initialise_llm()
    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()
    ragas_llm, ragas_embeddings = load_ragas_models()

    all_results: list[pd.DataFrame] = []

    for config in CHUNKING_STRATEGY_CONFIGS:
        chunk_size: int = config["size"]
        chunk_overlap: int = config["overlap"]

        print(f"\n--- Testing Chunk Config: size={chunk_size}, overlap={chunk_overlap} ---")

        index: VectorStoreIndex = get_or_build_index(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_model=embed_model_to_test,
        )

        query_engine: BaseQueryEngine = index.as_query_engine(
            similarity_top_k=SIMILARITY_TOP_K,
            llm=llm_to_test,
        )

        qa_dataset: Dataset = generate_qa_dataset(
            query_engine=query_engine,
            questions=questions,
            ground_truths=ground_truths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_top_k=SIMILARITY_TOP_K,
        )

        print("--- 🧪 Running Ragas evaluation for chunking... ---")

        results_df: pd.DataFrame = evaluate_with_rate_limit(
            qa_dataset=qa_dataset,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )

        results_df["chunk_size"] = chunk_size
        results_df["chunk_overlap"] = chunk_overlap
        results_df["similarity_top_k"] = SIMILARITY_TOP_K

        all_results.append(results_df)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)
    save_results(final_df, "chunking_evaluation")
    print("--- ✅ Chunking Strategy Evaluation Complete ---")


def evaluate_reranker_strategies() -> None:
    """
    Stage 3 (school guide):
    Evaluate reranker configs ON TOP of best chunking strategy.
    Here best chunking strategy = current CHUNK_SIZE/CHUNK_OVERLAP in src/config.py.
    """
    print("\n--- 🚀 Stage 3: Evaluating Reranker Strategies ---")

    llm_to_test: Groq = initialise_llm()
    embed_model_to_test: HuggingFaceEmbedding = get_embedding_model()

    questions, ground_truths = get_evaluation_data()
    ragas_llm, ragas_embeddings = load_ragas_models()

    # Build/load index once (best chunking already set in src/config.py)
    index: VectorStoreIndex = get_or_build_index(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embed_model=embed_model_to_test,
    )

    all_results: list[pd.DataFrame] = []

    for config in RERANKER_CONFIGS:
        retriever_k: int = config["retriever_k"]
        reranker_n: int = config["reranker_n"]

        print(
            f"--- Testing Reranker Config: retrieve_k={retriever_k}, rerank_n={reranker_n} ---"
        )

        retriever = index.as_retriever(similarity_top_k=retriever_k)

        reranker = SentenceTransformerRerank(
            top_n=reranker_n,
            model=RERANKER_MODEL_NAME,
        )

        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker],
            llm=llm_to_test,
        )

        # Build/Load QA dataset (cache key uses similarity_top_k)
        qa_dataset: Dataset = generate_qa_dataset(
            query_engine=query_engine,
            questions=questions,
            ground_truths=ground_truths,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            similarity_top_k=retriever_k,  # broad retrieval is the "top_k" for caching
        )

        print("--- 🧪 Running Ragas evaluation for reranker... ---")

        results_df: pd.DataFrame = evaluate_with_rate_limit(
            qa_dataset=qa_dataset,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )

        results_df["chunk_size"] = CHUNK_SIZE
        results_df["chunk_overlap"] = CHUNK_OVERLAP
        results_df["retriever_k"] = retriever_k
        results_df["reranker_n"] = reranker_n
        results_df["reranker_model"] = RERANKER_MODEL_NAME

        all_results.append(results_df)

    final_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)
    save_results(final_df, "reranker_evaluation")
    print("--- ✅ Reranker Strategy Evaluation Complete ---")