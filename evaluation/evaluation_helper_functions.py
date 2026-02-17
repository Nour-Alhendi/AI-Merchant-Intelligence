import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.executor import Executor
from ragas.llms.base import LlamaIndexLLMWrapper

from evaluation.evaluation_config import (
    EVALUATION_RESULTS_PATH,
    EXPERIMENTAL_VECTOR_STORES_PATH,
    QA_DATASET_CACHE_PATH,
    SLEEP_PER_QUESTION,
    SLEEP_PER_EVALUATION,
    EVALUATION_METRICS,
)
from evaluation.evaluation_questions import EVALUATION_DATA

from src.config import DOCS_PATH


# =========================================================
# 1️⃣ Load Evaluation Questions
# =========================================================

def get_evaluation_data() -> tuple[list[str], list[str]]:
    questions = [item["question"] for item in EVALUATION_DATA]
    ground_truths = [item["ground_truth"] for item in EVALUATION_DATA]
    return questions, ground_truths


# =========================================================
# 2️⃣ Build or Load Vector Store
# =========================================================

def get_or_build_index(
    chunk_size: int,
    chunk_overlap: int,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:

    vector_store_id = f"vs_chunk_{chunk_size}_overlap_{chunk_overlap}"
    path = EXPERIMENTAL_VECTOR_STORES_PATH / vector_store_id

    EXPERIMENTAL_VECTOR_STORES_PATH.mkdir(parents=True, exist_ok=True)

    if path.exists() and any(path.iterdir()):
        print(f"--- Loading existing index: {vector_store_id} ---")
        storage_context = StorageContext.from_defaults(persist_dir=str(path))
        return load_index_from_storage(storage_context, embed_model=embed_model)

    print(f"--- Creating new index: {vector_store_id} ---")

    documents: list[Document] = SimpleDirectoryReader(
        input_dir=DOCS_PATH.as_posix(),
        required_exts=[".text", ".pdf", ".md"],
        recursive=True,
    ).load_data()

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter],
        embed_model=embed_model,
    )

    path.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(path))

    print(f"--- Saved index to {vector_store_id} ---")
    return index


# =========================================================
# 3️⃣ Generate QA Dataset (WITH CACHE)
# =========================================================

def generate_qa_dataset(
    query_engine: BaseQueryEngine,
    questions: list[str],
    ground_truths: list[str],
    chunk_size: int,
    chunk_overlap: int,
    similarity_top_k: int,
) -> Dataset:

    QA_DATASET_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    cache_file = (
        QA_DATASET_CACHE_PATH
        / f"qa_chunk_{chunk_size}_overlap_{chunk_overlap}_topk_{similarity_top_k}.jsonl"
    )

    # ✅ Load from cache if exists
    if cache_file.exists():
        print(f"--- ✅ Loading QA dataset from cache: {cache_file.name} ---")
        rows = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return Dataset.from_list(rows)

    # ❌ Otherwise create it
    print(f"--- 🧪 Creating QA dataset: {cache_file.name} ---")

    rows = []

    for i, question in enumerate(questions):
        print(f"Q {i+1}/{len(questions)}: {question[:50]}...")

        response = query_engine.query(question)

        row = {
            "question": question,
            "answer": str(response),
            "contexts": [node.get_content() for node in response.source_nodes],
            "ground_truth": ground_truths[i],
        }

        rows.append(row)

        if i + 1 < len(questions):
            time.sleep(SLEEP_PER_QUESTION)

    # 💾 Save cache
    with open(cache_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"--- 💾 Saved QA dataset cache to: {cache_file} ---")

    return Dataset.from_list(rows)


# =========================================================
# 4️⃣ Evaluation Without Rate Limit
# =========================================================

def evaluate_without_rate_limit(
    qa_dataset: Dataset,
    ragas_llm: LlamaIndexLLMWrapper,
    ragas_embeddings: HuggingFaceEmbeddings,
) -> pd.DataFrame:

    print("--- ⚡ Running evaluation without rate limiting... ---")

    result: EvaluationResult | Executor = evaluate(
        dataset=qa_dataset,
        metrics=EVALUATION_METRICS,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True,
    )

    print("--- ✅ Evaluation complete! ---")
    return result.to_pandas()


# =========================================================
# 5️⃣ Evaluation WITH Rate Limit
# =========================================================

def evaluate_with_rate_limit(
    qa_dataset: Dataset,
    ragas_llm: LlamaIndexLLMWrapper,
    ragas_embeddings: HuggingFaceEmbeddings,
) -> pd.DataFrame:

    print("--- 🐢 Running evaluation with rate limiting... ---")

    partial_results = []

    for i, row in enumerate(qa_dataset):
        print(f"Evaluating {i+1}/{len(qa_dataset)}: '{row['question'][:60]}...'")

        single_row = Dataset.from_dict(
            {key: [value] for key, value in row.items()}
        )

        result = evaluate(
            dataset=single_row,
            metrics=EVALUATION_METRICS,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=True,
        )

        partial_results.append(result.to_pandas())

        if i + 1 < len(qa_dataset):
            print(f"Sleeping {SLEEP_PER_EVALUATION}s to respect rate limits...")
            time.sleep(SLEEP_PER_EVALUATION)

    print("--- ✅ Evaluation complete! ---")
    return pd.concat(partial_results, ignore_index=True)


# =========================================================
# 6️⃣ Save Results
# =========================================================

def save_results(results_df: pd.DataFrame, filename_prefix: str) -> None:

    EVALUATION_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    detailed_path = (
        EVALUATION_RESULTS_PATH
        / f"{filename_prefix}_detailed_{timestamp}.csv"
    )

    results_df.to_csv(detailed_path, index=False)
    print(f"--- 💾 Detailed results saved to: {detailed_path} ---")

    param_cols = [
    col for col in ["chunk_size", "chunk_overlap", "similarity_top_k", "retriever_k", "reranker_n", "reranker_model"]
    if col in results_df.columns
    ]
    
    if param_cols:
        summary = results_df.groupby(param_cols).mean(numeric_only=True)
        summary_path = (
            EVALUATION_RESULTS_PATH
            / f"{filename_prefix}_summary_{timestamp}.csv"
        )
        summary.to_csv(summary_path)
        print(f"--- 💾 Summary saved to: {summary_path} ---")