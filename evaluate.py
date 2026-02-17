from evaluation.evaluation_engine import (
    evaluate_chunking_strategies,
    evaluate_reranker_strategies,
)

if __name__ == "__main__":
    # Stage 2
    evaluate_chunking_strategies()

    # Stage 3
    evaluate_reranker_strategies()