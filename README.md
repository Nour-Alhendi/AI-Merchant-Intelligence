# AI Merchant Intelligence (Hybrid Structured + RAG)

A production-style fintech analytics assistant that answers merchant risk questions using:
1) **Structured analytics** computed from transaction data (ground truth)
2) **RAG knowledge base** (policies & explanations)
3) **LLM routing** to decide which system should answer each question

## Why this project
**Key idea:** In real fintech risk systems, core numbers must be deterministic and auditable.  
The LLM is used for explanation and context, not for inventing metrics.

## Features
- Structured merchant KPI computation from CSV
  - total_transactions, total_revenue, average_transaction
  - refunds + refund_ratio
  - chargebacks + chargeback_rate
  - frauds + fraud_rate
  - cross_border_ratio
- Policy thresholds (from knowledge base)
  - refund_ratio > 30%
  - chargeback_rate > 2%
  - large_transactions > 10,000
  - high_cross_border > 25%
- RAG pipeline using LlamaIndex + HuggingFace embeddings
- Cross-encoder reranker for better retrieval relevance
- FastAPI service + Streamlit UI
- Synthetic dataset generator (~1M transactions / 5 years)

## Architecture (high level)
User -> Streamlit UI -> FastAPI (/ask) -> HybridRouter
- GLOBAL_STRUCTURED: dataset-wide aggregation
- MERCHANT_STRUCTURED: merchant-specific KPIs
- RAG_ONLY: definitions/policy explanations
- HYBRID: KPIs + policy explanation

## Quickstart

### Create environment
```bash
conda create -n rag-project-env python=3.12 -y
conda activate rag-project-env
pip install -r requirements.txt


## Environment Variables
Create a `.env` file in the project root and add:
GROQ_API_KEY=your_groq_api_key_here
The application will not start without a valid API key.


## Example questions
Which merchant has the highest chargeback rate?
Is merchant m_00062 above the 2% chargeback threshold?
Explain why high cross-border ratio increases fraud risk.
Compare fraud rate vs chargeback rate.
Does merchant M122 violate policy thresholds?