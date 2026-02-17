# src/router.py
import json
import re
from typing import Optional, Dict, Any

from llama_index.core.base.llms.types import ChatMessage


ROUTER_SYSTEM_PROMPT = """
You are a routing classifier for a fintech analytics assistant.
Decide which tool should answer the user's question.

Return ONLY valid JSON.

Schema:
{
  "route": "GLOBAL_STRUCTURED" | "MERCHANT_STRUCTURED" | "RAG_ONLY" | "HYBRID",
  "merchant_id": string | null,
  "intent": string
}

Rules:
- If the question asks to compare, rank, top, highest, lowest, or "which merchant" across merchants -> GLOBAL_STRUCTURED.
- If it asks about a specific merchant (mentions merchant id like m_00062) -> MERCHANT_STRUCTURED.
- If it asks definitions/policy explanations without needing numbers -> RAG_ONLY.
- If it asks to interpret KPIs AND reference policy thresholds/actions -> HYBRID.
"""

# Accepts: m_00062, m00062, m 62, M4, M 00057, etc.
MERCHANT_ID_REGEX = re.compile(r"\bm[\s_]?0*(\d{1,5})\b", re.IGNORECASE)

# --- Extract merchant id in flexible formats and normalize to m_00000 ---
def _extract_merchant_id(text: str) -> Optional[str]:
    m = MERCHANT_ID_REGEX.search(text)
    if not m:
        return None

    number = int(m.group(1))
    return f"m_{number:05d}"

# --- Ask the router LLM for a JSON routing decision (with optional merchant_id) ---
def route_question(router_llm, question: str) -> Dict[str, Any]:
    """
    router_llm is a LlamaIndex LLM (Groq). We call chat() with ChatMessage list.
    """

    # help the model by giving merchant_id if present
    mid = _extract_merchant_id(question)

    user_text = question
    if mid:
        user_text += f"\n\nDetected merchant_id: {mid}"

    messages = [
        ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_text),
    ]

    resp = router_llm.chat(messages)
    text = resp.message.content.strip()

    decision = json.loads(text)

    # Ensure merchant_id is filled if we detected one locally
    if decision.get("merchant_id") is None and mid is not None:
        decision["merchant_id"] = mid

    return decision


class HybridRouter:
    """
    Orchestrator: decides route via router_llm and then calls analytics and/or RAG chat_engine.
    """

    def __init__(self, analytics_engine, chat_engine, router_llm):
        self.analytics = analytics_engine
        self.chat = chat_engine
        self.router_llm = router_llm
# Route and return an answer with a route label
    def answer(self, question: str) -> dict:
        decision = route_question(self.router_llm, question)
        route = decision.get("route", "RAG_ONLY")
        merchant_id = decision.get("merchant_id", None)

        q_lower = question.lower()

        # 1) Global structured analytics (across all merchants)
        if route == "GLOBAL_STRUCTURED":
            # implement the most important global intents first
            if "chargeback" in q_lower:
                stats = self.analytics.highest_chargeback_merchant()
                if not stats:
                    return {"route": "structured_global", "answer": "No transaction data available to compute chargeback rates."}

                answer = (
                    f"Merchant **{stats['merchant_id']}** has the highest chargeback rate.\n\n"
                    f"- Total transactions: {stats['total_transactions']}\n"
                    f"- Chargebacks: {stats['chargebacks']}\n"
                    f"- Chargeback rate: {stats['chargeback_rate']*100:.2f}%\n\n"
                    f"Policy check: > **2%** chargeback rate → **elevated risk**."
                )
                return {"route": "structured_global", "answer": answer}

            # fallback if global metric not implemented
            return {"route": "structured_global", "answer": "Global structured intent detected, but not implemented for this metric yet."}

        # 2) Merchant-specific KPIs (structured + optionally explained by LLM)
        if route == "MERCHANT_STRUCTURED":
            if not merchant_id:
                return {"route": "merchant_structured", "answer": "Please provide a merchant_id (e.g., m_00062)."}
            ctx = self.analytics.build_structured_context(merchant_id)
            # If user wants explanation, send ctx into chat_engine
            enhanced_prompt = f"{ctx}\n\nUser question:\n{question}"
            answer = str(self.chat.chat(enhanced_prompt))
            return {"route": "merchant_structured", "answer": answer}

        # 3) Hybrid: structured context + policy explanation via RAG
        if route == "HYBRID":
            structured = ""
            if merchant_id:
                structured = self.analytics.build_structured_context(merchant_id)

            enhanced_prompt = f"{structured}\n\nUser question:\n{question}"
            answer = str(self.chat.chat(enhanced_prompt))
            return {"route": "hybrid", "answer": answer}

        # 4) RAG-only: definitions/policy/general explanations
        answer = str(self.chat.chat(question))
        return {"route": "rag", "answer": answer}