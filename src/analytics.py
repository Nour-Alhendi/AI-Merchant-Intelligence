from pathlib import Path
import pandas as pd

# --- Structured KPI and risk analytics for merchants ---
class AnalyticsEngine:
    def __init__(self, merchants_path: Path, transactions_path: Path):
        self.merchants = pd.read_csv(merchants_path)
        self.transactions = pd.read_csv(transactions_path)

        if "timestamp" in self.transactions.columns:
            self.transactions["timestamp"] = pd.to_datetime(
                self.transactions["timestamp"], errors="coerce"
            )

        for col in ["is_refund", "is_fraud", "is_chargeback"]:
            if col in self.transactions.columns:
                self.transactions[col] = self.transactions[col].astype(int)

# --- Return aggregated KPIs for a merchant ---
    def merchant_summary(self, merchant_id: str) -> dict:
        df = self.transactions[self.transactions["merchant_id"] == merchant_id]

        return {
            "merchant_id": merchant_id,
            "total_transactions": int(len(df)),
            "total_revenue": float(df["amount"].sum()) if len(df) > 0 else 0.0,
            "average_transaction": float(df["amount"].mean()) if len(df) > 0 else 0.0,
            "refunds": int(df["is_refund"].sum()) if len(df) > 0 else 0,
            "frauds": int(df["is_fraud"].sum()) if len(df) > 0 else 0,
            "chargebacks": int(df["is_chargeback"].sum()) if len(df) > 0 else 0,
            "refund_ratio": float(df["is_refund"].mean()) if len(df) > 0 else 0.0,
            "chargeback_rate": float(df["is_chargeback"].mean()) if len(df) > 0 else 0.0,
            "fraud_rate": float(df["is_fraud"].mean()) if len(df) > 0 else 0.0,
        }
# --- Return merchant with highest chargeback rate ---
    def highest_chargeback_merchant(self) -> dict | None:
        if self.transactions.empty:
            return None

        grouped = (
            self.transactions
            .groupby("merchant_id")["is_chargeback"]
            .agg(total_transactions="count", chargebacks="sum")
            .reset_index()
        )

        grouped = grouped[grouped["total_transactions"] > 0]
        if grouped.empty:
            return None

        grouped["chargeback_rate"] = grouped["chargebacks"] / grouped["total_transactions"]

        top = grouped.sort_values("chargeback_rate", ascending=False).iloc[0]

        return {
            "merchant_id": str(top["merchant_id"]),
            "total_transactions": int(top["total_transactions"]),
            "chargebacks": int(top["chargebacks"]),
            "chargeback_rate": float(top["chargeback_rate"]),
        }

# --- Return daily revenue time series for a merchant ---
    def daily_revenue(self, merchant_id: str):
        df = self.transactions[self.transactions["merchant_id"] == merchant_id].copy()
        if "timestamp" not in df.columns or df.empty:
            return None

        df["date"] = df["timestamp"].dt.date
        return df.groupby("date")["amount"].sum().reset_index()

# --- Compute rule-based fraud and risk flags ---
    def anomaly_flags(self, merchant_id: str) -> dict:
        df = self.transactions[self.transactions["merchant_id"] == merchant_id]
        if df.empty:
            return {
                "refund_ratio": 0.0,
                "high_refund_ratio": False,
                "large_transactions": False,
                "high_chargeback_rate": False,
                "cross_border_ratio": 0.0,
                "high_cross_border": False,
            }

        refund_ratio = float(df["is_refund"].mean())
        chargeback_rate = float(df["is_chargeback"].mean())
        cross_border_ratio = float((df["transaction_country"] != df["merchant_country"]).mean())

        return {
            "refund_ratio": refund_ratio,
            "high_refund_ratio": bool(refund_ratio > 0.30),
            "large_transactions": bool((df["amount"] > 10000).any()),
            "high_chargeback_rate": bool(chargeback_rate > 0.02),
            "cross_border_ratio": cross_border_ratio,
            "high_cross_border": bool(cross_border_ratio > 0.25),
        }

# --- Convert KPIs and risk flags into LLM-ready context ---
    def build_structured_context(self, merchant_id: str) -> str:
        s = self.merchant_summary(merchant_id)
        f = self.anomaly_flags(merchant_id)

        context = f"""
STRUCTURED MERCHANT ANALYTICS (from transactions dataset)
Merchant ID: {s['merchant_id']}
Total transactions: {s['total_transactions']}
Total revenue: {s['total_revenue']}
Average transaction: {s['average_transaction']}
Refunds: {s['refunds']} (refund_ratio: {s['refund_ratio']})
Chargebacks: {s['chargebacks']} (chargeback_rate: {s['chargeback_rate']})
Frauds: {s['frauds']} (fraud_rate: {s['fraud_rate']})

Flags:
- high_refund_ratio (>30%): {f['high_refund_ratio']}
- high_chargeback_rate (>2%): {f['high_chargeback_rate']}
- large_transactions (>10k): {f['large_transactions']}
- cross_border_ratio: {f['cross_border_ratio']}
- high_cross_border (>25%): {f['high_cross_border']}
""".strip()

        return context