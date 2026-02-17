# data/generate_transactions.py
# Creates synthetic structured merchant + transaction data for 5 years.
# Outputs:
#   - data/merchants.csv
#   - data/transactions.csv
#
# Run from project root:
#   python data/generate_transactions.py

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# =========================================================
# CONFIG (tuned for ~1M rows over 5 years)
# =========================================================
@dataclass
class GenConfig:
    seed: int = 42

    # last 5 years (calendar-style)
    start_date: str = "2021-01-01"
    end_date: str = "2025-12-31"

    # dataset target size
    target_transactions: int = 1_000_000

    # merchants
    n_merchants: int = 220

    # base tx/day per merchant (sampled)
    base_txs_per_day_min: int = 1
    base_txs_per_day_max: int = 5

    # baseline rates (synthetic)
    base_refund_rate: float = 0.03
    base_fraud_rate: float = 0.02
    base_chargeback_rate: float = 0.006

    # IMPORTANT: output directory name inside project root
    out_dir_name: str = "data"


CFG = GenConfig()


# =========================================================
# DOMAIN SETUP
# =========================================================
COUNTRIES = ["DE", "FR", "NL", "ES", "IT", "UK", "US", "VN", "PL", "SE"]
CURRENCIES = {
    "DE": "EUR",
    "FR": "EUR",
    "NL": "EUR",
    "ES": "EUR",
    "IT": "EUR",
    "PL": "PLN",
    "SE": "SEK",
    "UK": "GBP",
    "US": "USD",
    "VN": "VND",
}

PAYMENT_METHODS = ["card", "bank_transfer", "wallet", "bnpl"]
PM_PROBS = [0.72, 0.10, 0.12, 0.06]

INDUSTRIES = ["ecommerce", "travel", "gaming", "electronics", "grocery", "services", "marketplace"]
INDUSTRY_PROBS = [0.34, 0.10, 0.10, 0.12, 0.14, 0.12, 0.08]

INDUSTRY_RISK_BIAS = {
    "gaming": 0.75,
    "travel": 0.55,
    "electronics": 0.52,
    "marketplace": 0.48,
    "ecommerce": 0.42,
    "services": 0.30,
    "grocery": 0.18,
}

# purely synthetic "high risk geos"
HIGH_RISK_GEOS = {"US", "VN"}


# =========================================================
# HELPERS
# =========================================================
def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def daterange(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="D")


def seasonality_multiplier(day: pd.Timestamp) -> float:
    m = day.month
    if m in (11, 12):  # peak
        return 1.30
    if m in (6, 7, 8):  # summer dip
        return 0.90
    if m in (1, 2):  # post-holiday slight dip
        return 0.95
    return 1.00


def weekday_multiplier(day: pd.Timestamp) -> float:
    wd = day.weekday()  # Mon=0
    return 1.10 if wd in (5, 6) else 1.00


# =========================================================
# GENERATE MERCHANTS
# =========================================================
def generate_merchants(g: np.random.Generator, cfg: GenConfig) -> pd.DataFrame:
    merchant_ids = [f"m_{i:05d}" for i in range(1, cfg.n_merchants + 1)]
    industries = g.choice(INDUSTRIES, size=cfg.n_merchants, p=np.array(INDUSTRY_PROBS))
    countries = g.choice(COUNTRIES, size=cfg.n_merchants)

    # onboarding between 2019-01-01 and end_date-180d
    start_onb = pd.Timestamp("2019-01-01")
    end_onb = pd.Timestamp(cfg.end_date) - pd.Timedelta(days=180)
    onb_days = (end_onb - start_onb).days
    onboarding_dates = [
        start_onb + pd.Timedelta(days=int(g.integers(0, onb_days))) for _ in range(cfg.n_merchants)
    ]

    base_txs = g.integers(cfg.base_txs_per_day_min, cfg.base_txs_per_day_max + 1, size=cfg.n_merchants)

    # avg ticket by industry (lognormal)
    avg_ticket_params = {
        "grocery": (math.log(25), 0.35),
        "services": (math.log(60), 0.55),
        "ecommerce": (math.log(80), 0.65),
        "electronics": (math.log(200), 0.75),
        "marketplace": (math.log(120), 0.70),
        "travel": (math.log(450), 0.80),
        "gaming": (math.log(35), 0.60),
    }

    avg_ticket = []
    baseline_risk = []
    for ind, c in zip(industries, countries):
        mu, sigma = avg_ticket_params[ind]
        avg_ticket.append(float(g.lognormal(mean=mu, sigma=sigma)))

        bias = INDUSTRY_RISK_BIAS.get(ind, 0.4)
        geo_bias = 0.10 if c in HIGH_RISK_GEOS else 0.0
        r = clamp(float(g.normal(loc=bias + geo_bias, scale=0.12)), 0.05, 0.95)
        baseline_risk.append(r)

    df = pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "industry": industries,
            "merchant_country": countries,
            "currency": [CURRENCIES[c] for c in countries],
            "onboarding_date": pd.to_datetime(onboarding_dates),
            "base_avg_txs_per_day": base_txs,
            "avg_ticket_amount": np.round(np.array(avg_ticket), 2),
            "baseline_risk_score": np.round(np.array(baseline_risk), 3),
        }
    )
    return df


# =========================================================
# GENERATE TRANSACTIONS
# =========================================================
def estimate_expected_total(merchants: pd.DataFrame, days: pd.DatetimeIndex) -> float:
    # rough expected multiplier across time
    mean_mult = 1.02
    return float(merchants["base_avg_txs_per_day"].sum() * len(days) * mean_mult)


def generate_transactions(
    g: np.random.Generator,
    cfg: GenConfig,
    merchants: pd.DataFrame,
) -> pd.DataFrame:
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    days = daterange(start, end)

    expected = estimate_expected_total(merchants, days)
    scale = cfg.target_transactions / expected if expected > 0 else 1.0
    scale = clamp(scale, 0.2, 3.0)

    tx_rows: List[dict] = []
    tx_id_counter = 1

    for _, m in merchants.iterrows():
        mid = str(m["merchant_id"])
        m_country = str(m["merchant_country"])
        currency = str(m["currency"])
        base_lambda = float(m["base_avg_txs_per_day"]) * scale
        avg_ticket = float(m["avg_ticket_amount"])
        risk = float(m["baseline_risk_score"])
        onboard = pd.Timestamp(m["onboarding_date"])

        active_days = days[days >= max(onboard, start)]
        if len(active_days) == 0:
            continue

        spike_prob = clamp(0.03 + risk * 0.10, 0.02, 0.15)
        n_spikes = int(g.binomial(n=4, p=spike_prob))
        spike_starts = []
        for _ in range(n_spikes):
            if len(active_days) < 60:
                break
            s = active_days[int(g.integers(0, len(active_days) - 10))]
            spike_starts.append(s)

        for d in active_days:
            lam = base_lambda * seasonality_multiplier(d) * weekday_multiplier(d)

            for s in spike_starts:
                if s <= d <= s + pd.Timedelta(days=5):
                    lam *= float(g.uniform(2.0, 4.0))

            n = int(g.poisson(lam=lam))
            if n <= 0:
                continue

            # timestamps within day
            hours = g.integers(0, 24, size=n)
            minutes = g.integers(0, 60, size=n)
            seconds = g.integers(0, 60, size=n)
            ts = (
                pd.to_datetime(d.date())
                + pd.to_timedelta(hours, unit="h")
                + pd.to_timedelta(minutes, unit="m")
                + pd.to_timedelta(seconds, unit="s")
            )

            # amounts: lognormal around avg_ticket
            sigma = 0.60 + 0.40 * risk
            mu = math.log(max(avg_ticket, 5.0)) - (sigma**2) / 2
            amounts = g.lognormal(mean=mu, sigma=sigma, size=n)

            # occasional high-amount cluster
            if g.random() < (0.02 + 0.08 * risk):
                idx = g.choice(n, size=max(1, n // 15), replace=False)
                amounts[idx] *= float(g.uniform(5.0, 15.0))

            amounts = np.clip(amounts, 1.0, None)

            # cross-border probability
            cross_border_prob = clamp(0.05 + 0.20 * risk, 0.03, 0.30)
            tx_countries = []
            for _i in range(n):
                if g.random() < cross_border_prob:
                    other = str(g.choice([c for c in COUNTRIES if c != m_country]))
                    tx_countries.append(other)
                else:
                    tx_countries.append(m_country)

            payment_methods = g.choice(PAYMENT_METHODS, size=n, p=np.array(PM_PROBS))

            # refunds
            refund_prob = clamp(cfg.base_refund_rate + 0.05 * risk, 0.01, 0.18)
            is_refund = g.random(n) < refund_prob

            # fraud probability
            is_night = (hours <= 5) | (hours >= 23)
            cross_border = np.array([c != m_country for c in tx_countries])
            geo_risky = np.array([c in HIGH_RISK_GEOS for c in tx_countries])
            high_amount = amounts > (avg_ticket * (6.0 + 5.0 * risk))

            fraud_p = (
                cfg.base_fraud_rate
                + 0.05 * risk
                + 0.02 * is_night.astype(float)
                + 0.03 * cross_border.astype(float)
                + 0.04 * geo_risky.astype(float)
                + 0.05 * high_amount.astype(float)
            )
            fraud_p = np.clip(fraud_p, 0.001, 0.55)
            is_fraud = (g.random(n) < fraud_p) & (~is_refund)

            # chargebacks
            cb_base = cfg.base_chargeback_rate + 0.008 * risk
            cb_from_fraud = 0.55
            cb_p = np.where(is_fraud, cb_from_fraud, cb_base)
            cb_p = np.clip(cb_p, 0.0005, 0.70)
            is_chargeback = (g.random(n) < cb_p) & (~is_refund)

            for i in range(n):
                tx_rows.append(
                    {
                        "transaction_id": f"t_{tx_id_counter:09d}",
                        "merchant_id": mid,
                        "timestamp": ts[i],
                        "amount": float(round(amounts[i], 2)),
                        "currency": currency,
                        "merchant_country": m_country,
                        "transaction_country": tx_countries[i],
                        "payment_method": str(payment_methods[i]),
                        "is_refund": int(is_refund[i]),
                        "is_fraud": int(is_fraud[i]),
                        "is_chargeback": int(is_chargeback[i]),
                    }
                )
                tx_id_counter += 1

    tx = pd.DataFrame(tx_rows)
    if not tx.empty:
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])
        tx.sort_values(["timestamp", "transaction_id"], inplace=True, ignore_index=True)
    return tx


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    # project root = parent of /data
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    out_dir = PROJECT_ROOT / CFG.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    g = rng(CFG.seed)

    print("Generating merchants...")
    merchants = generate_merchants(g, CFG)

    print("Generating transactions (~target)...")
    tx = generate_transactions(g, CFG, merchants)

    merchants_path = out_dir / "merchants.csv"
    tx_path = out_dir / "transactions.csv"

    merchants.to_csv(merchants_path, index=False)
    tx.to_csv(tx_path, index=False)

    print("\n✅ Done.")
    print(f"Saved: {merchants_path}")
    print(f"Saved: {tx_path}")
    print(f"Merchants: {len(merchants):,}")
    print(f"Transactions: {len(tx):,}")

    if not tx.empty:
        fraud_rate = float(tx["is_fraud"].mean())
        cb_rate = float(tx["is_chargeback"].mean())
        refund_rate = float(tx["is_refund"].mean())
        print(f"Fraud rate: {fraud_rate:.3%}")
        print(f"Chargeback rate: {cb_rate:.3%}")
        print(f"Refund rate: {refund_rate:.3%}")


if __name__ == "__main__":
    main()