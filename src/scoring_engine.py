"""
Scoring Engine for CrediVist
Computes 4 sub-scores and combines them into a Final Trust Score (0–900).
"""

import numpy as np
import pandas as pd


# ─── Sub-Score Weights ──────────────────────────────────────────────────────
SUB_SCORE_WEIGHTS = {
    "financial_stability": 0.35,
    "payment_discipline": 0.30,
    "digital_behavior": 0.20,
    "work_reliability": 0.15,
}

# Maximum score (like CIBIL range)
MAX_SCORE = 900
MIN_SCORE = 300  # minimum floor


# ─── 1. Financial Stability Score (0–100) ───────────────────────────────────
def financial_stability_score(row: pd.Series) -> dict:
    """
    Components: Income Stability + Cash Flow Ratio + Savings Behavior
    """
    income_stability = row["feat_income_stability"] * 100
    # Bonus for positive trend
    trend_bonus = max(row["feat_income_trend"] * 10, 0)
    cash_flow = row["feat_cash_flow_ratio"] * 100
    savings = row["feat_savings_score"] * 100

    raw = (income_stability * 0.40 + cash_flow * 0.35 + savings * 0.25 + trend_bonus)
    score = np.clip(raw, 0, 100)

    return {
        "sub_financial_stability": round(float(score), 2),
        "detail_income_stability": round(float(income_stability), 2),
        "detail_cash_flow": round(float(cash_flow), 2),
        "detail_savings": round(float(savings), 2),
        "detail_trend_bonus": round(float(trend_bonus), 2),
    }


# ─── 2. Payment Discipline Score (0–100) ────────────────────────────────────
def payment_discipline_score(row: pd.Series) -> dict:
    """
    Components: Utility Timeliness + EMI-like Behavior + Recharge Pattern
    """
    utility = row["feat_utility_score"] * 100
    emi = row["feat_emi_score"] * 100
    recharge = row["recharge_regularity"] * 100

    raw = utility * 0.40 + emi * 0.35 + recharge * 0.25
    score = np.clip(raw, 0, 100)

    return {
        "sub_payment_discipline": round(float(score), 2),
        "detail_utility": round(float(utility), 2),
        "detail_emi": round(float(emi), 2),
        "detail_recharge": round(float(recharge), 2),
    }


# ─── 3. Digital Behavior Score (0–100) ──────────────────────────────────────
def digital_behavior_score(row: pd.Series) -> dict:
    """
    Components: Transaction Regularity + Expense Categorization
    """
    txn_reg = row["feat_txn_regularity"] * 100
    expense = row["feat_expense_score"] * 100

    raw = txn_reg * 0.50 + expense * 0.50
    score = np.clip(raw, 0, 100)

    return {
        "sub_digital_behavior": round(float(score), 2),
        "detail_txn_regularity": round(float(txn_reg), 2),
        "detail_expense": round(float(expense), 2),
    }


# ─── 4. Work Reliability Score (0–100) ──────────────────────────────────────
def work_reliability_score(row: pd.Series) -> dict:
    """
    Components: Platform Tenure + Rating + Active Days + Income Diversity
    """
    work_rel = row["feat_work_reliability"] * 100
    diversity = row["feat_income_diversity"] * 100
    shock = row["feat_shock_recovery"] * 100

    raw = work_rel * 0.45 + diversity * 0.25 + shock * 0.30
    score = np.clip(raw, 0, 100)

    return {
        "sub_work_reliability": round(float(score), 2),
        "detail_work_rel": round(float(work_rel), 2),
        "detail_diversity": round(float(diversity), 2),
        "detail_shock_recovery": round(float(shock), 2),
    }


# ─── Compute Base Trust Score ───────────────────────────────────────────────
def compute_base_score(row: pd.Series) -> dict:
    """
    Weighted combination of all 4 sub-scores → base score (0–100).
    Then mapped to 300–900 range.
    """
    fin = financial_stability_score(row)
    pay = payment_discipline_score(row)
    dig = digital_behavior_score(row)
    wrk = work_reliability_score(row)

    base_100 = (
        fin["sub_financial_stability"] * SUB_SCORE_WEIGHTS["financial_stability"] +
        pay["sub_payment_discipline"] * SUB_SCORE_WEIGHTS["payment_discipline"] +
        dig["sub_digital_behavior"] * SUB_SCORE_WEIGHTS["digital_behavior"] +
        wrk["sub_work_reliability"] * SUB_SCORE_WEIGHTS["work_reliability"]
    )

    # Map to 300–900
    base_score = MIN_SCORE + (base_100 / 100) * (MAX_SCORE - MIN_SCORE)
    base_score = np.clip(base_score, MIN_SCORE, MAX_SCORE)

    result = {
        "base_score_100": round(float(base_100), 2),
        "base_trust_score": round(float(base_score), 0),
    }
    result.update(fin)
    result.update(pay)
    result.update(dig)
    result.update(wrk)
    return result


def compute_data_confidence(row: pd.Series = None) -> float:
    """
    Compute a data-driven confidence score (0.0–1.0) based on how
    many data signals are present and how rich the data is.
    """
    if row is None:
        return 0.65  # default when no data available

    checks = []

    # 1. Income history length (6 months ideal)
    try:
        import json as _json
        incomes = _json.loads(row["monthly_incomes"]) if isinstance(
            row.get("monthly_incomes"), str
        ) else row.get("monthly_incomes", [])
        checks.append(min(len(incomes) / 6, 1.0))
    except Exception:
        checks.append(0.3)

    # 2. Transaction volume (150+ is high confidence)
    txn_count = float(row.get("total_transactions", 0))
    checks.append(min(txn_count / 150, 1.0))

    # 3. Utility bill history present
    total_bills = float(row.get("total_bills", 0))
    checks.append(min(total_bills / 12, 1.0))

    # 4. Platform tenure (12+ months ideal)
    tenure = float(row.get("tenure_months", 0))
    checks.append(min(tenure / 12, 1.0))

    # 5. Savings data available
    has_savings_data = (
        row.get("has_recurring_savings", 0) != 0
        or float(row.get("avg_monthly_savings", 0)) > 0
    )
    checks.append(1.0 if has_savings_data else 0.4)

    # 6. Income diversity (multiple sources = more reliable)
    sources = int(row.get("num_income_sources", 1))
    checks.append(min(sources / 3, 1.0))

    # Weighted average
    weights = [0.20, 0.25, 0.15, 0.15, 0.10, 0.15]
    raw_confidence = sum(c * w for c, w in zip(checks, weights))

    # Floor at 0.40, cap at 0.95
    return round(float(np.clip(raw_confidence, 0.40, 0.95)), 2)


def compute_final_score(base_score: float, risk_probability: float,
                        row: pd.Series = None) -> dict:
    """
    FinalScore = BaseScore × (1 - RiskProbability)
    Maps result back to 300–900 scale.
    Confidence is now data-driven based on actual data completeness.
    """
    adjusted_100 = (base_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE) * 100
    adjusted_100 = adjusted_100 * (1 - risk_probability)
    final = MIN_SCORE + (adjusted_100 / 100) * (MAX_SCORE - MIN_SCORE)
    final = np.clip(final, MIN_SCORE, MAX_SCORE)

    # Determine grade
    if final >= 750:
        grade = "Excellent"
        color = "#22c55e"
    elif final >= 650:
        grade = "Good"
        color = "#84cc16"
    elif final >= 500:
        grade = "Fair"
        color = "#eab308"
    elif final >= 400:
        grade = "Poor"
        color = "#f97316"
    else:
        grade = "Very Poor"
        color = "#ef4444"

    # Data-driven confidence
    confidence = compute_data_confidence(row)

    return {
        "final_trust_score": round(float(final), 0),
        "risk_probability": round(float(risk_probability), 4),
        "grade": grade,
        "grade_color": color,
        "confidence": confidence,
    }


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply scoring to entire DataFrame.
    Returns DataFrame with all score columns.
    """
    scores = df.apply(compute_base_score, axis=1, result_type="expand")
    return pd.concat([df, scores], axis=1)


def get_score_breakdown(row: pd.Series) -> dict:
    """
    Return a structured breakdown for display in UI.
    """
    return {
        "Financial Stability": {
            "score": row.get("sub_financial_stability", 0),
            "weight": f"{SUB_SCORE_WEIGHTS['financial_stability']*100:.0f}%",
            "components": {
                "Income Stability": row.get("detail_income_stability", 0),
                "Cash Flow Health": row.get("detail_cash_flow", 0),
                "Savings Discipline": row.get("detail_savings", 0),
                "Income Trend Bonus": row.get("detail_trend_bonus", 0),
            }
        },
        "Payment Discipline": {
            "score": row.get("sub_payment_discipline", 0),
            "weight": f"{SUB_SCORE_WEIGHTS['payment_discipline']*100:.0f}%",
            "components": {
                "Utility Bill Timeliness": row.get("detail_utility", 0),
                "EMI-like Behavior": row.get("detail_emi", 0),
                "Recharge Regularity": row.get("detail_recharge", 0),
            }
        },
        "Digital Behavior": {
            "score": row.get("sub_digital_behavior", 0),
            "weight": f"{SUB_SCORE_WEIGHTS['digital_behavior']*100:.0f}%",
            "components": {
                "Transaction Regularity": row.get("detail_txn_regularity", 0),
                "Essential Expense Ratio": row.get("detail_expense", 0),
            }
        },
        "Work Reliability": {
            "score": row.get("sub_work_reliability", 0),
            "weight": f"{SUB_SCORE_WEIGHTS['work_reliability']*100:.0f}%",
            "components": {
                "Platform Performance": row.get("detail_work_rel", 0),
                "Income Diversity": row.get("detail_diversity", 0),
                "Shock Recovery": row.get("detail_shock_recovery", 0),
            }
        },
    }
