"""
Feature Engineering Module for CrediVist
Implements all 10 alternative credit scoring features (A–J).
"""

import numpy as np
import pandas as pd
import json


# ─── A. Income Stability Index ──────────────────────────────────────────────
def income_stability_index(monthly_incomes: list) -> dict:
    """
    Measures how stable the user's income is.
    Returns stability score (0-1) and income trend (slope).
    """
    arr = np.array(monthly_incomes, dtype=float)
    mean_inc = arr.mean()
    std_inc = arr.std()
    stability = 1 - (std_inc / (mean_inc + 1e-9))
    stability = np.clip(stability, 0, 1)

    # Income trend via linear regression slope
    slope = np.polyfit(range(len(arr)), arr, 1)[0]
    # Normalize trend: positive = growing
    trend_norm = np.clip(slope / (mean_inc + 1e-9), -1, 1)

    return {
        "stability_score": round(float(stability), 4),
        "income_trend": round(float(trend_norm), 4),
        "mean_income": round(float(mean_inc), 2),
        "income_std": round(float(std_inc), 2)
    }


# ─── B. Cash Flow Health Ratio ──────────────────────────────────────────────
def cash_flow_health_ratio(net_income: float, fixed_expenses: float) -> dict:
    """
    (NetIncome - FixedExpenses) / TotalIncome
    > 0.4 → Strong | 0.2-0.4 → Moderate | < 0.2 → Risk
    """
    total_income = net_income
    ratio = (net_income - fixed_expenses) / (total_income + 1e-9)
    ratio = np.clip(ratio, 0, 1)

    if ratio > 0.4:
        category = "Strong"
    elif ratio >= 0.2:
        category = "Moderate"
    else:
        category = "Risk"

    return {
        "ratio": round(float(ratio), 4),
        "category": category
    }


# ─── C. Income Source Diversity ─────────────────────────────────────────────
def income_source_diversity(num_sources: int, max_sources: int = 5) -> dict:
    """
    DiversityScore = NumberOfIncomeSources / MaxPossibleSources
    """
    score = min(num_sources / max_sources, 1.0)
    return {
        "num_sources": num_sources,
        "diversity_score": round(float(score), 4),
        "is_diversified": num_sources > 1
    }


# ─── D. Utility Bill Timeliness Score ───────────────────────────────────────
def utility_timeliness_score(on_time: int, total_bills: int,
                              avg_delay_days: float,
                              delay_penalty_weight: float = 0.02) -> dict:
    """
    UtilityScore = OnTimeRate - (DelayPenaltyWeight × AvgDelay)
    """
    on_time_rate = on_time / (total_bills + 1e-9)
    penalty = delay_penalty_weight * avg_delay_days
    score = max(on_time_rate - penalty, 0)
    return {
        "on_time_rate": round(float(on_time_rate), 4),
        "delay_penalty": round(float(penalty), 4),
        "utility_score": round(float(score), 4)
    }


# ─── E. EMI-Like Pattern Detection ─────────────────────────────────────────
def emi_pattern_score(recurring_payments: int, consistency: float) -> dict:
    """
    Detects recurring payment patterns (same amount ± variance, same date).
    Higher recurring & consistency → bonus score.
    """
    max_recurring = 5
    recurrence_norm = min(recurring_payments / max_recurring, 1.0)
    score = (recurrence_norm * 0.5 + consistency * 0.5)
    return {
        "recurring_payments": recurring_payments,
        "consistency": round(float(consistency), 4),
        "emi_score": round(float(score), 4)
    }


# ─── F. Transaction Regularity ──────────────────────────────────────────────
def transaction_regularity(txn_regularity_score: float,
                           total_transactions: int) -> dict:
    """
    Lower variance in weekly transaction count → higher stability.
    txn_regularity_score already computed as 1 - (std/mean).
    """
    # Volume bonus: more transactions = more data confidence
    volume_factor = min(total_transactions / 150, 1.0)  # cap at 150 txns
    adjusted = txn_regularity_score * 0.7 + volume_factor * 0.3
    return {
        "regularity_score": round(float(txn_regularity_score), 4),
        "volume_factor": round(float(volume_factor), 4),
        "adjusted_score": round(float(adjusted), 4)
    }


# ─── G. Expense Categorization ──────────────────────────────────────────────
def expense_categorization(essential_ratio: float) -> dict:
    """
    EssentialRatio = EssentialSpending / TotalSpending
    Higher essential ratio → safer borrower.
    """
    if essential_ratio >= 0.65:
        risk_level = "Low"
    elif essential_ratio >= 0.45:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "essential_ratio": round(float(essential_ratio), 4),
        "risk_level": risk_level,
        "expense_score": round(float(np.clip(essential_ratio, 0, 1)), 4)
    }


# ─── H. Savings Behavior ───────────────────────────────────────────────────
def savings_behavior(has_recurring_savings: bool,
                     min_balance_maintained: bool,
                     avg_monthly_savings: float,
                     mean_income: float) -> dict:
    """
    Measures savings discipline.
    """
    savings_rate = avg_monthly_savings / (mean_income + 1e-9)
    savings_rate = np.clip(savings_rate, 0, 1)

    component_score = (
        (0.3 if has_recurring_savings else 0) +
        (0.3 if min_balance_maintained else 0) +
        (0.4 * savings_rate)
    )
    return {
        "savings_rate": round(float(savings_rate), 4),
        "has_recurring_savings": has_recurring_savings,
        "min_balance_maintained": min_balance_maintained,
        "savings_score": round(float(component_score), 4)
    }


# ─── I. Platform Tenure & Rating ───────────────────────────────────────────
def platform_tenure_rating(tenure_months: int, rating: float,
                           active_days: int,
                           max_tenure: int = 48) -> dict:
    """
    TenureScore = MonthsActive / MaxMonths
    RatingScore = Rating / 5
    """
    tenure_score = min(tenure_months / max_tenure, 1.0)
    rating_score = rating / 5.0
    activity_score = min(active_days / 30, 1.0)

    combined = tenure_score * 0.35 + rating_score * 0.40 + activity_score * 0.25
    return {
        "tenure_score": round(float(tenure_score), 4),
        "rating_score": round(float(rating_score), 4),
        "activity_score": round(float(activity_score), 4),
        "work_reliability_score": round(float(combined), 4)
    }


# ─── J. Shock Recovery Score ───────────────────────────────────────────────
def shock_recovery(monthly_incomes: list) -> dict:
    """
    Enhanced shock recovery: detects ALL income dips (>15%),
    grades severity, and measures recovery speed, completeness,
    and post-shock trajectory.

    Scoring:
      0.40 × speed_score
      0.35 × completeness_score
      0.25 × trajectory_score

    No shock ever → 0.85 (unverified resilience, not 1.0).
    Survived shock + fast recovery → up to 1.0 (proven resilience).
    """
    arr = np.array(monthly_incomes, dtype=float)
    n = len(arr)

    # Detect ALL shocks (not just the first)
    shocks = []
    for i in range(1, n):
        ratio = arr[i] / (arr[i - 1] + 1e-9)
        if ratio < 0.85:  # any drop >15%
            if ratio < 0.5:
                severity = "severe"
            elif ratio < 0.7:
                severity = "moderate"
            else:
                severity = "mild"

            # Measure recovery from this dip
            recovery_months = 0
            recovered = False
            for j in range(i + 1, n):
                recovery_months += 1
                if arr[j] >= arr[i - 1] * 0.9:
                    recovered = True
                    break
            if not recovered and i + 1 < n:
                recovery_months = n - i  # still recovering

            completeness = float(
                min(arr[i:].max() / (arr[i - 1] + 1e-9), 1.0)
            )

            shocks.append({
                "month": i,
                "drop_pct": round((1 - ratio) * 100, 1),
                "severity": severity,
                "recovery_months": recovery_months,
                "recovered": recovered,
                "completeness": completeness,
            })

    if shocks:
        # Speed score: inverse of avg recovery time
        avg_recovery = np.mean([s["recovery_months"] for s in shocks])
        speed_score = max(0, 1 - avg_recovery / n)

        # Completeness: avg recovery completeness
        completeness_score = float(
            np.mean([s["completeness"] for s in shocks])
        )

        # Trajectory: income slope after last shock
        last_shock = shocks[-1]["month"]
        if last_shock < n - 1:
            post_shock = arr[last_shock:]
            slope = np.polyfit(range(len(post_shock)), post_shock, 1)[0]
            trajectory_score = float(
                np.clip(slope / (arr.mean() + 1e-9) + 0.5, 0, 1)
            )
        else:
            trajectory_score = 0.3

        resilience = (
            0.40 * speed_score
            + 0.35 * completeness_score
            + 0.25 * trajectory_score
        )
    else:
        # Never faced a shock → unverified resilience
        resilience = 0.85

    return {
        "had_shock": len(shocks) > 0,
        "num_shocks": len(shocks),
        "shock_details": shocks,
        "resilience_score": round(float(np.clip(resilience, 0, 1)), 4),
    }


# ─── Master Feature Extractor ──────────────────────────────────────────────
def extract_all_features(row: pd.Series) -> dict:
    """
    Given a single row from the dataset, compute all feature scores.
    Returns a flat dictionary of all engineered features.
    """
    monthly_incomes = json.loads(row["monthly_incomes"]) if isinstance(row["monthly_incomes"], str) else row["monthly_incomes"]

    # A. Income Stability
    inc_stab = income_stability_index(monthly_incomes)

    # B. Cash Flow Health
    cf = cash_flow_health_ratio(row["mean_income"], row["fixed_expenses"])

    # C. Income Diversity
    div = income_source_diversity(row["num_income_sources"])

    # D. Utility Timeliness
    util = utility_timeliness_score(
        row["on_time_payments"], row["total_bills"], row["avg_delay_days"]
    )

    # E. EMI Pattern
    emi = emi_pattern_score(row["recurring_payments_detected"],
                            row["emi_consistency_score"])

    # F. Transaction Regularity
    txn = transaction_regularity(row["txn_regularity_score"],
                                  row["total_transactions"])

    # G. Expense Categorization
    exp = expense_categorization(row["essential_ratio"])

    # H. Savings Behavior
    sav = savings_behavior(
        bool(row["has_recurring_savings"]),
        bool(row["min_balance_maintained"]),
        row["avg_monthly_savings"],
        row["mean_income"]
    )

    # I. Platform Tenure & Rating
    plat = platform_tenure_rating(
        row["tenure_months"], row["platform_rating"],
        row["active_days_per_month"]
    )

    # J. Shock Recovery
    shock = shock_recovery(monthly_incomes)

    return {
        # A
        "feat_income_stability": inc_stab["stability_score"],
        "feat_income_trend": inc_stab["income_trend"],
        # B
        "feat_cash_flow_ratio": cf["ratio"],
        "feat_cash_flow_category": cf["category"],
        # C
        "feat_income_diversity": div["diversity_score"],
        # D
        "feat_utility_score": util["utility_score"],
        # E
        "feat_emi_score": emi["emi_score"],
        # F
        "feat_txn_regularity": txn["adjusted_score"],
        # G
        "feat_expense_score": exp["expense_score"],
        "feat_expense_risk": exp["risk_level"],
        # H
        "feat_savings_score": sav["savings_score"],
        # I
        "feat_work_reliability": plat["work_reliability_score"],
        # J
        "feat_shock_recovery": shock["resilience_score"],
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to entire DataFrame.
    Returns DataFrame with all engineered feature columns added.
    """
    features = df.apply(extract_all_features, axis=1, result_type="expand")
    return pd.concat([df, features], axis=1)
