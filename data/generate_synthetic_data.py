"""
Synthetic Data Generator for CrediVist
Generates realistic alternative credit data for underbanked / gig-economy users.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timedelta

np.random.seed(42)

# ── Configuration ───────────────────────────────────────────────────────────
NUM_USERS = 1000
MONTHS = 6  # trailing months of history
PLATFORMS = ["Swiggy", "Uber", "Zomato", "Ola", "Dunzo", "UrbanClap"]
EXPENSE_CATEGORIES = ["Rent", "Food", "Transport", "Utilities", "Entertainment",
                      "Healthcare", "Education", "Savings", "Miscellaneous"]


def _monthly_incomes(profile: str) -> list:
    """Return 6-month income history based on risk profile."""
    if profile == "good":
        base = np.random.randint(18000, 45000)
        noise = np.random.normal(0, base * 0.05, MONTHS)
    elif profile == "moderate":
        base = np.random.randint(12000, 30000)
        noise = np.random.normal(0, base * 0.15, MONTHS)
    else:  # risky
        base = np.random.randint(8000, 20000)
        noise = np.random.normal(0, base * 0.30, MONTHS)
    incomes = (base + noise).clip(min=1000).astype(int).tolist()
    return incomes


def _generate_transactions(monthly_incomes: list, profile: str) -> list:
    """Generate 6-month transaction list for a user."""
    transactions = []
    base_date = datetime(2025, 7, 1)
    for month_idx, income in enumerate(monthly_incomes):
        month_start = base_date + timedelta(days=30 * month_idx)
        # Number of transactions per month
        if profile == "good":
            n_txns = np.random.randint(25, 45)
        elif profile == "moderate":
            n_txns = np.random.randint(15, 30)
        else:
            n_txns = np.random.randint(8, 20)

        remaining = income * 0.85  # spend ~85% of income
        for t in range(n_txns):
            day_offset = np.random.randint(0, 28)
            txn_date = month_start + timedelta(days=day_offset)
            category = np.random.choice(
                EXPENSE_CATEGORIES,
                p=[0.25, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.10, 0.05]
            )
            amount = round(remaining / (n_txns - t) * np.random.uniform(0.3, 1.7), 2)
            amount = max(amount, 10)
            remaining -= amount
            transactions.append({
                "date": txn_date.strftime("%Y-%m-%d"),
                "category": category,
                "amount": round(amount, 2),
                "type": "debit"
            })
            if remaining <= 0:
                break
    return transactions


def _utility_bills(profile: str) -> dict:
    """Generate utility bill payment history."""
    total_bills = np.random.randint(12, 24)
    if profile == "good":
        on_time = int(total_bills * np.random.uniform(0.85, 1.0))
        avg_delay = round(np.random.uniform(0, 2), 1)
    elif profile == "moderate":
        on_time = int(total_bills * np.random.uniform(0.60, 0.85))
        avg_delay = round(np.random.uniform(2, 7), 1)
    else:
        on_time = int(total_bills * np.random.uniform(0.30, 0.65))
        avg_delay = round(np.random.uniform(5, 15), 1)
    return {
        "total_bills": total_bills,
        "on_time_payments": on_time,
        "avg_delay_days": avg_delay
    }


def _recharge_pattern(profile: str) -> dict:
    """Mobile recharge regularity."""
    if profile == "good":
        return {"monthly_recharges": 6, "regularity_score": round(np.random.uniform(0.8, 1.0), 2)}
    elif profile == "moderate":
        return {"monthly_recharges": np.random.randint(4, 6),
                "regularity_score": round(np.random.uniform(0.5, 0.8), 2)}
    else:
        return {"monthly_recharges": np.random.randint(1, 4),
                "regularity_score": round(np.random.uniform(0.2, 0.5), 2)}


def _platform_info(profile: str) -> dict:
    """Gig platform tenure and rating."""
    n_platforms = np.random.randint(1, 4) if profile != "risky" else np.random.randint(1, 2)
    platforms = np.random.choice(PLATFORMS, size=n_platforms, replace=False).tolist()
    if profile == "good":
        tenure_months = np.random.randint(12, 48)
        rating = round(np.random.uniform(4.3, 5.0), 1)
        active_days = np.random.randint(22, 30)
    elif profile == "moderate":
        tenure_months = np.random.randint(6, 24)
        rating = round(np.random.uniform(3.8, 4.5), 1)
        active_days = np.random.randint(15, 25)
    else:
        tenure_months = np.random.randint(1, 12)
        rating = round(np.random.uniform(3.0, 4.0), 1)
        active_days = np.random.randint(5, 18)
    return {
        "platforms": platforms,
        "num_platforms": len(platforms),
        "tenure_months": tenure_months,
        "rating": rating,
        "active_days_per_month": active_days
    }


def _savings_info(profile: str) -> dict:
    """Savings behaviour."""
    if profile == "good":
        return {
            "has_recurring_savings": True,
            "min_balance_maintained": True,
            "avg_monthly_savings": int(np.random.randint(2000, 8000))
        }
    elif profile == "moderate":
        return {
            "has_recurring_savings": np.random.choice([True, False]),
            "min_balance_maintained": np.random.choice([True, False]),
            "avg_monthly_savings": int(np.random.randint(500, 3000))
        }
    else:
        return {
            "has_recurring_savings": False,
            "min_balance_maintained": False,
            "avg_monthly_savings": int(np.random.randint(0, 500))
        }


def _emi_like_payments(profile: str) -> dict:
    """Recurring EMI-like payment detection signals."""
    if profile == "good":
        return {
            "recurring_payments_detected": np.random.randint(2, 5),
            "consistency_score": round(np.random.uniform(0.8, 1.0), 2)
        }
    elif profile == "moderate":
        return {
            "recurring_payments_detected": np.random.randint(1, 3),
            "consistency_score": round(np.random.uniform(0.5, 0.8), 2)
        }
    else:
        return {
            "recurring_payments_detected": np.random.randint(0, 1),
            "consistency_score": round(np.random.uniform(0.1, 0.5), 2)
        }


def _fixed_expenses_ratio(profile: str, mean_income: float) -> float:
    """Return fixed expenses as a fraction of income."""
    if profile == "good":
        return round(mean_income * np.random.uniform(0.25, 0.45), 2)
    elif profile == "moderate":
        return round(mean_income * np.random.uniform(0.40, 0.65), 2)
    else:
        return round(mean_income * np.random.uniform(0.60, 0.85), 2)


def generate_dataset(n: int = NUM_USERS) -> pd.DataFrame:
    """Generate full synthetic dataset."""
    records = []
    for i in range(n):
        # Assign risk profile (determines ground truth)
        profile = np.random.choice(["good", "moderate", "risky"], p=[0.40, 0.35, 0.25])
        default_label = 0 if profile == "good" else (
            np.random.choice([0, 1], p=[0.75, 0.25]) if profile == "moderate"
            else np.random.choice([0, 1], p=[0.35, 0.65])
        )

        monthly_incomes = _monthly_incomes(profile)
        mean_income = np.mean(monthly_incomes)
        fixed_expenses = _fixed_expenses_ratio(profile, mean_income)
        utility = _utility_bills(profile)
        recharge = _recharge_pattern(profile)
        platform = _platform_info(profile)
        savings = _savings_info(profile)
        emi = _emi_like_payments(profile)
        transactions = _generate_transactions(monthly_incomes, profile)

        # Expense breakdown from transactions
        total_spend = sum(t["amount"] for t in transactions)
        essential_spend = sum(t["amount"] for t in transactions
                             if t["category"] in ["Rent", "Food", "Transport", "Utilities",
                                                   "Healthcare", "Education"])
        essential_ratio = round(essential_spend / total_spend, 4) if total_spend > 0 else 0

        # Transaction regularity - std of weekly txn counts
        txn_weeks = {}
        for t in transactions:
            wk = datetime.strptime(t["date"], "%Y-%m-%d").isocalendar()[1]
            txn_weeks[wk] = txn_weeks.get(wk, 0) + 1
        txn_regularity = round(1 - min(np.std(list(txn_weeks.values())) / (np.mean(list(txn_weeks.values())) + 1e-9), 1), 4)

        # Shock recovery (simulate)
        incomes_arr = np.array(monthly_incomes, dtype=float)
        dips = np.where((incomes_arr[1:] / (incomes_arr[:-1] + 1e-9)) < 0.7)[0]
        if len(dips) > 0:
            recovery_months = 0
            dip_idx = dips[0] + 1
            for j in range(dip_idx, len(incomes_arr)):
                recovery_months += 1
                if incomes_arr[j] >= incomes_arr[dip_idx - 1] * 0.9:
                    break
            shock_recovery = max(0, 1 - recovery_months / MONTHS)
        else:
            shock_recovery = 1.0

        record = {
            "user_id": f"USR{i+1:04d}",
            "profile": profile,
            # Income features
            "monthly_incomes": json.dumps(monthly_incomes),
            "mean_income": round(mean_income, 2),
            "income_std": round(float(np.std(monthly_incomes)), 2),
            "income_trend": round(float(np.polyfit(range(MONTHS), monthly_incomes, 1)[0]), 2),
            # Cash flow
            "fixed_expenses": fixed_expenses,
            "cash_flow_health_ratio": round((mean_income - fixed_expenses) / (mean_income + 1e-9), 4),
            # Income diversity
            "num_income_sources": platform["num_platforms"],
            # Utility bills
            "total_bills": utility["total_bills"],
            "on_time_payments": utility["on_time_payments"],
            "avg_delay_days": utility["avg_delay_days"],
            # Recharge
            "recharge_regularity": recharge["regularity_score"],
            # EMI-like
            "recurring_payments_detected": emi["recurring_payments_detected"],
            "emi_consistency_score": emi["consistency_score"],
            # Transaction behaviour
            "total_transactions": len(transactions),
            "txn_regularity_score": txn_regularity,
            "essential_ratio": essential_ratio,
            # Savings
            "has_recurring_savings": int(savings["has_recurring_savings"]),
            "min_balance_maintained": int(savings["min_balance_maintained"]),
            "avg_monthly_savings": savings["avg_monthly_savings"],
            # Platform
            "platforms": json.dumps(platform["platforms"]),
            "tenure_months": platform["tenure_months"],
            "platform_rating": platform["rating"],
            "active_days_per_month": platform["active_days_per_month"],
            # Shock recovery
            "shock_recovery_score": round(shock_recovery, 4),
            # Target
            "default": default_label,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def main():
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "credit_data.csv")
    df.to_csv(out_path, index=False)
    print(f"[CrediVist] Generated {len(df)} synthetic user records → {out_path}")
    print(f"  Default rate: {df['default'].mean():.2%}")
    print(f"  Profiles: {df['profile'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
