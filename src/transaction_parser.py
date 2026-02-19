"""
Transaction Parser & Auto-Categorizer for CrediVist
Parses bank/UPI transaction CSVs and auto-extracts credit scoring features.
"""

import numpy as np
import pandas as pd
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict


# ─── Category Keywords (Indian Banking Context) ────────────────────────────
CATEGORY_KEYWORDS = {
    "Rent": [
        "rent", "house rent", "apartment", "pg ", "hostel", "accommodation",
        "landlord", "society maintenance", "maintenance charge", "flat rent",
    ],
    "Food": [
        "swiggy", "zomato", "food", "restaurant", "cafe", "mess", "canteen",
        "grocery", "bigbasket", "blinkit", "instamart", "dunzo", "zepto",
        "dmart", "reliance fresh", "kitchen", "hotel", "bakery", "dairy",
    ],
    "Transport": [
        "uber ride", "ola ride", "rapido", "metro", "bus ticket", "fuel",
        "petrol", "diesel", "parking", "toll", "fastag", "irctc", "train",
        "flight", "makemytrip", "redbus", "cab", "auto ride",
    ],
    "Utilities": [
        "electricity", "water bill", "gas bill", "broadband", "wifi",
        "internet", "tata power", "bescom", "mseb", "piped gas", "lpg",
        "cylinder", "power bill", "electric",
    ],
    "Recharge": [
        "recharge", "prepaid", "mobile recharge", "dth", "tata sky",
        "dish tv", "d2h", "jio", "airtel", "vi ", "bsnl", "postpaid",
    ],
    "Entertainment": [
        "netflix", "hotstar", "disney", "prime video", "spotify", "gaana",
        "movie", "pvr", "inox", "game", "playstation", "steam",
        "youtube premium", "apple music", "subscription",
    ],
    "Healthcare": [
        "hospital", "medical", "pharmacy", "doctor", "clinic", "health",
        "apollo", "medplus", "1mg", "pharmeasy", "practo", "diagnostic",
        "lab test", "medicine",
    ],
    "Education": [
        "school", "college", "tuition", "course", "udemy", "coursera",
        "education", "fees", "exam", "unacademy", "byju", "coaching",
    ],
    "EMI": [
        "emi", "loan", "repayment", "instalment", "installment",
        "bajaj finserv", "personal loan", "car loan", "home loan",
    ],
    "Insurance": [
        "insurance", "lic", "premium", "policy", "health insurance",
        "term plan", "cover",
    ],
    "Savings": [
        "savings", "fixed deposit", "fd ", "rd ", "mutual fund", "sip",
        "invest", "ppf", "nps", "groww", "zerodha", "kuvera", "smallcase",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "meesho", "nykaa",
        "croma", "reliance digital", "shopping",
    ],
    "Transfer": [
        "self transfer", "own account", "self txn",
    ],
}

# Platforms that indicate gig income
GIG_PLATFORMS = {
    "Swiggy": ["swiggy", "bundl technologies"],
    "Uber": ["uber", "uber india"],
    "Zomato": ["zomato", "zomato media"],
    "Ola": ["ola", "ani technologies"],
    "Dunzo": ["dunzo"],
    "UrbanClap": ["urbanclap", "urban company"],
    "Rapido": ["rapido"],
}

INCOME_KEYWORDS = [
    "salary", "payment received", "cashback", "refund",
    "commission", "earning", "payout", "incentive", "bonus",
    "settlement", "wages", "stipend", "credit",
]

# Essential categories for expense ratio
ESSENTIAL_CATEGORIES = [
    "Rent", "Food", "Transport", "Utilities", "Healthcare",
    "Education", "EMI", "Insurance",
]


# ─── Column Detection ───────────────────────────────────────────────────────
DATE_PATTERNS = [
    "date", "txn date", "transaction date", "value date",
    "posting date", "value dt",
]
DESC_PATTERNS = [
    "description", "narration", "remarks", "transaction remarks",
    "particulars", "details", "memo",
]
DEBIT_PATTERNS = [
    "debit", "withdrawal", "withdrawal amt", "withdrawal amount", "dr",
]
CREDIT_PATTERNS = [
    "credit", "deposit", "deposit amt", "deposit amount", "cr",
]
AMOUNT_PATTERNS = ["amount", "txn amount", "transaction amount"]
BALANCE_PATTERNS = [
    "balance", "closing balance", "available balance", "running balance",
]


def _match_column(columns, patterns):
    """Find the best matching column name from a list of patterns."""
    cols_lower = {c.lower().strip(): c for c in columns}
    for pattern in patterns:
        for col_lower, col_original in cols_lower.items():
            if pattern in col_lower:
                return col_original
    return None


# ─── Transaction Parser Class ──────────────────────────────────────────────
class TransactionParser:
    """Parses bank/UPI transaction files and auto-extracts credit features."""

    def __init__(self):
        self.raw_df = None
        self.parsed_df = None
        self.profile = None

    # ─── File Parsing ───────────────────────────────────────────────────
    def parse_file(self, file, file_type="csv") -> pd.DataFrame:
        """Parse uploaded CSV/Excel into standardized format."""
        if file_type == "csv":
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, encoding=encoding)
                    if len(df.columns) >= 3:
                        break
                except Exception:
                    continue
            else:
                file.seek(0)
                df = pd.read_csv(file, encoding="utf-8", sep=";")
        elif file_type in ("xlsx", "xls"):
            df = pd.read_excel(file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        self.raw_df = df
        return self.standardize_columns(df)

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and standardize column names."""
        columns = df.columns.tolist()

        date_col = _match_column(columns, DATE_PATTERNS)
        desc_col = _match_column(columns, DESC_PATTERNS)
        debit_col = _match_column(columns, DEBIT_PATTERNS)
        credit_col = _match_column(columns, CREDIT_PATTERNS)
        amount_col = _match_column(columns, AMOUNT_PATTERNS)
        balance_col = _match_column(columns, BALANCE_PATTERNS)

        standardized = pd.DataFrame()

        # Date
        if date_col:
            standardized["date"] = pd.to_datetime(
                df[date_col], dayfirst=True, errors="coerce"
            )
        else:
            standardized["date"] = pd.to_datetime(
                df.iloc[:, 0], dayfirst=True, errors="coerce"
            )

        # Description
        if desc_col:
            standardized["description"] = df[desc_col].astype(str).str.strip()
        else:
            # Fallback: second column
            standardized["description"] = df.iloc[:, 1].astype(str).str.strip()

        # Amount and type
        if debit_col and credit_col:
            debit = pd.to_numeric(
                df[debit_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0)
            credit = pd.to_numeric(
                df[credit_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0)
            standardized["amount"] = np.where(credit > 0, credit, debit)
            standardized["type"] = np.where(credit > 0, "credit", "debit")
            standardized["debit_amount"] = debit
            standardized["credit_amount"] = credit
        elif amount_col:
            amt = pd.to_numeric(
                df[amount_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            ).fillna(0)
            standardized["amount"] = amt.abs()
            standardized["type"] = np.where(amt >= 0, "credit", "debit")
            standardized["debit_amount"] = np.where(amt < 0, amt.abs(), 0)
            standardized["credit_amount"] = np.where(amt >= 0, amt, 0)
        else:
            raise ValueError(
                "Could not detect amount columns. "
                "Please ensure your file has 'Debit'/'Credit' or 'Amount' columns."
            )

        # Balance
        if balance_col:
            standardized["balance"] = pd.to_numeric(
                df[balance_col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )
        else:
            standardized["balance"] = np.nan

        # Clean up
        standardized = standardized.dropna(subset=["date"])
        standardized = standardized[standardized["amount"] > 0]
        standardized = standardized.sort_values("date").reset_index(drop=True)

        self.parsed_df = standardized
        return standardized

    # ─── Auto-Categorization ────────────────────────────────────────────
    def auto_categorize(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Categorize each transaction using keyword matching."""
        if df is None:
            df = self.parsed_df.copy()
        else:
            df = df.copy()

        categories = []
        confidences = []

        for _, row in df.iterrows():
            desc = str(row.get("description", "")).lower()
            txn_type = row.get("type", "debit")
            cat, conf = self._categorize_single(desc, txn_type)
            categories.append(cat)
            confidences.append(conf)

        df["category"] = categories
        df["category_confidence"] = confidences
        self.parsed_df = df
        return df

    def _categorize_single(self, description: str, txn_type: str) -> tuple:
        """Categorize a single transaction. Returns (category, confidence)."""
        desc_lower = description.lower()

        # Check for income first (credit transactions)
        if txn_type == "credit":
            # Check gig platforms
            for platform, keywords in GIG_PLATFORMS.items():
                for kw in keywords:
                    if kw in desc_lower:
                        return "Income", 0.95
            # Check general income keywords
            for kw in INCOME_KEYWORDS:
                if kw in desc_lower:
                    return "Income", 0.85
            # Generic credit → likely income
            return "Income", 0.60

        # For debit transactions, check categories
        best_cat = "Miscellaneous"
        best_conf = 0.0

        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in desc_lower:
                    conf = min(0.70 + len(kw) * 0.02, 0.95)
                    if conf > best_conf:
                        best_cat = category
                        best_conf = conf

        if best_conf == 0:
            best_conf = 0.30

        return best_cat, best_conf

    # ─── Gig Platform Detection ─────────────────────────────────────────
    def detect_gig_platforms(self, df: pd.DataFrame = None) -> dict:
        """Detect gig platform income sources from transactions."""
        if df is None:
            df = self.parsed_df

        detected = {}
        for platform, keywords in GIG_PLATFORMS.items():
            platform_txns = df[
                df["description"].str.lower().apply(
                    lambda d: any(kw in str(d) for kw in keywords)
                )
            ]
            if len(platform_txns) > 0:
                months_active = platform_txns["date"].dt.to_period("M").nunique()
                detected[platform] = {
                    "transactions": len(platform_txns),
                    "months_active": months_active,
                    "total_amount": float(platform_txns["amount"].sum()),
                }
        return detected

    # ─── Feature Extraction ─────────────────────────────────────────────
    def extract_profile(self, df: pd.DataFrame = None,
                        platform_rating: float = 4.0,
                        active_days: int = 20) -> pd.Series:
        """
        Extract a complete user profile from parsed & categorized transactions.
        Returns a Series compatible with the existing scoring pipeline.
        """
        if df is None:
            df = self.parsed_df

        if "category" not in df.columns:
            df = self.auto_categorize(df)

        # ── Monthly Income & Expenses ──
        credits = df[df["type"] == "credit"].copy()
        debits = df[df["type"] == "debit"].copy()

        if len(credits) > 0:
            credits["month"] = credits["date"].dt.to_period("M")
        if len(debits) > 0:
            debits["month"] = debits["date"].dt.to_period("M")

        # Group by month
        monthly_income = (
            credits.groupby("month")["amount"].sum()
            if len(credits) > 0 else pd.Series(dtype=float)
        )
        monthly_expenses = (
            debits.groupby("month")["amount"].sum()
            if len(debits) > 0 else pd.Series(dtype=float)
        )

        # Get all months
        all_months = sorted(set(
            list(monthly_income.index) + list(monthly_expenses.index)
        ))
        recent_months = all_months[-6:] if len(all_months) >= 6 else all_months

        monthly_incomes = [
            float(monthly_income.get(m, 0)) for m in recent_months
        ]
        # Pad to at least 3 months
        while len(monthly_incomes) < 3:
            monthly_incomes.insert(
                0, monthly_incomes[0] if monthly_incomes else 10000
            )

        mean_income = float(np.mean(monthly_incomes))
        income_std = float(np.std(monthly_incomes))

        # ── Fixed Expenses ──
        fixed_expenses = self._detect_fixed_expenses(debits, mean_income)

        # ── Utility Bills ──
        utility_info = self._analyze_utility_payments(df)

        # ── Recharge Pattern ──
        recharge_info = self._analyze_recharge_pattern(df, len(recent_months))

        # ── EMI-like Patterns ──
        emi_info = self._detect_emi_patterns(debits)

        # ── Transaction Regularity ──
        txn_regularity = self._compute_txn_regularity(df)

        # ── Expense Categorization ──
        essential_ratio = self._compute_essential_ratio(debits)

        # ── Savings Behavior ──
        savings_info = self._analyze_savings(
            df, monthly_incomes, monthly_expenses, recent_months
        )

        # ── Platform Detection ──
        gig_platforms = self.detect_gig_platforms(df)
        num_income_sources = max(len(gig_platforms), 1)

        # Estimate tenure from data
        if gig_platforms:
            max_tenure = max(p["months_active"] for p in gig_platforms.values())
        else:
            max_tenure = len(recent_months)

        platform_names = (
            list(gig_platforms.keys()) if gig_platforms else ["Direct"]
        )

        # Income trend
        if len(monthly_incomes) >= 2:
            trend = float(
                np.polyfit(range(len(monthly_incomes)), monthly_incomes, 1)[0]
            )
        else:
            trend = 0.0

        profile = pd.Series({
            "user_id": "UPLOADED_USER",
            "profile": "uploaded",
            "monthly_incomes": json.dumps([int(x) for x in monthly_incomes]),
            "mean_income": round(mean_income, 2),
            "income_std": round(income_std, 2),
            "income_trend": round(trend, 2),
            "fixed_expenses": round(fixed_expenses, 2),
            "cash_flow_health_ratio": round(
                (mean_income - fixed_expenses) / (mean_income + 1e-9), 4
            ),
            "num_income_sources": num_income_sources,
            "total_bills": utility_info["total_bills"],
            "on_time_payments": utility_info["on_time_payments"],
            "avg_delay_days": utility_info["avg_delay_days"],
            "recharge_regularity": recharge_info["regularity_score"],
            "recurring_payments_detected": emi_info["recurring_count"],
            "emi_consistency_score": emi_info["consistency_score"],
            "total_transactions": len(df),
            "txn_regularity_score": txn_regularity,
            "essential_ratio": essential_ratio,
            "has_recurring_savings": int(savings_info["has_recurring_savings"]),
            "min_balance_maintained": int(savings_info["min_balance_maintained"]),
            "avg_monthly_savings": savings_info["avg_monthly_savings"],
            "platforms": json.dumps(platform_names),
            "tenure_months": max_tenure,
            "platform_rating": platform_rating,
            "active_days_per_month": active_days,
            "default": 0,
        })

        self.profile = profile
        return profile

    # ─── Internal Feature Extractors ────────────────────────────────────
    def _detect_fixed_expenses(self, debits: pd.DataFrame,
                               mean_income: float) -> float:
        """Detect fixed/recurring expenses from debit transactions."""
        if len(debits) == 0:
            return mean_income * 0.5

        debits = debits.copy()
        if "month" not in debits.columns:
            debits["month"] = debits["date"].dt.to_period("M")

        n_months = debits["month"].nunique()
        if n_months < 2:
            return float(debits["amount"].sum() / max(n_months, 1) * 0.6)

        # Round amounts to nearest 100 for matching
        debits["rounded"] = debits["amount"].apply(lambda x: round(x / 100) * 100)
        amt_month_counts = debits.groupby("rounded")["month"].nunique()

        # Amounts appearing in ≥50% of months are likely fixed
        threshold = max(n_months * 0.5, 2)
        fixed_amounts = amt_month_counts[amt_month_counts >= threshold].index.tolist()

        if fixed_amounts:
            return float(sum(fixed_amounts))
        else:
            return float(mean_income * 0.45)

    def _analyze_utility_payments(self, df: pd.DataFrame) -> dict:
        """Analyze utility payment behavior."""
        utility_txns = df[
            df["category"].isin(["Utilities", "Recharge"])
            & (df["type"] == "debit")
        ]

        if len(utility_txns) == 0:
            return {
                "total_bills": 6,
                "on_time_payments": 4,
                "avg_delay_days": 3.0,
            }

        total_bills = len(utility_txns)

        # Estimate timeliness: bills paid in first 10 days = on-time
        day_of_month = utility_txns["date"].dt.day
        on_time = int((day_of_month <= 10).sum())

        # Average delay estimate (assume due on 5th)
        avg_delay = float(max(day_of_month.mean() - 5, 0))

        return {
            "total_bills": total_bills,
            "on_time_payments": on_time,
            "avg_delay_days": round(avg_delay, 1),
        }

    def _analyze_recharge_pattern(self, df: pd.DataFrame,
                                  n_months: int) -> dict:
        """Analyze mobile recharge regularity."""
        recharge_txns = df[
            (df["category"] == "Recharge") & (df["type"] == "debit")
        ]

        if len(recharge_txns) == 0:
            return {"monthly_recharges": 0, "regularity_score": 0.5}

        months_with_recharge = (
            recharge_txns["date"].dt.to_period("M").nunique()
        )
        regularity = min(months_with_recharge / max(n_months, 1), 1.0)

        return {
            "monthly_recharges": len(recharge_txns),
            "regularity_score": round(regularity, 2),
        }

    def _detect_emi_patterns(self, debits: pd.DataFrame) -> dict:
        """Detect EMI-like recurring payment patterns."""
        if len(debits) < 3:
            return {"recurring_count": 0, "consistency_score": 0.3}

        debits = debits.copy()
        if "month" not in debits.columns:
            debits["month"] = debits["date"].dt.to_period("M")

        # Round amounts to nearest ₹50 for matching
        debits["rounded"] = debits["amount"].apply(lambda x: round(x / 50) * 50)

        # Find amounts appearing in 3+ months
        amt_month = debits.groupby("rounded")["month"].nunique()
        recurring = amt_month[amt_month >= 3]

        if len(recurring) == 0:
            return {"recurring_count": 0, "consistency_score": 0.3}

        n_months = debits["month"].nunique()
        best_consistency = float(recurring.max() / max(n_months, 1))

        return {
            "recurring_count": len(recurring),
            "consistency_score": round(min(best_consistency, 1.0), 2),
        }

    def _compute_txn_regularity(self, df: pd.DataFrame) -> float:
        """Compute transaction regularity (weekly consistency)."""
        if len(df) < 7:
            return 0.5

        df_copy = df.copy()
        df_copy["week"] = df_copy["date"].dt.isocalendar().week.astype(int)
        weekly_counts = df_copy.groupby("week").size()

        if len(weekly_counts) < 2:
            return 0.5

        mean_txns = weekly_counts.mean()
        std_txns = weekly_counts.std()

        regularity = 1 - min(std_txns / (mean_txns + 1e-9), 1.0)
        return round(float(regularity), 4)

    def _compute_essential_ratio(self, debits: pd.DataFrame) -> float:
        """Compute essential vs total spending ratio."""
        if len(debits) == 0:
            return 0.65

        total_spend = debits["amount"].sum()
        essential_spend = debits[
            debits["category"].isin(ESSENTIAL_CATEGORIES)
        ]["amount"].sum()

        if total_spend == 0:
            return 0.65

        return round(float(essential_spend / total_spend), 4)

    def _analyze_savings(self, df, monthly_incomes, monthly_expenses_series,
                         recent_months) -> dict:
        """Analyze savings behavior from transactions."""
        # Check for savings category transactions
        savings_txns = df[
            (df["category"] == "Savings") & (df["type"] == "debit")
        ]
        has_recurring_savings = len(savings_txns) >= 2

        # Check minimum balance
        if "balance" in df.columns and df["balance"].notna().any():
            min_balance = df["balance"].min()
            min_balance_maintained = min_balance > 1000
        else:
            min_balance_maintained = False

        # Monthly savings = income - expenses
        monthly_savings = []
        for i, m in enumerate(recent_months):
            inc = monthly_incomes[i] if i < len(monthly_incomes) else 0
            exp = float(monthly_expenses_series.get(m, 0))
            sav = max(inc - exp, 0)
            monthly_savings.append(sav)

        avg_savings = (
            float(np.mean(monthly_savings)) if monthly_savings else 0
        )

        return {
            "has_recurring_savings": has_recurring_savings,
            "min_balance_maintained": min_balance_maintained,
            "avg_monthly_savings": round(avg_savings, 2),
        }

    # ─── Summary Methods ────────────────────────────────────────────────
    def get_parsing_summary(self) -> dict:
        """Get a human-readable summary of parsed data."""
        df = self.parsed_df
        if df is None or len(df) == 0:
            return {
                "total_transactions": 0, "date_range": "N/A",
                "months_covered": 0, "total_credits": 0,
                "total_debits": 0, "total_income": 0, "total_expenses": 0,
            }
        return {
            "total_transactions": len(df),
            "date_range": (
                f"{df['date'].min().strftime('%d %b %Y')} — "
                f"{df['date'].max().strftime('%d %b %Y')}"
            ),
            "months_covered": df["date"].dt.to_period("M").nunique(),
            "total_credits": int((df["type"] == "credit").sum()),
            "total_debits": int((df["type"] == "debit").sum()),
            "total_income": float(
                df[df["type"] == "credit"]["amount"].sum()
            ),
            "total_expenses": float(
                df[df["type"] == "debit"]["amount"].sum()
            ),
        }

    def get_category_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get spending summary by category."""
        if df is None:
            df = self.parsed_df

        debits = df[df["type"] == "debit"]
        if len(debits) == 0:
            return pd.DataFrame(
                columns=["category", "count", "total", "average", "percentage"]
            )

        summary = debits.groupby("category").agg(
            count=("amount", "count"),
            total=("amount", "sum"),
            average=("amount", "mean"),
        ).sort_values("total", ascending=False)

        summary["percentage"] = (
            summary["total"] / summary["total"].sum() * 100
        ).round(1)
        return summary.reset_index()

    def get_monthly_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get monthly income vs expense summary."""
        if df is None:
            df = self.parsed_df

        df = df.copy()
        df["month"] = df["date"].dt.to_period("M").astype(str)

        monthly = (
            df.groupby(["month", "type"])["amount"]
            .sum()
            .unstack(fill_value=0)
        )

        if "credit" not in monthly.columns:
            monthly["credit"] = 0
        if "debit" not in monthly.columns:
            monthly["debit"] = 0

        monthly["net_savings"] = monthly["credit"] - monthly["debit"]
        return monthly.reset_index()


# ─── Sample Statement Generator ────────────────────────────────────────────
def generate_sample_statement(n_months: int = 6) -> pd.DataFrame:
    """
    Generate a realistic sample Indian bank statement for demo purposes.
    Produces ~25-35 transactions per month with realistic descriptions.
    """
    np.random.seed(42)
    transactions = []
    base_date = datetime(2025, 1, 1)
    balance = 25000.0

    for month in range(n_months):
        month_start = base_date + timedelta(days=30 * month)

        # ═══ CREDITS ═══

        # Salary (1st of month)
        salary = 28000 + np.random.randint(-2000, 3000)
        balance += salary
        transactions.append({
            "Date": (month_start + timedelta(days=0)).strftime("%d-%m-%Y"),
            "Description": "NEFT-SALARY-TECHCORP SOLUTIONS PVT LTD",
            "Debit": "", "Credit": salary,
            "Balance": round(balance, 2),
        })

        # Gig platform income (3-4 times)
        gig_descs = [
            "UPI-SWIGGY-BUNDL TECHNOLOGIES",
            "UPI-ZOMATO MEDIA PVT LTD",
            "UPI-UBER INDIA SYSTEMS",
            "UPI-SWIGGY-PAYOUT",
        ]
        for w in range(np.random.randint(3, 5)):
            gig_amt = np.random.randint(1500, 4500)
            day = 7 * w + np.random.randint(2, 7)
            balance += gig_amt
            transactions.append({
                "Date": (month_start + timedelta(days=min(day, 28))).strftime("%d-%m-%Y"),
                "Description": np.random.choice(gig_descs),
                "Debit": "", "Credit": gig_amt,
                "Balance": round(balance, 2),
            })

        # Cashback (occasionally)
        if np.random.random() > 0.6:
            cb = np.random.randint(50, 300)
            balance += cb
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(10, 25))).strftime("%d-%m-%Y"),
                "Description": "UPI-CASHBACK-PHONEPE",
                "Debit": "", "Credit": cb,
                "Balance": round(balance, 2),
            })

        # ═══ DEBITS ═══

        # Rent (5th)
        rent = 8000
        balance -= rent
        transactions.append({
            "Date": (month_start + timedelta(days=4)).strftime("%d-%m-%Y"),
            "Description": "UPI-RENT PAYMENT TO LANDLORD",
            "Debit": rent, "Credit": "",
            "Balance": round(balance, 2),
        })

        # Electricity (8-12th)
        elec = np.random.randint(700, 1500)
        balance -= elec
        transactions.append({
            "Date": (month_start + timedelta(days=np.random.randint(7, 12))).strftime("%d-%m-%Y"),
            "Description": "BESCOM ELECTRICITY BILL PAYMENT",
            "Debit": elec, "Credit": "",
            "Balance": round(balance, 2),
        })

        # Mobile recharge
        recharge = np.random.choice([199, 249, 299, 399, 449])
        balance -= recharge
        transactions.append({
            "Date": (month_start + timedelta(days=np.random.randint(1, 5))).strftime("%d-%m-%Y"),
            "Description": "JIO PREPAID RECHARGE",
            "Debit": recharge, "Credit": "",
            "Balance": round(balance, 2),
        })

        # Food delivery (5-8 times)
        food_descs = [
            "UPI-SWIGGY ORDER", "UPI-ZOMATO ORDER",
            "UPI-BLINKIT GROCERY", "UPI-BIGBASKET",
        ]
        for _ in range(np.random.randint(5, 9)):
            food = np.random.randint(150, 650)
            balance -= food
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(0, 28))).strftime("%d-%m-%Y"),
                "Description": np.random.choice(food_descs),
                "Debit": food, "Credit": "",
                "Balance": round(balance, 2),
            })

        # Transport (3-5 times)
        transport_descs = [
            "UPI-UBER RIDE", "UPI-OLA RIDE",
            "UPI-RAPIDO AUTO", "BMTC BUS PASS",
        ]
        for _ in range(np.random.randint(3, 6)):
            transport = np.random.randint(80, 450)
            balance -= transport
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(0, 28))).strftime("%d-%m-%Y"),
                "Description": np.random.choice(transport_descs),
                "Debit": transport, "Credit": "",
                "Balance": round(balance, 2),
            })

        # Shopping (1-2 times)
        for _ in range(np.random.randint(1, 3)):
            shop = np.random.randint(300, 2500)
            balance -= shop
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(5, 27))).strftime("%d-%m-%Y"),
                "Description": np.random.choice([
                    "AMAZON PAY INDIA", "FLIPKART INTERNET",
                    "MYNTRA DESIGNS", "UPI-RELIANCE TRENDS",
                ]),
                "Debit": shop, "Credit": "",
                "Balance": round(balance, 2),
            })

        # Entertainment (0-2 times)
        for _ in range(np.random.randint(0, 3)):
            ent = np.random.randint(149, 649)
            balance -= ent
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(0, 28))).strftime("%d-%m-%Y"),
                "Description": np.random.choice([
                    "NETFLIX SUBSCRIPTION", "SPOTIFY PREMIUM",
                    "PVR CINEMAS TICKET", "HOTSTAR VIP",
                ]),
                "Debit": ent, "Credit": "",
                "Balance": round(balance, 2),
            })

        # SIP / Savings (monthly)
        sip = np.random.choice([500, 1000, 2000, 2500])
        balance -= sip
        transactions.append({
            "Date": (month_start + timedelta(days=np.random.randint(3, 7))).strftime("%d-%m-%Y"),
            "Description": "SIP-GROWW-MUTUAL FUND",
            "Debit": sip, "Credit": "",
            "Balance": round(balance, 2),
        })

        # Healthcare (occasional)
        if np.random.random() > 0.7:
            med = np.random.randint(200, 1500)
            balance -= med
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(0, 28))).strftime("%d-%m-%Y"),
                "Description": np.random.choice([
                    "APOLLO PHARMACY", "UPI-PRACTO CONSULTATION",
                    "MEDPLUS MEDICINES",
                ]),
                "Debit": med, "Credit": "",
                "Balance": round(balance, 2),
            })

        # Society maintenance (quarterly)
        if month % 3 == 0:
            maint = np.random.randint(1500, 3000)
            balance -= maint
            transactions.append({
                "Date": (month_start + timedelta(days=np.random.randint(1, 5))).strftime("%d-%m-%Y"),
                "Description": "SOCIETY MAINTENANCE CHARGE",
                "Debit": maint, "Credit": "",
                "Balance": round(balance, 2),
            })

    df = pd.DataFrame(transactions)
    # Sort by date
    df["_sort_date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("_sort_date").drop(columns=["_sort_date"])
    df = df.reset_index(drop=True)

    return df
