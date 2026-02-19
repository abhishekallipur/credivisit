"""
Explainability Module for CrediVist
Uses SHAP to explain individual predictions and global model behavior.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.ml_model import ML_FEATURES

# Human-readable feature names for display
FEATURE_LABELS = {
    "feat_income_stability": "Income Stability",
    "feat_income_trend": "Income Trend",
    "feat_cash_flow_ratio": "Cash Flow Health",
    "feat_income_diversity": "Income Diversity",
    "feat_utility_score": "Utility Bill Timeliness",
    "feat_emi_score": "EMI-like Behavior",
    "feat_txn_regularity": "Transaction Regularity",
    "feat_expense_score": "Essential Expense Ratio",
    "feat_savings_score": "Savings Discipline",
    "feat_work_reliability": "Work Reliability",
    "feat_shock_recovery": "Shock Recovery",
    "recharge_regularity": "Recharge Regularity",
    "mean_income": "Mean Monthly Income",
    "income_std": "Income Variability",
    "num_income_sources": "Number of Income Sources",
    "tenure_months": "Platform Tenure (Months)",
    "platform_rating": "Platform Rating",
    "active_days_per_month": "Active Days / Month",
    "avg_monthly_savings": "Monthly Savings (₹)",
    "total_transactions": "Total Transactions",
}


class ScoreExplainer:
    """SHAP-based explainability for CrediVist predictions."""

    def __init__(self, model):
        """
        Args:
            model: Trained CreditRiskModel instance.
        """
        self.model = model
        self.explainer = None
        self.shap_values = None

    def initialize(self, background_data: pd.DataFrame, n_background: int = 100):
        """Create SHAP explainer using background data."""
        if not HAS_SHAP:
            return

        X_bg = background_data[ML_FEATURES].copy()
        X_bg = X_bg.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Sample background
        if len(X_bg) > n_background:
            X_bg = X_bg.sample(n_background, random_state=42)

        # Use the primary model for SHAP
        if self.model.xgb_model is not None:
            self.explainer = shap.TreeExplainer(self.model.xgb_model)
        else:
            # For LR, use KernelExplainer
            X_bg_scaled = self.model.scaler.transform(X_bg)
            self.explainer = shap.LinearExplainer(
                self.model.lr_model, X_bg_scaled
            )

    def explain_single(self, row: pd.Series) -> dict:
        """
        Explain a single prediction.
        Returns top positive and negative contributors.
        """
        if not HAS_SHAP or self.explainer is None:
            return self._fallback_explanation(row)

        X = pd.DataFrame([row[ML_FEATURES]])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self.model.xgb_model is not None:
            sv = self.explainer.shap_values(X)
        else:
            X_scaled = self.model.scaler.transform(X)
            sv = self.explainer.shap_values(X_scaled)

        shap_vals = sv[0] if isinstance(sv, list) else sv[0]

        # Build explanation
        explanations = []
        for feat, val in zip(ML_FEATURES, shap_vals):
            label = FEATURE_LABELS.get(feat, feat)
            feature_value = float(row[feat]) if feat in row.index else 0
            explanations.append({
                "feature": label,
                "feature_key": feat,
                "shap_value": round(float(val), 4),
                "feature_value": round(feature_value, 4),
                "direction": "positive" if val > 0 else "negative",
            })

        # Sort by absolute impact
        explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        top_positive = [e for e in explanations if e["shap_value"] > 0][:5]
        top_negative = [e for e in explanations if e["shap_value"] < 0][:5]

        return {
            "all_contributions": explanations,
            "top_risk_factors": top_positive,
            "top_positive_factors": top_negative,  # negative SHAP = reduces risk
            "explanation_text": self._generate_text(top_positive, top_negative),
        }

    def _fallback_explanation(self, row: pd.Series) -> dict:
        """Rule-based explanation when SHAP is not available."""
        factors = []
        for feat in ML_FEATURES:
            if feat in row.index:
                label = FEATURE_LABELS.get(feat, feat)
                val = float(row[feat])
                # Simple threshold-based
                if val >= 0.7:
                    impact = "positive"
                    strength = val
                elif val <= 0.3:
                    impact = "negative"
                    strength = 1 - val
                else:
                    impact = "neutral"
                    strength = 0.5
                factors.append({
                    "feature": label,
                    "feature_key": feat,
                    "shap_value": round(strength if impact == "negative" else -strength, 4),
                    "feature_value": round(val, 4),
                    "direction": impact,
                })

        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_positive = [f for f in factors if f["direction"] == "negative"][:5]
        top_negative = [f for f in factors if f["direction"] == "positive"][:5]

        return {
            "all_contributions": factors,
            "top_risk_factors": top_negative,
            "top_positive_factors": top_positive,
            "explanation_text": self._generate_text(top_negative, top_positive),
        }

    def _generate_text(self, risk_factors: list, positive_factors: list) -> str:
        """Generate human-readable explanation text."""
        lines = []
        if positive_factors:
            lines.append("**Strengths:**")
            for f in positive_factors[:3]:
                lines.append(f"  • {f['feature']} is working in your favor")
        if risk_factors:
            lines.append("\n**Areas to Improve:**")
            for f in risk_factors[:3]:
                lines.append(f"  • {f['feature']} needs attention")
        return "\n".join(lines) if lines else "Score computed based on overall financial profile."

    def plot_waterfall(self, row: pd.Series) -> plt.Figure:
        """Generate a waterfall plot for a single prediction."""
        if not HAS_SHAP or self.explainer is None:
            return self._plot_bar_fallback(row)

        X = pd.DataFrame([row[ML_FEATURES]])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if self.model.xgb_model is not None:
            sv = self.explainer(X)
        else:
            X_scaled = self.model.scaler.transform(X)
            sv = self.explainer(X_scaled)

        # Rename features for display
        sv.feature_names = [FEATURE_LABELS.get(f, f) for f in ML_FEATURES]

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(sv[0], show=False)
        plt.tight_layout()
        return fig

    def _plot_bar_fallback(self, row: pd.Series) -> plt.Figure:
        """Simple bar chart when SHAP is unavailable."""
        features = []
        values = []
        for feat in ML_FEATURES[:10]:  # top 10
            if feat in row.index:
                features.append(FEATURE_LABELS.get(feat, feat))
                values.append(float(row[feat]))

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#22c55e" if v >= 0.5 else "#ef4444" for v in values]
        ax.barh(features, values, color=colors)
        ax.set_xlabel("Feature Value")
        ax.set_title("Feature Contribution to Credit Score")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        return fig

    def plot_global_importance(self, df: pd.DataFrame) -> plt.Figure:
        """Plot global feature importance across all users."""
        if not HAS_SHAP or self.explainer is None:
            # Use model's built-in feature importance
            imp = self.model.get_feature_importance()
            if not imp:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No importance data", ha="center", va="center")
                return fig

            labels = [FEATURE_LABELS.get(k, k) for k in imp.keys()]
            vals = list(imp.values())

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(labels[:10], vals[:10], color="#6366f1")
            ax.set_xlabel("Importance")
            ax.set_title("Global Feature Importance")
            plt.tight_layout()
            return fig

        X = df[ML_FEATURES].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        if len(X) > 200:
            X = X.sample(200, random_state=42)

        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X,
            feature_names=[FEATURE_LABELS.get(f, f) for f in ML_FEATURES],
            show=False, plot_type="bar"
        )
        plt.tight_layout()
        return fig
