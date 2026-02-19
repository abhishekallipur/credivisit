"""
ML Model Module for CrediVist
Trains XGBoost + Logistic Regression for default risk prediction.
Provides risk probability that adjusts the final trust score.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, accuracy_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Feature columns used for ML model
ML_FEATURES = [
    "feat_income_stability",
    "feat_income_trend",
    "feat_cash_flow_ratio",
    "feat_income_diversity",
    "feat_utility_score",
    "feat_emi_score",
    "feat_txn_regularity",
    "feat_expense_score",
    "feat_savings_score",
    "feat_work_reliability",
    "feat_shock_recovery",
    "recharge_regularity",
    "mean_income",
    "income_std",
    "num_income_sources",
    "tenure_months",
    "platform_rating",
    "active_days_per_month",
    "avg_monthly_savings",
    "total_transactions",
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class CreditRiskModel:
    """Hybrid ML model for default risk prediction."""

    def __init__(self):
        self.xgb_model = None
        self.lr_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.metrics = {}

    def train(self, df: pd.DataFrame, target_col: str = "default"):
        """Train both XGBoost and Logistic Regression models."""
        X = df[ML_FEATURES].copy()
        y = df[target_col].copy()

        # Handle any NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── XGBoost ─────────────────────────────────────────────────────
        if HAS_XGBOOST:
            self.xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
            self.xgb_model.fit(X_train, y_train)

            xgb_pred = self.xgb_model.predict(X_test)
            xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]

            self.metrics["xgboost"] = {
                "accuracy": round(accuracy_score(y_test, xgb_pred), 4),
                "roc_auc": round(roc_auc_score(y_test, xgb_proba), 4),
                "confusion_matrix": confusion_matrix(y_test, xgb_pred).tolist(),
                "classification_report": classification_report(y_test, xgb_pred, output_dict=True),
            }

        # ── Logistic Regression ─────────────────────────────────────────
        self.lr_model = LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        )
        self.lr_model.fit(X_train_scaled, y_train)

        lr_pred = self.lr_model.predict(X_test_scaled)
        lr_proba = self.lr_model.predict_proba(X_test_scaled)[:, 1]

        self.metrics["logistic_regression"] = {
            "accuracy": round(accuracy_score(y_test, lr_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, lr_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
            "classification_report": classification_report(y_test, lr_pred, output_dict=True),
        }

        # Cross-validation
        cv_scores = cross_val_score(self.lr_model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
        self.metrics["cross_val_auc"] = {
            "mean": round(cv_scores.mean(), 4),
            "std": round(cv_scores.std(), 4),
        }

        self.is_trained = True
        return self.metrics

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict default risk probability.
        Uses ensemble of XGBoost + LR if both available.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")

        X = df[ML_FEATURES].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if HAS_XGBOOST and self.xgb_model is not None:
            xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
            X_scaled = self.scaler.transform(X)
            lr_proba = self.lr_model.predict_proba(X_scaled)[:, 1]
            # Ensemble: 60% XGBoost + 40% LR
            risk = 0.6 * xgb_proba + 0.4 * lr_proba
        else:
            X_scaled = self.scaler.transform(X)
            risk = self.lr_model.predict_proba(X_scaled)[:, 1]

        return risk

    def predict_single(self, row: pd.Series) -> float:
        """Predict risk for a single user."""
        df = pd.DataFrame([row])
        return float(self.predict_risk(df)[0])

    def get_feature_importance(self) -> dict:
        """Return feature importance from XGBoost (or LR coefficients)."""
        if HAS_XGBOOST and self.xgb_model is not None:
            importances = self.xgb_model.feature_importances_
        elif self.lr_model is not None:
            importances = np.abs(self.lr_model.coef_[0])
        else:
            return {}

        imp_dict = dict(zip(ML_FEATURES, importances))
        # Sort descending
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        return imp_dict

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            path = os.path.join(MODELS_DIR, "credit_risk_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "xgb_model": self.xgb_model,
                "lr_model": self.lr_model,
                "scaler": self.scaler,
                "metrics": self.metrics,
                "is_trained": self.is_trained,
            }, f)
        return path

    def load(self, path: str = None):
        """Load model from disk."""
        if path is None:
            path = os.path.join(MODELS_DIR, "credit_risk_model.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.xgb_model = data["xgb_model"]
        self.lr_model = data["lr_model"]
        self.scaler = data["scaler"]
        self.metrics = data["metrics"]
        self.is_trained = data["is_trained"]
        return self
