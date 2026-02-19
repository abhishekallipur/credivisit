"""Quick smoke test for CrediVist pipeline."""
import pandas as pd
from src.feature_engineering import engineer_features
from src.scoring_engine import compute_all_scores, compute_final_score
from src.ml_model import CreditRiskModel

# Load data
df = pd.read_csv("data/credit_data.csv")
print(f"Loaded {len(df)} users")

# Feature engineering
df = engineer_features(df)
feat_cols = [c for c in df.columns if c.startswith("feat_")]
print(f"Features engineered: {feat_cols}")

# Scoring
df = compute_all_scores(df)
print(f"Base scores computed. Mean: {df['base_trust_score'].mean():.0f}")

# ML Model
model = CreditRiskModel()
metrics = model.train(df)
for name, m in metrics.items():
    if "accuracy" in m:
        print(f"{name}: accuracy={m['accuracy']:.4f}, auc={m['roc_auc']:.4f}")

# Risk prediction
risk = model.predict_risk(df)
df["risk_probability"] = risk

# Final scores
for idx, row in df.head(5).iterrows():
    fs = compute_final_score(row["base_trust_score"], row["risk_probability"], row)
    print(f"{row['user_id']}: Score={fs['final_trust_score']:.0f} Grade={fs['grade']} Risk={row['risk_probability']:.2%}")

model.save()
print("Model saved successfully!")
print("Pipeline test PASSED")
