"""Quick test for the transaction parser module."""
import io
from src.transaction_parser import TransactionParser, generate_sample_statement

# 1. Generate sample bank statement
sample = generate_sample_statement()
print(f"Sample statement: {len(sample)} transactions")
print(f"Columns: {sample.columns.tolist()}")
print(sample.head(5).to_string())

# 2. Parse it
parser = TransactionParser()
csv_buf = io.StringIO(sample.to_csv(index=False))
parsed = parser.parse_file(csv_buf, "csv")
categorized = parser.auto_categorize(parsed)

print(f"\nParsed: {len(categorized)} transactions")
print(f"Categories: {categorized['category'].value_counts().to_dict()}")

# 3. Get summaries
psummary = parser.get_parsing_summary()
print(f"\nParsing Summary: {psummary}")

cat_summary = parser.get_category_summary()
print(f"\nCategory Summary:\n{cat_summary.to_string()}")

monthly = parser.get_monthly_summary()
print(f"\nMonthly Summary:\n{monthly.to_string()}")

# 4. Extract profile
profile = parser.extract_profile()
print(f"\n--- Extracted Profile ---")
for key, val in profile.items():
    print(f"  {key}: {val}")

# 5. Test with full pipeline
from src.feature_engineering import extract_all_features
from src.scoring_engine import compute_base_score, compute_final_score

features = extract_all_features(profile)
for key, val in features.items():
    if not isinstance(val, str):
        profile[key] = val

base_result = compute_base_score(profile)
for key, val in base_result.items():
    profile[key] = val

print(f"\nBase Trust Score: {profile['base_trust_score']}")
print(f"Sub-scores:")
print(f"  Financial Stability: {profile['sub_financial_stability']}")
print(f"  Payment Discipline:  {profile['sub_payment_discipline']}")
print(f"  Digital Behavior:    {profile['sub_digital_behavior']}")
print(f"  Work Reliability:    {profile['sub_work_reliability']}")

# ML prediction (needs trained model)
from src.ml_model import CreditRiskModel
import pandas as pd
from src.feature_engineering import engineer_features
from src.scoring_engine import compute_all_scores

raw_df = pd.read_csv("data/credit_data.csv")
raw_df = engineer_features(raw_df)
raw_df = compute_all_scores(raw_df)
model = CreditRiskModel()
model.train(raw_df)

risk_prob = model.predict_single(profile)
final = compute_final_score(float(profile["base_trust_score"]), risk_prob, profile)

print(f"\nRisk Probability: {risk_prob:.2%}")
print(f"Final Trust Score: {final['final_trust_score']:.0f}")
print(f"Grade: {final['grade']}")
print(f"Confidence: {final['confidence']:.0%}")
print(f"\n✅ Transaction Parser Pipeline Test PASSED!")
