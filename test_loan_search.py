"""
Test suite for Loan Search & Eligibility Checker
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.loan_engine import (
    search_loans, check_loan_eligibility, get_all_loans_catalog,
    get_loan_categories, TRANSACTION_LOANS, PERSONA_LOANS,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")

print("=" * 60)
print("LOAN SEARCH & ELIGIBILITY CHECKER TESTS")
print("=" * 60)

# ── Test 1: Full Catalog ──
print("\n--- Test 1: Full Catalog ---")
catalog = get_all_loans_catalog()
print(f"  Total loans: {len(catalog)}")
check("Catalog has 30+ loans", len(catalog) >= 30)
check("Each loan has source tag", all("source" in l for l in catalog))
check("Each loan has key", all("key" in l for l in catalog))

# ── Test 2: Categories ──
print("\n--- Test 2: Loan Categories ---")
cats = get_loan_categories()
print(f"  Categories: {cats}")
check("Has Agriculture category", "Agriculture" in cats)
check("Has Personal category", "Personal" in cats)
check("Has multiple categories", len(cats) >= 5)

# ── Test 3: Search by keyword ──
print("\n--- Test 3: Keyword Search ---")
r = search_loans(query="KCC")
check("KCC search finds Kisan Credit Card", len(r) == 1 and "Kisan" in r[0]["name"])

r = search_loans(query="gold")
check("Gold search finds 3+ gold loans", len(r) >= 3)

r = search_loans(query="mudra")
check("Mudra search finds 4+ loans", len(r) >= 4)

r = search_loans(query="education")
check("Education search finds 2+ loans", len(r) >= 2)

r = search_loans(query="nonexistent_xyz_123")
check("Bad search returns empty", len(r) == 0)

r = search_loans(query="")
check("Empty search returns all", len(r) == len(catalog))

# ── Test 4: Filter by source ──
print("\n--- Test 4: Source Filter ---")
txn = search_loans(source_filter="transaction")
check("Transaction filter returns 9 loans", len(txn) == len(TRANSACTION_LOANS))

persona = search_loans(source_filter="persona")
total_persona = sum(len(v) for v in PERSONA_LOANS.values())
check("Persona filter returns all persona loans", len(persona) == total_persona)

# ── Test 5: Filter by persona ──
print("\n--- Test 5: Persona Filter ---")
for p_name, p_key in [("farmer", "farmer"), ("student", "student"),
                       ("street_vendor", "street_vendor"),
                       ("homemaker", "homemaker"),
                       ("general_no_bank", "general_no_bank")]:
    r = search_loans(persona_filter=p_key)
    expected = len(PERSONA_LOANS[p_key])
    check(f"{p_name} filter: {len(r)} == {expected}", len(r) == expected)

# ── Test 6: Subsidy filter ──
print("\n--- Test 6: Subsidy Filter ---")
subsidized = search_loans(subsidy_filter=True)
check("Subsidized loans > 5", len(subsidized) > 5)
check("All results have subsidy", all(l.get("subsidy") for l in subsidized))

# ── Test 7: Collateral filter ──
print("\n--- Test 7: Collateral Filter ---")
no_col = search_loans(collateral_filter="no")
check("No-collateral loans exist", len(no_col) > 10)
check("All are collateral-free", all(not l.get("collateral") for l in no_col))

yes_col = search_loans(collateral_filter="yes")
check("Collateral loans exist", len(yes_col) > 3)
check("All require collateral", all(l.get("collateral") for l in yes_col))

# ── Test 8: Max rate filter ──
print("\n--- Test 8: Rate Filter ---")
low_rate = search_loans(max_rate=8.0)
check("Low rate filter works", len(low_rate) > 0)
check("All within rate limit", all(l["interest_range"][0] <= 8.0 for l in low_rate))

# ── Test 9: Min amount filter ──
print("\n--- Test 9: Amount Filter ---")
big_loans = search_loans(min_amount=1000000)
check("Big loan filter works", len(big_loans) > 0)
check("All meet min amount", all(l["amount_range"][1] >= 1000000 for l in big_loans))

# ── Test 10: Combined filters ──
print("\n--- Test 10: Combined Filters ---")
r = search_loans(persona_filter="farmer", subsidy_filter=True)
check("Farmer + subsidy combo works", len(r) >= 3)

r = search_loans(query="loan", collateral_filter="no", max_rate=15.0)
check("Multi-filter combo works", len(r) > 0)

# ══════════════════════════════════════════════════════════════
# ELIGIBILITY CHECKER TESTS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ELIGIBILITY CHECKER TESTS")
print("=" * 60)

# ── Test 11: Eligible — good score, good income ──
print("\n--- Test 11: Fully Eligible ---")
result = check_loan_eligibility(
    loan_key="personal_loan", source="transaction",
    score=700, monthly_income=30000, monthly_expenses=10000,
)
check("Verdict is ELIGIBLE", result["verdict"] == "ELIGIBLE")
check("Has loan details", bool(result["loan_details"]))
check("No fail reasons for core checks", not any(
    "below" in r.lower() for r in result["reasons_fail"]
))
check("Max amount > 0", result["loan_details"]["max_eligible_amount"] > 0)
check("EMI > 0", result["loan_details"]["emi"] > 0)

# ── Test 12: Not eligible — low score ──
print("\n--- Test 12: Low Score ---")
result = check_loan_eligibility(
    loan_key="home_loan", source="transaction",
    score=450, monthly_income=15000,
)
check("Verdict is NOT_ELIGIBLE", result["verdict"] == "NOT_ELIGIBLE")
check("Score fail reason present", any("Score" in r and "below" in r for r in result["reasons_fail"]))
check("Gap analysis has score gap", any(g["check"] == "Credit Score" for g in result["gap_analysis"]))

# ── Test 13: Not eligible — low income ──
print("\n--- Test 13: Low Income ---")
result = check_loan_eligibility(
    loan_key="personal_loan", source="transaction",
    score=650, monthly_income=5000,
)
check("Income fail reason present", any("income" in r.lower() and "below" in r.lower() for r in result["reasons_fail"]))

# ── Test 14: Very poor score — tier blocks all ──
print("\n--- Test 14: Very Poor Score ---")
result = check_loan_eligibility(
    loan_key="personal_loan", source="transaction",
    score=350, monthly_income=20000,
)
check("Verdict NOT_ELIGIBLE for score 350", result["verdict"] == "NOT_ELIGIBLE")
check("Tier block reason present", any("Very Poor" in r or "too low" in r for r in result["reasons_fail"]))

# ── Test 15: Persona loan — farmer KCC eligible ──
print("\n--- Test 15: Farmer KCC Eligible ---")
result = check_loan_eligibility(
    loan_key="kcc", source="persona", persona="farmer",
    score=600, monthly_income=0,
    persona_data={"owns_land": True, "crops_per_year": 2, "land_acres": 3},
)
check("KCC eligible for farmer", result["verdict"] in ("ELIGIBLE", "ELIGIBLE_WITH_CAUTION"))
check("Income estimated (was 0)", result["repayment_capacity"]["monthly_income"] > 0)
check("Owns land criteria met", any("Owns Land" in r for r in result["reasons_pass"]))

# ── Test 16: Persona loan — student education ──
print("\n--- Test 16: Student Education Loan ---")
result = check_loan_eligibility(
    loan_key="education_loan", source="persona", persona="student",
    score=550, monthly_income=5000,
    persona_data={"score_value": 550},
)
check("Student edu loan has verdict", result["verdict"] in ("ELIGIBLE", "ELIGIBLE_WITH_CAUTION", "MICRO_ONLY"))

# ── Test 17: Desired amount exceeds max ──
print("\n--- Test 17: Amount Exceeds Max ---")
result = check_loan_eligibility(
    loan_key="emergency_loan", source="transaction",
    score=550, monthly_income=10000,
    desired_amount=500000,
)
check("Has amount gap", any("Desired amount" in r or "exceeds" in r.lower() for r in result["reasons_fail"]))

# ── Test 18: Loan not found ──
print("\n--- Test 18: Loan Not Found ---")
result = check_loan_eligibility(
    loan_key="nonexistent_loan_xyz", source="transaction",
    score=700, monthly_income=30000,
)
check("Verdict is LOAN_NOT_FOUND", result["verdict"] == "LOAN_NOT_FOUND")

# ── Test 19: Improvement steps present when not eligible ──
print("\n--- Test 19: Improvement Steps ---")
result = check_loan_eligibility(
    loan_key="home_loan", source="transaction",
    score=450, monthly_income=10000,
)
check("Has improvement steps", len(result["improvement_steps"]) > 0)
check("Has gap analysis", len(result["gap_analysis"]) > 0)

# ── Test 20: Eligible with desired amount within range ──
print("\n--- Test 20: Desired Amount Within Range ---")
result = check_loan_eligibility(
    loan_key="personal_loan", source="transaction",
    score=700, monthly_income=30000,
    desired_amount=50000, desired_tenure=24,
)
check("Eligible with specific amount", result["verdict"] == "ELIGIBLE")
check("Amount matches requested", result["loan_details"]["actual_amount"] == 50000)

# ── Test 21: Existing EMI reduces capacity ──
print("\n--- Test 21: High Existing EMI ---")
result_no_emi = check_loan_eligibility(
    loan_key="personal_loan", source="transaction",
    score=700, monthly_income=30000,
)
result_high_emi = check_loan_eligibility(
    loan_key="personal_loan", source="transaction",
    score=700, monthly_income=30000, existing_emi=14000,
)
if result_no_emi["loan_details"] and result_high_emi["loan_details"]:
    check("High EMI reduces max amount",
          result_high_emi["loan_details"]["max_eligible_amount"] <
          result_no_emi["loan_details"]["max_eligible_amount"])
else:
    check("High EMI reduces eligibility",
          result_high_emi["verdict"] != "ELIGIBLE" or
          not result_high_emi["loan_details"])

# ── Test 22: All persona catalogs are searchable ──
print("\n--- Test 22: All Persona Loans Checkable ---")
for persona_key, persona_catalog in PERSONA_LOANS.items():
    first_loan_key = list(persona_catalog.keys())[0]
    result = check_loan_eligibility(
        loan_key=first_loan_key, source="persona", persona=persona_key,
        score=700, monthly_income=20000,
    )
    check(f"{persona_key}: {first_loan_key} check works",
          result["verdict"] != "LOAN_NOT_FOUND")

# ── Test 23: All transaction loans are checkable ──
print("\n--- Test 23: All Transaction Loans Checkable ---")
for loan_key in TRANSACTION_LOANS:
    result = check_loan_eligibility(
        loan_key=loan_key, source="transaction",
        score=700, monthly_income=30000,
    )
    check(f"Transaction {loan_key} check works",
          result["verdict"] != "LOAN_NOT_FOUND")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)
if failed == 0:
    print("ALL LOAN SEARCH & ELIGIBILITY TESTS PASSED!")
else:
    print(f"WARNING: {failed} test(s) FAILED")
    sys.exit(1)
