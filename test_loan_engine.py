"""
Comprehensive Test Suite for Loan Recommendation Engine
========================================================
Tests all components: EMI calculation, repayment capacity, score tiers,
transaction-based recommendations, persona-based recommendations,
loan comparison, financial tips, and seasonal recommendations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.loan_engine import (
    calculate_emi, calculate_total_interest,
    analyze_repayment_capacity, max_loan_from_emi,
    get_score_tier, get_transaction_loan_recommendations,
    get_persona_loan_recommendations, compare_loans,
    get_financial_tips, get_seasonal_recommendations,
    generate_repayment_schedule, SCORE_TIERS,
    TRANSACTION_LOANS, PERSONA_LOANS,
)


def test_emi_calculation():
    """Test EMI formula with known values."""
    # ₹1,00,000 at 12% for 12 months → EMI ≈ ₹8,885
    emi = calculate_emi(100000, 12.0, 12)
    assert 8800 < emi < 8900, f"EMI should be ~₹8,885, got {emi}"

    # ₹5,00,000 at 10% for 60 months → EMI ≈ ₹10,624
    emi2 = calculate_emi(500000, 10.0, 60)
    assert 10600 < emi2 < 10650, f"EMI should be ~₹10,624, got {emi2}"

    # Zero rate → simple division
    emi3 = calculate_emi(12000, 0, 12)
    assert emi3 == 1000.0, f"Zero rate EMI should be 1000, got {emi3}"

    # Edge cases
    assert calculate_emi(0, 12, 12) == 0.0
    assert calculate_emi(100000, 12, 0) == 0.0

    print("  ✓ EMI calculation: PASS")


def test_total_interest():
    """Test total interest calculation."""
    interest = calculate_total_interest(100000, 12.0, 12)
    # Total paid = EMI × 12, interest = total - principal
    assert 6000 < interest < 7000, f"Interest should be ~₹6,619, got {interest}"

    # Zero rate → zero interest
    assert calculate_total_interest(100000, 0, 12) == 0.0

    print("  ✓ Total interest: PASS")


def test_repayment_capacity():
    """Test repayment capacity analysis."""
    # Healthy borrower
    rep = analyze_repayment_capacity(
        monthly_income=50000,
        monthly_expenses=20000,
        existing_emi=5000,
        foir_cap=0.40,
    )
    assert rep["monthly_income"] == 50000
    assert rep["max_total_emi"] == 20000  # 50K × 0.40
    assert rep["max_new_emi"] == 15000   # 20K - 5K
    assert rep["verdict"] == "ELIGIBLE"
    assert len(rep["risk_flags"]) == 0

    # Over-leveraged borrower
    rep2 = analyze_repayment_capacity(
        monthly_income=30000,
        monthly_expenses=15000,
        existing_emi=18000,
        foir_cap=0.40,
    )
    assert rep2["max_new_emi"] == 0  # 12K - 18K = negative → 0
    assert rep2["verdict"] == "NOT_ELIGIBLE"
    assert any("Over-leveraged" in f for f in rep2["risk_flags"])

    # Zero income
    rep3 = analyze_repayment_capacity(0, 0, 0, 0.40)
    assert rep3["verdict"] == "NOT_ELIGIBLE"

    print("  ✓ Repayment capacity analysis: PASS")


def test_max_loan_from_emi():
    """Test reverse EMI calculation."""
    # If EMI budget is ₹8,885, rate 12%, 12 months → principal ~₹1,00,000
    principal = max_loan_from_emi(8885, 12.0, 12)
    assert 99000 < principal < 101000, f"Principal should be ~1L, got {principal}"

    # Zero EMI
    assert max_loan_from_emi(0, 12, 12) == 0.0

    # Zero rate
    principal2 = max_loan_from_emi(1000, 0, 12)
    assert principal2 == 12000.0

    print("  ✓ Max loan from EMI: PASS")


def test_score_tiers():
    """Test score tier resolution."""
    # Excellent
    tier = get_score_tier(800)
    assert tier["grade"] == "Excellent"
    assert tier["max_simultaneous_loans"] == 3

    # Good
    tier2 = get_score_tier(700)
    assert tier2["grade"] == "Good"

    # Fair
    tier3 = get_score_tier(600)
    assert tier3["grade"] == "Fair"

    # Poor
    tier4 = get_score_tier(450)
    assert tier4["grade"] == "Poor"

    # Very Poor
    tier5 = get_score_tier(350)
    assert tier5["grade"] == "Very Poor"
    assert tier5["max_simultaneous_loans"] == 0

    # Edge: exactly 750
    tier6 = get_score_tier(750)
    assert tier6["grade"] == "Excellent"

    print("  ✓ Score tiers: PASS")


def test_transaction_loan_recommendations_excellent():
    """Test recommendations for excellent score with bank history."""
    recs = get_transaction_loan_recommendations(
        score=800,
        monthly_income=50000,
        monthly_expenses=20000,
        existing_emi=0,
    )

    assert recs["score"] == 800
    assert recs["tier"]["grade"] == "Excellent"
    assert recs["total_eligible"] > 0
    assert recs["max_simultaneous_loans"] == 3
    assert recs["pre_approval_status"] == "Pre-Approved"
    assert recs["source"] == "transaction"

    # Should include personal loan, credit card at minimum
    eligible_names = [l["name"] for l in recs["eligible_loans"]]
    assert "Personal Loan" in eligible_names, f"Personal Loan missing from {eligible_names}"
    assert "Credit Card" in eligible_names, f"Credit Card missing from {eligible_names}"

    # Each loan should have EMI calculated
    for loan in recs["eligible_loans"]:
        assert loan["emi"] >= 0
        assert loan["max_loan_amount"] >= 0
        assert loan["effective_rate"] > 0

    # Repayment capacity should show healthy stats
    assert recs["repayment_capacity"]["verdict"] in ("ELIGIBLE", "ELIGIBLE_WITH_CAUTION")

    print("  ✓ Transaction recs (excellent): PASS")


def test_transaction_loan_recommendations_poor():
    """Test recommendations for poor score."""
    recs = get_transaction_loan_recommendations(
        score=420,
        monthly_income=15000,
        monthly_expenses=8000,
    )

    assert recs["tier"]["grade"] == "Poor"
    assert recs["max_simultaneous_loans"] == 1

    # Should have fewer eligible loans
    assert recs["total_eligible"] < 5

    # Gold loan and secured loans should still be available
    if recs["total_eligible"] > 0:
        has_secured = any(
            l["collateral_required"] for l in recs["eligible_loans"]
        )
        # At minimum gold loan / FD loan should be there
        eligible_keys = [l["key"] for l in recs["eligible_loans"]]
        assert ("gold_loan" in eligible_keys or "loan_against_fd" in eligible_keys), \
            f"Expected secured loans for poor tier, got {eligible_keys}"

    print("  ✓ Transaction recs (poor): PASS")


def test_transaction_loan_recommendations_very_poor():
    """Test recommendations for very poor score."""
    recs = get_transaction_loan_recommendations(
        score=350,
        monthly_income=8000,
    )

    assert recs["tier"]["grade"] == "Very Poor"
    assert recs["max_simultaneous_loans"] == 0
    assert recs["total_eligible"] == 0

    # Should have improvement path
    assert len(recs["improvement_path"]) > 0

    print("  ✓ Transaction recs (very poor): PASS")


def test_persona_farmer_loans():
    """Test farmer-specific loan recommendations."""
    recs = get_persona_loan_recommendations(
        persona="farmer",
        score=700,
        persona_data={
            "owns_land": True,
            "land_acres": 5,
            "crops_per_year": 2,
            "has_warehouse_receipt": True,
            "years_on_land": 10,
        },
    )

    assert recs["persona"] == "farmer"
    assert recs["source"] == "alternative_profile"
    assert recs["total_eligible"] > 0
    assert recs["estimated_monthly_income"] > 0

    # KCC should be eligible
    eligible_names = [l["name"] for l in recs["eligible_loans"]]
    assert "Kisan Credit Card (KCC)" in eligible_names, \
        f"KCC missing from {eligible_names}"

    # KCC should have subsidy info
    kcc = [l for l in recs["eligible_loans"] if l["key"] == "kcc"][0]
    assert kcc["subsidy"] is not None
    assert kcc["effective_rate"] <= 7.0  # subsidized rate
    assert kcc["collateral_required"] is False

    print("  ✓ Farmer loan recs: PASS")


def test_persona_student_loans():
    """Test student-specific loan recommendations."""
    recs = get_persona_loan_recommendations(
        persona="student",
        score=650,
        persona_data={
            "score_value": 8.0,
            "has_internship": True,
            "monthly_earnings": 5000,
        },
    )

    assert recs["persona"] == "student"
    assert recs["total_eligible"] > 0

    eligible_names = [l["name"] for l in recs["eligible_loans"]]
    assert "Education Loan (Vidya Lakshmi)" in eligible_names

    # Education loan should be subsidized
    edu = [l for l in recs["eligible_loans"] if l["key"] == "education_loan"][0]
    assert edu["subsidy"] is not None

    print("  ✓ Student loan recs: PASS")


def test_persona_vendor_loans():
    """Test street vendor loan recommendations."""
    recs = get_persona_loan_recommendations(
        persona="street_vendor",
        score=550,
        persona_data={
            "has_license": True,
            "avg_daily_income": 800,
            "working_days_per_month": 25,
            "years_in_trade": 5,
        },
    )

    assert recs["persona"] == "street_vendor"
    assert recs["total_eligible"] > 0

    eligible_names = [l["name"] for l in recs["eligible_loans"]]
    assert "PM SVANidhi (Street Vendor Loan)" in eligible_names

    # SVANidhi should have 7% subsidy
    svanidhi = [l for l in recs["eligible_loans"] if l["key"] == "pm_svanidhi"][0]
    assert svanidhi["subsidy"] is not None
    assert "7%" in svanidhi["subsidy"]

    print("  ✓ Vendor loan recs: PASS")


def test_persona_homemaker_loans():
    """Test homemaker loan recommendations."""
    recs = get_persona_loan_recommendations(
        persona="homemaker",
        score=600,
        persona_data={
            "is_shg_member": True,
            "has_enterprise": True,
            "household_income": 25000,
            "monthly_revenue": 5000,
        },
    )

    assert recs["persona"] == "homemaker"
    assert recs["total_eligible"] > 0

    eligible_names = [l["name"] for l in recs["eligible_loans"]]
    assert "SHG Group Loan (NRLM)" in eligible_names

    print("  ✓ Homemaker loan recs: PASS")


def test_persona_general_loans():
    """Test general (no bank) loan recommendations."""
    recs = get_persona_loan_recommendations(
        persona="general_no_bank",
        score=500,
        persona_data={
            "is_group_member": True,
            "rent_amount": 3000,
        },
    )

    assert recs["persona"] == "general_no_bank"
    assert recs["total_eligible"] > 0

    # Gold loan should always be available (min score 350)
    eligible_keys = [l["key"] for l in recs["eligible_loans"]]
    assert "gold_loan_g" in eligible_keys

    print("  ✓ General loan recs: PASS")


def test_loan_comparison():
    """Test loan comparison/ranking."""
    recs = get_transaction_loan_recommendations(
        score=750,
        monthly_income=40000,
    )

    top = compare_loans(recs["eligible_loans"])
    assert len(top) <= 3
    assert len(top) > 0

    # Should be sorted by composite score
    for loan in top:
        assert loan["eligible"] is True
        assert "_composite_score" in loan

    # First should have highest composite
    if len(top) > 1:
        assert top[0]["_composite_score"] >= top[1]["_composite_score"]

    print("  ✓ Loan comparison: PASS")


def test_financial_tips():
    """Test financial literacy tips generation."""
    # Farmer tips
    tips = get_financial_tips(persona="farmer", score=600)
    assert len(tips) > 0
    tip_titles = [t["title"] for t in tips]
    assert any("KCC" in t for t in tip_titles), f"Missing KCC tip in {tip_titles}"

    # Student tips
    tips2 = get_financial_tips(persona="student", score=500)
    assert len(tips2) > 0
    titles2 = [t["title"] for t in tips2]
    assert any("Education" in t or "Vidya" in t for t in titles2)

    # Vendor tips
    tips3 = get_financial_tips(persona="street_vendor", score=400)
    assert len(tips3) > 0
    titles3 = [t["title"] for t in tips3]
    assert any("SVANidhi" in t for t in titles3)

    # General (low score)
    tips4 = get_financial_tips(score=400)
    assert any("Build Credit" in t["title"] for t in tips4)

    # With subsidized loans
    fake_loans = [{"subsidy": "Test subsidy", "interest_saved_via_subsidy": 5000}]
    tips5 = get_financial_tips(eligible_loans=fake_loans)
    titles5 = [t["title"] for t in tips5]
    assert any("Subsidy" in t for t in titles5)

    print("  ✓ Financial tips: PASS")


def test_seasonal_recommendations():
    """Test farmer seasonal recommendations."""
    # Farmer in Feb → should get Rabi or Zaid info
    recs = get_seasonal_recommendations("farmer", "Feb")
    assert len(recs) > 0

    # Non-farmer → empty
    recs2 = get_seasonal_recommendations("student", "Feb")
    assert len(recs2) == 0

    # Farmer in Jun → Kharif active
    recs3 = get_seasonal_recommendations("farmer", "Jun")
    assert len(recs3) > 0
    assert any("Kharif" in r["season"] for r in recs3)

    print("  ✓ Seasonal recommendations: PASS")


def test_repayment_schedule():
    """Test repayment schedule generation."""
    schedule = generate_repayment_schedule(100000, 12.0, 12)
    assert len(schedule) == 12

    # First month should have more interest, less principal
    assert schedule[0]["interest"] > schedule[-1]["interest"]
    assert schedule[0]["principal"] < schedule[-1]["principal"]

    # Last balance should be ~0
    assert schedule[-1]["balance"] < 10, f"Final balance should be ~0, got {schedule[-1]['balance']}"

    # All EMIs should be equal
    emis = set(s["emi"] for s in schedule)
    assert len(emis) == 1

    # Empty case
    empty = generate_repayment_schedule(0, 12, 12)
    assert len(empty) == 0

    print("  ✓ Repayment schedule: PASS")


def test_end_to_end_transaction_flow():
    """End-to-end: score → tier → recommendations → comparison → tips."""
    score = 720
    income = 35000

    recs = get_transaction_loan_recommendations(
        score=score, monthly_income=income,
        monthly_expenses=12000, existing_emi=3000,
    )

    assert recs["tier"]["grade"] == "Good"
    assert recs["total_eligible"] > 0

    # Compare top loans
    top = compare_loans(recs["eligible_loans"])
    assert len(top) > 0

    # Get tips
    tips = get_financial_tips(
        score=score,
        eligible_loans=recs["eligible_loans"],
    )
    assert len(tips) > 0

    # Get improvement path
    assert len(recs["improvement_path"]) > 0
    upgrade = [i for i in recs["improvement_path"] if i["type"] == "score_upgrade"]
    assert len(upgrade) > 0
    assert upgrade[0]["target_score"] == 750

    # Verify EMI for top loan
    if top:
        best = top[0]
        emi = calculate_emi(
            best["recommended_amount"],
            best["effective_rate"],
            best["suggested_tenure"],
        )
        assert abs(emi - best["emi"]) < 1  # should match

    print("  ✓ E2E transaction flow: PASS")


def test_end_to_end_persona_flow():
    """End-to-end: persona → score → persona loans → comparison → seasonal."""
    score = 750
    persona = "farmer"
    data = {
        "owns_land": True,
        "land_acres": 4.5,
        "crops_per_year": 3,
        "has_warehouse_receipt": True,
    }

    recs = get_persona_loan_recommendations(
        persona=persona, score=score, persona_data=data,
    )

    assert recs["tier"]["grade"] == "Excellent"
    assert recs["total_eligible"] > 0

    # Compare
    top = compare_loans(recs["eligible_loans"])
    assert len(top) > 0

    # Seasonal
    seasonal = get_seasonal_recommendations(persona)
    # May or may not have results depending on month mapping

    # Tips
    tips = get_financial_tips(
        persona=persona, score=score,
        eligible_loans=recs["eligible_loans"],
    )
    assert len(tips) > 0

    print("  ✓ E2E persona flow: PASS")


def test_all_persona_catalogs_exist():
    """Verify all 5 personas have loan catalogs."""
    for persona in ["farmer", "student", "street_vendor", "homemaker", "general_no_bank"]:
        assert persona in PERSONA_LOANS, f"Missing catalog for {persona}"
        assert len(PERSONA_LOANS[persona]) > 0, f"Empty catalog for {persona}"

    print("  ✓ All persona catalogs exist: PASS")


def test_interest_saved_via_subsidy():
    """Test subsidy interest savings are calculated."""
    recs = get_persona_loan_recommendations(
        persona="farmer", score=700,
        persona_data={"owns_land": True, "crops_per_year": 2},
    )

    subsidized = [l for l in recs["eligible_loans"] if l.get("subsidy")]
    assert len(subsidized) > 0, "Farmer should have subsidized loans"

    # At least one should show interest saved
    has_savings = any(l.get("interest_saved_via_subsidy", 0) > 0 for l in subsidized)
    assert has_savings, "Subsidized loans should show interest savings"

    print("  ✓ Subsidy interest savings: PASS")


# ─── Run All Tests ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Loan Recommendation Engine — Test Suite")
    print("=" * 60)

    tests = [
        test_emi_calculation,
        test_total_interest,
        test_repayment_capacity,
        test_max_loan_from_emi,
        test_score_tiers,
        test_transaction_loan_recommendations_excellent,
        test_transaction_loan_recommendations_poor,
        test_transaction_loan_recommendations_very_poor,
        test_persona_farmer_loans,
        test_persona_student_loans,
        test_persona_vendor_loans,
        test_persona_homemaker_loans,
        test_persona_general_loans,
        test_loan_comparison,
        test_financial_tips,
        test_seasonal_recommendations,
        test_repayment_schedule,
        test_end_to_end_transaction_flow,
        test_end_to_end_persona_flow,
        test_all_persona_catalogs_exist,
        test_interest_saved_via_subsidy,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: FAIL — {e}")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"✅ ALL {passed} TESTS PASSED!")
    else:
        print(f"❌ {failed} FAILED, {passed} passed")
    print("=" * 60)
