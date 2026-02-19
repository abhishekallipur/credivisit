"""
Test Alternative Profiles Module — All 5 Persona Scoring Paths
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.alternative_profiles import (
    PERSONAS, compute_persona_score, get_persona_form_fields,
    get_improvement_tips
)

def test_farmer():
    data = {
        "owns_land": True, "land_acres": 3.5, "years_on_land": 8,
        "seasons_active": 6, "crops_per_year": 2, "yield_trend": "stable",
        "has_pm_kisan": True, "has_crop_insurance": True,
        "has_soil_health_card": False, "kcc_holder": True,
        "sells_at_mandi": True, "has_warehouse_receipt": False,
        "uses_enam": False, "avg_trips_per_month": 3,
        "references_count": 3, "is_group_member": True,
        "years_in_community": 10, "has_local_business_reference": True,
        "bills_per_year": 12, "on_time_pct": 85,
        "has_electricity": True, "has_water": True, "has_gas": False,
        "recharge_frequency": "monthly", "has_smartphone": True,
        "uses_upi_basic": False, "avg_monthly_recharge": 199,
    }
    result = compute_persona_score("farmer", data)
    print(f"\n🌾 FARMER: Score={result['trust_score']:.0f}, Grade={result['grade']}, "
          f"Confidence={result['confidence']:.0%}")
    print(f"   Breakdown:")
    for k, v in result["criteria_breakdown"].items():
        print(f"   {v['label']}: {v['score']:.2%} — {v['detail']}")
    assert 300 <= result["trust_score"] <= 900
    assert result["grade"] in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
    return result

def test_student():
    data = {
        "score_type": "cgpa", "score_value": 7.8, "education_level": "ug",
        "backlog_count": 1,
        "scholarships_received": 2, "total_scholarship_value": 25000,
        "merit_based": True,
        "cert_count": 3, "has_govt_certification": False,
        "platform_certs": ["Coursera", "NPTEL"],
        "attendance_pct": 82,
        "has_part_time": True, "monthly_earnings": 8000, "months_active": 4,
        "references_count": 2, "is_group_member": False,
        "years_in_community": 3, "has_local_business_reference": False,
        "recharge_frequency": "monthly", "has_smartphone": True,
        "uses_upi_basic": True, "avg_monthly_recharge": 399,
        "institution_tier": 2, "branch_demand": "high", "has_internship": True,
    }
    result = compute_persona_score("student", data)
    print(f"\n🎓 STUDENT: Score={result['trust_score']:.0f}, Grade={result['grade']}, "
          f"Confidence={result['confidence']:.0%}")
    print(f"   Breakdown:")
    for k, v in result["criteria_breakdown"].items():
        print(f"   {v['label']}: {v['score']:.2%} — {v['detail']}")
    assert 300 <= result["trust_score"] <= 900
    return result

def test_street_vendor():
    data = {
        "avg_daily_income": 800, "working_days_per_month": 26,
        "seasonal_variation": "low",
        "pays_rent": True, "rent_amount": 3000, "on_time_pct": 90,
        "months_of_history": 24,
        "bills_per_year": 12, "has_electricity": True,
        "has_water": True, "has_gas": True,
        "savings_method": "chit_fund", "monthly_savings": 2000,
        "months_saving": 18, "is_shg_member": False,
        "references_count": 4, "is_group_member": True,
        "years_in_community": 7, "has_local_business_reference": True,
        "recharge_frequency": "weekly", "has_smartphone": True,
        "uses_upi_basic": True, "avg_monthly_recharge": 249,
        "years_in_trade": 8, "same_location": True, "has_license": False,
    }
    result = compute_persona_score("street_vendor", data)
    print(f"\n🏪 STREET VENDOR: Score={result['trust_score']:.0f}, Grade={result['grade']}, "
          f"Confidence={result['confidence']:.0%}")
    print(f"   Breakdown:")
    for k, v in result["criteria_breakdown"].items():
        print(f"   {v['label']}: {v['score']:.2%} — {v['detail']}")
    assert 300 <= result["trust_score"] <= 900
    return result

def test_homemaker():
    data = {
        "household_income": 30000, "household_expenses": 22000,
        "manages_budget": True, "dependents": 4,
        "savings_method": "shg", "monthly_savings": 1500,
        "months_saving": 24, "is_shg_member": True,
        "references_count": 3, "is_group_member": True,
        "years_in_community": 12, "has_local_business_reference": False,
        "bills_per_year": 12, "on_time_pct": 90,
        "has_electricity": True, "has_water": True, "has_gas": True,
        "has_enterprise": True, "enterprise_type": "Tiffin Service",
        "monthly_revenue": 8000, "months_active": 18,
        "recharge_frequency": "monthly", "has_smartphone": True,
        "uses_upi_basic": True, "avg_monthly_recharge": 199,
        "cert_count": 1, "has_govt_certification": True, "platform_certs": [],
    }
    result = compute_persona_score("homemaker", data)
    print(f"\n🏠 HOMEMAKER: Score={result['trust_score']:.0f}, Grade={result['grade']}, "
          f"Confidence={result['confidence']:.0%}")
    print(f"   Breakdown:")
    for k, v in result["criteria_breakdown"].items():
        print(f"   {v['label']}: {v['score']:.2%} — {v['detail']}")
    assert 300 <= result["trust_score"] <= 900
    return result

def test_general_no_bank():
    data = {
        "recharge_frequency": "weekly", "has_smartphone": False,
        "uses_upi_basic": False, "avg_monthly_recharge": 149,
        "bills_per_year": 6, "on_time_pct": 70,
        "has_electricity": True, "has_water": False, "has_gas": False,
        "pays_rent": True, "rent_amount": 2000, "on_time_pct": 75,
        "months_of_history": 6,
        "references_count": 2, "is_group_member": False,
        "years_in_community": 3, "has_local_business_reference": False,
        "savings_method": "cash_at_home", "monthly_savings": 300,
        "months_saving": 4, "is_shg_member": False,
        "has_aadhaar": True, "has_pan": False,
        "has_voter_id": True, "has_ration_card": True,
        "q1_financial_planning": 3, "q2_risk_awareness": 2,
        "q3_goal_orientation": 3, "q4_repayment_intent": 4,
        "q5_responsibility": 4,
    }
    result = compute_persona_score("general_no_bank", data)
    print(f"\n👤 GENERAL (No Bank): Score={result['trust_score']:.0f}, Grade={result['grade']}, "
          f"Confidence={result['confidence']:.0%}")
    print(f"   Breakdown:")
    for k, v in result["criteria_breakdown"].items():
        print(f"   {v['label']}: {v['score']:.2%} — {v['detail']}")
    assert 300 <= result["trust_score"] <= 900
    return result

def test_improvement_tips():
    result = compute_persona_score("general_no_bank", {})
    tips = get_improvement_tips("general_no_bank", result)
    print(f"\n💡 IMPROVEMENT TIPS for General (minimal data):")
    for tip in tips:
        print(f"   [{tip['impact'].upper()}] {tip['action']} "
              f"(current: {tip['current_score']:.0%})")
    assert len(tips) > 0

def test_form_fields():
    for persona in PERSONAS:
        fields = get_persona_form_fields(persona)
        total = sum(len(g["fields"]) for g in fields)
        print(f"\n📝 {PERSONAS[persona]['label']}: {len(fields)} sections, {total} input fields")
    assert len(get_persona_form_fields("farmer")) > 0

if __name__ == "__main__":
    print("=" * 60)
    print("Alternative Profiles — Full Test Suite")
    print("=" * 60)

    test_farmer()
    test_student()
    test_street_vendor()
    test_homemaker()
    test_general_no_bank()
    test_improvement_tips()
    test_form_fields()

    print("\n" + "=" * 60)
    print("✅ ALL PERSONA TESTS PASSED!")
    print("=" * 60)
