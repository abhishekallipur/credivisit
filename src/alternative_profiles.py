"""
Alternative Profiles Module for CrediVist
==========================================
Enables credit scoring for individuals WITHOUT bank/UPI transaction history.
Supports persona-specific data collection & scoring for:
  1. Farmers
  2. Students
  3. Street Vendors / Informal Workers
  4. Homemakers
  5. General (No Bank Account)

Each persona uses unique alternative data signals relevant to their livelihood.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


# ─── Persona Definitions ────────────────────────────────────────────────────

PERSONAS = {
    "farmer": {
        "label": "🌾 Farmer",
        "description": "Earns through agriculture — crop sales, dairy, govt subsidies",
        "criteria_weights": {
            "land_asset": 0.20,
            "crop_consistency": 0.20,
            "subsidy_linkage": 0.15,
            "market_engagement": 0.15,
            "community_trust": 0.10,
            "utility_discipline": 0.10,
            "mobile_behaviour": 0.10,
        },
    },
    "student": {
        "label": "🎓 Student",
        "description": "Currently studying — potential-based scoring",
        "criteria_weights": {
            "academic_performance": 0.25,
            "scholarship_history": 0.15,
            "skill_certifications": 0.15,
            "attendance_discipline": 0.10,
            "part_time_income": 0.10,
            "community_trust": 0.10,
            "mobile_behaviour": 0.10,
            "future_potential": 0.05,
        },
    },
    "street_vendor": {
        "label": "🏪 Street Vendor / Informal Worker",
        "description": "Daily earnings through informal trade — no fixed salary",
        "criteria_weights": {
            "daily_income_consistency": 0.20,
            "rental_discipline": 0.15,
            "utility_discipline": 0.15,
            "savings_habit": 0.15,
            "community_trust": 0.15,
            "mobile_behaviour": 0.10,
            "years_in_trade": 0.10,
        },
    },
    "homemaker": {
        "label": "🏠 Homemaker",
        "description": "Manages household — tracks expenses, savings groups, micro-enterprise",
        "criteria_weights": {
            "household_budgeting": 0.20,
            "savings_habit": 0.20,
            "community_trust": 0.15,
            "utility_discipline": 0.15,
            "micro_enterprise": 0.10,
            "mobile_behaviour": 0.10,
            "skill_certifications": 0.10,
        },
    },
    "general_no_bank": {
        "label": "👤 General (No Bank Account)",
        "description": "No formal banking — scored on lifestyle signals",
        "criteria_weights": {
            "mobile_behaviour": 0.20,
            "utility_discipline": 0.20,
            "rental_discipline": 0.15,
            "community_trust": 0.15,
            "savings_habit": 0.15,
            "id_verification": 0.10,
            "psychometric": 0.05,
        },
    },
}


# ─── Individual Criteria Scorers (0–1) ──────────────────────────────────────

def score_land_asset(data: Dict) -> Dict:
    """
    Farmer: Land ownership / lease stability.
    Inputs: owns_land (bool), land_acres (float), years_on_land (int)
    """
    owns = data.get("owns_land", False)
    acres = float(data.get("land_acres", 0))
    years = int(data.get("years_on_land", 0))

    ownership_score = 0.6 if owns else 0.3
    area_score = min(acres / 5.0, 1.0)  # 5+ acres = max
    tenure_score = min(years / 10.0, 1.0)  # 10+ years = max

    score = ownership_score * 0.40 + area_score * 0.30 + tenure_score * 0.30
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Land / Asset Ownership",
        "detail": f"{'Owns' if owns else 'Leases'} {acres:.1f} acres, {years} yrs"
    }


def score_crop_consistency(data: Dict) -> Dict:
    """
    Farmer: How consistent are crop yields / sales across seasons.
    Inputs: seasons_active (int), crops_per_year (int), yield_trend ("up"/"stable"/"down")
    """
    seasons = min(int(data.get("seasons_active", 0)) / 6, 1.0)
    crops = min(int(data.get("crops_per_year", 0)) / 3, 1.0)
    trend = data.get("yield_trend", "stable")
    trend_map = {"up": 1.0, "stable": 0.7, "down": 0.3}
    trend_score = trend_map.get(trend, 0.5)

    score = seasons * 0.35 + crops * 0.30 + trend_score * 0.35
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Crop / Yield Consistency",
        "detail": f"{data.get('seasons_active', 0)} seasons, {data.get('crops_per_year', 0)} crops/yr, trend: {trend}"
    }


def score_subsidy_linkage(data: Dict) -> Dict:
    """
    Farmer: Linked to government schemes (PM-KISAN, crop insurance, soil card).
    Inputs: has_pm_kisan (bool), has_crop_insurance (bool), has_soil_health_card (bool),
            kcc_holder (bool)
    """
    schemes = [
        data.get("has_pm_kisan", False),
        data.get("has_crop_insurance", False),
        data.get("has_soil_health_card", False),
        data.get("kcc_holder", False),
    ]
    linked = sum(1 for s in schemes if s)
    score = min(linked / 3, 1.0)  # 3+ out of 4 = max
    active_names = []
    if data.get("has_pm_kisan"): active_names.append("PM-KISAN")
    if data.get("has_crop_insurance"): active_names.append("Crop Ins.")
    if data.get("has_soil_health_card"): active_names.append("Soil Card")
    if data.get("kcc_holder"): active_names.append("KCC")

    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Government Subsidy Linkage",
        "detail": f"{linked}/4 schemes linked: {', '.join(active_names) or 'None'}"
    }


def score_market_engagement(data: Dict) -> Dict:
    """
    Farmer: Sells at mandis, has warehouse receipts, uses e-NAM.
    Inputs: sells_at_mandi (bool), has_warehouse_receipt (bool),
            uses_enam (bool), avg_trips_per_month (int)
    """
    mandi = 0.3 if data.get("sells_at_mandi", False) else 0.0
    warehouse = 0.25 if data.get("has_warehouse_receipt", False) else 0.0
    enam = 0.2 if data.get("uses_enam", False) else 0.0
    trips = min(int(data.get("avg_trips_per_month", 0)) / 4, 1.0) * 0.25

    score = mandi + warehouse + enam + trips
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Market Engagement",
        "detail": f"Mandi: {'✓' if data.get('sells_at_mandi') else '✗'}, "
                  f"Warehouse: {'✓' if data.get('has_warehouse_receipt') else '✗'}, "
                  f"e-NAM: {'✓' if data.get('uses_enam') else '✗'}"
    }


def score_academic_performance(data: Dict) -> Dict:
    """
    Student: Academic grades / percentage / CGPA.
    Inputs: score_type ("percentage" or "cgpa"), score_value (float),
            education_level ("school"/"ug"/"pg"), backlog_count (int)
    """
    stype = data.get("score_type", "percentage")
    val = float(data.get("score_value", 0))

    if stype == "cgpa":
        normalized = min(val / 10.0, 1.0)
    else:
        normalized = min(val / 100.0, 1.0)

    # Penalty for backlogs
    backlogs = int(data.get("backlog_count", 0))
    backlog_penalty = min(backlogs * 0.1, 0.4)

    # Level bonus
    level = data.get("education_level", "school")
    level_bonus = {"school": 0.0, "ug": 0.05, "pg": 0.10}.get(level, 0.0)

    score = normalized - backlog_penalty + level_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Academic Performance",
        "detail": f"{val} {'CGPA' if stype == 'cgpa' else '%'} ({level.upper()}), {backlogs} backlogs"
    }


def score_scholarship_history(data: Dict) -> Dict:
    """
    Student: Scholarship count and value.
    Inputs: scholarships_received (int), total_scholarship_value (float),
            merit_based (bool)
    """
    count = int(data.get("scholarships_received", 0))
    value = float(data.get("total_scholarship_value", 0))
    merit = data.get("merit_based", False)

    count_score = min(count / 3, 1.0)
    value_score = min(value / 50000, 1.0)  # ₹50k+ = max
    merit_bonus = 0.15 if merit else 0.0

    score = count_score * 0.40 + value_score * 0.45 + merit_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Scholarship History",
        "detail": f"{count} scholarships, ₹{value:,.0f} total, Merit: {'✓' if merit else '✗'}"
    }


def score_skill_certifications(data: Dict) -> Dict:
    """
    Student / Homemaker: Vocational or online certifications.
    Inputs: cert_count (int), has_govt_certification (bool),
            platform_certs (list of str like ["NPTEL", "Coursera", "NSDC"])
    """
    count = int(data.get("cert_count", 0))
    govt = data.get("has_govt_certification", False)
    platforms = data.get("platform_certs", [])

    count_score = min(count / 5, 1.0)
    govt_bonus = 0.20 if govt else 0.0
    platform_score = min(len(platforms) / 3, 1.0) * 0.3

    score = count_score * 0.50 + platform_score + govt_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Skill Certifications",
        "detail": f"{count} certs, Govt: {'✓' if govt else '✗'}, Platforms: {', '.join(platforms) or 'None'}"
    }


def score_attendance_discipline(data: Dict) -> Dict:
    """
    Student: Attendance percentage.
    Inputs: attendance_pct (float 0-100)
    """
    pct = float(data.get("attendance_pct", 0))
    score = min(pct / 90, 1.0)  # 90%+ = max score
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Attendance Discipline",
        "detail": f"{pct:.0f}% attendance"
    }


def score_part_time_income(data: Dict) -> Dict:
    """
    Student: Earns from tutoring, freelancing, part-time jobs.
    Inputs: has_part_time (bool), monthly_earnings (float),
            months_active (int)
    """
    if not data.get("has_part_time", False):
        return {"score": 0.30, "label": "Part-time / Freelance Income",
                "detail": "No part-time income"}

    earnings = float(data.get("monthly_earnings", 0))
    months = int(data.get("months_active", 0))

    earn_score = min(earnings / 10000, 1.0)
    consistency = min(months / 6, 1.0)

    score = earn_score * 0.50 + consistency * 0.50
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Part-time / Freelance Income",
        "detail": f"₹{earnings:,.0f}/month for {months} months"
    }


def score_future_potential(data: Dict) -> Dict:
    """
    Student: Expected placement / career potential based on institution tier and branch.
    Inputs: institution_tier (1-4), branch_demand ("high"/"medium"/"low"),
            has_internship (bool)
    """
    tier = int(data.get("institution_tier", 4))
    tier_score = {1: 1.0, 2: 0.75, 3: 0.50, 4: 0.30}.get(tier, 0.30)

    demand = data.get("branch_demand", "medium")
    demand_score = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(demand, 0.5)

    internship_bonus = 0.15 if data.get("has_internship", False) else 0.0

    score = tier_score * 0.45 + demand_score * 0.40 + internship_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Future Earning Potential",
        "detail": f"Tier {tier}, Demand: {demand}, Internship: {'✓' if data.get('has_internship') else '✗'}"
    }


def score_daily_income_consistency(data: Dict) -> Dict:
    """
    Street Vendor: How consistent are daily earnings.
    Inputs: avg_daily_income (float), working_days_per_month (int),
            seasonal_variation ("low"/"medium"/"high")
    """
    daily = float(data.get("avg_daily_income", 0))
    days = int(data.get("working_days_per_month", 0))
    variation = data.get("seasonal_variation", "medium")

    income_score = min(daily * days / 15000, 1.0)  # ₹15k/month = max
    day_consistency = min(days / 26, 1.0)  # 26+ days = max
    var_map = {"low": 1.0, "medium": 0.6, "high": 0.3}
    stability = var_map.get(variation, 0.5)

    score = income_score * 0.35 + day_consistency * 0.35 + stability * 0.30
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Daily Income Consistency",
        "detail": f"₹{daily:,.0f}/day × {days} days, Seasonal var: {variation}"
    }


def score_rental_discipline(data: Dict) -> Dict:
    """
    Vendor / General: Regular rent or stall fee payments.
    Inputs: pays_rent (bool), rent_amount (float), on_time_pct (float 0-100),
            months_of_history (int)
    """
    if not data.get("pays_rent", False):
        return {"score": 0.40, "label": "Rental Payment Discipline",
                "detail": "No rent data available"}

    rent = float(data.get("rent_amount", 0))
    on_time = float(data.get("on_time_pct", 0)) / 100
    history = min(int(data.get("months_of_history", 0)) / 12, 1.0)

    affordability = min(rent / 5000, 1.0) * 0.15  # shows financial capacity
    discipline = on_time * 0.55
    track_record = history * 0.30

    score = affordability + discipline + track_record
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Rental Payment Discipline",
        "detail": f"₹{rent:,.0f}/month, {on_time*100:.0f}% on-time, {data.get('months_of_history', 0)} months"
    }


def score_utility_discipline(data: Dict) -> Dict:
    """
    Universal: Electricity, water, gas bill payment regularity.
    Inputs: bills_per_year (int), on_time_pct (float 0-100),
            has_electricity (bool), has_water (bool), has_gas (bool)
    """
    bills = min(int(data.get("bills_per_year", 0)) / 12, 1.0)
    on_time = float(data.get("on_time_pct", 80)) / 100
    services = sum([
        data.get("has_electricity", False),
        data.get("has_water", False),
        data.get("has_gas", False),
    ])
    service_score = min(services / 2, 1.0)

    score = bills * 0.30 + on_time * 0.45 + service_score * 0.25
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Utility Bill Discipline",
        "detail": f"{data.get('bills_per_year', 0)} bills/yr, {on_time*100:.0f}% on-time, {services}/3 services"
    }


def score_savings_habit(data: Dict) -> Dict:
    """
    Universal: Savings groups, chit funds, post office, cash savings.
    Inputs: savings_method (str), monthly_savings (float),
            months_saving (int), is_shg_member (bool)
    """
    method = data.get("savings_method", "none")
    monthly = float(data.get("monthly_savings", 0))
    months = int(data.get("months_saving", 0))
    shg = data.get("is_shg_member", False)

    method_scores = {
        "shg": 0.9, "chit_fund": 0.8, "post_office": 0.85,
        "cash_at_home": 0.5, "gold": 0.6, "bank": 0.9, "none": 0.1
    }
    method_score = method_scores.get(method, 0.3)
    amount_score = min(monthly / 3000, 1.0)  # ₹3k/month savings = max
    consistency = min(months / 12, 1.0)
    shg_bonus = 0.10 if shg else 0.0

    score = method_score * 0.30 + amount_score * 0.30 + consistency * 0.30 + shg_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Savings Discipline",
        "detail": f"₹{monthly:,.0f}/month via {method}, {months} months, SHG: {'✓' if shg else '✗'}"
    }


def score_community_trust(data: Dict) -> Dict:
    """
    Universal: Social references, group membership, local reputation.
    Inputs: references_count (int), is_group_member (bool),
            group_type (str), years_in_community (int),
            has_local_business_reference (bool)
    """
    refs = min(int(data.get("references_count", 0)) / 3, 1.0)
    group = data.get("is_group_member", False)
    years = min(int(data.get("years_in_community", 0)) / 5, 1.0)
    biz_ref = data.get("has_local_business_reference", False)

    group_score = 0.25 if group else 0.0
    biz_bonus = 0.10 if biz_ref else 0.0

    score = refs * 0.35 + group_score + years * 0.30 + biz_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Community Trust Network",
        "detail": f"{data.get('references_count', 0)} references, Group: {'✓' if group else '✗'}, "
                  f"{data.get('years_in_community', 0)} yrs in community"
    }


def score_mobile_behaviour(data: Dict) -> Dict:
    """
    Universal: Mobile recharge regularity, smartphone usage, app engagement.
    Inputs: recharge_frequency ("daily"/"weekly"/"monthly"/"irregular"),
            has_smartphone (bool), uses_upi_basic (bool),
            avg_monthly_recharge (float)
    """
    freq_map = {"daily": 0.5, "weekly": 0.7, "monthly": 0.9, "irregular": 0.3}
    freq = data.get("recharge_frequency", "irregular")
    freq_score = freq_map.get(freq, 0.3)

    smartphone = 0.20 if data.get("has_smartphone", False) else 0.0
    upi = 0.15 if data.get("uses_upi_basic", False) else 0.0
    recharge = min(float(data.get("avg_monthly_recharge", 0)) / 500, 1.0) * 0.25

    score = freq_score * 0.40 + smartphone + upi + recharge
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Mobile Behaviour & Digital Footprint",
        "detail": f"Recharge: {freq}, Smartphone: {'✓' if data.get('has_smartphone') else '✗'}, "
                  f"UPI: {'✓' if data.get('uses_upi_basic') else '✗'}"
    }


def score_years_in_trade(data: Dict) -> Dict:
    """
    Vendor: How long the person has been doing this work.
    Inputs: years_in_trade (int), same_location (bool), has_license (bool)
    """
    years = int(data.get("years_in_trade", 0))
    same_loc = data.get("same_location", False)
    license_ = data.get("has_license", False)

    year_score = min(years / 10, 1.0)
    loc_bonus = 0.15 if same_loc else 0.0
    lic_bonus = 0.10 if license_ else 0.0

    score = year_score * 0.75 + loc_bonus + lic_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Years in Trade",
        "detail": f"{years} years, Same location: {'✓' if same_loc else '✗'}, License: {'✓' if license_ else '✗'}"
    }


def score_household_budgeting(data: Dict) -> Dict:
    """
    Homemaker: Ability to manage household expenses within budget.
    Inputs: household_income (float), household_expenses (float),
            manages_budget (bool), dependents (int)
    """
    income = float(data.get("household_income", 0))
    expenses = float(data.get("household_expenses", 0))
    manages = data.get("manages_budget", False)
    dependents = int(data.get("dependents", 0))

    if income > 0:
        ratio = (income - expenses) / income
        ratio_score = np.clip(ratio, 0, 1)
    else:
        ratio_score = 0.2

    manage_bonus = 0.20 if manages else 0.0
    # Efficiency: managing more dependents well shows capability
    dep_efficiency = min(dependents / 5, 1.0) * 0.15 if manages else 0.0

    score = ratio_score * 0.65 + manage_bonus + dep_efficiency
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Household Budget Management",
        "detail": f"Income: ₹{income:,.0f}, Expenses: ₹{expenses:,.0f}, "
                  f"Manages: {'✓' if manages else '✗'}, {dependents} dependents"
    }


def score_micro_enterprise(data: Dict) -> Dict:
    """
    Homemaker: Running small business — tiffin, tailoring, pickles, etc.
    Inputs: has_enterprise (bool), enterprise_type (str),
            monthly_revenue (float), months_active (int)
    """
    if not data.get("has_enterprise", False):
        return {"score": 0.25, "label": "Micro Enterprise",
                "detail": "No micro-enterprise"}

    revenue = float(data.get("monthly_revenue", 0))
    months = int(data.get("months_active", 0))

    rev_score = min(revenue / 10000, 1.0)
    consistency = min(months / 12, 1.0)

    score = rev_score * 0.50 + consistency * 0.40 + 0.10  # 0.10 base for having enterprise
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Micro Enterprise",
        "detail": f"{data.get('enterprise_type', 'N/A')}, ₹{revenue:,.0f}/month, {months} months"
    }


def score_id_verification(data: Dict) -> Dict:
    """
    General: Government IDs verified.
    Inputs: has_aadhaar (bool), has_pan (bool), has_voter_id (bool),
            has_ration_card (bool)
    """
    ids = {
        "Aadhaar": data.get("has_aadhaar", False),
        "PAN": data.get("has_pan", False),
        "Voter ID": data.get("has_voter_id", False),
        "Ration Card": data.get("has_ration_card", False),
    }
    verified = [k for k, v in ids.items() if v]
    count = len(verified)
    # Aadhaar is most important
    aadhaar_bonus = 0.20 if data.get("has_aadhaar") else 0.0
    base = min(count / 3, 1.0) * 0.80

    score = base + aadhaar_bonus
    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Identity Verification",
        "detail": f"{count}/4 IDs: {', '.join(verified) or 'None'}"
    }


def score_psychometric(data: Dict) -> Dict:
    """
    General: Simple psychometric assessment (5 questions, scored 1-5 each).
    Inputs: q1_financial_planning (int 1-5), q2_risk_awareness (int 1-5),
            q3_goal_orientation (int 1-5), q4_repayment_intent (int 1-5),
            q5_responsibility (int 1-5)
    """
    questions = [
        int(data.get("q1_financial_planning", 3)),
        int(data.get("q2_risk_awareness", 3)),
        int(data.get("q3_goal_orientation", 3)),
        int(data.get("q4_repayment_intent", 3)),
        int(data.get("q5_responsibility", 3)),
    ]
    avg = np.mean(questions)
    score = (avg - 1) / 4  # Map 1-5 to 0-1

    return {
        "score": round(float(np.clip(score, 0, 1)), 4),
        "label": "Psychometric Assessment",
        "detail": f"Avg response: {avg:.1f}/5"
    }


# ─── Criteria Registry (maps criteria name → scorer function) ───────────────

CRITERIA_REGISTRY = {
    "land_asset": score_land_asset,
    "crop_consistency": score_crop_consistency,
    "subsidy_linkage": score_subsidy_linkage,
    "market_engagement": score_market_engagement,
    "academic_performance": score_academic_performance,
    "scholarship_history": score_scholarship_history,
    "skill_certifications": score_skill_certifications,
    "attendance_discipline": score_attendance_discipline,
    "part_time_income": score_part_time_income,
    "future_potential": score_future_potential,
    "daily_income_consistency": score_daily_income_consistency,
    "rental_discipline": score_rental_discipline,
    "utility_discipline": score_utility_discipline,
    "savings_habit": score_savings_habit,
    "community_trust": score_community_trust,
    "mobile_behaviour": score_mobile_behaviour,
    "years_in_trade": score_years_in_trade,
    "household_budgeting": score_household_budgeting,
    "micro_enterprise": score_micro_enterprise,
    "id_verification": score_id_verification,
    "psychometric": score_psychometric,
}


# ─── Main Scoring Functions ─────────────────────────────────────────────────

def compute_persona_score(persona: str, data: Dict) -> Dict:
    """
    Compute the alternative credit score for a given persona.

    Args:
        persona: One of the keys in PERSONAS dict
        data: Dict of all input fields collected from the user

    Returns:
        Dict with final score (300-900), grade, criteria breakdown, confidence
    """
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}. Must be one of {list(PERSONAS.keys())}")

    config = PERSONAS[persona]
    weights = config["criteria_weights"]

    criteria_results = {}
    weighted_total = 0.0
    total_weight = 0.0
    filled_count = 0

    for criterion, weight in weights.items():
        scorer = CRITERIA_REGISTRY.get(criterion)
        if scorer is None:
            continue

        result = scorer(data)
        criteria_results[criterion] = result
        weighted_total += result["score"] * weight
        total_weight += weight

        # Track how many criteria have meaningful data
        if result["score"] > 0.20:
            filled_count += 1

    # Normalize if weights don't sum to 1
    if total_weight > 0:
        base_100 = (weighted_total / total_weight) * 100
    else:
        base_100 = 30

    # Map to 300-900
    MIN_SCORE, MAX_SCORE = 300, 900
    trust_score = MIN_SCORE + (base_100 / 100) * (MAX_SCORE - MIN_SCORE)
    trust_score = np.clip(trust_score, MIN_SCORE, MAX_SCORE)

    # Confidence based on data completeness
    total_criteria = len(weights)
    confidence = round(float(np.clip(filled_count / total_criteria, 0.30, 0.95)), 2)

    # Grade
    if trust_score >= 750:
        grade, color = "Excellent", "#22c55e"
    elif trust_score >= 650:
        grade, color = "Good", "#84cc16"
    elif trust_score >= 500:
        grade, color = "Fair", "#eab308"
    elif trust_score >= 400:
        grade, color = "Poor", "#f97316"
    else:
        grade, color = "Very Poor", "#ef4444"

    return {
        "persona": persona,
        "persona_label": config["label"],
        "base_score_100": round(float(base_100), 2),
        "trust_score": round(float(trust_score), 0),
        "grade": grade,
        "grade_color": color,
        "confidence": confidence,
        "criteria_breakdown": criteria_results,
        "criteria_count": total_criteria,
        "filled_count": filled_count,
    }


def get_persona_form_fields(persona: str) -> list:
    """
    Return the list of form fields required for a given persona,
    along with field metadata (type, label, options, etc.).
    """
    field_definitions = {
        # ── Farmer Fields ──
        "land_asset": [
            {"key": "owns_land", "label": "Do you own agricultural land?", "type": "boolean"},
            {"key": "land_acres", "label": "Land area (acres)", "type": "number", "min": 0.0, "max": 100.0, "default": 1.0},
            {"key": "years_on_land", "label": "Years on this land", "type": "number", "min": 0, "max": 50, "default": 5},
        ],
        "crop_consistency": [
            {"key": "seasons_active", "label": "Seasons actively farmed", "type": "number", "min": 0, "max": 30, "default": 4},
            {"key": "crops_per_year", "label": "Number of crop cycles per year", "type": "number", "min": 0, "max": 4, "default": 2},
            {"key": "yield_trend", "label": "Recent yield trend", "type": "select", "options": ["up", "stable", "down"], "default": "stable"},
        ],
        "subsidy_linkage": [
            {"key": "has_pm_kisan", "label": "Enrolled in PM-KISAN?", "type": "boolean"},
            {"key": "has_crop_insurance", "label": "Have Crop Insurance (PMFBY)?", "type": "boolean"},
            {"key": "has_soil_health_card", "label": "Have Soil Health Card?", "type": "boolean"},
            {"key": "kcc_holder", "label": "Kisan Credit Card (KCC) holder?", "type": "boolean"},
        ],
        "market_engagement": [
            {"key": "sells_at_mandi", "label": "Sell produce at Mandi?", "type": "boolean"},
            {"key": "has_warehouse_receipt", "label": "Have warehouse receipts?", "type": "boolean"},
            {"key": "uses_enam", "label": "Use e-NAM (online mandi)?", "type": "boolean"},
            {"key": "avg_trips_per_month", "label": "Average mandi trips per month", "type": "number", "min": 0, "max": 30, "default": 2},
        ],
        # ── Student Fields ──
        "academic_performance": [
            {"key": "score_type", "label": "Score type", "type": "select", "options": ["percentage", "cgpa"], "default": "percentage"},
            {"key": "score_value", "label": "Your score / CGPA", "type": "number", "min": 0, "max": 100, "default": 70.0},
            {"key": "education_level", "label": "Education level", "type": "select", "options": ["school", "ug", "pg"], "default": "ug"},
            {"key": "backlog_count", "label": "Number of backlogs", "type": "number", "min": 0, "max": 20, "default": 0},
        ],
        "scholarship_history": [
            {"key": "scholarships_received", "label": "Scholarships received", "type": "number", "min": 0, "max": 20, "default": 0},
            {"key": "total_scholarship_value", "label": "Total scholarship value (₹)", "type": "number", "min": 0, "max": 1000000, "default": 0},
            {"key": "merit_based", "label": "Merit-based scholarship?", "type": "boolean"},
        ],
        "skill_certifications": [
            {"key": "cert_count", "label": "Number of certifications", "type": "number", "min": 0, "max": 50, "default": 0},
            {"key": "has_govt_certification", "label": "Government certification (NSDC, etc.)?", "type": "boolean"},
            {"key": "platform_certs", "label": "Certification platforms (comma-separated)", "type": "text", "default": ""},
        ],
        "attendance_discipline": [
            {"key": "attendance_pct", "label": "Attendance percentage", "type": "number", "min": 0, "max": 100, "default": 75},
        ],
        "part_time_income": [
            {"key": "has_part_time", "label": "Have part-time / freelance work?", "type": "boolean"},
            {"key": "monthly_earnings", "label": "Monthly earnings (₹)", "type": "number", "min": 0, "max": 100000, "default": 0},
            {"key": "months_active", "label": "Months active", "type": "number", "min": 0, "max": 60, "default": 0},
        ],
        "future_potential": [
            {"key": "institution_tier", "label": "Institution tier", "type": "select", "options": ["1", "2", "3", "4"], "default": "3"},
            {"key": "branch_demand", "label": "Branch/course demand", "type": "select", "options": ["high", "medium", "low"], "default": "medium"},
            {"key": "has_internship", "label": "Completed internship?", "type": "boolean"},
        ],
        # ── Street Vendor / Informal Worker Fields ──
        "daily_income_consistency": [
            {"key": "avg_daily_income", "label": "Average daily income (₹)", "type": "number", "min": 0, "max": 50000, "default": 500},
            {"key": "working_days_per_month", "label": "Working days per month", "type": "number", "min": 0, "max": 31, "default": 25},
            {"key": "seasonal_variation", "label": "Seasonal income variation", "type": "select", "options": ["low", "medium", "high"], "default": "medium"},
        ],
        "rental_discipline": [
            {"key": "pays_rent", "label": "Do you pay rent / stall fee?", "type": "boolean"},
            {"key": "rent_amount", "label": "Monthly rent (₹)", "type": "number", "min": 0, "max": 50000, "default": 2000},
            {"key": "on_time_pct", "label": "% of rent paid on time", "type": "number", "min": 0, "max": 100, "default": 80},
            {"key": "months_of_history", "label": "Months of rental history", "type": "number", "min": 0, "max": 240, "default": 12},
        ],
        # ── Universal Fields ──
        "utility_discipline": [
            {"key": "bills_per_year", "label": "Utility bills paid per year", "type": "number", "min": 0, "max": 36, "default": 12},
            {"key": "on_time_pct", "label": "% paid on time", "type": "number", "min": 0, "max": 100, "default": 80},
            {"key": "has_electricity", "label": "Electricity connection?", "type": "boolean", "default": True},
            {"key": "has_water", "label": "Water bill?", "type": "boolean"},
            {"key": "has_gas", "label": "Gas connection?", "type": "boolean"},
        ],
        "savings_habit": [
            {"key": "savings_method", "label": "Primary savings method", "type": "select",
             "options": ["shg", "chit_fund", "post_office", "cash_at_home", "gold", "bank", "none"],
             "default": "cash_at_home"},
            {"key": "monthly_savings", "label": "Monthly savings (₹)", "type": "number", "min": 0, "max": 100000, "default": 500},
            {"key": "months_saving", "label": "Months saving consistently", "type": "number", "min": 0, "max": 120, "default": 6},
            {"key": "is_shg_member", "label": "Self Help Group (SHG) member?", "type": "boolean"},
        ],
        "community_trust": [
            {"key": "references_count", "label": "Number of character references", "type": "number", "min": 0, "max": 10, "default": 2},
            {"key": "is_group_member", "label": "Member of community group?", "type": "boolean"},
            {"key": "group_type", "label": "Group type", "type": "text", "default": ""},
            {"key": "years_in_community", "label": "Years in current community", "type": "number", "min": 0, "max": 50, "default": 5},
            {"key": "has_local_business_reference", "label": "Have local business reference?", "type": "boolean"},
        ],
        "mobile_behaviour": [
            {"key": "recharge_frequency", "label": "Mobile recharge frequency", "type": "select",
             "options": ["daily", "weekly", "monthly", "irregular"], "default": "monthly"},
            {"key": "has_smartphone", "label": "Have smartphone?", "type": "boolean"},
            {"key": "uses_upi_basic", "label": "Use any UPI / digital payment?", "type": "boolean"},
            {"key": "avg_monthly_recharge", "label": "Monthly recharge amount (₹)", "type": "number", "min": 0, "max": 5000, "default": 299},
        ],
        "years_in_trade": [
            {"key": "years_in_trade", "label": "Years in current trade/work", "type": "number", "min": 0, "max": 50, "default": 5},
            {"key": "same_location", "label": "Same location throughout?", "type": "boolean"},
            {"key": "has_license", "label": "Have trade/vendor license?", "type": "boolean"},
        ],
        # ── Homemaker Fields ──
        "household_budgeting": [
            {"key": "household_income", "label": "Total household income (₹/month)", "type": "number", "min": 0, "max": 500000, "default": 20000},
            {"key": "household_expenses", "label": "Total household expenses (₹/month)", "type": "number", "min": 0, "max": 500000, "default": 15000},
            {"key": "manages_budget", "label": "Do you manage household budget?", "type": "boolean", "default": True},
            {"key": "dependents", "label": "Number of dependents", "type": "number", "min": 0, "max": 15, "default": 3},
        ],
        "micro_enterprise": [
            {"key": "has_enterprise", "label": "Run any home-based business?", "type": "boolean"},
            {"key": "enterprise_type", "label": "Business type (tiffin, tailoring, etc.)", "type": "text", "default": ""},
            {"key": "monthly_revenue", "label": "Monthly revenue (₹)", "type": "number", "min": 0, "max": 500000, "default": 0},
            {"key": "months_active", "label": "Months business active", "type": "number", "min": 0, "max": 120, "default": 0},
        ],
        # ── General No-Bank Fields ──
        "id_verification": [
            {"key": "has_aadhaar", "label": "Have Aadhaar card?", "type": "boolean", "default": True},
            {"key": "has_pan", "label": "Have PAN card?", "type": "boolean"},
            {"key": "has_voter_id", "label": "Have Voter ID?", "type": "boolean"},
            {"key": "has_ration_card", "label": "Have Ration Card?", "type": "boolean"},
        ],
        "psychometric": [
            {"key": "q1_financial_planning", "label": "I plan my expenses before spending (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
            {"key": "q2_risk_awareness", "label": "I understand borrowing has costs (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
            {"key": "q3_goal_orientation", "label": "I save for future goals (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
            {"key": "q4_repayment_intent", "label": "I always repay what I owe (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
            {"key": "q5_responsibility", "label": "I feel responsible for my family's finances (1-5)", "type": "number", "min": 1, "max": 5, "default": 3},
        ],
    }

    config = PERSONAS.get(persona, {})
    fields = []
    for criterion in config.get("criteria_weights", {}):
        criterion_fields = field_definitions.get(criterion, [])
        fields.append({
            "criterion": criterion,
            "label": CRITERIA_REGISTRY[criterion]({}).__class__.__name__ if criterion in CRITERIA_REGISTRY else criterion,
            "fields": criterion_fields
        })

    return fields


def get_improvement_tips(persona: str, result: Dict) -> list:
    """
    Generate persona-specific improvement suggestions based on weak criteria.
    """
    tips = []
    breakdown = result.get("criteria_breakdown", {})

    for criterion, info in breakdown.items():
        if info["score"] < 0.50:
            tip = _get_tip(persona, criterion, info["score"])
            if tip:
                tips.append(tip)

    # Sort by impact (lowest scores first)
    tips.sort(key=lambda t: t["current_score"])
    return tips[:5]  # Top 5 suggestions


def _get_tip(persona: str, criterion: str, score: float) -> Optional[Dict]:
    """Return a tip for improving a specific criterion."""
    tip_map = {
        "land_asset": {
            "action": "Secure formal land records or lease agreements",
            "impact": "high",
            "description": "Documented land tenure significantly boosts your profile. Register lease if renting."
        },
        "crop_consistency": {
            "action": "Diversify crops and maintain yield records",
            "impact": "high",
            "description": "Growing 2-3 crops per year with documented yield history improves consistency score."
        },
        "subsidy_linkage": {
            "action": "Enroll in PM-KISAN and get Kisan Credit Card",
            "impact": "medium",
            "description": "Government scheme linkage signals stability. Apply at nearest CSC or bank."
        },
        "market_engagement": {
            "action": "Register on e-NAM and use mandi receipts",
            "impact": "medium",
            "description": "Formal market participation with receipts creates a documentable trading history."
        },
        "academic_performance": {
            "action": "Clear backlogs and improve current semester grades",
            "impact": "high",
            "description": "Each backlog cleared adds +10% to your academic score."
        },
        "scholarship_history": {
            "action": "Apply for merit and need-based scholarships",
            "impact": "medium",
            "description": "Even small scholarships signal merit. Check NSP (National Scholarship Portal)."
        },
        "skill_certifications": {
            "action": "Complete free NPTEL/Coursera/NSDC certifications",
            "impact": "medium",
            "description": "Government-recognized certifications (NSDC, PMKVY) carry extra weight."
        },
        "attendance_discipline": {
            "action": "Maintain 90%+ attendance",
            "impact": "low",
            "description": "Regular attendance signals discipline. Aim for 90%+ for full score."
        },
        "part_time_income": {
            "action": "Start tutoring, freelancing, or campus work",
            "impact": "medium",
            "description": "Even ₹5,000/month earned consistently for 6 months significantly boosts this score."
        },
        "daily_income_consistency": {
            "action": "Work more days per month and reduce seasonal gaps",
            "impact": "high",
            "description": "26+ working days/month with low seasonal variation scores highest."
        },
        "rental_discipline": {
            "action": "Pay rent on time and keep receipts",
            "impact": "medium",
            "description": "On-time rent payments with 12+ months history strongly boost your score."
        },
        "utility_discipline": {
            "action": "Pay all utility bills on time",
            "impact": "medium",
            "description": "Consistent bill payments across electricity, water, gas show financial discipline."
        },
        "savings_habit": {
            "action": "Join an SHG or start systematic savings",
            "impact": "high",
            "description": "Even ₹500/month saved consistently in SHG or post office scores well."
        },
        "community_trust": {
            "action": "Join a community group and get references",
            "impact": "medium",
            "description": "3+ character references and group membership signal social trustworthiness."
        },
        "mobile_behaviour": {
            "action": "Use monthly recharge plans and try UPI payments",
            "impact": "low",
            "description": "Regular monthly recharges on a smartphone with basic UPI usage improves this score."
        },
        "years_in_trade": {
            "action": "Stay at same location and get a vendor license",
            "impact": "medium",
            "description": "Stability in location and formal licensing show commitment to the trade."
        },
        "household_budgeting": {
            "action": "Track income vs expenses and reduce deficit",
            "impact": "high",
            "description": "Keeping household expenses below 60% of income scores highest."
        },
        "micro_enterprise": {
            "action": "Start a small home business or formalize existing one",
            "impact": "medium",
            "description": "Even a small tiffin or tailoring service active for 12+ months adds to your profile."
        },
        "id_verification": {
            "action": "Get Aadhaar and at least one more government ID",
            "impact": "high",
            "description": "Aadhaar linkage is essential. PAN and Voter ID further strengthen your identity score."
        },
        "psychometric": {
            "action": "Improve financial awareness and planning",
            "impact": "low",
            "description": "Understanding borrowing costs and planning expenses wisely improves this assessment."
        },
    }

    tip_info = tip_map.get(criterion)
    if tip_info:
        return {
            "criterion": criterion,
            "current_score": score,
            "action": tip_info["action"],
            "impact": tip_info["impact"],
            "description": tip_info["description"],
        }
    return None
