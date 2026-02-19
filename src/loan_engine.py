"""
Loan Recommendation Engine for CrediVist
=========================================
Maps trust score + financial profile → eligible loan products with:
  - Loan types (persona-aware & transaction-aware)
  - Maximum loan amounts & count limits
  - Interest rates, tenure, EMI estimates
  - Eligibility checklists
  - Government subsidy info
  - Repayment capacity analysis
  - Credit improvement paths
  - Document checklists

Supports two pathways:
  1. Transaction-based (Upload & Score) — verified income, precise FOIR
  2. Alternative profile-based (Alternative Score) — persona-specific schemes
"""

import math
from typing import Dict, List, Any, Optional


# ─── Score-Based Tier System ─────────────────────────────────────────────────

SCORE_TIERS = {
    "excellent": {
        "range": (750, 900),
        "grade": "Excellent",
        "max_simultaneous_loans": 3,
        "max_exposure_multiplier": 6.0,   # × monthly income
        "base_interest_range": (9.0, 12.0),
        "max_tenure_months": 60,
        "foir_cap": 0.50,                  # max 50% of income as EMI
        "collateral_required": False,
        "processing_fee_pct": 1.0,
        "pre_approval": "Pre-Approved",
        "color": "#22c55e",
    },
    "good": {
        "range": (650, 749),
        "grade": "Good",
        "max_simultaneous_loans": 2,
        "max_exposure_multiplier": 4.0,
        "base_interest_range": (13.0, 16.0),
        "max_tenure_months": 36,
        "foir_cap": 0.40,
        "collateral_required": False,
        "processing_fee_pct": 1.5,
        "pre_approval": "Likely Approved",
        "color": "#84cc16",
    },
    "fair": {
        "range": (500, 649),
        "grade": "Fair",
        "max_simultaneous_loans": 1,
        "max_exposure_multiplier": 2.0,
        "base_interest_range": (18.0, 22.0),
        "max_tenure_months": 18,
        "foir_cap": 0.35,
        "collateral_required": False,
        "processing_fee_pct": 2.0,
        "pre_approval": "Needs Review",
        "color": "#eab308",
    },
    "poor": {
        "range": (400, 499),
        "grade": "Poor",
        "max_simultaneous_loans": 1,
        "max_exposure_multiplier": 1.0,
        "base_interest_range": (24.0, 30.0),
        "max_tenure_months": 12,
        "foir_cap": 0.30,
        "collateral_required": True,
        "processing_fee_pct": 2.5,
        "pre_approval": "High Risk",
        "color": "#f97316",
    },
    "very_poor": {
        "range": (300, 399),
        "grade": "Very Poor",
        "max_simultaneous_loans": 0,
        "max_exposure_multiplier": 0,
        "base_interest_range": (0, 0),
        "max_tenure_months": 0,
        "foir_cap": 0,
        "collateral_required": True,
        "processing_fee_pct": 0,
        "pre_approval": "Not Eligible",
        "color": "#ef4444",
    },
}


# ─── Loan Product Catalog ───────────────────────────────────────────────────

# Transaction-based loan products (for people WITH bank history)
TRANSACTION_LOANS = {
    "personal_loan": {
        "name": "Personal Loan",
        "icon": "💰",
        "category": "Personal",
        "min_score": 550,
        "min_income": 10000,
        "amount_range": (25000, 500000),
        "interest_range": (10.5, 24.0),
        "tenure_range": (6, 60),
        "collateral": False,
        "processing_fee": "1-2%",
        "description": "Unsecured personal loan for any purpose",
        "lenders": ["Banks", "NBFCs", "Fintech"],
        "documents": ["Aadhaar Card", "PAN Card", "Bank Statement (6 months)",
                       "Salary Slip / Income Proof"],
        "subsidy": None,
    },
    "business_loan": {
        "name": "Business Loan / Working Capital",
        "icon": "🏢",
        "category": "Business",
        "min_score": 600,
        "min_income": 15000,
        "amount_range": (50000, 1000000),
        "interest_range": (12.0, 22.0),
        "tenure_range": (12, 60),
        "collateral": False,
        "processing_fee": "1.5-2.5%",
        "description": "Working capital or business expansion loan",
        "lenders": ["Banks", "NBFCs", "SIDBI"],
        "documents": ["Business Registration", "Bank Statement (12 months)",
                       "GST Returns", "ITR (2 years)"],
        "subsidy": None,
    },
    "home_loan": {
        "name": "Home Loan",
        "icon": "🏠",
        "category": "Property",
        "min_score": 700,
        "min_income": 25000,
        "amount_range": (500000, 5000000),
        "interest_range": (8.5, 12.0),
        "tenure_range": (60, 360),
        "collateral": True,
        "processing_fee": "0.5-1%",
        "description": "Long-term secured loan for property purchase",
        "lenders": ["Banks", "HFCs"],
        "documents": ["Aadhaar Card", "PAN Card", "Bank Statement (12 months)",
                       "Salary Slip (6 months)", "Property Documents", "ITR (3 years)"],
        "subsidy": "PMAY — up to ₹2.67L interest subsidy for EWS/LIG",
    },
    "vehicle_loan": {
        "name": "Vehicle / Auto Loan",
        "icon": "🚗",
        "category": "Vehicle",
        "min_score": 600,
        "min_income": 20000,
        "amount_range": (100000, 1500000),
        "interest_range": (9.0, 15.0),
        "tenure_range": (12, 84),
        "collateral": True,
        "processing_fee": "1-2%",
        "description": "Secured loan for two-wheeler or four-wheeler purchase",
        "lenders": ["Banks", "NBFCs"],
        "documents": ["Aadhaar Card", "PAN Card", "Bank Statement (6 months)",
                       "Income Proof", "Vehicle Quotation"],
        "subsidy": None,
    },
    "education_loan": {
        "name": "Education Loan",
        "icon": "🎓",
        "category": "Education",
        "min_score": 500,
        "min_income": 0,
        "amount_range": (100000, 2000000),
        "interest_range": (8.0, 14.0),
        "tenure_range": (60, 180),
        "collateral": False,
        "processing_fee": "0-1%",
        "description": "Loan for higher education (Vidya Lakshmi portal eligible)",
        "lenders": ["Banks", "Vidya Lakshmi Portal"],
        "documents": ["Aadhaar Card", "Admission Letter", "Fee Structure",
                       "Academic Records", "Co-applicant Income Proof"],
        "subsidy": "Interest subsidy for EWS under CSIS scheme (family income < ₹4.5L)",
    },
    "gold_loan": {
        "name": "Gold Loan",
        "icon": "🪙",
        "category": "Secured",
        "min_score": 400,
        "min_income": 0,
        "amount_range": (10000, 2500000),
        "interest_range": (7.0, 15.0),
        "tenure_range": (3, 36),
        "collateral": True,
        "processing_fee": "0.5-1%",
        "description": "Quick secured loan against gold ornaments",
        "lenders": ["Banks", "Muthoot", "Manappuram", "IIFL"],
        "documents": ["Aadhaar Card", "Gold for Pledging"],
        "subsidy": None,
    },
    "credit_card": {
        "name": "Credit Card",
        "icon": "💳",
        "category": "Revolving",
        "min_score": 650,
        "min_income": 15000,
        "amount_range": (15000, 300000),
        "interest_range": (24.0, 42.0),
        "tenure_range": (0, 0),
        "collateral": False,
        "processing_fee": "₹0-₹500 annual fee",
        "description": "Revolving credit line for purchases",
        "lenders": ["Banks"],
        "documents": ["Aadhaar Card", "PAN Card", "Bank Statement (3 months)",
                       "Salary Slip"],
        "subsidy": None,
    },
    "loan_against_fd": {
        "name": "Loan Against FD / MF",
        "icon": "🏦",
        "category": "Secured",
        "min_score": 400,
        "min_income": 0,
        "amount_range": (10000, 1000000),
        "interest_range": (6.5, 10.0),
        "tenure_range": (1, 60),
        "collateral": True,
        "processing_fee": "0-0.5%",
        "description": "Loan against Fixed Deposit or Mutual Fund holdings",
        "lenders": ["Banks", "AMCs"],
        "documents": ["FD Receipt / MF Statement", "KYC Documents"],
        "subsidy": None,
    },
    "emergency_loan": {
        "name": "Emergency / Instant Loan",
        "icon": "🚨",
        "category": "Personal",
        "min_score": 450,
        "min_income": 8000,
        "amount_range": (5000, 100000),
        "interest_range": (15.0, 36.0),
        "tenure_range": (1, 12),
        "collateral": False,
        "processing_fee": "2-3%",
        "description": "Quick disbursal micro loan for emergencies",
        "lenders": ["Fintech Apps", "NBFCs"],
        "documents": ["Aadhaar Card", "PAN Card", "Bank Statement (3 months)"],
        "subsidy": None,
    },
}


# Persona-specific loan products (for people WITHOUT bank history)
PERSONA_LOANS = {
    "farmer": {
        "kcc": {
            "name": "Kisan Credit Card (KCC)",
            "icon": "🌾",
            "category": "Agriculture",
            "min_score": 450,
            "amount_range": (25000, 300000),
            "interest_range": (4.0, 7.0),
            "tenure_range": (12, 60),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Crop loan + working capital at subsidized rates. "
                           "3% interest subvention by GoI.",
            "lenders": ["Cooperative Banks", "Regional Rural Banks",
                        "Commercial Banks"],
            "documents": ["Aadhaar Card", "Land Records / Patta",
                          "Crop Sowing Certificate", "Passport Photo"],
            "subsidy": "4% interest subvention (effective rate ~4% for prompt repayment)",
            "eligibility_criteria": ["owns_land", "crops_per_year"],
        },
        "crop_loan": {
            "name": "Crop Loan",
            "icon": "🌱",
            "category": "Agriculture",
            "min_score": 400,
            "amount_range": (10000, 200000),
            "interest_range": (4.0, 9.0),
            "tenure_range": (6, 12),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Short-term loan for crop sowing, seeds, fertilizers.",
            "lenders": ["Cooperative Banks", "NABARD"],
            "documents": ["Aadhaar Card", "Land Records", "Sowing Certificate"],
            "subsidy": "Interest subvention under Modified Interest Subvention Scheme (MISS)",
            "eligibility_criteria": ["owns_land"],
        },
        "farm_equipment": {
            "name": "Farm Equipment Loan",
            "icon": "🚜",
            "category": "Agriculture",
            "min_score": 550,
            "amount_range": (50000, 1000000),
            "interest_range": (8.0, 12.0),
            "tenure_range": (24, 84),
            "collateral": True,
            "processing_fee": "1%",
            "description": "Loan for tractors, tillers, pumps, irrigation equipment.",
            "lenders": ["Banks", "NABARD", "Mahindra Finance"],
            "documents": ["Aadhaar Card", "Land Records", "Equipment Quotation",
                          "Income Proof"],
            "subsidy": "Subsidy under SMAM scheme (25-50% for small/marginal farmers)",
            "eligibility_criteria": ["owns_land", "land_acres"],
        },
        "dairy_poultry": {
            "name": "Dairy / Poultry / Fishery Loan",
            "icon": "🐄",
            "category": "Allied Agriculture",
            "min_score": 500,
            "amount_range": (25000, 500000),
            "interest_range": (7.0, 11.0),
            "tenure_range": (12, 60),
            "collateral": False,
            "processing_fee": "0.5-1%",
            "description": "Loan for dairy cattle, poultry farm, inland fisheries.",
            "lenders": ["NABARD", "Cooperative Banks", "AHDF"],
            "documents": ["Aadhaar Card", "Project Report", "Land/Shed Proof"],
            "subsidy": "25-33% subsidy under DEDS/PMMSY",
            "eligibility_criteria": [],
        },
        "solar_pump": {
            "name": "Solar Pump Loan (PM-KUSUM)",
            "icon": "☀️",
            "category": "Agriculture",
            "min_score": 450,
            "amount_range": (20000, 300000),
            "interest_range": (5.0, 8.0),
            "tenure_range": (12, 84),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Solar irrigation pump — 60% subsidy by Central+State Govt.",
            "lenders": ["NABARD", "State Energy Dept", "Banks"],
            "documents": ["Aadhaar Card", "Land Records", "Electricity Connection"],
            "subsidy": "60% capital subsidy (30% Central + 30% State)",
            "eligibility_criteria": ["owns_land"],
        },
        "warehouse_receipt": {
            "name": "Warehouse Receipt Loan",
            "icon": "🏭",
            "category": "Agriculture",
            "min_score": 500,
            "amount_range": (10000, 500000),
            "interest_range": (6.0, 9.0),
            "tenure_range": (3, 12),
            "collateral": True,
            "processing_fee": "0.5%",
            "description": "Loan against stored crop in registered warehouse.",
            "lenders": ["WDRA", "Banks", "NABARD"],
            "documents": ["Warehouse Receipt", "Aadhaar Card", "Land Records"],
            "subsidy": None,
            "eligibility_criteria": ["has_warehouse_receipt"],
        },
    },
    "student": {
        "education_loan": {
            "name": "Education Loan (Vidya Lakshmi)",
            "icon": "🎓",
            "category": "Education",
            "min_score": 450,
            "amount_range": (100000, 2000000),
            "interest_range": (8.0, 12.0),
            "tenure_range": (60, 180),
            "collateral": False,
            "processing_fee": "₹0-₹500",
            "description": "Central Sector Education Loan via Vidya Lakshmi Portal. "
                           "Moratorium during study period + 1 year.",
            "lenders": ["Banks (SBI, PNB, BOB)", "Vidya Lakshmi Portal"],
            "documents": ["Admission Letter", "Fee Structure", "Academic Records",
                          "Aadhaar Card", "Co-applicant Income Proof"],
            "subsidy": "Full interest subsidy during moratorium for family income < ₹4.5L/yr (CSIS)",
            "eligibility_criteria": ["score_value"],
        },
        "skill_loan": {
            "name": "Skill Development Loan",
            "icon": "🛠️",
            "category": "Education",
            "min_score": 400,
            "amount_range": (5000, 150000),
            "interest_range": (8.0, 12.0),
            "tenure_range": (3, 84),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Loan for vocational/skill training courses (NSDC approved).",
            "lenders": ["Banks", "NSDC"],
            "documents": ["Course Admission Letter", "Aadhaar Card",
                          "Training Center NSDC Affiliation"],
            "subsidy": "Interest subsidy for courses under PMKVY",
            "eligibility_criteria": [],
        },
        "device_loan": {
            "name": "Laptop / Device Loan",
            "icon": "💻",
            "category": "Personal",
            "min_score": 500,
            "amount_range": (10000, 100000),
            "interest_range": (12.0, 18.0),
            "tenure_range": (3, 24),
            "collateral": False,
            "processing_fee": "1-2%",
            "description": "EMI-based purchase of laptop, tablet, or smartphone.",
            "lenders": ["Bajaj Finserv", "HDB Financial", "ZestMoney"],
            "documents": ["Student ID", "Aadhaar Card", "Address Proof"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "startup_loan": {
            "name": "Startup India Seed Loan",
            "icon": "🚀",
            "category": "Business",
            "min_score": 600,
            "amount_range": (100000, 500000),
            "interest_range": (10.0, 14.0),
            "tenure_range": (12, 60),
            "collateral": False,
            "processing_fee": "1%",
            "description": "Seed funding loan for student/graduate entrepreneurs.",
            "lenders": ["SIDBI", "Startup India Fund"],
            "documents": ["Business Plan", "Aadhaar Card", "Degree Certificate",
                          "Incorporation Certificate"],
            "subsidy": "80% guarantee cover under CGTMSE",
            "eligibility_criteria": ["has_internship"],
        },
    },
    "street_vendor": {
        "pm_svanidhi": {
            "name": "PM SVANidhi (Street Vendor Loan)",
            "icon": "🏪",
            "category": "Micro Enterprise",
            "min_score": 350,
            "amount_range": (10000, 50000),
            "interest_range": (0.0, 7.0),
            "tenure_range": (12, 12),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Government micro-credit for street vendors. "
                           "₹10K (1st), ₹20K (2nd), ₹50K (3rd tranche). "
                           "7% interest subsidy + cashback for digital payments.",
            "lenders": ["Banks", "MFIs", "NBFCs", "SHGs"],
            "documents": ["Vendor Certificate / LoR from ULB/Municipality",
                          "Aadhaar Card", "Bank Account"],
            "subsidy": "7% interest subsidy + ₹1,200/yr digital payment cashback",
            "eligibility_criteria": ["has_license"],
        },
        "mudra_shishu": {
            "name": "Mudra Loan — Shishu (up to ₹50K)",
            "icon": "🌱",
            "category": "Micro Enterprise",
            "min_score": 400,
            "amount_range": (10000, 50000),
            "interest_range": (10.0, 14.0),
            "tenure_range": (12, 36),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Mudra Shishu loan for micro-business working capital.",
            "lenders": ["Banks", "MFIs", "NBFCs"],
            "documents": ["Aadhaar Card", "Business Proof / Address Proof",
                          "Passport Photo"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "mudra_kishore": {
            "name": "Mudra Loan — Kishore (₹50K–₹5L)",
            "icon": "📈",
            "category": "Micro Enterprise",
            "min_score": 550,
            "amount_range": (50000, 500000),
            "interest_range": (11.0, 16.0),
            "tenure_range": (12, 60),
            "collateral": False,
            "processing_fee": "0.5-1%",
            "description": "Mudra Kishore for business expansion / equipment purchase.",
            "lenders": ["Banks", "NBFCs", "SIDBI"],
            "documents": ["Aadhaar Card", "Business Plan / Invoice",
                          "Proof of existing business (2+ years)"],
            "subsidy": None,
            "eligibility_criteria": ["years_in_trade"],
        },
        "cart_equipment": {
            "name": "Cart / Equipment Loan",
            "icon": "🛒",
            "category": "Micro Enterprise",
            "min_score": 450,
            "amount_range": (5000, 100000),
            "interest_range": (12.0, 18.0),
            "tenure_range": (6, 36),
            "collateral": False,
            "processing_fee": "1%",
            "description": "Loan for carts, cooking equipment, tools, display units.",
            "lenders": ["MFIs", "Cooperative Banks"],
            "documents": ["Aadhaar Card", "Vendor License", "Equipment Quotation"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "working_capital_micro": {
            "name": "Working Capital Micro Loan",
            "icon": "💵",
            "category": "Micro Enterprise",
            "min_score": 400,
            "amount_range": (5000, 75000),
            "interest_range": (12.0, 24.0),
            "tenure_range": (3, 12),
            "collateral": False,
            "processing_fee": "1-2%",
            "description": "Short-term working capital for daily stock purchase.",
            "lenders": ["MFIs", "SHGs", "Fintech Apps"],
            "documents": ["Aadhaar Card", "Address Proof"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
    },
    "homemaker": {
        "shg_loan": {
            "name": "SHG Group Loan (NRLM)",
            "icon": "👩‍👩‍👧‍👦",
            "category": "Group Lending",
            "min_score": 350,
            "amount_range": (10000, 200000),
            "interest_range": (4.0, 12.0),
            "tenure_range": (12, 36),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Joint Liability Group / SHG loan under NRLM/DAY-NRLM. "
                           "3% interest subvention for women SHGs.",
            "lenders": ["NABARD", "Cooperative Banks", "SHG Federations"],
            "documents": ["SHG Registration", "Minutes of Meeting",
                          "Member Aadhaar Cards", "Group Savings Passbook"],
            "subsidy": "3% interest subvention for women SHGs (effective rate ~4%)",
            "eligibility_criteria": ["is_shg_member"],
        },
        "mudra_shishu_w": {
            "name": "Mudra Loan — Shishu (Women)",
            "icon": "🌸",
            "category": "Micro Enterprise",
            "min_score": 400,
            "amount_range": (10000, 50000),
            "interest_range": (10.0, 14.0),
            "tenure_range": (12, 36),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Mudra Shishu for women-led micro enterprises — "
                           "tiffin service, tailoring, beauty parlour, etc.",
            "lenders": ["Banks", "MFIs"],
            "documents": ["Aadhaar Card", "Address Proof",
                          "Business Plan (1-page)"],
            "subsidy": None,
            "eligibility_criteria": ["has_enterprise"],
        },
        "standup_india": {
            "name": "Stand-Up India Loan (Women)",
            "icon": "🏗️",
            "category": "Enterprise",
            "min_score": 550,
            "amount_range": (100000, 10000000),
            "interest_range": (8.0, 12.0),
            "tenure_range": (36, 84),
            "collateral": False,
            "processing_fee": "0.5%",
            "description": "Loan for women entrepreneurs — manufacturing, "
                           "services, or trading. At least 1 per bank branch.",
            "lenders": ["Scheduled Commercial Banks"],
            "documents": ["Aadhaar Card", "PAN Card", "Business Plan",
                          "Proof of SC/ST/Woman status", "Project Report"],
            "subsidy": "CGTMSE guarantee cover, margin money 25%",
            "eligibility_criteria": ["has_enterprise"],
        },
        "micro_enterprise_w": {
            "name": "Micro Enterprise Loan",
            "icon": "🏠",
            "category": "Micro Enterprise",
            "min_score": 500,
            "amount_range": (25000, 200000),
            "interest_range": (12.0, 18.0),
            "tenure_range": (12, 48),
            "collateral": False,
            "processing_fee": "1%",
            "description": "Individual micro enterprise loan for home-based business.",
            "lenders": ["MFIs", "NBFCs", "Cooperative Banks"],
            "documents": ["Aadhaar Card", "Business Proof", "Savings Passbook"],
            "subsidy": None,
            "eligibility_criteria": ["has_enterprise"],
        },
        "gold_loan_w": {
            "name": "Gold Loan",
            "icon": "🪙",
            "category": "Secured",
            "min_score": 350,
            "amount_range": (5000, 500000),
            "interest_range": (7.0, 12.0),
            "tenure_range": (3, 24),
            "collateral": True,
            "processing_fee": "0.5%",
            "description": "Quick secured loan against gold jewellery.",
            "lenders": ["Muthoot", "Manappuram", "Banks"],
            "documents": ["Aadhaar Card", "Gold for Pledging"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
    },
    "general_no_bank": {
        "mudra_shishu_g": {
            "name": "Mudra Loan — Shishu",
            "icon": "🌱",
            "category": "Micro Enterprise",
            "min_score": 400,
            "amount_range": (10000, 50000),
            "interest_range": (10.0, 14.0),
            "tenure_range": (12, 36),
            "collateral": False,
            "processing_fee": "₹0",
            "description": "Basic micro loan for starting a small business.",
            "lenders": ["Banks", "MFIs", "NBFCs"],
            "documents": ["Aadhaar Card", "Address Proof", "Passport Photo"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "personal_micro": {
            "name": "Personal Micro Loan",
            "icon": "💵",
            "category": "Personal",
            "min_score": 450,
            "amount_range": (5000, 50000),
            "interest_range": (15.0, 26.0),
            "tenure_range": (3, 12),
            "collateral": False,
            "processing_fee": "2%",
            "description": "Small unsecured personal loan from MFIs/Fintech.",
            "lenders": ["MFIs", "Fintech Apps", "SHGs"],
            "documents": ["Aadhaar Card", "Address Proof"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "emergency_micro": {
            "name": "Emergency Micro Loan",
            "icon": "🚨",
            "category": "Emergency",
            "min_score": 350,
            "amount_range": (1000, 25000),
            "interest_range": (18.0, 30.0),
            "tenure_range": (1, 6),
            "collateral": False,
            "processing_fee": "2-3%",
            "description": "Quick-disbursal emergency loan for immediate needs.",
            "lenders": ["MFIs", "Fintech Apps"],
            "documents": ["Aadhaar Card"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "gold_loan_g": {
            "name": "Gold Loan",
            "icon": "🪙",
            "category": "Secured",
            "min_score": 350,
            "amount_range": (5000, 500000),
            "interest_range": (7.0, 12.0),
            "tenure_range": (3, 24),
            "collateral": True,
            "processing_fee": "0.5%",
            "description": "Secured loan against gold. No income proof needed.",
            "lenders": ["Muthoot", "Manappuram", "Banks"],
            "documents": ["Aadhaar Card", "Gold for Pledging"],
            "subsidy": None,
            "eligibility_criteria": [],
        },
        "jlg_loan": {
            "name": "Joint Liability Group (JLG) Loan",
            "icon": "🤝",
            "category": "Group Lending",
            "min_score": 400,
            "amount_range": (10000, 100000),
            "interest_range": (12.0, 18.0),
            "tenure_range": (6, 24),
            "collateral": False,
            "processing_fee": "1%",
            "description": "Loan through a group of 5-10 members. "
                           "Group guarantee replaces collateral.",
            "lenders": ["NABARD", "MFIs", "Cooperative Banks"],
            "documents": ["JLG Formation Docs", "Member Aadhaar Cards",
                          "Group Agreement"],
            "subsidy": None,
            "eligibility_criteria": ["is_group_member"],
        },
    },
}


# ─── EMI Calculator ─────────────────────────────────────────────────────────

def calculate_emi(principal: float, annual_rate: float,
                  tenure_months: int) -> float:
    """
    Standard EMI formula: EMI = P × r × (1+r)^n / ((1+r)^n - 1)
    Returns monthly EMI amount.
    """
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / tenure_months

    r = annual_rate / (12 * 100)  # monthly rate
    n = tenure_months
    emi = principal * r * math.pow(1 + r, n) / (math.pow(1 + r, n) - 1)
    return round(emi, 2)


def calculate_total_interest(principal: float, annual_rate: float,
                             tenure_months: int) -> float:
    """Total interest payable over the loan tenure."""
    emi = calculate_emi(principal, annual_rate, tenure_months)
    total_paid = emi * tenure_months
    return round(total_paid - principal, 2)


# ─── Repayment Capacity Analysis ────────────────────────────────────────────

def analyze_repayment_capacity(monthly_income: float,
                               monthly_expenses: float = 0,
                               existing_emi: float = 0,
                               foir_cap: float = 0.40) -> Dict:
    """
    Compute repayment capacity using FOIR (Fixed Obligation to Income Ratio).
    
    Args:
        monthly_income: Verified or declared monthly income
        monthly_expenses: Known fixed monthly expenses (rent, utilities, etc.)
        existing_emi: Sum of all existing EMI obligations
        foir_cap: Max fraction of income allowed for total EMIs (score-based)
    
    Returns:
        Dict with max affordable EMI, FOIR, risk flags, etc.
    """
    if monthly_income <= 0:
        return {
            "monthly_income": 0,
            "monthly_expenses": monthly_expenses,
            "existing_emi": existing_emi,
            "disposable_income": 0,
            "max_total_emi": 0,
            "max_new_emi": 0,
            "current_foir": 0,
            "foir_limit": foir_cap,
            "foir_headroom": 0,
            "risk_flags": ["Zero or negative income detected"],
            "verdict": "NOT_ELIGIBLE",
        }

    disposable = monthly_income - monthly_expenses
    max_total_emi = monthly_income * foir_cap
    max_new_emi = max(max_total_emi - existing_emi, 0)
    current_foir = (existing_emi / monthly_income) if monthly_income > 0 else 0
    headroom = max(foir_cap - current_foir, 0)

    risk_flags = []
    if current_foir > 0.50:
        risk_flags.append("Over-leveraged: Existing EMIs exceed 50% of income")
    if current_foir > foir_cap:
        risk_flags.append(f"FOIR exceeded: {current_foir:.0%} > {foir_cap:.0%} limit")
    if disposable < monthly_income * 0.20:
        risk_flags.append("Low disposable income: Less than 20% of earnings remain")
    if existing_emi > 0 and max_new_emi < 1000:
        risk_flags.append("Very limited new EMI capacity")

    if max_new_emi <= 0:
        verdict = "NOT_ELIGIBLE"
    elif max_new_emi < 2000:
        verdict = "MICRO_ONLY"
    elif len(risk_flags) > 0:
        verdict = "ELIGIBLE_WITH_CAUTION"
    else:
        verdict = "ELIGIBLE"

    return {
        "monthly_income": round(monthly_income, 2),
        "monthly_expenses": round(monthly_expenses, 2),
        "existing_emi": round(existing_emi, 2),
        "disposable_income": round(disposable, 2),
        "max_total_emi": round(max_total_emi, 2),
        "max_new_emi": round(max_new_emi, 2),
        "current_foir": round(current_foir, 4),
        "foir_limit": foir_cap,
        "foir_headroom": round(headroom, 4),
        "risk_flags": risk_flags,
        "verdict": verdict,
    }


def max_loan_from_emi(max_emi: float, annual_rate: float,
                      tenure_months: int) -> float:
    """
    Reverse EMI calculation: given max affordable EMI, find max principal.
    P = EMI × ((1+r)^n - 1) / (r × (1+r)^n)
    """
    if max_emi <= 0 or tenure_months <= 0:
        return 0.0
    if annual_rate <= 0:
        return max_emi * tenure_months

    r = annual_rate / (12 * 100)
    n = tenure_months
    principal = max_emi * (math.pow(1 + r, n) - 1) / (r * math.pow(1 + r, n))
    return round(principal, 2)


# ─── Score Tier Resolver ────────────────────────────────────────────────────

def get_score_tier(score: float) -> Dict:
    """Return the tier config for a given trust score."""
    for tier_key, tier in SCORE_TIERS.items():
        low, high = tier["range"]
        if low <= score <= high:
            return {**tier, "tier_key": tier_key}
    # Default: very poor
    return {**SCORE_TIERS["very_poor"], "tier_key": "very_poor"}


# ─── Transaction-Based Recommendations ──────────────────────────────────────

def get_transaction_loan_recommendations(
    score: float,
    monthly_income: float,
    monthly_expenses: float = 0,
    existing_emi: float = 0,
    profile_data: Optional[Dict] = None,
) -> Dict:
    """
    Generate loan recommendations for users WITH transaction history.
    
    Args:
        score: Final trust score (300-900)
        monthly_income: Average monthly income (from transactions)
        monthly_expenses: Fixed monthly expenses (rent, utilities, etc.)
        existing_emi: Detected existing EMI payments
        profile_data: Additional profile data (savings, gig info, etc.)
    
    Returns:
        Complete loan recommendation package
    """
    tier = get_score_tier(score)
    repayment = analyze_repayment_capacity(
        monthly_income, monthly_expenses, existing_emi, tier["foir_cap"]
    )

    # Filter eligible loans
    eligible_loans = []
    ineligible_loans = []

    for loan_key, loan in TRANSACTION_LOANS.items():
        eligibility = _check_transaction_loan_eligibility(
            loan, score, monthly_income, tier, repayment
        )

        loan_detail = _build_loan_detail(
            loan_key, loan, score, monthly_income, tier, repayment, eligibility
        )

        if eligibility["eligible"]:
            eligible_loans.append(loan_detail)
        else:
            ineligible_loans.append(loan_detail)

    # Sort eligible by best fit (highest max amount first)
    eligible_loans.sort(key=lambda x: x["max_loan_amount"], reverse=True)

    # Limit simultaneous loans
    max_loans = tier["max_simultaneous_loans"]

    # Credit improvement path
    improvement = _get_credit_improvement_path(score, tier, repayment)

    return {
        "score": score,
        "tier": tier,
        "repayment_capacity": repayment,
        "eligible_loans": eligible_loans,
        "ineligible_loans": ineligible_loans,
        "max_simultaneous_loans": max_loans,
        "total_eligible": len(eligible_loans),
        "max_total_exposure": round(monthly_income * tier["max_exposure_multiplier"], 2),
        "pre_approval_status": tier["pre_approval"],
        "improvement_path": improvement,
        "source": "transaction",
    }


def _check_transaction_loan_eligibility(
    loan: Dict, score: float, income: float,
    tier: Dict, repayment: Dict
) -> Dict:
    """Check if user is eligible for a specific transaction-based loan."""
    reasons = []
    eligible = True

    if score < loan["min_score"]:
        eligible = False
        reasons.append(f"Score {score:.0f} below minimum {loan['min_score']}")

    if income < loan["min_income"]:
        eligible = False
        reasons.append(f"Income ₹{income:,.0f} below minimum ₹{loan['min_income']:,}")

    if repayment["verdict"] == "NOT_ELIGIBLE":
        eligible = False
        reasons.append("Repayment capacity insufficient")

    if tier["max_simultaneous_loans"] == 0:
        eligible = False
        reasons.append("Score too low for any loans")

    return {"eligible": eligible, "reasons": reasons}


def _build_loan_detail(
    loan_key: str, loan: Dict, score: float, income: float,
    tier: Dict, repayment: Dict, eligibility: Dict
) -> Dict:
    """Build detailed loan recommendation with EMI calculations."""
    # Determine effective interest rate based on score
    loan_rate_low, loan_rate_high = loan["interest_range"]
    tier_rate_low, tier_rate_high = tier["base_interest_range"]

    # Better score → closer to lower rate
    if score >= 750:
        effective_rate = loan_rate_low
    elif score >= 650:
        effective_rate = loan_rate_low + (loan_rate_high - loan_rate_low) * 0.3
    elif score >= 500:
        effective_rate = loan_rate_low + (loan_rate_high - loan_rate_low) * 0.6
    else:
        effective_rate = loan_rate_high

    # Determine max loan amount based on repayment capacity
    max_amount_by_income = income * tier["max_exposure_multiplier"]
    max_amount_by_product = loan["amount_range"][1]
    min_amount = loan["amount_range"][0]

    if repayment["max_new_emi"] > 0:
        # Use mid-tenure for FOIR-based calculation
        mid_tenure = (loan["tenure_range"][0] + loan["tenure_range"][1]) // 2
        max_amount_by_emi = max_loan_from_emi(
            repayment["max_new_emi"], effective_rate, mid_tenure
        )
    else:
        max_amount_by_emi = 0

    max_loan = min(max_amount_by_income, max_amount_by_product, max_amount_by_emi)
    max_loan = max(max_loan, 0)

    # Recommended loan (80% of max for safety margin)
    recommended = min(max_loan * 0.8, max_amount_by_product)
    recommended = max(recommended, 0)

    # EMI for recommended amount at suggested tenure
    suggested_tenure = min(
        loan["tenure_range"][1],
        tier["max_tenure_months"]
    )
    if suggested_tenure == 0 and loan["tenure_range"][1] > 0:
        suggested_tenure = loan["tenure_range"][0]

    emi = calculate_emi(recommended, effective_rate, suggested_tenure)
    total_interest = calculate_total_interest(
        recommended, effective_rate, suggested_tenure
    )

    # Interest saved via subsidy
    interest_saved = 0
    if loan.get("subsidy"):
        # Rough estimate: subsidy saves ~3-5% interest
        market_interest = calculate_total_interest(
            recommended, effective_rate + 4, suggested_tenure
        )
        interest_saved = max(market_interest - total_interest, 0)

    return {
        "key": loan_key,
        "name": loan["name"],
        "icon": loan["icon"],
        "category": loan["category"],
        "eligible": eligibility["eligible"],
        "reasons": eligibility.get("reasons", []),
        "effective_rate": round(effective_rate, 2),
        "max_loan_amount": round(max_loan, 0),
        "recommended_amount": round(recommended, 0),
        "min_amount": min_amount,
        "emi": emi,
        "suggested_tenure": suggested_tenure,
        "tenure_range": loan["tenure_range"],
        "total_interest": total_interest,
        "interest_saved_via_subsidy": round(interest_saved, 0),
        "collateral_required": loan.get("collateral", False),
        "processing_fee": loan["processing_fee"],
        "description": loan["description"],
        "lenders": loan["lenders"],
        "documents": loan["documents"],
        "subsidy": loan.get("subsidy"),
    }


# ─── Persona-Based Recommendations ──────────────────────────────────────────

def get_persona_loan_recommendations(
    persona: str,
    score: float,
    persona_data: Optional[Dict] = None,
    monthly_income: float = 0,
) -> Dict:
    """
    Generate loan recommendations for users WITHOUT transaction history.
    Uses persona-specific government schemes and microfinance products.
    
    Args:
        persona: One of farmer/student/street_vendor/homemaker/general_no_bank
        score: Alternative trust score (300-900)
        persona_data: Dict of persona-specific data fields
        monthly_income: Estimated monthly income (if known)
    
    Returns:
        Complete loan recommendation package
    """
    tier = get_score_tier(score)
    persona_data = persona_data or {}

    # Get persona-specific loan catalog
    persona_loan_catalog = PERSONA_LOANS.get(persona, PERSONA_LOANS["general_no_bank"])

    # Estimate income if not provided
    if monthly_income <= 0:
        monthly_income = _estimate_income_from_persona(persona, persona_data)

    repayment = analyze_repayment_capacity(
        monthly_income, 0, 0, tier["foir_cap"]
    )

    eligible_loans = []
    ineligible_loans = []

    for loan_key, loan in persona_loan_catalog.items():
        eligibility = _check_persona_loan_eligibility(
            loan, score, persona_data, tier
        )

        loan_detail = _build_persona_loan_detail(
            loan_key, loan, score, monthly_income, tier, repayment, eligibility
        )

        if eligibility["eligible"]:
            eligible_loans.append(loan_detail)
        else:
            ineligible_loans.append(loan_detail)

    # Sort eligible loans by amount (descending)
    eligible_loans.sort(key=lambda x: x["max_loan_amount"], reverse=True)

    # Credit improvement path
    improvement = _get_credit_improvement_path(score, tier, repayment)

    return {
        "score": score,
        "persona": persona,
        "tier": tier,
        "repayment_capacity": repayment,
        "eligible_loans": eligible_loans,
        "ineligible_loans": ineligible_loans,
        "max_simultaneous_loans": tier["max_simultaneous_loans"],
        "total_eligible": len(eligible_loans),
        "estimated_monthly_income": round(monthly_income, 0),
        "pre_approval_status": tier["pre_approval"],
        "improvement_path": improvement,
        "source": "alternative_profile",
    }


def _estimate_income_from_persona(persona: str, data: Dict) -> float:
    """Estimate monthly income from persona-specific data."""
    if persona == "farmer":
        acres = float(data.get("land_acres", 2))
        # Average yield: ₹15,000-25,000 per acre per season, 2-3 seasons
        seasons = int(data.get("crops_per_year", 2))
        return acres * 18000 * seasons / 12

    elif persona == "student":
        part_time = float(data.get("monthly_earnings", 0))
        scholarship = float(data.get("total_scholarship_value", 0)) / 12
        return max(part_time + scholarship, 5000)

    elif persona == "street_vendor":
        daily = float(data.get("avg_daily_income", 500))
        days = int(data.get("working_days_per_month", 25))
        return daily * days

    elif persona == "homemaker":
        hh_income = float(data.get("household_income", 15000))
        enterprise_rev = float(data.get("monthly_revenue", 0))
        return hh_income + enterprise_rev

    elif persona == "general_no_bank":
        rent = float(data.get("rent_amount", 2000))
        # If they pay rent, income is likely 3-4x rent
        return max(rent * 3.5, 8000)

    return 10000  # default fallback


def _check_persona_loan_eligibility(
    loan: Dict, score: float, data: Dict, tier: Dict
) -> Dict:
    """Check eligibility for persona-specific loan."""
    reasons = []
    eligible = True
    criteria_met = []
    criteria_not_met = []

    if score < loan["min_score"]:
        eligible = False
        reasons.append(f"Score {score:.0f} below minimum {loan['min_score']}")

    if tier["max_simultaneous_loans"] == 0:
        eligible = False
        reasons.append("Score too low for any loans")

    # Check persona-specific eligibility criteria
    for criterion in loan.get("eligibility_criteria", []):
        val = data.get(criterion)
        if val and val not in (False, 0, "", "0", "none"):
            criteria_met.append(criterion)
        else:
            criteria_not_met.append(criterion)

    # If more than half of criteria are not met, flag it (but don't reject)
    if criteria_not_met:
        reasons.append(
            f"Missing data for: {', '.join(c.replace('_', ' ') for c in criteria_not_met)}"
        )

    return {
        "eligible": eligible,
        "reasons": reasons,
        "criteria_met": criteria_met,
        "criteria_not_met": criteria_not_met,
    }


def _build_persona_loan_detail(
    loan_key: str, loan: Dict, score: float, income: float,
    tier: Dict, repayment: Dict, eligibility: Dict
) -> Dict:
    """Build loan detail for persona-specific products."""
    # Interest rate: use loan's own range (often subsidized)
    loan_rate_low, loan_rate_high = loan["interest_range"]
    if score >= 700:
        effective_rate = loan_rate_low
    elif score >= 550:
        effective_rate = loan_rate_low + (loan_rate_high - loan_rate_low) * 0.4
    else:
        effective_rate = loan_rate_high

    # Max amount: capped by product limits (not FOIR for govt schemes)
    max_amount = loan["amount_range"][1]
    min_amount = loan["amount_range"][0]

    # If income is known, also apply a reasonable income multiplier
    if income > 0:
        income_cap = income * tier["max_exposure_multiplier"]
        if income_cap > 0 and income_cap < max_amount:
            max_amount = income_cap

    # Recommended: 60% of max for alternative profiles (conservative)
    recommended = min(max_amount * 0.6, loan["amount_range"][1])
    recommended = max(recommended, min_amount)

    # Tenure
    suggested_tenure = loan["tenure_range"][1]
    if tier["max_tenure_months"] > 0:
        suggested_tenure = min(suggested_tenure, tier["max_tenure_months"])
    if suggested_tenure == 0:
        suggested_tenure = loan["tenure_range"][0]

    emi = calculate_emi(recommended, effective_rate, suggested_tenure)
    total_interest = calculate_total_interest(
        recommended, effective_rate, suggested_tenure
    )

    # Interest saved if subsidy exists
    interest_saved = 0
    if loan.get("subsidy"):
        market_rate = effective_rate + 5  # assume 5% higher without subsidy
        market_interest = calculate_total_interest(
            recommended, market_rate, suggested_tenure
        )
        interest_saved = max(market_interest - total_interest, 0)

    return {
        "key": loan_key,
        "name": loan["name"],
        "icon": loan["icon"],
        "category": loan["category"],
        "eligible": eligibility["eligible"],
        "reasons": eligibility.get("reasons", []),
        "criteria_met": eligibility.get("criteria_met", []),
        "criteria_not_met": eligibility.get("criteria_not_met", []),
        "effective_rate": round(effective_rate, 2),
        "max_loan_amount": round(max_amount, 0),
        "recommended_amount": round(recommended, 0),
        "min_amount": min_amount,
        "emi": emi,
        "suggested_tenure": suggested_tenure,
        "tenure_range": loan["tenure_range"],
        "total_interest": total_interest,
        "interest_saved_via_subsidy": round(interest_saved, 0),
        "collateral_required": loan.get("collateral", False),
        "processing_fee": loan["processing_fee"],
        "description": loan["description"],
        "lenders": loan["lenders"],
        "documents": loan["documents"],
        "subsidy": loan.get("subsidy"),
    }


# ─── Credit Improvement Path ────────────────────────────────────────────────

def _get_credit_improvement_path(
    score: float, tier: Dict, repayment: Dict
) -> List[Dict]:
    """
    Generate actionable steps to improve score → unlock better loans.
    """
    improvements = []
    current_tier_key = tier["tier_key"]

    # Map current tier to next tier unlock
    tier_upgrades = {
        "very_poor": ("poor", 400, "Micro Loans"),
        "poor": ("fair", 500, "Standard Micro Loans"),
        "fair": ("good", 650, "Personal & Business Loans"),
        "good": ("excellent", 750, "Premium Loans & Credit Cards"),
    }

    if current_tier_key in tier_upgrades:
        next_key, target_score, unlocks = tier_upgrades[current_tier_key]
        gap = target_score - score
        next_tier = SCORE_TIERS[next_key]

        improvements.append({
            "type": "score_upgrade",
            "title": f"Reach {target_score} to unlock {unlocks}",
            "gap": round(gap, 0),
            "current_score": score,
            "target_score": target_score,
            "benefit": (
                f"Max loans: {tier['max_simultaneous_loans']} → "
                f"{next_tier['max_simultaneous_loans']} | "
                f"Interest: {tier['base_interest_range'][0]}-"
                f"{tier['base_interest_range'][1]}% → "
                f"{next_tier['base_interest_range'][0]}-"
                f"{next_tier['base_interest_range'][1]}%"
            ),
            "actions": _get_score_improvement_actions(score, gap),
        })

    # Repayment capacity improvements
    if repayment.get("risk_flags"):
        for flag in repayment["risk_flags"]:
            improvements.append({
                "type": "financial_health",
                "title": flag,
                "actions": [
                    "Reduce existing EMI burden by prepaying high-interest loans",
                    "Increase income through supplementary earnings",
                    "Reduce fixed expenses to improve disposable income",
                ],
            })

    # If score is already excellent, show maintenance tips
    if current_tier_key == "excellent":
        improvements.append({
            "type": "maintenance",
            "title": "Maintain your excellent score",
            "actions": [
                "Continue on-time payments across all obligations",
                "Keep FOIR below 40% for best rates",
                "Diversify income sources for higher limits",
                "Build 6+ months emergency fund",
            ],
        })

    return improvements


def _get_score_improvement_actions(score: float, gap: float) -> List[str]:
    """Contextual actions based on current score gap."""
    actions = []

    if score < 450:
        actions.extend([
            "Start regular mobile recharges (monthly plans)",
            "Pay utility bills on time for 3+ months",
            "Get Aadhaar and at least one more government ID",
            "Join a Self-Help Group (SHG) or community group",
        ])
    elif score < 600:
        actions.extend([
            "Maintain 6+ months of consistent income records",
            "Build a savings habit — even ₹500/month helps",
            "Pay all utility bills within due date",
            "Get references from community members or employers",
        ])
    elif score < 750:
        actions.extend([
            "Maintain transaction consistency (avoid long gaps)",
            "Reduce expense-to-income ratio below 60%",
            "Build recurring savings (SIP, RD, or SHG deposits)",
            "Establish 12+ month positive payment history",
        ])

    if gap <= 30:
        actions.insert(0, f"You're only {gap:.0f} points away — focus on consistency!")

    return actions


# ─── Repayment Schedule Generator ───────────────────────────────────────────

def generate_repayment_schedule(
    principal: float, annual_rate: float,
    tenure_months: int, start_month: str = "Jan 2026"
) -> List[Dict]:
    """
    Generate a month-by-month EMI repayment schedule.
    """
    if principal <= 0 or tenure_months <= 0:
        return []

    emi = calculate_emi(principal, annual_rate, tenure_months)
    r = annual_rate / (12 * 100) if annual_rate > 0 else 0
    balance = principal
    schedule = []

    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    # Parse start month
    try:
        parts = start_month.split()
        start_month_idx = months.index(parts[0])
        start_year = int(parts[1])
    except (ValueError, IndexError):
        start_month_idx = 0
        start_year = 2026

    for i in range(tenure_months):
        interest_component = balance * r
        principal_component = emi - interest_component
        balance = max(balance - principal_component, 0)

        month_idx = (start_month_idx + i) % 12
        year = start_year + (start_month_idx + i) // 12

        schedule.append({
            "month": f"{months[month_idx]} {year}",
            "emi": round(emi, 2),
            "principal": round(principal_component, 2),
            "interest": round(interest_component, 2),
            "balance": round(balance, 2),
        })

    return schedule


# ─── Loan Comparison ────────────────────────────────────────────────────────

def compare_loans(loans: List[Dict]) -> List[Dict]:
    """
    Compare multiple eligible loans side by side.
    Returns top 3 loans ranked by a composite advantage score.
    """
    if not loans:
        return []

    scored = []
    for loan in loans:
        if not loan.get("eligible", False):
            continue

        # Composite score: lower rate + higher amount + lower total interest
        rate_score = max(0, 30 - loan["effective_rate"]) / 30  # lower is better
        amount_score = min(loan["max_loan_amount"] / 500000, 1.0)  # higher is better
        subsidy_score = 1.0 if loan.get("subsidy") else 0.0
        collateral_score = 0.0 if loan.get("collateral_required") else 0.5

        composite = (
            rate_score * 0.35 +
            amount_score * 0.30 +
            subsidy_score * 0.20 +
            collateral_score * 0.15
        )

        scored.append({**loan, "_composite_score": composite})

    scored.sort(key=lambda x: x["_composite_score"], reverse=True)
    return scored[:3]


# ─── Financial Literacy Tips ────────────────────────────────────────────────

def get_financial_tips(persona: str = None, score: float = 0,
                       eligible_loans: List[Dict] = None) -> List[Dict]:
    """
    Return contextual financial literacy tips based on persona and situation.
    """
    tips = []

    # General tips
    tips.append({
        "icon": "💡",
        "title": "Understand EMI Before You Borrow",
        "detail": "Your EMI should never exceed 40% of your monthly income. "
                  "Use our EMI calculator to plan."
    })

    if score < 600:
        tips.append({
            "icon": "📈",
            "title": "Build Credit History First",
            "detail": "Start with small loans (₹5K-₹10K), repay on time, "
                      "and your score will improve within 6 months."
        })

    # Persona-specific tips
    if persona == "farmer":
        tips.extend([
            {
                "icon": "🌾",
                "title": "KCC Has 4% Interest Subvention",
                "detail": "Kisan Credit Card loans have only 4% effective "
                          "interest if repaid within 1 year. Much cheaper than "
                          "informal moneylenders (36-60%)."
            },
            {
                "icon": "☀️",
                "title": "PM-KUSUM: 60% Subsidy on Solar Pumps",
                "detail": "Govt covers 60% of solar pump cost. You only pay 40%. "
                          "Saves ₹3,000-₹5,000/month on electricity."
            },
        ])

    elif persona == "student":
        tips.extend([
            {
                "icon": "🎓",
                "title": "Education Loan Interest Subsidy",
                "detail": "Family income < ₹4.5L/yr? You get full interest waiver "
                          "during study + 1 year moratorium under CSIS scheme."
            },
            {
                "icon": "📚",
                "title": "Vidya Lakshmi Portal",
                "detail": "Apply to multiple bank education loans with single form at "
                          "vidyalakshmi.co.in. Compare rates across 38 banks."
            },
        ])

    elif persona == "street_vendor":
        tips.extend([
            {
                "icon": "🏪",
                "title": "PM SVANidhi: ₹1,200 Digital Cashback",
                "detail": "Accept 50+ digital payments/month and earn ₹1,200/year "
                          "cashback. Plus 7% interest subsidy on your loan."
            },
            {
                "icon": "📈",
                "title": "SVANidhi 3-Tranche Growth",
                "detail": "Repay ₹10K → Get ₹20K → Repay → Get ₹50K. "
                          "Each stage unlocks a larger loan with better terms."
            },
        ])

    elif persona == "homemaker":
        tips.extend([
            {
                "icon": "👩‍👩‍👧‍👦",
                "title": "Women SHG Loans at 4% Interest",
                "detail": "Self-Help Group members get 3% interest subvention. "
                          "Effective rate is only 4% — much lower than market."
            },
            {
                "icon": "🏗️",
                "title": "Stand-Up India: ₹10L-₹1Cr for Women",
                "detail": "Every bank branch must give at least 1 loan of ₹10L-₹1Cr "
                          "to a woman entrepreneur. Ask your nearest bank."
            },
        ])

    elif persona == "general_no_bank":
        tips.extend([
            {
                "icon": "🤝",
                "title": "JLG: No Collateral, Group Guarantee",
                "detail": "Join a group of 5-10 people. The group's joint guarantee "
                          "replaces collateral. Available at MFIs and NABARD."
            },
            {
                "icon": "🪙",
                "title": "Gold Loan: Fastest Approval",
                "detail": "Gold loans need zero income proof and disburse in 30 minutes. "
                          "Interest is only 7-12% vs 36%+ from moneylenders."
            },
        ])

    # Loan-specific tips
    if eligible_loans:
        subsidy_loans = [l for l in eligible_loans if l.get("subsidy")]
        if subsidy_loans:
            total_saved = sum(
                l.get("interest_saved_via_subsidy", 0) for l in subsidy_loans
            )
            if total_saved > 0:
                tips.append({
                    "icon": "💰",
                    "title": f"Potential Subsidy Savings: ₹{total_saved:,.0f}",
                    "detail": f"You have {len(subsidy_loans)} subsidized loan(s) available. "
                              f"Government schemes can save you ₹{total_saved:,.0f} in interest."
                })

    return tips


# ─── Seasonal Recommendations (Farmers) ─────────────────────────────────────

CROP_SEASONS = {
    "Kharif": {
        "months": ["Jun", "Jul", "Aug", "Sep", "Oct"],
        "apply_by": "May",
        "crops": "Rice, Maize, Soybean, Cotton, Groundnut, Jowar",
        "loan_type": "crop_loan",
    },
    "Rabi": {
        "months": ["Oct", "Nov", "Dec", "Jan", "Feb"],
        "apply_by": "Sep",
        "crops": "Wheat, Mustard, Gram, Barley, Peas",
        "loan_type": "crop_loan",
    },
    "Zaid": {
        "months": ["Mar", "Apr", "May", "Jun"],
        "apply_by": "Feb",
        "crops": "Watermelon, Muskmelon, Cucumber, Moong",
        "loan_type": "crop_loan",
    },
}


def get_all_loans_catalog() -> List[Dict]:
    """
    Return a flat list of ALL loans (transaction + all persona catalogs)
    with source/persona tags for browsing/searching.
    """
    catalog = []

    # Transaction-based loans
    for key, loan in TRANSACTION_LOANS.items():
        catalog.append({
            **loan,
            "key": key,
            "source": "transaction",
            "persona": None,
            "eligibility_criteria": [],
        })

    # Persona-based loans
    for persona_key, persona_loans in PERSONA_LOANS.items():
        for key, loan in persona_loans.items():
            catalog.append({
                **loan,
                "key": key,
                "source": "persona",
                "persona": persona_key,
                "eligibility_criteria": loan.get("eligibility_criteria", []),
            })

    return catalog


def search_loans(
    query: str = "",
    category: str = "",
    source_filter: str = "",      # "transaction", "persona", or ""
    persona_filter: str = "",     # "farmer", "student", ...
    collateral_filter: str = "",  # "yes", "no", ""
    subsidy_filter: bool = False,
    max_rate: float = 0,
    min_amount: float = 0,
) -> List[Dict]:
    """
    Search and filter the full loan catalog.

    Args:
        query: Free-text search (matched against name, category, description, lenders)
        category: Filter by loan category (e.g., "Agriculture", "Personal")
        source_filter: "transaction" or "persona" or "" for all
        persona_filter: Filter to a specific persona's loans
        collateral_filter: "yes" | "no" | "" (any)
        subsidy_filter: If True, only show subsidized loans
        max_rate: If > 0, only loans with min interest rate <= this value
        min_amount: If > 0, only loans with max amount >= this value

    Returns:
        Filtered list of loan dicts with source/persona tags
    """
    catalog = get_all_loans_catalog()
    results = []

    query_lower = query.strip().lower()

    for loan in catalog:
        # --- Text search ---
        if query_lower:
            searchable = " ".join([
                loan.get("name", ""),
                loan.get("category", ""),
                loan.get("description", ""),
                " ".join(loan.get("lenders", [])),
                loan.get("key", ""),
                loan.get("persona", "") or "",
                loan.get("subsidy", "") or "",
            ]).lower()
            if query_lower not in searchable:
                continue

        # --- Category filter ---
        if category and loan.get("category", "").lower() != category.lower():
            continue

        # --- Source filter ---
        if source_filter:
            if loan.get("source", "") != source_filter:
                continue

        # --- Persona filter ---
        if persona_filter:
            if loan.get("persona", "") != persona_filter:
                continue

        # --- Collateral filter ---
        if collateral_filter == "no" and loan.get("collateral", False):
            continue
        if collateral_filter == "yes" and not loan.get("collateral", False):
            continue

        # --- Subsidy filter ---
        if subsidy_filter and not loan.get("subsidy"):
            continue

        # --- Max interest rate filter ---
        if max_rate > 0:
            loan_min_rate = loan.get("interest_range", (0, 0))[0]
            if loan_min_rate > max_rate:
                continue

        # --- Min amount filter ---
        if min_amount > 0:
            loan_max_amount = loan.get("amount_range", (0, 0))[1]
            if loan_max_amount < min_amount:
                continue

        results.append(loan)

    return results


def check_loan_eligibility(
    loan_key: str,
    source: str,
    persona: str = "",
    score: float = 0,
    monthly_income: float = 0,
    monthly_expenses: float = 0,
    existing_emi: float = 0,
    persona_data: Optional[Dict] = None,
    desired_amount: float = 0,
    desired_tenure: int = 0,
) -> Dict:
    """
    Check eligibility for ONE specific loan given user-provided details.

    Returns a comprehensive verdict with:
      - eligible: bool
      - verdict: str (ELIGIBLE / ELIGIBLE_WITH_CAUTION / NOT_ELIGIBLE / MICRO_ONLY)
      - reasons_pass: list of checks that passed
      - reasons_fail: list of checks that failed
      - gap_analysis: what the user needs to improve
      - loan_details: computed EMI, max amount, etc. if eligible
      - improvement_steps: actionable steps to become eligible
    """
    persona_data = persona_data or {}

    # Resolve the loan from catalog
    loan = None
    if source == "transaction":
        loan = TRANSACTION_LOANS.get(loan_key)
    elif source == "persona" and persona:
        persona_catalog = PERSONA_LOANS.get(persona, {})
        loan = persona_catalog.get(loan_key)

    if not loan:
        return {
            "eligible": False,
            "verdict": "LOAN_NOT_FOUND",
            "loan_name": loan_key,
            "reasons_pass": [],
            "reasons_fail": [f"Loan '{loan_key}' not found in {source}/{persona} catalog"],
            "gap_analysis": [],
            "loan_details": {},
            "improvement_steps": [],
        }

    tier = get_score_tier(score)
    foir_cap = tier["foir_cap"]

    # Estimate income if persona and not provided
    if monthly_income <= 0 and source == "persona" and persona:
        monthly_income = _estimate_income_from_persona(persona, persona_data)

    repayment = analyze_repayment_capacity(
        monthly_income, monthly_expenses, existing_emi, foir_cap
    )

    # --- Run all eligibility checks ---
    reasons_pass = []
    reasons_fail = []
    gap_analysis = []

    # 1. Score check
    min_score = loan.get("min_score", 0)
    if score >= min_score:
        reasons_pass.append(
            f"Score {score:.0f} meets minimum requirement of {min_score}"
        )
    else:
        gap = min_score - score
        reasons_fail.append(
            f"Score {score:.0f} is below minimum {min_score} (need +{gap:.0f} points)"
        )
        gap_analysis.append({
            "check": "Credit Score",
            "current": f"{score:.0f}",
            "required": f"{min_score}",
            "gap": f"+{gap:.0f} points needed",
            "difficulty": "Medium" if gap <= 50 else "Hard",
        })

    # 2. Income check (transaction loans)
    min_income = loan.get("min_income", 0)
    if source == "transaction" and min_income > 0:
        if monthly_income >= min_income:
            reasons_pass.append(
                f"Monthly income Rs.{monthly_income:,.0f} meets minimum Rs.{min_income:,}"
            )
        else:
            gap_income = min_income - monthly_income
            reasons_fail.append(
                f"Monthly income Rs.{monthly_income:,.0f} below minimum Rs.{min_income:,} "
                f"(need +Rs.{gap_income:,.0f})"
            )
            gap_analysis.append({
                "check": "Monthly Income",
                "current": f"Rs.{monthly_income:,.0f}",
                "required": f"Rs.{min_income:,}",
                "gap": f"+Rs.{gap_income:,.0f}/month needed",
                "difficulty": "Hard",
            })

    # 3. Tier check (max simultaneous loans)
    if tier["max_simultaneous_loans"] > 0:
        reasons_pass.append(
            f"Score tier '{tier['grade']}' allows up to {tier['max_simultaneous_loans']} loan(s)"
        )
    else:
        reasons_fail.append(
            "Score tier 'Very Poor' does not allow any loans. Improve score to 400+ first."
        )
        gap_analysis.append({
            "check": "Loan Tier Access",
            "current": tier["grade"],
            "required": "Poor (400+)",
            "gap": f"+{400 - score:.0f} points to unlock loan access",
            "difficulty": "Hard",
        })

    # 4. Repayment capacity check
    if repayment["verdict"] == "NOT_ELIGIBLE":
        reasons_fail.append(
            "Repayment capacity insufficient — income is zero or FOIR exceeded"
        )
        gap_analysis.append({
            "check": "Repayment Capacity",
            "current": f"Max new EMI: Rs.{repayment['max_new_emi']:,.0f}",
            "required": "Positive EMI capacity",
            "gap": "Reduce existing EMIs or increase income",
            "difficulty": "Hard",
        })
    elif repayment["verdict"] == "MICRO_ONLY":
        reasons_pass.append(
            f"Limited repayment capacity: max new EMI Rs.{repayment['max_new_emi']:,.0f}"
        )
    else:
        reasons_pass.append(
            f"Repayment capacity healthy: max new EMI Rs.{repayment['max_new_emi']:,.0f}"
        )

    # 5. Persona-specific eligibility criteria
    for criterion in loan.get("eligibility_criteria", []):
        val = persona_data.get(criterion)
        label = criterion.replace("_", " ").title()
        if val and val not in (False, 0, "", "0", "none"):
            reasons_pass.append(f"Criteria '{label}' is met")
        else:
            reasons_fail.append(
                f"Criteria '{label}' not verified — provide proof/data to strengthen application"
            )

    # --- Determine overall eligibility ---
    eligible = len(reasons_fail) == 0
    has_score_fail = any("Score" in r and "below" in r for r in reasons_fail)
    has_income_fail = any("income" in r.lower() and "below" in r.lower() for r in reasons_fail)
    has_tier_fail = any("Loan Tier" in r or "Very Poor" in r for r in reasons_fail)

    if has_tier_fail:
        verdict = "NOT_ELIGIBLE"
    elif has_score_fail and has_income_fail:
        verdict = "NOT_ELIGIBLE"
    elif has_score_fail or has_income_fail:
        verdict = "NOT_ELIGIBLE"
    elif repayment["verdict"] == "NOT_ELIGIBLE":
        verdict = "NOT_ELIGIBLE"
    elif repayment["verdict"] == "MICRO_ONLY":
        verdict = "MICRO_ONLY"
    elif len(reasons_fail) > 0:
        verdict = "ELIGIBLE_WITH_CAUTION"
    else:
        verdict = "ELIGIBLE"

    # --- Calculate loan details if eligible or close ---
    loan_details = {}
    if verdict in ("ELIGIBLE", "ELIGIBLE_WITH_CAUTION", "MICRO_ONLY"):
        # Effective interest rate based on score
        rate_low, rate_high = loan["interest_range"]
        if score >= 750:
            effective_rate = rate_low
        elif score >= 650:
            effective_rate = rate_low + (rate_high - rate_low) * 0.3
        elif score >= 500:
            effective_rate = rate_low + (rate_high - rate_low) * 0.6
        else:
            effective_rate = rate_high

        # Max loan amount
        max_by_product = loan["amount_range"][1]
        max_by_income = monthly_income * tier["max_exposure_multiplier"] if monthly_income > 0 else max_by_product

        if repayment["max_new_emi"] > 0:
            mid_tenure = desired_tenure if desired_tenure > 0 else (
                (loan["tenure_range"][0] + loan["tenure_range"][1]) // 2
            )
            max_by_emi = max_loan_from_emi(repayment["max_new_emi"], effective_rate, mid_tenure)
        else:
            max_by_emi = 0

        max_amount = min(max_by_product, max_by_income, max_by_emi) if max_by_emi > 0 else min(max_by_product, max_by_income)
        max_amount = max(max_amount, 0)

        # Use desired amount or recommended
        actual_amount = desired_amount if 0 < desired_amount <= max_amount else min(max_amount * 0.8, max_by_product)
        actual_amount = max(actual_amount, 0)

        # Tenure
        actual_tenure = desired_tenure if desired_tenure > 0 else loan["tenure_range"][1]
        if tier["max_tenure_months"] > 0:
            actual_tenure = min(actual_tenure, tier["max_tenure_months"])
        actual_tenure = max(actual_tenure, loan["tenure_range"][0])
        actual_tenure = min(actual_tenure, loan["tenure_range"][1])

        emi = calculate_emi(actual_amount, effective_rate, actual_tenure)
        total_interest = calculate_total_interest(actual_amount, effective_rate, actual_tenure)
        total_payable = actual_amount + total_interest

        # Check if desired amount exceeds max
        amount_ok = True
        if desired_amount > 0 and desired_amount > max_amount:
            amount_ok = False
            reasons_fail.append(
                f"Desired amount Rs.{desired_amount:,.0f} exceeds maximum eligible Rs.{max_amount:,.0f}"
            )
            gap_analysis.append({
                "check": "Loan Amount",
                "current": f"Max eligible: Rs.{max_amount:,.0f}",
                "required": f"Rs.{desired_amount:,.0f}",
                "gap": f"Exceeds by Rs.{desired_amount - max_amount:,.0f}",
                "difficulty": "Medium",
            })
            eligible = False
            if verdict == "ELIGIBLE":
                verdict = "ELIGIBLE_WITH_CAUTION"

        # Check if EMI is affordable
        if emi > repayment["max_new_emi"] and repayment["max_new_emi"] > 0:
            reasons_fail.append(
                f"EMI Rs.{emi:,.0f} exceeds affordable EMI Rs.{repayment['max_new_emi']:,.0f}"
            )
            gap_analysis.append({
                "check": "EMI Affordability",
                "current": f"Affordable: Rs.{repayment['max_new_emi']:,.0f}/mo",
                "required": f"Rs.{emi:,.0f}/mo",
                "gap": f"Reduce amount or extend tenure",
                "difficulty": "Medium",
            })

        loan_details = {
            "effective_rate": round(effective_rate, 2),
            "max_eligible_amount": round(max_amount, 0),
            "actual_amount": round(actual_amount, 0),
            "actual_tenure_months": actual_tenure,
            "emi": round(emi, 2),
            "total_interest": round(total_interest, 2),
            "total_payable": round(total_payable, 2),
            "processing_fee": loan["processing_fee"],
            "collateral_required": loan.get("collateral", False),
            "subsidy": loan.get("subsidy"),
            "documents_needed": loan.get("documents", []),
            "lenders": loan.get("lenders", []),
            "amount_ok": amount_ok,
        }

    # --- Improvement steps ---
    improvement_steps = []
    for gap in gap_analysis:
        if gap["check"] == "Credit Score":
            improvement_steps.extend([
                f"Improve score by {gap['gap']} — pay bills on time, maintain consistent income records",
                "Use Score Builder page to see exactly which criteria to improve",
            ])
        elif gap["check"] == "Monthly Income":
            improvement_steps.extend([
                f"Increase income by {gap['gap']} — consider supplementary income sources",
                "After income improves, upload 6-month bank statement to verify",
            ])
        elif gap["check"] == "Loan Tier Access":
            improvement_steps.extend([
                "Focus on consistently paying utility bills on time",
                "Get registered with Aadhaar + PAN for identity verification",
                "Join an SHG or JLG group for initial credit access",
            ])
        elif gap["check"] == "Repayment Capacity":
            improvement_steps.extend([
                "Pay off or reduce existing loan EMIs first",
                "Reduce monthly fixed expenses",
                "Consider increasing income before applying",
            ])
        elif gap["check"] == "Loan Amount":
            improvement_steps.extend([
                f"Apply for a lower amount (max eligible: {loan_details.get('max_eligible_amount', 'N/A')})",
                "Improve score / income to increase eligibility limit",
            ])
        elif gap["check"] == "EMI Affordability":
            improvement_steps.extend([
                "Choose a longer tenure to reduce monthly EMI",
                "Apply for a smaller amount to bring EMI within budget",
                "Pay off existing EMIs to free up capacity",
            ])

    if not improvement_steps and not eligible:
        improvement_steps.append(
            "Improve your credit score and income to become eligible for this loan."
        )

    # If eligible, add positive reinforcement
    if eligible and verdict == "ELIGIBLE":
        improvement_steps = [
            "You are fully eligible! Gather the required documents and apply.",
            "Compare rates across multiple lenders for the best deal.",
        ]

    return {
        "eligible": eligible,
        "verdict": verdict,
        "loan_name": loan.get("name", loan_key),
        "loan_icon": loan.get("icon", ""),
        "loan_category": loan.get("category", ""),
        "loan_description": loan.get("description", ""),
        "source": source,
        "persona": persona,
        "score_used": score,
        "tier": tier["grade"],
        "reasons_pass": reasons_pass,
        "reasons_fail": reasons_fail,
        "gap_analysis": gap_analysis,
        "loan_details": loan_details,
        "improvement_steps": improvement_steps,
        "repayment_capacity": repayment,
    }


def get_loan_categories() -> List[str]:
    """Return all unique loan categories across the full catalog."""
    catalog = get_all_loans_catalog()
    categories = sorted(set(loan.get("category", "") for loan in catalog if loan.get("category")))
    return categories


def get_seasonal_recommendations(persona: str,
                                 current_month: str = "Feb") -> List[Dict]:
    """
    For farmers: recommend loans aligned to crop seasons.
    """
    if persona != "farmer":
        return []

    recs = []
    for season, info in CROP_SEASONS.items():
        if current_month in info["months"] or current_month == info["apply_by"]:
            recs.append({
                "season": season,
                "status": "Active" if current_month in info["months"] else "Apply Now",
                "crops": info["crops"],
                "recommended_loan": info["loan_type"],
                "advice": (
                    f"{season} season {'is active' if current_month in info['months'] else 'starts soon'}. "
                    f"Apply for crop loan by {info['apply_by']} for best rates."
                ),
            })

    if not recs:
        # Find next upcoming season
        all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        current_idx = all_months.index(current_month) if current_month in all_months else 0
        for season, info in CROP_SEASONS.items():
            apply_idx = all_months.index(info["apply_by"]) if info["apply_by"] in all_months else 0
            if apply_idx > current_idx:
                recs.append({
                    "season": season,
                    "status": f"Upcoming — Apply by {info['apply_by']}",
                    "crops": info["crops"],
                    "recommended_loan": info["loan_type"],
                    "advice": (
                        f"Plan ahead: {season} crop loan applications open in "
                        f"{info['apply_by']}. Prepare land records and sowing certificate."
                    ),
                })
                break

    return recs
