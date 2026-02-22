"""
Document Analyzer for CrediVist — Alternative Profiles
=======================================================
Automatically extracts credit-scoring data from uploaded documents.
Supports: PDF, CSV, Excel, Images (via text patterns), plain text.

Each persona has specific extraction logic:
  - Farmer: Land records, PM-KISAN statements, mandi receipts
  - Student: Marksheets, scholarship letters, certificates
  - Street Vendor: Sales registers, rent receipts, utility bills
  - Homemaker: Expense diaries, SHG passbooks, utility bills
  - General: ID documents, recharge history, utility bills
"""

import re
import io
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ─── PDF Text Extraction ────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file (text-based)."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return ""


def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Extract text and/or DataFrame from an uploaded file.
    Now supports images and scanned PDFs via OCR engine.
    Returns (text, dataframe_or_none).
    """
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    df = None

    # ── Image files — use OCR ────────────────────────────────────────
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]:
        try:
            from src.ocr_engine import ocr_image
            text = ocr_image(file_bytes)
            if not text.strip():
                logger.info(f"OCR returned empty for image {filename}")
        except ImportError:
            logger.warning("OCR engine not available for image processing")

    # ── PDF files — try text first, then OCR for scanned ────────────
    elif ext == ".pdf":
        text = extract_text_from_pdf(file_bytes)

        # If very little text extracted, PDF is likely scanned — try OCR
        if len(text.strip()) < 50:
            try:
                from src.ocr_engine import ocr_pdf_pages
                ocr_text = ocr_pdf_pages(file_bytes)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
            except ImportError:
                logger.info("OCR engine not available for scanned PDF")

        # Also try table extraction from PDF
        try:
            from src.ocr_engine import extract_tables_from_pdf
            tables = extract_tables_from_pdf(file_bytes)
            if tables:
                df = tables[0]
                # Append table text to main text
                for t in tables:
                    text += "\n" + t.to_string()
        except ImportError:
            pass

    elif ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            text = df.to_string()
        except Exception:
            text = file_bytes.decode("utf-8", errors="ignore")

    elif ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            text = df.to_string()
        except Exception:
            pass

    elif ext == ".txt":
        text = file_bytes.decode("utf-8", errors="ignore")

    elif ext in [".json"]:
        try:
            data = json.loads(file_bytes.decode("utf-8"))
            text = json.dumps(data, indent=2)
        except Exception:
            text = file_bytes.decode("utf-8", errors="ignore")

    else:
        # Try as text
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            pass

    return text, df


# ─── Utility: Number Extraction ─────────────────────────────────────────────

def find_amounts(text: str) -> List[float]:
    """Find all currency amounts in text (₹, Rs, INR patterns)."""
    patterns = [
        r'[₹₨]\s*([\d,]+(?:\.\d{2})?)',
        r'Rs\.?\s*([\d,]+(?:\.\d{2})?)',
        r'INR\s*([\d,]+(?:\.\d{2})?)',
        r'Amount[:\s]*([\d,]+(?:\.\d{2})?)',
    ]
    amounts = []
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            val = m.group(1).replace(",", "")
            try:
                amounts.append(float(val))
            except ValueError:
                pass
    return amounts


def find_percentage(text: str, keywords: List[str]) -> Optional[float]:
    """Find percentage near given keywords."""
    for kw in keywords:
        patterns = [
            rf'{kw}[:\s]*(\d+(?:\.\d+)?)\s*%',
            rf'{kw}[:\s]*(\d+(?:\.\d+)?)\s*percent',
            rf'(\d+(?:\.\d+)?)\s*%\s*{kw}',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
    return None


def find_number_near(text: str, keywords: List[str]) -> Optional[float]:
    """Find a number near given keywords."""
    for kw in keywords:
        patterns = [
            rf'{kw}[:\s]*(\d+(?:\.\d+)?)',
            rf'(\d+(?:\.\d+)?)\s*{kw}',
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
    return None


def text_contains_any(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the keywords."""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def count_dates(text: str) -> int:
    """Count date-like patterns in text to estimate record count."""
    date_patterns = [
        r'\d{2}[/-]\d{2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{2}[/-]\d{2}',
        r'\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{2,4}',
    ]
    count = 0
    for p in date_patterns:
        count += len(re.findall(p, text, re.IGNORECASE))
    return count


# ─── Auto-Detect Persona ────────────────────────────────────────────────────

PERSONA_KEYWORDS = {
    "farmer": [
        "land", "acre", "hectare", "crop", "kharif", "rabi", "mandi",
        "agricultural", "farm", "soil", "pm-kisan", "kisan", "kcc",
        "pmfby", "crop insurance", "harvest", "yield", "seed", "fertilizer",
        "irrigation", "patta", "rtc", "khata", "wheat", "rice", "paddy",
        "sugarcane", "cotton", "dairy", "livestock", "cattle",
    ],
    "student": [
        "cgpa", "gpa", "semester", "marksheet", "marks", "grade",
        "examination", "university", "college", "school", "scholarship",
        "certificate", "coursera", "nptel", "udemy", "backlog",
        "attendance", "enrollment", "admission", "degree", "diploma",
        "transcript", "academic", "student", "roll no",
    ],
    "street_vendor": [
        "daily sales", "stall", "vendor", "hawker", "cart", "shop",
        "daily income", "daily earning", "trade license", "municipal",
        "rent receipt", "stall fee", "market fee", "daily collection",
        "cash sale", "footfall",
    ],
    "homemaker": [
        "household", "tiffin", "tailoring", "pickle", "papad",
        "home business", "micro enterprise", "shg", "self help group",
        "mahila", "women group", "kitchen", "homemade", "handicraft",
        "dependents", "family budget", "household expense",
    ],
    "general_no_bank": [
        "aadhaar", "aadhar", "pan card", "voter id", "ration card",
        "identity", "recharge", "prepaid", "mobile number",
        "electricity bill", "water bill", "gas bill",
    ],
}


def auto_detect_persona(text: str) -> Tuple[str, float]:
    """
    Detect the most likely persona from document text.
    Returns (persona_key, confidence).
    """
    text_lower = text.lower()
    scores = {}

    for persona, keywords in PERSONA_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[persona] = count / len(keywords) if keywords else 0

    if not scores or max(scores.values()) == 0:
        return "general_no_bank", 0.3

    best = max(scores, key=scores.get)
    confidence = min(scores[best] * 3, 0.95)  # Scale up, cap at 95%
    return best, round(confidence, 2)


# ─── Persona-Specific Extractors ────────────────────────────────────────────

def extract_farmer_data(text: str, df: Optional[pd.DataFrame]) -> Dict:
    """Extract farmer-related data from documents."""
    data = {}

    # Land ownership
    data["owns_land"] = text_contains_any(text, [
        "owner", "patta", "khata", "land owner", "own name",
        "registered in the name", "title deed"
    ])

    # Land area
    acres = find_number_near(text, ["acre", "acres", "acr"])
    hectares = find_number_near(text, ["hectare", "hectares", "ha"])
    if acres:
        data["land_acres"] = min(acres, 100)
    elif hectares:
        data["land_acres"] = min(hectares * 2.47, 100)
    else:
        data["land_acres"] = 2.0

    # Years on land
    years = find_number_near(text, ["years", "year", "since"])
    if years and years < 100:
        data["years_on_land"] = int(years)
    else:
        data["years_on_land"] = 5

    # Crop info
    crop_keywords = ["kharif", "rabi", "zaid", "summer", "winter", "crop"]
    seasons = sum(1 for k in crop_keywords if k in text.lower())
    data["seasons_active"] = max(min(seasons * 2, 12), 2)

    crops_found = sum(1 for c in [
        "wheat", "rice", "paddy", "maize", "cotton", "sugarcane",
        "soybean", "groundnut", "mustard", "potato", "onion",
        "tomato", "vegetables", "pulses", "dal", "millets"
    ] if c in text.lower())
    data["crops_per_year"] = max(min(crops_found, 4), 1)

    # Yield trend
    if text_contains_any(text, ["increase", "growth", "improved", "higher"]):
        data["yield_trend"] = "up"
    elif text_contains_any(text, ["decrease", "decline", "reduced", "lower"]):
        data["yield_trend"] = "down"
    else:
        data["yield_trend"] = "stable"

    # Government schemes
    data["has_pm_kisan"] = text_contains_any(text, [
        "pm-kisan", "pm kisan", "pradhan mantri kisan", "kisan samman"
    ])
    data["has_crop_insurance"] = text_contains_any(text, [
        "pmfby", "crop insurance", "fasal bima", "pradhan mantri fasal"
    ])
    data["has_soil_health_card"] = text_contains_any(text, [
        "soil health card", "soil card", "soil test"
    ])
    data["kcc_holder"] = text_contains_any(text, [
        "kcc", "kisan credit card", "kisan credit"
    ])

    # Market engagement
    data["sells_at_mandi"] = text_contains_any(text, [
        "mandi", "market yard", "apmc", "agricultural market"
    ])
    data["has_warehouse_receipt"] = text_contains_any(text, [
        "warehouse", "godown", "storage receipt", "wdra"
    ])
    data["uses_enam"] = text_contains_any(text, [
        "e-nam", "enam", "national agriculture market"
    ])
    mandi_trips = find_number_near(text, ["trip", "visit", "mandi"])
    data["avg_trips_per_month"] = int(min(mandi_trips or 2, 30))

    # Community
    data.update(_extract_community_data(text))

    # Utility
    data.update(_extract_utility_data(text, df))

    # Mobile
    data.update(_extract_mobile_data(text))

    return data


def extract_student_data(text: str, df: Optional[pd.DataFrame]) -> Dict:
    """Extract student-related data from documents."""
    data = {}

    # Academic performance
    cgpa = find_number_near(text, ["cgpa", "cpi", "gpa", "spi"])
    pct = find_percentage(text, ["marks", "score", "percentage", "aggregate", "total"])

    if cgpa and cgpa <= 10:
        data["score_type"] = "cgpa"
        data["score_value"] = round(cgpa, 2)
    elif pct and pct <= 100:
        data["score_type"] = "percentage"
        data["score_value"] = round(pct, 1)
    else:
        # Try to find any score-like number
        score_match = re.search(
            r'(?:total|aggregate|overall|final)[:\s]*(\d+(?:\.\d+)?)',
            text, re.IGNORECASE
        )
        if score_match:
            val = float(score_match.group(1))
            if val <= 10:
                data["score_type"] = "cgpa"
                data["score_value"] = round(val, 2)
            elif val <= 100:
                data["score_type"] = "percentage"
                data["score_value"] = round(val, 1)
            else:
                data["score_type"] = "percentage"
                data["score_value"] = 65.0
        else:
            data["score_type"] = "percentage"
            data["score_value"] = 65.0

    # Education level — check PG first (more specific), then UG
    if text_contains_any(text, ["post graduate", "m.tech", "mba", "m.sc", "m.a.",
                                 "master of", "masters"]):
        data["education_level"] = "pg"
    elif text_contains_any(text, ["b.tech", "b.e.", "b.e ", "bba", "b.sc", "b.com",
                                   "b.a.", "b.a ", "bachelor", "engineering",
                                   "degree", "undergraduate", "usn"]):
        data["education_level"] = "ug"
    elif text_contains_any(text, ["class 10", "class 12", "sslc", "hsc",
                                   "cbse", "icse", "school"]):
        data["education_level"] = "school"
    else:
        data["education_level"] = "ug"  # default for college docs

    # Backlogs — look for explicit count near keyword
    backlog_match = re.search(
        r'(?:backlog|arrear|kt)[s]?[:\s]*(\d+)',
        text, re.IGNORECASE
    )
    if backlog_match:
        data["backlog_count"] = int(min(int(backlog_match.group(1)), 20))
    elif text_contains_any(text, ["no backlog", "no arrear", "clear", "0 backlog"]):
        data["backlog_count"] = 0
    else:
        data["backlog_count"] = 0

    # Scholarships
    scholarship_keywords = ["scholarship", "merit award", "fellowship", "stipend", "bursary"]
    scholarship_count = sum(1 for k in scholarship_keywords if k in text.lower())
    data["scholarships_received"] = max(scholarship_count, 0)

    amounts = find_amounts(text)
    scholarship_amounts = [a for a in amounts if 500 <= a <= 500000]
    data["total_scholarship_value"] = sum(scholarship_amounts) if scholarship_amounts else 0

    data["merit_based"] = text_contains_any(text, [
        "merit", "topper", "rank", "distinction", "first class"
    ])

    # Certifications
    cert_platforms = []
    platform_list = ["nptel", "coursera", "udemy", "edx", "swayam", "nsdc",
                     "pmkvy", "skill india", "google", "microsoft", "aws"]
    for p in platform_list:
        if p in text.lower():
            cert_platforms.append(p.upper() if len(p) <= 5 else p.title())
    data["platform_certs"] = cert_platforms
    data["cert_count"] = max(len(cert_platforms), 0)
    data["has_govt_certification"] = text_contains_any(text, [
        "nsdc", "pmkvy", "skill india", "government certified", "govt cert"
    ])

    # Attendance
    att_pct = find_percentage(text, ["attendance", "present"])
    data["attendance_pct"] = att_pct if att_pct else 75

    # Part-time income
    part_time_keywords = ["freelance", "part-time", "part time", "tutoring",
                          "tuition", "intern", "stipend", "earning"]
    data["has_part_time"] = text_contains_any(text, part_time_keywords)
    if data["has_part_time"]:
        # Look for earnings/stipend amount near relevant keywords
        earn_match = re.search(
            r'(?:earning|stipend|salary|freelance|part.?time)[^₹₨]*[₹₨Rs.INR\s]*([\d,]+)',
            text, re.IGNORECASE
        )
        if earn_match:
            data["monthly_earnings"] = min(float(earn_match.group(1).replace(',','')), 100000)
        else:
            earnings = [a for a in amounts if 1000 <= a <= 30000]
            data["monthly_earnings"] = earnings[0] if earnings else 5000
        months = find_number_near(text, ["month", "months"])
        data["months_active"] = int(min(months or 3, 60))
    else:
        data["monthly_earnings"] = 0
        data["months_active"] = 0

    # Future potential
    tier_map = {"iit": 1, "nit": 1, "iiit": 1, "bits": 1, "tier 1": 1, "tier-1": 1,
                "tier 2": 2, "tier-2": 2, "state university": 2,
                "tier 3": 3, "tier-3": 3, "private": 3}
    data["institution_tier"] = 3
    for keyword, tier in tier_map.items():
        if keyword in text.lower():
            data["institution_tier"] = tier
            break

    high_demand = ["computer", "cse", "it", "information technology", "data science",
                   "ai", "artificial intelligence", "electronics", "ece", "mechanical"]
    low_demand = ["arts", "history", "philosophy", "library"]
    if text_contains_any(text, high_demand):
        data["branch_demand"] = "high"
    elif text_contains_any(text, low_demand):
        data["branch_demand"] = "low"
    else:
        data["branch_demand"] = "medium"

    data["has_internship"] = text_contains_any(text, [
        "intern", "internship", "industrial training", "summer training"
    ])

    # Community
    data.update(_extract_community_data(text))

    # Mobile
    data.update(_extract_mobile_data(text))

    return data


def extract_vendor_data(text: str, df: Optional[pd.DataFrame]) -> Dict:
    """Extract street vendor / informal worker data."""
    data = {}

    # Daily income
    daily = find_number_near(text, ["daily income", "daily earning", "per day",
                                     "daily sale", "daily collection"])
    if daily and daily < 50000:
        data["avg_daily_income"] = daily
    else:
        amounts = find_amounts(text)
        small_amounts = [a for a in amounts if 100 <= a <= 5000]
        data["avg_daily_income"] = np.mean(small_amounts) if small_amounts else 500

    # Working days from DataFrame
    if df is not None and len(df) > 0:
        data["working_days_per_month"] = min(len(df), 31)
        if "amount" in [c.lower() for c in df.columns]:
            amt_col = [c for c in df.columns if c.lower() == "amount"][0]
            daily_vals = pd.to_numeric(df[amt_col], errors="coerce").dropna()
            if len(daily_vals) > 0:
                data["avg_daily_income"] = round(daily_vals.mean(), 0)
                cv = daily_vals.std() / (daily_vals.mean() + 1e-9)
                if cv < 0.3:
                    data["seasonal_variation"] = "low"
                elif cv < 0.6:
                    data["seasonal_variation"] = "medium"
                else:
                    data["seasonal_variation"] = "high"
    else:
        days = find_number_near(text, ["working day", "days worked", "business day"])
        data["working_days_per_month"] = int(min(days or 25, 31))

    if "seasonal_variation" not in data:
        if text_contains_any(text, ["regular", "consistent", "steady", "stable"]):
            data["seasonal_variation"] = "low"
        elif text_contains_any(text, ["seasonal", "fluctuate", "varies"]):
            data["seasonal_variation"] = "high"
        else:
            data["seasonal_variation"] = "medium"

    # Rental
    data["pays_rent"] = text_contains_any(text, [
        "rent", "stall fee", "market fee", "shop rent", "lease"
    ])
    if data["pays_rent"]:
        rent_amounts = [a for a in find_amounts(text) if 500 <= a <= 50000]
        data["rent_amount"] = rent_amounts[0] if rent_amounts else 2000
        data["on_time_pct"] = find_percentage(text, ["on time", "timely", "paid"]) or 80
        months = find_number_near(text, ["month", "year"])
        if months and months <= 30:
            data["months_of_history"] = int(months * 12)
        elif months and months <= 360:
            data["months_of_history"] = int(months)
        else:
            data["months_of_history"] = 12
    else:
        data["rent_amount"] = 0
        data["on_time_pct"] = 0
        data["months_of_history"] = 0

    # Years in trade
    years = find_number_near(text, ["years in", "experience", "since", "doing this for"])
    data["years_in_trade"] = int(min(years or 5, 50))
    data["same_location"] = text_contains_any(text, [
        "same location", "same place", "same spot", "same area", "permanent"
    ])
    data["has_license"] = text_contains_any(text, [
        "license", "licence", "permit", "registration", "trade license",
        "vendor license", "fssai"
    ])

    # Utility, Community, Mobile, Savings
    data.update(_extract_utility_data(text, df))
    data.update(_extract_savings_data(text))
    data.update(_extract_community_data(text))
    data.update(_extract_mobile_data(text))

    return data


def extract_homemaker_data(text: str, df: Optional[pd.DataFrame]) -> Dict:
    """Extract homemaker-related data."""
    data = {}

    # Household budgeting
    income = find_number_near(text, ["household income", "family income", "total income",
                                      "monthly income"])
    expense = find_number_near(text, ["household expense", "family expense", "total expense",
                                       "monthly expense", "expenditure"])

    if df is not None and len(df) > 0:
        # Try to compute from expense diary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            totals = df[numeric_cols].sum()
            # Look for income and expense columns
            for col in numeric_cols:
                col_lower = col.lower()
                if "income" in col_lower or "credit" in col_lower or "earning" in col_lower:
                    income = float(totals[col])
                elif "expense" in col_lower or "debit" in col_lower or "spent" in col_lower:
                    expense = float(totals[col])

    data["household_income"] = income if income and income > 0 else 20000
    data["household_expenses"] = expense if expense and expense > 0 else 15000
    data["manages_budget"] = text_contains_any(text, [
        "budget", "manage", "plan", "track", "record", "diary", "register"
    ]) or (df is not None)  # If they have a spreadsheet, they manage budget
    dependents = find_number_near(text, ["dependent", "children", "family member", "members"])
    data["dependents"] = int(min(dependents or 3, 15))

    # Micro enterprise
    enterprise_keywords = {
        "tiffin": "Tiffin Service", "tailoring": "Tailoring",
        "pickle": "Pickle Making", "papad": "Papad Making",
        "handicraft": "Handicraft", "beauty": "Beauty Parlour",
        "tuition": "Home Tuition", "crèche": "Day Care",
        "candle": "Candle Making", "embroidery": "Embroidery",
        "dairy": "Dairy Products", "baking": "Home Baking",
    }
    data["has_enterprise"] = False
    data["enterprise_type"] = ""
    for kw, etype in enterprise_keywords.items():
        if kw in text.lower():
            data["has_enterprise"] = True
            data["enterprise_type"] = etype
            break

    if data["has_enterprise"]:
        rev = find_number_near(text, ["revenue", "income", "earning", "monthly", "sale"])
        data["monthly_revenue"] = min(rev or 5000, 500000)
        months = find_number_near(text, ["month", "running for", "active for"])
        data["months_active"] = int(min(months or 6, 120))
    else:
        data["monthly_revenue"] = 0
        data["months_active"] = 0

    # Skill certifications
    cert_kws = ["nsdc", "pmkvy", "skill india", "certificate", "training", "course"]
    data["cert_count"] = sum(1 for k in cert_kws if k in text.lower())
    data["has_govt_certification"] = text_contains_any(text, ["nsdc", "pmkvy", "skill india", "govt"])
    data["platform_certs"] = []

    # Shared extractors
    data.update(_extract_utility_data(text, df))
    data.update(_extract_savings_data(text))
    data.update(_extract_community_data(text))
    data.update(_extract_mobile_data(text))

    return data


def extract_general_data(text: str, df: Optional[pd.DataFrame]) -> Dict:
    """Extract data for general (no bank account) persona."""
    data = {}

    # ID verification
    data["has_aadhaar"] = text_contains_any(text, ["aadhaar", "aadhar", "uid", "unique identification"])
    data["has_pan"] = text_contains_any(text, ["pan card", "pan no", "permanent account number"])
    data["has_voter_id"] = text_contains_any(text, ["voter", "election card", "epic"])
    data["has_ration_card"] = text_contains_any(text, ["ration card", "ration", "bpl card", "apl card"])

    # If ID doc is uploaded, mark aadhaar as present even if text extraction fails
    if text_contains_any(text, ["government of india", "male", "female",
                                 "date of birth", "dob", "address"]):
        if not data["has_aadhaar"] and not data["has_pan"]:
            data["has_aadhaar"] = True

    # Psychometric defaults (can't extract from docs, set moderate defaults)
    data["q1_financial_planning"] = 3
    data["q2_risk_awareness"] = 3
    data["q3_goal_orientation"] = 3
    data["q4_repayment_intent"] = 4
    data["q5_responsibility"] = 4

    # If they uploaded documents, they show some responsibility
    if len(text) > 100:
        data["q1_financial_planning"] = 4
        data["q5_responsibility"] = 4

    # Shared extractors
    data.update(_extract_utility_data(text, df))
    data.update(_extract_savings_data(text))
    data.update(_extract_community_data(text))
    data.update(_extract_mobile_data(text))
    data.update(_extract_rental_data(text))

    return data


# ─── Shared Sub-Extractors ──────────────────────────────────────────────────

def _extract_utility_data(text: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """Extract utility bill information."""
    data = {}
    data["has_electricity"] = text_contains_any(text, [
        "electricity", "electric", "bescom", "msedcl", "kseb", "discom",
        "power bill", "eb bill", "light bill"
    ])
    data["has_water"] = text_contains_any(text, [
        "water bill", "water supply", "bwssb", "jal board", "water charge"
    ])
    data["has_gas"] = text_contains_any(text, [
        "gas", "lpg", "cylinder", "bharat gas", "hp gas", "indane", "piped gas"
    ])

    services = sum([data["has_electricity"], data["has_water"], data["has_gas"]])

    # Estimate bills per year
    bill_dates = count_dates(text)
    data["bills_per_year"] = min(max(bill_dates, services * 4), 36)

    # On-time percentage
    on_time = find_percentage(text, ["on time", "timely", "before due"])
    if on_time:
        data["on_time_pct"] = min(on_time, 100)
    elif text_contains_any(text, ["overdue", "late fee", "penalty", "delayed"]):
        data["on_time_pct"] = 60
    elif text_contains_any(text, ["paid", "receipt", "payment received"]):
        data["on_time_pct"] = 85
    else:
        data["on_time_pct"] = 75

    return data


def _extract_savings_data(text: str) -> Dict:
    """Extract savings habit information."""
    data = {}

    if text_contains_any(text, ["self help group", "shg", "mahila group", "bachat gat"]):
        data["savings_method"] = "shg"
        data["is_shg_member"] = True
    elif text_contains_any(text, ["chit fund", "chitty", "chit"]):
        data["savings_method"] = "chit_fund"
        data["is_shg_member"] = False
    elif text_contains_any(text, ["post office", "postal saving", "nsc", "kvp", "rd receipt"]):
        data["savings_method"] = "post_office"
        data["is_shg_member"] = False
    elif text_contains_any(text, ["gold", "jewel", "ornament"]):
        data["savings_method"] = "gold"
        data["is_shg_member"] = False
    elif text_contains_any(text, ["saving", "deposit", "bank"]):
        data["savings_method"] = "bank"
        data["is_shg_member"] = False
    else:
        data["savings_method"] = "cash_at_home"
        data["is_shg_member"] = False

    savings_amt = find_number_near(text, ["saving", "deposit", "contribution", "monthly saving"])
    data["monthly_savings"] = min(savings_amt or 500, 100000)

    months = find_number_near(text, ["months saving", "saving for", "since"])
    data["months_saving"] = int(min(months or 6, 120))

    return data


def _extract_community_data(text: str) -> Dict:
    """Extract community trust information."""
    data = {}

    # References
    ref_count = 0
    ref_patterns = [
        r'reference[s]?\s*[:\-]?\s*(\d+)',
        r'(\d+)\s*reference',
        r'recomme?nd(?:ation|ed)',
    ]
    for p in ref_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            try:
                ref_count = int(m.group(1))
            except (IndexError, ValueError):
                ref_count = 1
            break

    # Check for named references
    reference_names = re.findall(
        r'(?:reference|referee|recommend(?:ed by)?)[:\s]*([A-Z][a-z]+ [A-Z][a-z]+)',
        text
    )
    if reference_names:
        ref_count = max(ref_count, len(reference_names))

    data["references_count"] = min(ref_count, 10) if ref_count > 0 else 2

    data["is_group_member"] = text_contains_any(text, [
        "group", "member", "association", "union", "society", "committee",
        "shg", "federation", "cooperative"
    ])
    data["group_type"] = ""
    if data["is_group_member"]:
        for gtype in ["SHG", "Cooperative", "Trade Union", "Farmers Association",
                       "Vendors Association", "Mahila Mandal", "Youth Club"]:
            if gtype.lower() in text.lower():
                data["group_type"] = gtype
                break

    years = find_number_near(text, ["years in community", "living here", "resident since",
                                     "years at address", "residing for"])
    data["years_in_community"] = int(min(years or 5, 50))

    data["has_local_business_reference"] = text_contains_any(text, [
        "business reference", "shop owner", "employer", "contractor",
        "local business", "merchant"
    ])

    return data


def _extract_mobile_data(text: str) -> Dict:
    """Extract mobile behaviour data."""
    data = {}

    if text_contains_any(text, ["monthly plan", "monthly recharge", "postpaid"]):
        data["recharge_frequency"] = "monthly"
    elif text_contains_any(text, ["weekly recharge", "weekly plan"]):
        data["recharge_frequency"] = "weekly"
    elif text_contains_any(text, ["daily recharge", "daily data"]):
        data["recharge_frequency"] = "daily"
    else:
        data["recharge_frequency"] = "monthly"

    data["has_smartphone"] = text_contains_any(text, [
        "smartphone", "android", "iphone", "samsung", "redmi", "realme",
        "oppo", "vivo", "whatsapp"  # WhatsApp implies smartphone
    ])

    data["uses_upi_basic"] = text_contains_any(text, [
        "upi", "phonepe", "gpay", "google pay", "paytm", "bhim",
        "digital payment", "online payment", "qr code"
    ])

    recharge_amt = find_number_near(text, ["recharge", "plan"])
    data["avg_monthly_recharge"] = min(recharge_amt or 249, 5000)

    return data


def _extract_rental_data(text: str) -> Dict:
    """Extract rental discipline data."""
    data = {}
    data["pays_rent"] = text_contains_any(text, [
        "rent", "lease", "tenant", "rental", "house rent",
        "stall fee", "shop rent"
    ])

    if data["pays_rent"]:
        rent_amounts = [a for a in find_amounts(text) if 500 <= a <= 50000]
        data["rent_amount"] = rent_amounts[0] if rent_amounts else 2000
        data["on_time_pct"] = find_percentage(text, ["on time", "timely"]) or 80
        months = find_number_near(text, ["months", "month"])
        data["months_of_history"] = int(min(months or 12, 240))
    else:
        data["rent_amount"] = 0
        data["on_time_pct"] = 0
        data["months_of_history"] = 0

    return data


# ─── Document Relevance Validation ──────────────────────────────────────────

# Keywords that indicate a document is NOT credit-relevant (academic content,
# textbooks, study material, lecture notes, technical manuals, etc.)
_IRRELEVANT_KEYWORDS = [
    "chapter", "module", "syllabus", "lecture", "textbook", "lesson plan",
    "learning objective", "course outline", "table of contents",
    "bibliography", "references cited", "abstract", "introduction to",
    "theory of", "definition of", "algorithm", "theorem", "proof",
    "equation", "diagram", "figure", "appendix", "exercise",
    "question paper", "model paper", "previous year", "assignment",
    "experiment", "laboratory", "lab manual", "viva", "objective type",
    "multiple choice", "fill in the blank", "short answer", "long answer",
    "unit test", "internal assessment", "scheme of evaluation",
    "compiled by", "prepared by", "department of", "faculty of",
    "published by", "isbn", "edition", "page no",
]

# Keywords that indicate a document IS credit-relevant (financial,
# identity, employment, property, bills, etc.)
_RELEVANT_KEYWORDS = [
    # Identity
    "aadhaar", "aadhar", "pan card", "voter id", "ration card",
    "passport", "driving license", "uid",
    # Financial
    "bank statement", "account number", "ifsc", "transaction", "balance",
    "credit", "debit", "upi", "neft", "imps", "salary", "income",
    "payment", "receipt", "invoice", "bill amount", "due date",
    "emi", "loan", "interest", "principal", "repayment",
    # Employment / Gig
    "employer", "employee", "salary slip", "pay slip", "swiggy", "zomato",
    "uber", "ola", "rapido", "dunzo", "urban company", "freelance",
    "contract", "appointment letter", "offer letter",
    # Property / Land
    "land record", "patta", "khata", "survey no", "property",
    "title deed", "encumbrance",
    # Academic (credit-relevant: marksheets, certificates)
    "marksheet", "grade card", "cgpa", "sgpa", "result",
    "certificate of", "awarded to", "successfully completed",
    "scholarship", "merit",
    # Utility bills
    "electricity bill", "water bill", "gas bill", "consumer no",
    "meter reading", "bescom", "bwssb",
    # Savings / Insurance
    "savings", "fixed deposit", "recurring deposit", "insurance",
    "premium", "policy no", "pm-kisan", "kisan", "crop insurance",
    # Vendor / Business
    "trade license", "vendor", "daily sales", "stall", "shop rent",
    "mandi", "market fee",
    # SHG / Community
    "self help group", "shg", "cooperative", "chit fund",
    # Recharge / Mobile
    "recharge", "prepaid", "postpaid", "mobile plan",
]


def check_document_relevance(text: str) -> Dict[str, Any]:
    """
    Check whether uploaded document text is relevant for credit scoring.

    Returns:
        Dict with:
          - is_relevant (bool): True if document appears credit-relevant
          - relevance_score (float): 0.0–1.0 confidence of relevance
          - relevant_signals (list): matched credit-relevant keywords
          - irrelevant_signals (list): matched irrelevant keywords
          - reason (str): human-readable explanation
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())

    # Count matches
    relevant_matches = [kw for kw in _RELEVANT_KEYWORDS if kw in text_lower]
    irrelevant_matches = [kw for kw in _IRRELEVANT_KEYWORDS if kw in text_lower]

    relevant_count = len(relevant_matches)
    irrelevant_count = len(irrelevant_matches)

    # Check for monetary amounts (strong relevance signal)
    amounts = find_amounts(text)
    has_amounts = len(amounts) > 0

    # Check for dates (moderate relevance signal)
    date_count = count_dates(text)

    # Score calculation
    # Relevant signals boost the score, irrelevant signals reduce it
    relevance_score = 0.0

    if relevant_count > 0:
        relevance_score += min(relevant_count / 5, 0.5)  # up to 0.5 from keywords
    if has_amounts:
        relevance_score += 0.2  # monetary amounts are a strong signal
    if date_count >= 3:
        relevance_score += 0.1  # multiple dates suggest records/statements

    # Penalize for irrelevant content
    if irrelevant_count > 0:
        penalty = min(irrelevant_count / 8, 0.5)
        relevance_score -= penalty

    # If heavy irrelevant content with very few relevant signals
    if irrelevant_count >= 5 and relevant_count <= 2 and not has_amounts:
        relevance_score = max(relevance_score, 0) * 0.3  # heavy suppression

    # Very long text with no financial signals is likely a textbook/module
    if word_count > 2000 and relevant_count <= 1 and not has_amounts:
        relevance_score *= 0.3

    relevance_score = max(0.0, min(1.0, relevance_score))

    # Determine relevance
    is_relevant = relevance_score >= 0.15

    # Generate reason
    if not is_relevant:
        if irrelevant_count > relevant_count:
            reason = (
                "This document appears to be academic/study material "
                "(e.g., textbook, lecture notes, syllabus, module PDF) "
                "and does not contain credit-relevant financial data. "
                "Please upload documents like bank statements, ID cards, "
                "utility bills, marksheets, salary slips, or land records."
            )
        elif word_count > 2000 and not has_amounts:
            reason = (
                "This document is lengthy but contains no financial data "
                "(no monetary amounts, account numbers, or transaction records). "
                "Please upload credit-relevant documents instead."
            )
        else:
            reason = (
                "Could not find credit-relevant information in this document. "
                "Please upload documents such as bank statements, ID proofs, "
                "utility bills, marksheets, salary slips, or land records."
            )
    else:
        reason = ""

    return {
        "is_relevant": is_relevant,
        "relevance_score": round(relevance_score, 3),
        "relevant_signals": relevant_matches,
        "irrelevant_signals": irrelevant_matches,
        "reason": reason,
    }


# ─── Main Analysis Function ─────────────────────────────────────────────────

PERSONA_EXTRACTORS = {
    "farmer": extract_farmer_data,
    "student": extract_student_data,
    "street_vendor": extract_vendor_data,
    "homemaker": extract_homemaker_data,
    "general_no_bank": extract_general_data,
}


def analyze_documents(
    files: List[Tuple[str, bytes]],
    persona: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze uploaded documents and extract persona-specific data.

    Args:
        files: List of (filename, file_bytes) tuples
        persona: Persona key. If None, auto-detect from content.

    Returns:
        Dict with: detected_persona, confidence, extracted_data,
                    document_summaries, warnings
    """
    all_text = ""
    all_dfs = []
    doc_summaries = []
    parsed_documents = []   # structured parsed data from OCR engine
    ocr_used = False

    for filename, file_bytes in files:
        text, df = extract_text_from_file(file_bytes, filename)
        all_text += f"\n--- {filename} ---\n{text}\n"

        if df is not None:
            all_dfs.append(df)

        # Run structured document classification & parsing
        doc_parsed = {"filename": filename, "document_type": "unknown",
                      "parsed_data": {}, "ocr_used": False}
        try:
            from src.ocr_engine import classify_document, process_file_with_ocr, get_ocr_capabilities

            ext = os.path.splitext(filename)[1].lower()
            is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]

            if is_image:
                # Full OCR pipeline for images
                ocr_result = process_file_with_ocr(file_bytes, filename)
                doc_parsed = ocr_result
                if ocr_result.get("text") and len(ocr_result["text"]) > len(text):
                    text = ocr_result["text"]
                    all_text += f"\n{text}\n"
                ocr_used = ocr_used or ocr_result.get("ocr_used", False)
            else:
                # For non-image files, just classify the already-extracted text
                doc_type, doc_conf, parsed_data = classify_document(text)
                doc_parsed["document_type"] = doc_type
                doc_parsed["parsed_data"] = parsed_data
                doc_parsed["classification_confidence"] = doc_conf

        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Document parsing failed for {filename}: {e}")

        parsed_documents.append(doc_parsed)

        doc_summaries.append({
            "filename": filename,
            "text_length": len(text),
            "has_table": df is not None,
            "rows": len(df) if df is not None else 0,
            "amounts_found": len(find_amounts(text)),
            "dates_found": count_dates(text),
            "document_type": doc_parsed.get("document_type", "unknown"),
            "ocr_used": doc_parsed.get("ocr_used", False),
        })

    # Merge DataFrames if multiple tabular files
    merged_df = None
    if all_dfs:
        try:
            merged_df = pd.concat(all_dfs, ignore_index=True)
        except Exception:
            merged_df = all_dfs[0]

    # ── Relevance check ────────────────────────────────────────────────
    relevance = check_document_relevance(all_text)
    if not relevance["is_relevant"]:
        return {
            "detected_persona": persona or "unknown",
            "detection_confidence": 0.0,
            "extracted_data": {},
            "document_summaries": doc_summaries,
            "parsed_documents": parsed_documents,
            "detected_document_types": [],
            "ocr_used": ocr_used,
            "warnings": [relevance["reason"]],
            "total_text_length": len(all_text),
            "files_processed": len(files),
            "is_relevant": False,
            "relevance_score": relevance["relevance_score"],
            "relevance_reason": relevance["reason"],
        }
    # ───────────────────────────────────────────────────────────────────

    # Auto-detect persona if not specified
    detect_confidence = 0.0
    if persona is None:
        persona, detect_confidence = auto_detect_persona(all_text)
    else:
        detect_confidence = 0.95

        # ── Persona-document mismatch check ────────────────────────────
        # Even when user explicitly selected a persona, verify the
        # uploaded documents actually belong to that persona category.
        detected_persona, detected_conf = auto_detect_persona(all_text)
        if (
            detected_persona != persona
            and detected_conf >= 0.15           # auto-detect is reasonably sure
            and detected_persona != "general_no_bank"  # general is a catch-all
        ):
            # Double-check: does the document have ANY keywords for the
            # selected persona?  If yes at decent level, allow it.
            text_lower = all_text.lower()
            selected_kws = PERSONA_KEYWORDS.get(persona, [])
            selected_hits = sum(1 for kw in selected_kws if kw in text_lower)
            selected_ratio = selected_hits / len(selected_kws) if selected_kws else 0

            detected_kws = PERSONA_KEYWORDS.get(detected_persona, [])
            detected_hits = sum(1 for kw in detected_kws if kw in text_lower)
            detected_ratio = detected_hits / len(detected_kws) if detected_kws else 0

            # Reject if detected persona has significantly more signal
            # than the selected persona (clear mismatch)
            if detected_ratio > selected_ratio and selected_ratio < 0.15:
                from src.alternative_profiles import PERSONAS as _PERSONAS
                selected_label = _PERSONAS.get(
                    persona, {}).get("label", persona.replace("_", " ").title())
                detected_label = _PERSONAS.get(
                    detected_persona, {}).get("label", detected_persona.replace("_", " ").title())

                mismatch_reason = (
                    f"The uploaded documents appear to be **{detected_label}** documents, "
                    f"but you selected the **{selected_label}** persona. "
                    f"Please upload documents relevant to the {selected_label} category, "
                    f"or switch to the {detected_label} persona to score these documents correctly."
                )
                return {
                    "detected_persona": persona,
                    "detection_confidence": 0.0,
                    "extracted_data": {},
                    "document_summaries": doc_summaries,
                    "parsed_documents": parsed_documents,
                    "detected_document_types": [],
                    "ocr_used": ocr_used,
                    "warnings": [mismatch_reason],
                    "total_text_length": len(all_text),
                    "files_processed": len(files),
                    "is_relevant": True,
                    "persona_mismatch": True,
                    "actual_persona": detected_persona,
                    "actual_persona_confidence": detected_conf,
                    "mismatch_reason": mismatch_reason,
                    "relevance_score": relevance["relevance_score"],
                }
        # ────────────────────────────────────────────────────────────────

    # Extract data using persona-specific extractor (regex-based)
    extractor = PERSONA_EXTRACTORS.get(persona, extract_general_data)
    extracted_data = extractor(all_text, merged_df)

    # Merge structured data from document parsers (OCR engine)
    try:
        from src.ocr_engine import merge_parsed_into_persona_data
        structured_data = merge_parsed_into_persona_data(parsed_documents, persona)
        # Structured data overrides regex-extracted data (higher accuracy)
        for key, value in structured_data.items():
            if value is not None and value != "" and value != 0 and value != False:
                extracted_data[key] = value
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Structured data merge failed: {e}")

    # Warnings for low data
    warnings = []
    if len(all_text.strip()) < 50:
        if any(d.get("ocr_used") for d in doc_summaries):
            warnings.append("Limited text extracted even with OCR. Document may be low quality.")
        else:
            from src.ocr_engine import get_ocr_capabilities
            caps = get_ocr_capabilities()
            if not caps.get("tesseract"):
                warnings.append(
                    "Very little text extracted. Install Tesseract OCR for scanned document support. "
                    "PDF may be image-based — try uploading as image (JPG/PNG) or text format."
                )
            else:
                warnings.append("Very little text extracted. PDF may be image-based — try CSV or text format.")
    if not any(d["amounts_found"] > 0 for d in doc_summaries):
        warnings.append("No monetary amounts detected in documents.")
    if not any(d["dates_found"] > 0 for d in doc_summaries):
        warnings.append("No dates detected — historical patterns may be estimated.")
    if ocr_used:
        warnings.append("OCR was used to extract text from scanned documents. Verify extracted data for accuracy.")

    # Document types detected
    detected_doc_types = [d.get("document_type", "unknown") for d in parsed_documents
                          if d.get("document_type") != "unknown"]

    return {
        "detected_persona": persona,
        "detection_confidence": detect_confidence,
        "extracted_data": extracted_data,
        "document_summaries": doc_summaries,
        "parsed_documents": parsed_documents,
        "detected_document_types": detected_doc_types,
        "ocr_used": ocr_used,
        "warnings": warnings,
        "total_text_length": len(all_text),
        "files_processed": len(files),
        "is_relevant": True,
        "relevance_score": relevance["relevance_score"],
    }


# ─── Sample Document Generators ─────────────────────────────────────────────

def generate_sample_farmer_doc() -> str:
    """Generate a sample farmer document for testing/demo."""
    return """
LAND RECORD CERTIFICATE (RTC)
==========================================
Taluk: Hubli    District: Dharwad    State: Karnataka

Owner Name: Ramesh B. Patil
Survey No: 45/A    Khata No: 123
Land Area: 4.5 Acres
Land Type: Agricultural - Irrigated
Years of Ownership: 12 years (since 2014)

CROP HISTORY:
- Kharif 2024: Paddy (2 acres), Cotton (2.5 acres)
- Rabi 2024: Wheat (2 acres), Groundnut (2.5 acres)
- Kharif 2023: Paddy (4.5 acres)
- Rabi 2023: Wheat (4.5 acres)
Crops per year: 2 cycles
Yield Trend: Stable with slight increase

GOVERNMENT SCHEMES:
✓ PM-KISAN: Active (Beneficiary ID: PMKISAN/KA/2020/45678)
  - Installments received: 14 (₹2,000 each)
✓ PMFBY Crop Insurance: Active (Policy No: PMFBY/2024/KA/12345)
✓ Soil Health Card: Issued (Card No: SHC/KA/2022/7890)
✗ Kisan Credit Card (KCC): Not applied

MANDI ENGAGEMENT:
Sells at: APMC Hubli Market Yard
Average trips per month: 3
Has warehouse receipt: No
e-NAM registered: No

UTILITY PAYMENTS:
Electricity: HESCOM connection (Consumer No: HB-12345)
Bills per year: 12 (on time: 90%)
Water: Gram Panchayat supply
Gas: LPG connection (Bharat Gas)

COMMUNITY:
Member of: Raita Sangha (Farmers Association)
Years in community: 15
References: Suresh Kulkarni (neighbor farmer), Prakash Desai (merchant)
Local business reference: Krishna Traders (seed supplier)

MOBILE:
Phone: Samsung smartphone
Recharge: Monthly plan ₹299
UPI: Not used
"""


def generate_sample_student_doc() -> str:
    """Generate a sample student marksheet/profile for demo."""
    return """
SEMESTER GRADE CARD
==========================================
University: Visvesvaraya Technological University (VTU)
College: BVB College of Engineering & Technology, Hubli (Tier 2)
Branch: Computer Science & Engineering (CSE)
USN: 2VB21CS045

ACADEMIC PERFORMANCE:
Semester 1: SGPA 7.2
Semester 2: SGPA 7.5
Semester 3: SGPA 8.0
Semester 4: SGPA 7.8
Semester 5: SGPA 8.3
Semester 6: SGPA 8.1
Aggregate CGPA: 7.82 / 10.0
Backlogs: 0 (No backlogs)
Attendance: 87%

SCHOLARSHIPS & AWARDS:
1. NSP Scholarship (Merit-based): ₹25,000 per year (2 years)
   Total scholarship value: ₹50,000
2. College Merit Award: ₹5,000

CERTIFICATIONS:
1. NPTEL - Data Structures (Elite + Gold) - 12 weeks
2. Coursera - Machine Learning by Andrew Ng
3. Google Cloud Fundamentals
Govt certification: None (NSDC not applicable)

INTERNSHIP:
Summer Internship at Infosys, Bangalore (2 months)
Stipend: ₹15,000/month

PART-TIME WORK:
Freelance web development on Fiverr
Monthly earnings: ₹8,000
Active for: 6 months

COMMUNITY:
Member of college coding club
Years in city: 4
Reference: Prof. Shivakumar (HOD, CSE)
"""


def generate_sample_vendor_doc() -> str:
    """Generate a sample vendor daily sales register."""
    return """
DAILY SALES REGISTER — Ravi's Tea Stall
==========================================
Location: Keshwapur Circle, Hubli (Same location for 8 years)
Trade License: Municipal Corp License No: HMC/VND/2018/456

RENTAL:
Stall rent: ₹3,500/month (paid to Municipal Corporation)
Rent paid on time: 85% (missed 2 months in monsoon)
Months of rental history: 36

MONTHLY INCOME SUMMARY (Last 6 months):
Month       Working Days    Daily Avg    Monthly Total
Oct 2024        27          ₹900         ₹24,300
Sep 2024        25          ₹850         ₹21,250
Aug 2024        22          ₹700         ₹15,400
Jul 2024        20          ₹650         ₹13,000
Jun 2024        26          ₹880         ₹22,880
May 2024        28          ₹920         ₹25,760
Average daily income: ₹817
Seasonal variation: medium (monsoon dip)

UTILITY:
Electricity: HESCOM (Commercial) Bills per year: 12, on time: 80%
Water: Municipal supply
Gas: Commercial LPG (HP Gas) - 2 cylinders/month

SAVINGS:
Method: Chit fund (₹2,000/month contribution)
Months saving: 18 months
SHG member: No

COMMUNITY:
Member of: Hubli Vendors Association
Years in community: 8
References: Sunil (adjacent shop), Ravi (supplier), Meena (customer)
Local business reference: Wholesale supplier - Gupta Traders
"""


def generate_sample_homemaker_doc() -> str:
    """Generate a sample homemaker household diary."""
    return """
HOUSEHOLD BUDGET DIARY — Savita Devi
==========================================

HOUSEHOLD INCOME (Monthly):
Husband's salary: ₹22,000
Own tiffin service income: ₹7,000
Total household income: ₹29,000

HOUSEHOLD EXPENSES (Monthly):
Groceries: ₹6,000
Rent: ₹4,000
Children education: ₹3,000
Electricity bill: ₹800
Water: ₹200
Gas (LPG): ₹900
Transport: ₹1,500
Medical: ₹500
Miscellaneous: ₹2,000
Total household expenses: ₹18,900

Managing budget: Yes (maintains register since 2 years)
Number of dependents: 4 (2 children, 1 elderly parent, self)

MICRO ENTERPRISE: Tiffin Service
Running since: 24 months
Monthly revenue: ₹7,000
Customers: 8 regular tiffin subscribers (office workers nearby)

SAVINGS:
SHG: "Lakshmi Mahila Bachat Gat" — member since 3 years
Monthly contribution: ₹1,000
Total savings via SHG: ₹36,000

SKILLS:
PMKVY Certificate in Food Processing (Govt certified - Skill India)

UTILITY:
Electricity: BESCOM connection
Bills per year: 12 (on time: 95%)
Water: BWSSB
Gas: Indane LPG

COMMUNITY:
SHG member: Yes
Years in community: 10
References: Anita (SHG leader), Priya (neighbor), Suresh (tiffin customer)
Has local business reference: No

MOBILE:
Smartphone: Redmi (WhatsApp active)
Recharge: Monthly ₹199
Uses PhonePe for UPI payments
"""


def generate_sample_general_doc() -> str:
    """Generate a sample document for general (no bank) persona."""
    return """
IDENTITY DOCUMENTS — Lakshman Naik
==========================================

AADHAAR CARD:
Aadhaar Number: XXXX XXXX 4567
Name: Lakshman D. Naik
Date of Birth: 15-03-1985
Address: H.No 45, Tarihal Village, Hubli Taluk, Dharwad District
Gender: Male

VOTER ID:
EPIC Number: KA/09/XXX/XXXXXX
Name: Lakshman D Naik
Constituency: Hubli-Dharwad Central

RATION CARD:
Card Type: BPL (Below Poverty Line)
Card Number: KA-DW-00045678
Family members: 5
Valid since: 2019

PAN CARD: Not available

UTILITY BILLS:
Electricity: HESCOM connection (Domestic)
Consumer No: HB-98765
Bills paid: 8 per year (on time: 70%)
Water: Gram Panchayat — no formal bill
Gas: No piped gas, uses firewood and kerosene

MOBILE USAGE:
Phone: Basic feature phone (Nokia)
Recharge: Irregular, approximately ₹100-150/month
No smartphone, no UPI

RENTAL:
House rent: ₹1,500/month to landlord
Paying for: 4 years
On time: 80%

SAVINGS:
Method: Cash at home
Monthly savings: ₹300-500
No SHG membership
No bank account

COMMUNITY:
Lives in Tarihal village for 12 years
Employed as daily wage laborer (construction)
References: Manju (neighbor), Rajesh (contractor)
No group membership
"""


SAMPLE_GENERATORS = {
    "farmer": generate_sample_farmer_doc,
    "student": generate_sample_student_doc,
    "street_vendor": generate_sample_vendor_doc,
    "homemaker": generate_sample_homemaker_doc,
    "general_no_bank": generate_sample_general_doc,
}
