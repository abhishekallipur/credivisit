"""
OCR Engine for CrediVist — Scanned Document Processing
========================================================
Extracts text from scanned PDFs and images using multiple strategies:
  1. pytesseract (if Tesseract binary installed) — best accuracy
  2. PyPDF2 embedded text extraction — for digital/text PDFs
  3. PIL-based image analysis — basic metadata when OCR unavailable

Also includes:
  - ID card parsers (Aadhaar, PAN, Voter ID)
  - Marksheet table extractor
  - Table extraction from PDFs via tabula-py
  - Smart fallback chain for robustness
"""

import re
import io
import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ─── Capability Detection ───────────────────────────────────────────────────

_TESSERACT_AVAILABLE = False
_POPPLER_AVAILABLE = False
_TABULA_AVAILABLE = False

try:
    import pytesseract
    # Check common install paths on Windows
    _tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
            os.environ.get("USERNAME", "")
        ),
    ]
    for p in _tesseract_paths:
        if os.path.isfile(p):
            pytesseract.pytesseract.tesseract_cmd = p
            _TESSERACT_AVAILABLE = True
            break
    if not _TESSERACT_AVAILABLE:
        # Try default (on PATH)
        try:
            pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception:
            pass
except ImportError:
    pass

try:
    from PIL import Image, ImageFilter, ImageEnhance
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    # pdf2image needs poppler
    try:
        convert_from_bytes(b"%PDF-1.0\n1 0 obj\n<< >>\nendobj\n", fmt="png", dpi=50)
        _POPPLER_AVAILABLE = True
    except Exception:
        _POPPLER_AVAILABLE = False
except ImportError:
    _POPPLER_AVAILABLE = False

try:
    import tabula
    _TABULA_AVAILABLE = True
except ImportError:
    _TABULA_AVAILABLE = False


def get_ocr_capabilities() -> Dict[str, bool]:
    """Return which OCR capabilities are available."""
    return {
        "tesseract": _TESSERACT_AVAILABLE,
        "pil": _PIL_AVAILABLE,
        "poppler": _POPPLER_AVAILABLE,
        "tabula": _TABULA_AVAILABLE,
    }


# ─── Image Pre-processing ──────────────────────────────────────────────────

def preprocess_image_for_ocr(image: "Image.Image") -> "Image.Image":
    """
    Pre-process an image to improve OCR accuracy.
    - Convert to grayscale
    - Enhance contrast
    - Sharpen
    - Binarize (adaptive threshold via PIL)
    """
    if not _PIL_AVAILABLE:
        return image

    # Convert to grayscale
    img = image.convert("L")

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # Simple binarization: threshold at 128
    img = img.point(lambda x: 0 if x < 140 else 255, "1")

    # Convert back to grayscale for tesseract
    img = img.convert("L")

    return img


# ─── Core OCR Functions ────────────────────────────────────────────────────

def ocr_image(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Extract text from an image using the best available method.
    Falls back gracefully if Tesseract is unavailable.
    """
    if not _PIL_AVAILABLE:
        return ""

    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        logger.warning(f"Cannot open image: {e}")
        return ""

    # Strategy 1: Tesseract OCR (best quality)
    if _TESSERACT_AVAILABLE:
        try:
            processed = preprocess_image_for_ocr(img)
            text = pytesseract.image_to_string(
                processed,
                lang=lang,
                config="--psm 6 --oem 3",  # Assume uniform block of text
            )
            if text.strip():
                return text.strip()
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")

    # Strategy 2: PIL-based basic analysis (metadata extraction)
    return _extract_image_metadata(img)


def _extract_image_metadata(img: "Image.Image") -> str:
    """
    When OCR isn't available, extract what we can from image metadata
    and basic pixel analysis to provide contextual hints.
    """
    parts = []

    # Image metadata
    width, height = img.size
    parts.append(f"Image size: {width}x{height}")
    parts.append(f"Mode: {img.mode}")

    # EXIF data (may contain camera info, date, etc.)
    try:
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                if isinstance(value, str) and len(value) < 200:
                    parts.append(f"EXIF {tag_id}: {value}")
    except Exception:
        pass

    # Check if it looks like a document (predominantly white background)
    try:
        gray = img.convert("L")
        pixels = list(gray.getdata())
        avg_brightness = sum(pixels) / len(pixels)
        white_pct = sum(1 for p in pixels if p > 200) / len(pixels)

        if white_pct > 0.6:
            parts.append("Document type: Likely a scanned document (high white background)")
        elif white_pct > 0.3:
            parts.append("Document type: Mixed content document")
        else:
            parts.append("Document type: Photo or colored document")

        # Rough text density estimation
        dark_pixels = sum(1 for p in pixels if p < 100)
        text_density = dark_pixels / len(pixels)
        if text_density > 0.15:
            parts.append("Text density: High (likely text-heavy document)")
        elif text_density > 0.05:
            parts.append("Text density: Medium")
        else:
            parts.append("Text density: Low (may be mostly images/logos)")
    except Exception:
        pass

    return "\n".join(parts)


def ocr_pdf_pages(pdf_bytes: bytes, lang: str = "eng", max_pages: int = 20) -> str:
    """
    Extract text from a scanned PDF by converting pages to images and OCR-ing.
    Falls back to PyPDF2 if pdf2image/poppler unavailable.
    """
    all_text = ""

    # Strategy 1: pdf2image + Tesseract (best for scanned PDFs)
    if _POPPLER_AVAILABLE and _TESSERACT_AVAILABLE:
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt="png",
                                         first_page=1, last_page=max_pages)
            for i, img in enumerate(images):
                processed = preprocess_image_for_ocr(img)
                page_text = pytesseract.image_to_string(
                    processed, lang=lang,
                    config="--psm 6 --oem 3",
                )
                if page_text.strip():
                    all_text += f"\n--- Page {i+1} ---\n{page_text}\n"
            if all_text.strip():
                return all_text.strip()
        except Exception as e:
            logger.warning(f"pdf2image OCR failed: {e}")

    # Strategy 2: PyPDF2 text extraction (for text-based PDFs)
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for i, page in enumerate(reader.pages[:max_pages]):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                all_text += f"\n--- Page {i+1} ---\n{page_text}\n"
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {e}")

    # Strategy 3: If PDF has embedded images, try to extract them
    if not all_text.strip():
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages[:max_pages]:
                if hasattr(page, "images"):
                    for img_obj in page.images:
                        img_text = ocr_image(img_obj.data, lang=lang)
                        if img_text:
                            all_text += f"\n{img_text}\n"
        except Exception as e:
            logger.warning(f"PDF image extraction failed: {e}")

    return all_text.strip()


def extract_tables_from_pdf(pdf_bytes: bytes) -> List["pd.DataFrame"]:
    """
    Extract tables from PDF using tabula-py (Java-based, very accurate).
    Falls back to empty list if tabula is unavailable.
    """
    if not _TABULA_AVAILABLE:
        return []

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            temp_path = f.name

        try:
            import tabula
            tables = tabula.read_pdf(temp_path, pages="all", multiple_tables=True,
                                      silent=True)
            return [t for t in tables if len(t) > 0]
        finally:
            os.unlink(temp_path)
    except Exception as e:
        logger.warning(f"Tabula table extraction failed: {e}")
        return []


# ─── ID Card Parsers ────────────────────────────────────────────────────────

def parse_aadhaar_card(text: str) -> Dict[str, Any]:
    """
    Parse Aadhaar card text (from OCR or text PDF).
    Extracts: name, DOB, gender, aadhaar number, address.
    """
    data = {
        "document_type": "aadhaar",
        "has_aadhaar": True,
        "valid": False,
    }

    # Aadhaar number: 12 digits, often in groups of 4
    aadhaar_patterns = [
        r'(\d{4}\s*\d{4}\s*\d{4})',
        r'(\d{12})',
    ]
    for p in aadhaar_patterns:
        m = re.search(p, text)
        if m:
            num = re.sub(r'\s', '', m.group(1))
            if len(num) == 12:
                # Mask for privacy
                data["aadhaar_masked"] = f"XXXX XXXX {num[-4:]}"
                data["valid"] = True
                break

    # Name
    name_patterns = [
        r'(?:name|naam)[:\s]*([A-Z][a-zA-Z\s\.]+)',
        r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?)',
    ]
    for p in name_patterns:
        m = re.search(p, text, re.MULTILINE | re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) > 3 and len(name) < 80:
                data["name"] = name
                break

    # Date of Birth
    dob_patterns = [
        r'(?:DOB|Date of Birth|Birth|Year of Birth)[:\s]*([\d]{2}[/-][\d]{2}[/-][\d]{2,4})',
        r'(?:DOB|Date of Birth)[:\s]*([\d]{4})',
        r'(\d{2}/\d{2}/\d{4})',
    ]
    for p in dob_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            data["dob"] = m.group(1)
            break

    # Gender
    if re.search(r'\b(male|पुरुष)\b', text, re.IGNORECASE):
        data["gender"] = "Male"
    elif re.search(r'\b(female|महिला|स्त्री)\b', text, re.IGNORECASE):
        data["gender"] = "Female"

    # Address (rough extraction)
    addr_m = re.search(
        r'(?:address|पता)[:\s]*(.+?)(?:\n\n|\Z)',
        text, re.IGNORECASE | re.DOTALL
    )
    if addr_m:
        data["address"] = addr_m.group(1).strip()[:200]

    return data


def parse_pan_card(text: str) -> Dict[str, Any]:
    """
    Parse PAN card text.
    Extracts: PAN number, name, DOB.
    """
    data = {
        "document_type": "pan",
        "has_pan": True,
        "valid": False,
    }

    # PAN: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
    pan_match = re.search(r'([A-Z]{5}\d{4}[A-Z])', text.upper())
    if pan_match:
        data["pan_number"] = pan_match.group(1)
        data["valid"] = True

    # Name
    name_m = re.search(r'(?:name|naam)[:\s]*([A-Z][a-zA-Z\s\.]+)', text, re.IGNORECASE)
    if name_m:
        data["name"] = name_m.group(1).strip()

    # DOB
    dob_m = re.search(r'(\d{2}/\d{2}/\d{4})', text)
    if dob_m:
        data["dob"] = dob_m.group(1)

    return data


def parse_voter_id(text: str) -> Dict[str, Any]:
    """
    Parse Voter ID / EPIC card text.
    Extracts: EPIC number, name, address.
    """
    data = {
        "document_type": "voter_id",
        "has_voter_id": True,
        "valid": False,
    }

    # EPIC number: typically starts with state code letters + digits
    epic_patterns = [
        r'(?:EPIC|No)[:\s]*([A-Z]{2,3}/?\d{2,3}/?\w+/?\d+)',
        r'([A-Z]{3}\d{7})',
    ]
    for p in epic_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            data["epic_number"] = m.group(1)
            data["valid"] = True
            break

    # Name
    name_m = re.search(r'(?:name|elector)[:\s]*([A-Z][a-zA-Z\s\.]+)', text, re.IGNORECASE)
    if name_m:
        data["name"] = name_m.group(1).strip()

    # Age / DOB
    age_m = re.search(r'(?:age|आयु)[:\s]*(\d{2})', text, re.IGNORECASE)
    if age_m:
        data["age"] = int(age_m.group(1))

    return data


def parse_ration_card(text: str) -> Dict[str, Any]:
    """Parse Ration Card text."""
    data = {
        "document_type": "ration_card",
        "has_ration_card": True,
        "valid": False,
    }

    # Card number
    card_m = re.search(r'(?:card|no)[:\s]*([A-Z]{2}[-/]?\w+[-/]?\d+)', text, re.IGNORECASE)
    if card_m:
        data["card_number"] = card_m.group(1)
        data["valid"] = True

    # Card type
    if re.search(r'\bBPL\b|below poverty', text, re.IGNORECASE):
        data["card_type"] = "BPL"
    elif re.search(r'\bAPL\b|above poverty', text, re.IGNORECASE):
        data["card_type"] = "APL"
    elif re.search(r'\bAAY\b|antyodaya', text, re.IGNORECASE):
        data["card_type"] = "AAY"

    # Family members
    members_m = re.search(r'(?:members|family)[:\s]*(\d+)', text, re.IGNORECASE)
    if members_m:
        data["family_members"] = int(members_m.group(1))

    return data


# ─── Marksheet / Grade Card Parsers ─────────────────────────────────────────

def parse_marksheet(text: str) -> Dict[str, Any]:
    """
    Parse marksheet / grade card / transcript.
    Extracts: CGPA/percentage, subjects, backlogs, university, branch.
    """
    data = {
        "document_type": "marksheet",
    }

    # University / Institution
    uni_patterns = [
        r'(?:university|vishwavidyalaya|deemed)[:\s]*(.+?)(?:\n|$)',
        r'^([A-Z][A-Za-z\s]+(?:University|Institute|College))',
    ]
    for p in uni_patterns:
        m = re.search(p, text, re.IGNORECASE | re.MULTILINE)
        if m:
            data["university"] = m.group(1).strip()[:100]
            break

    # USN / Roll Number
    usn_patterns = [
        r'(?:USN|Roll\s*No|Reg\s*No|Enroll)[:\s.]*([A-Z0-9]{8,15})',
        r'(?:USN|Roll)[:\s.]*(\d{2}[A-Z]{2,4}\d{2,3}[A-Z]{2,3}\d{2,3})',
    ]
    for p in usn_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            data["usn"] = m.group(1)
            break

    # Branch / Department
    branch_patterns = [
        r'(?:branch|dept|department|programme|program|course)[:\s]*(.+?)(?:\n|$)',
        r'(Computer Science|Information Technology|Electronics|Mechanical|Civil|'
        r'Electrical|Chemical|Data Science|AI|Artificial Intelligence)',
    ]
    for p in branch_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            data["branch"] = m.group(1).strip()[:60]
            break

    # CGPA / SGPA extraction
    sgpas = []
    for m in re.finditer(r'(?:SGPA|SPI)(?!.*CGPA)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE):
        val = float(m.group(1))
        if 0 < val <= 10:
            sgpas.append(val)
    data["sgpa_list"] = sgpas

    # Aggregate CGPA
    cgpa_m = re.search(r'(?:CGPA|CPI|Aggregate|Overall|Cumulative)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if cgpa_m:
        val = float(cgpa_m.group(1))
        if 0 < val <= 10:
            data["cgpa"] = val
    elif sgpas:
        data["cgpa"] = round(np.mean(sgpas), 2)

    # Percentage
    pct_m = re.search(r'(?:percentage|percent|marks)[:\s]*(\d+\.?\d*)\s*%?', text, re.IGNORECASE)
    if pct_m:
        val = float(pct_m.group(1))
        if 0 < val <= 100:
            data["percentage"] = val

    # Semester count
    sem_matches = re.findall(r'sem(?:ester)?\s*(\d+)', text, re.IGNORECASE)
    if sem_matches:
        data["semesters"] = max(int(s) for s in sem_matches)

    # Subject-wise marks (table format)
    subjects = []
    # Pattern: Subject Name ... marks/grade
    for m in re.finditer(
        r'([A-Za-z][A-Za-z\s&]+?)\s+(\d{2,3})\s*/?\s*(\d{2,3})?\s*(P|F|AB|pass|fail)?',
        text, re.IGNORECASE
    ):
        subj = {
            "name": m.group(1).strip(),
            "marks": int(m.group(2)),
        }
        if m.group(3):
            subj["max_marks"] = int(m.group(3))
        if m.group(4):
            subj["status"] = m.group(4).upper()
        subjects.append(subj)
    data["subjects"] = subjects

    # Calculate percentage from subjects if not found directly
    if "percentage" not in data and subjects:
        total = sum(s["marks"] for s in subjects)
        max_total = sum(s.get("max_marks", 100) for s in subjects)
        if max_total > 0:
            data["percentage"] = round(total / max_total * 100, 1)

    # Backlogs
    failed = sum(1 for s in subjects if s.get("status") in ["F", "FAIL", "AB"])
    backlog_m = re.search(r'backlog[s]?[:\s]*(\d+)', text, re.IGNORECASE)
    if backlog_m:
        data["backlogs"] = int(backlog_m.group(1))
    elif re.search(r'no\s*backlog|0\s*backlog|all\s*pass', text, re.IGNORECASE):
        data["backlogs"] = 0
    else:
        data["backlogs"] = failed

    # Class / Division
    if re.search(r'first\s*class\s*with\s*dist|distinction', text, re.IGNORECASE):
        data["division"] = "Distinction"
    elif re.search(r'first\s*class', text, re.IGNORECASE):
        data["division"] = "First Class"
    elif re.search(r'second\s*class', text, re.IGNORECASE):
        data["division"] = "Second Class"

    return data


def parse_certificate(text: str) -> Dict[str, Any]:
    """
    Parse a certificate document (course completion, skill certification, etc.).
    """
    data = {
        "document_type": "certificate",
    }

    # Certificate title
    title_m = re.search(
        r'(?:certificate|certif)[:\s]+(?:of|in|for)[:\s]+(.+?)(?:\n|$)',
        text, re.IGNORECASE
    )
    if title_m:
        data["title"] = title_m.group(1).strip()[:100]

    # Platform
    platforms = {
        "nptel": "NPTEL", "coursera": "Coursera", "udemy": "Udemy",
        "edx": "edX", "swayam": "SWAYAM", "google": "Google",
        "microsoft": "Microsoft", "aws": "AWS", "nsdc": "NSDC",
        "pmkvy": "PMKVY", "skill india": "Skill India",
    }
    for kw, name in platforms.items():
        if kw in text.lower():
            data["platform"] = name
            break

    # Grade / Score
    grade_m = re.search(r'(?:grade|score|result)[:\s]*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if grade_m:
        data["grade"] = grade_m.group(1).strip()

    # Is government certification
    data["is_govt"] = any(k in text.lower() for k in ["nsdc", "pmkvy", "skill india",
                                                        "government", "govt"])

    # Date
    date_m = re.search(r'(?:date|issued)[:\s]*(\d{2}[/-]\d{2}[/-]\d{2,4})', text, re.IGNORECASE)
    if date_m:
        data["issue_date"] = date_m.group(1)

    return data


def parse_land_record(text: str) -> Dict[str, Any]:
    """
    Parse land record document (RTC, Patta, Khata).
    """
    data = {
        "document_type": "land_record",
    }

    # Survey / Khata number
    survey_m = re.search(r'(?:survey|sy|khata|patta)[:\s.]*(?:no)?[:\s.]*(\w+)', text, re.IGNORECASE)
    if survey_m:
        data["survey_no"] = survey_m.group(1)

    # Land area
    acres_m = re.search(r'(\d+\.?\d*)\s*(?:acres?|ac)', text, re.IGNORECASE)
    hectare_m = re.search(r'(\d+\.?\d*)\s*(?:hectares?|ha)', text, re.IGNORECASE)
    guntha_m = re.search(r'(\d+\.?\d*)\s*(?:gunthas?|guntha)', text, re.IGNORECASE)

    if acres_m:
        data["land_acres"] = float(acres_m.group(1))
    elif hectare_m:
        data["land_acres"] = float(hectare_m.group(1)) * 2.47
    elif guntha_m:
        data["land_acres"] = float(guntha_m.group(1)) / 40  # 40 guntha = 1 acre

    # Land type
    if re.search(r'irrigat', text, re.IGNORECASE):
        data["land_type"] = "irrigated"
    elif re.search(r'rain.?fed|dry', text, re.IGNORECASE):
        data["land_type"] = "rainfed"

    # Owner name
    owner_m = re.search(r'(?:owner|name|holder)[:\s]*([A-Z][a-zA-Z\s\.]+)', text, re.IGNORECASE)
    if owner_m:
        data["owner_name"] = owner_m.group(1).strip()

    return data


def parse_utility_bill(text: str) -> Dict[str, Any]:
    """Parse utility bill (electricity, water, gas)."""
    data = {
        "document_type": "utility_bill",
    }

    # Bill type
    if re.search(r'electric|bescom|hescom|msedcl|power|eb\s', text, re.IGNORECASE):
        data["utility_type"] = "electricity"
        data["has_electricity"] = True
    elif re.search(r'water|bwssb|jal\s', text, re.IGNORECASE):
        data["utility_type"] = "water"
        data["has_water"] = True
    elif re.search(r'gas|lpg|cylinder|piped\s*gas', text, re.IGNORECASE):
        data["utility_type"] = "gas"
        data["has_gas"] = True

    # Consumer number
    consumer_m = re.search(r'(?:consumer|account|conn)[:\s.]*(?:no)?[:\s.]*([A-Z]{0,3}[-/]?\d{5,15})',
                            text, re.IGNORECASE)
    if consumer_m:
        data["consumer_no"] = consumer_m.group(1)

    # Amount
    from src.document_analyzer import find_amounts
    amounts = find_amounts(text)
    if amounts:
        data["bill_amount"] = amounts[0]

    # Due date / payment status
    if re.search(r'paid|payment received|thank you', text, re.IGNORECASE):
        data["paid"] = True
    elif re.search(r'overdue|pending|due', text, re.IGNORECASE):
        data["paid"] = False

    return data


# ─── Smart Document Classifier ──────────────────────────────────────────────

DOCUMENT_TYPES = {
    "aadhaar": {
        "keywords": ["aadhaar", "aadhar", "unique identification", "uid",
                      "government of india", "enrolment"],
        "parser": parse_aadhaar_card,
    },
    "pan": {
        "keywords": ["permanent account number", "pan card", "income tax department"],
        "parser": parse_pan_card,
    },
    "voter_id": {
        "keywords": ["election commission", "voter", "epic", "elector"],
        "parser": parse_voter_id,
    },
    "ration_card": {
        "keywords": ["ration card", "bpl", "apl", "antyodaya", "food supply"],
        "parser": parse_ration_card,
    },
    "marksheet": {
        "keywords": ["marksheet", "grade card", "transcript", "semester",
                      "cgpa", "sgpa", "examination", "university",
                      "subject", "marks obtained"],
        "parser": parse_marksheet,
    },
    "certificate": {
        "keywords": ["certificate", "completion", "course", "certified",
                      "awarded to", "successfully completed"],
        "parser": parse_certificate,
    },
    "land_record": {
        "keywords": ["land record", "rtc", "patta", "khata", "survey no",
                      "land area", "agricultural land"],
        "parser": parse_land_record,
    },
    "utility_bill": {
        "keywords": ["electricity bill", "water bill", "gas bill",
                      "consumer no", "bill amount", "meter reading",
                      "billing period", "bescom", "hescom"],
        "parser": parse_utility_bill,
    },
}


def classify_document(text: str) -> Tuple[str, float, Dict]:
    """
    Classify a document and extract structured data from it.

    Returns:
        (document_type, confidence, extracted_data)
    """
    text_lower = text.lower()
    scores = {}

    for doc_type, config in DOCUMENT_TYPES.items():
        count = sum(1 for kw in config["keywords"] if kw in text_lower)
        scores[doc_type] = count / len(config["keywords"]) if config["keywords"] else 0

    if not scores or max(scores.values()) == 0:
        return "unknown", 0.0, {}

    best_type = max(scores, key=scores.get)
    confidence = min(scores[best_type] * 4, 0.95)

    # Run the appropriate parser
    parser = DOCUMENT_TYPES[best_type]["parser"]
    parsed_data = parser(text)

    return best_type, confidence, parsed_data


# ─── Enhanced File Processing ───────────────────────────────────────────────

def process_file_with_ocr(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Process a file with full OCR support. Handles:
    - Text PDFs (PyPDF2)
    - Scanned PDFs (pdf2image + Tesseract)
    - Images (Tesseract OCR)
    - CSV / Excel (pandas)
    - Text files

    Returns:
        Dict with: text, dataframe, document_type, parsed_data,
                    ocr_used, tables, warnings
    """
    import pandas as pd

    ext = os.path.splitext(filename)[1].lower()
    result = {
        "filename": filename,
        "text": "",
        "dataframe": None,
        "document_type": "unknown",
        "parsed_data": {},
        "ocr_used": False,
        "tables": [],
        "warnings": [],
    }

    # ── Image files ──────────────────────────────────────────────────
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]:
        text = ocr_image(file_bytes)
        result["text"] = text
        result["ocr_used"] = _TESSERACT_AVAILABLE

        if not text.strip() and not _TESSERACT_AVAILABLE:
            result["warnings"].append(
                f"Tesseract OCR not installed — cannot extract text from image '{filename}'. "
                "Install Tesseract for full OCR support."
            )

    # ── PDF files ────────────────────────────────────────────────────
    elif ext == ".pdf":
        # First try PyPDF2 for text-based PDFs
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            result["text"] = text
        except Exception:
            pass

        # If very little text, it's probably a scanned PDF — use OCR
        if len(result["text"].strip()) < 50:
            ocr_text = ocr_pdf_pages(file_bytes)
            if ocr_text:
                result["text"] = ocr_text
                result["ocr_used"] = True
            elif not _TESSERACT_AVAILABLE:
                result["warnings"].append(
                    f"PDF '{filename}' appears to be scanned (image-based). "
                    "Install Tesseract OCR for text extraction from scanned documents."
                )

        # Try table extraction
        tables = extract_tables_from_pdf(file_bytes)
        if tables:
            result["tables"] = tables
            result["dataframe"] = tables[0]  # Use first table as primary

    # ── CSV files ────────────────────────────────────────────────────
    elif ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            result["dataframe"] = df
            result["text"] = df.to_string()
        except Exception:
            result["text"] = file_bytes.decode("utf-8", errors="ignore")

    # ── Excel files ──────────────────────────────────────────────────
    elif ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
            result["dataframe"] = df
            result["text"] = df.to_string()
        except Exception:
            pass

    # ── Text / JSON files ────────────────────────────────────────────
    elif ext == ".txt":
        result["text"] = file_bytes.decode("utf-8", errors="ignore")

    elif ext == ".json":
        try:
            data = json.loads(file_bytes.decode("utf-8"))
            result["text"] = json.dumps(data, indent=2)
        except Exception:
            result["text"] = file_bytes.decode("utf-8", errors="ignore")

    else:
        try:
            result["text"] = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            result["warnings"].append(f"Unsupported file format: {ext}")

    # ── Classify the document ────────────────────────────────────────
    if result["text"].strip():
        doc_type, doc_conf, parsed_data = classify_document(result["text"])
        result["document_type"] = doc_type
        result["parsed_data"] = parsed_data
        result["classification_confidence"] = doc_conf

    return result


# ─── Merge parsed data into persona-compatible format ────────────────────────

def merge_parsed_into_persona_data(
    parsed_docs: List[Dict[str, Any]],
    persona: str,
) -> Dict[str, Any]:
    """
    Merge structured data from multiple parsed documents into
    a flat dict compatible with persona scoring functions.
    """
    merged = {}

    for doc in parsed_docs:
        parsed = doc.get("parsed_data", {})
        doc_type = doc.get("document_type", "unknown")

        if doc_type == "aadhaar":
            merged["has_aadhaar"] = parsed.get("valid", False)
            if "name" in parsed:
                merged["applicant_name"] = parsed["name"]

        elif doc_type == "pan":
            merged["has_pan"] = parsed.get("valid", False)

        elif doc_type == "voter_id":
            merged["has_voter_id"] = parsed.get("valid", False)

        elif doc_type == "ration_card":
            merged["has_ration_card"] = parsed.get("valid", False)
            if "family_members" in parsed:
                merged["dependents"] = parsed["family_members"]

        elif doc_type == "marksheet":
            if "cgpa" in parsed:
                merged["score_type"] = "cgpa"
                merged["score_value"] = parsed["cgpa"]
            elif "percentage" in parsed:
                merged["score_type"] = "percentage"
                merged["score_value"] = parsed["percentage"]

            if "backlogs" in parsed:
                merged["backlog_count"] = parsed["backlogs"]

            if "branch" in parsed:
                branch_lower = parsed["branch"].lower()
                high_demand = ["computer", "cse", "it", "data science", "ai",
                               "electronics", "ece"]
                if any(k in branch_lower for k in high_demand):
                    merged["branch_demand"] = "high"
                else:
                    merged["branch_demand"] = "medium"

            if "university" in parsed:
                uni_lower = parsed["university"].lower()
                if any(k in uni_lower for k in ["iit", "nit", "iiit", "bits"]):
                    merged["institution_tier"] = 1
                elif any(k in uni_lower for k in ["vtu", "anna", "mumbai", "pune"]):
                    merged["institution_tier"] = 2
                else:
                    merged["institution_tier"] = 3

            if "usn" in parsed:
                merged["education_level"] = "ug"

            if "semesters" in parsed:
                merged["education_level"] = "ug" if parsed["semesters"] <= 8 else "pg"

        elif doc_type == "certificate":
            if "platform" in parsed:
                existing = merged.get("platform_certs", [])
                existing.append(parsed["platform"])
                merged["platform_certs"] = existing
                merged["cert_count"] = len(existing)

            if parsed.get("is_govt"):
                merged["has_govt_certification"] = True

        elif doc_type == "land_record":
            merged["owns_land"] = True
            if "land_acres" in parsed:
                merged["land_acres"] = parsed["land_acres"]
            if "owner_name" in parsed:
                merged["applicant_name"] = parsed["owner_name"]

        elif doc_type == "utility_bill":
            for key in ["has_electricity", "has_water", "has_gas"]:
                if parsed.get(key):
                    merged[key] = True
            if "bill_amount" in parsed:
                merged["bills_per_year"] = merged.get("bills_per_year", 0) + 6
            if "paid" in parsed:
                merged["on_time_pct"] = 90 if parsed["paid"] else 60

    return merged
