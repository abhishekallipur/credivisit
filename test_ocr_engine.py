"""
Test OCR Engine & Enhanced Document Analyzer
=============================================
Tests:
  1. OCR capability detection
  2. Document classification (marksheet, ID cards, certificates, etc.)
  3. ID card parsing (Aadhaar, PAN, Voter ID, Ration Card)
  4. Marksheet parsing
  5. Certificate parsing
  6. Land record parsing
  7. Utility bill parsing
  8. Merge parsed data into persona format
  9. End-to-end: image/scanned doc → persona score
  10. Integration with existing document_analyzer pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ocr_engine import (
    get_ocr_capabilities,
    classify_document,
    parse_aadhaar_card,
    parse_pan_card,
    parse_voter_id,
    parse_ration_card,
    parse_marksheet,
    parse_certificate,
    parse_land_record,
    parse_utility_bill,
    merge_parsed_into_persona_data,
    process_file_with_ocr,
)
from src.document_analyzer import (
    analyze_documents,
    auto_detect_persona,
    SAMPLE_GENERATORS,
)
from src.alternative_profiles import compute_persona_score


def test_ocr_capabilities():
    """Test that capability detection works without errors."""
    caps = get_ocr_capabilities()
    assert isinstance(caps, dict)
    assert "tesseract" in caps
    assert "pil" in caps
    assert "poppler" in caps
    assert "tabula" in caps
    print(f"  OCR Capabilities: {caps}")
    print("  ✓ OCR capability detection works")


def test_aadhaar_parsing():
    """Test Aadhaar card text parsing."""
    aadhaar_text = """
    GOVERNMENT OF INDIA
    UNIQUE IDENTIFICATION AUTHORITY
    
    Name: Ramesh Kumar Patil
    Date of Birth: 15/06/1985
    Gender: Male
    
    Aadhaar Number: 8765 4321 0987
    
    Address: H.No 23, 2nd Cross, Keshwapur
    Hubli, Dharwad District, Karnataka - 580023
    """
    result = parse_aadhaar_card(aadhaar_text)
    assert result["has_aadhaar"] is True
    assert result["valid"] is True
    assert result["aadhaar_masked"] == "XXXX XXXX 0987"
    assert "gender" in result
    assert result["gender"] == "Male"
    print(f"  Aadhaar: valid={result['valid']}, masked={result['aadhaar_masked']}")
    print("  ✓ Aadhaar parsing works")


def test_pan_parsing():
    """Test PAN card text parsing."""
    pan_text = """
    INCOME TAX DEPARTMENT
    PERMANENT ACCOUNT NUMBER CARD
    
    Name: RAMESH KUMAR PATIL
    PAN: ABCDE1234F
    Date of Birth: 15/06/1985
    """
    result = parse_pan_card(pan_text)
    assert result["has_pan"] is True
    assert result["valid"] is True
    assert result["pan_number"] == "ABCDE1234F"
    print(f"  PAN: valid={result['valid']}, number={result['pan_number']}")
    print("  ✓ PAN parsing works")


def test_voter_id_parsing():
    """Test Voter ID parsing."""
    voter_text = """
    ELECTION COMMISSION OF INDIA
    ELECTOR'S PHOTO IDENTITY CARD
    
    Name: Ramesh Kumar Patil
    EPIC Number: ABC1234567
    Age: 38
    """
    result = parse_voter_id(voter_text)
    assert result["has_voter_id"] is True
    assert result["valid"] is True
    print(f"  Voter ID: valid={result['valid']}")
    print("  ✓ Voter ID parsing works")


def test_ration_card_parsing():
    """Test Ration Card parsing."""
    ration_text = """
    DEPARTMENT OF FOOD & CIVIL SUPPLIES
    RATION CARD
    
    Card No: KA-DW-00012345
    Category: BPL (Below Poverty Line)
    Family Members: 5
    """
    result = parse_ration_card(ration_text)
    assert result["has_ration_card"] is True
    assert result["valid"] is True
    assert result.get("card_type") == "BPL"
    assert result.get("family_members") == 5
    print(f"  Ration Card: valid={result['valid']}, type={result.get('card_type')}")
    print("  ✓ Ration Card parsing works")


def test_marksheet_parsing():
    """Test marksheet / grade card parsing."""
    marksheet_text = """
    VISVESVARAYA TECHNOLOGICAL UNIVERSITY
    SEMESTER GRADE CARD
    
    University: VTU, Belagavi
    USN: 2VB21CS045
    Branch: Computer Science & Engineering
    
    Semester 1: SGPA 7.2
    Semester 2: SGPA 7.5
    Semester 3: SGPA 8.0
    Semester 4: SGPA 7.8
    Semester 5: SGPA 8.3
    Semester 6: SGPA 8.1
    
    Aggregate CGPA: 7.82
    Backlogs: 0 (No backlogs)
    
    Result: First Class with Distinction
    
    Engineering Mathematics   85/100  P
    Data Structures          78/100  P
    Computer Networks        92/100  P
    Operating Systems        88/100  P
    """
    result = parse_marksheet(marksheet_text)
    assert result["document_type"] == "marksheet"
    assert abs(result.get("cgpa", 0) - 7.82) < 0.01
    assert result.get("backlogs", -1) == 0
    assert len(result.get("sgpa_list", [])) == 6
    assert result.get("semesters") == 6
    assert len(result.get("subjects", [])) >= 3
    assert result.get("division") == "Distinction"
    print(f"  Marksheet: CGPA={result['cgpa']}, backlogs={result['backlogs']}, "
          f"subjects={len(result.get('subjects', []))}, semesters={result.get('semesters')}")
    print("  ✓ Marksheet parsing works")


def test_certificate_parsing():
    """Test certificate document parsing."""
    cert_text = """
    CERTIFICATE OF COMPLETION
    
    This is to certify that Ramesh Kumar has
    successfully completed the course:
    
    Certificate of Completion in Machine Learning
    
    Platform: Coursera
    Grade: 92%
    Date Issued: 15/08/2024
    """
    result = parse_certificate(cert_text)
    assert result["document_type"] == "certificate"
    assert result.get("platform") == "Coursera"
    print(f"  Certificate: platform={result.get('platform')}")
    print("  ✓ Certificate parsing works")


def test_land_record_parsing():
    """Test land record parsing."""
    land_text = """
    LAND RECORD CERTIFICATE (RTC)
    
    Survey No: 45/A
    Khata No: 123
    Owner Name: Ramesh B. Patil
    Land Area: 4.5 Acres
    Land Type: Agricultural - Irrigated
    """
    result = parse_land_record(land_text)
    assert result["document_type"] == "land_record"
    assert abs(result.get("land_acres", 0) - 4.5) < 0.01
    assert result.get("land_type") == "irrigated"
    print(f"  Land Record: acres={result.get('land_acres')}, type={result.get('land_type')}")
    print("  ✓ Land record parsing works")


def test_utility_bill_parsing():
    """Test utility bill parsing."""
    bill_text = """
    HESCOM ELECTRICITY BILL
    
    Consumer No: HB-12345
    Billing Period: Oct 2024
    Meter Reading: 4567
    Bill Amount: ₹1,250
    Due Date: 15/11/2024
    Status: Payment received. Thank you.
    """
    result = parse_utility_bill(bill_text)
    assert result["document_type"] == "utility_bill"
    assert result.get("utility_type") == "electricity"
    assert result.get("has_electricity") is True
    assert result.get("paid") is True
    print(f"  Utility Bill: type={result.get('utility_type')}, paid={result.get('paid')}")
    print("  ✓ Utility bill parsing works")


def test_document_classification():
    """Test automatic document classification."""
    # Marksheet
    doc_type, conf, data = classify_document(
        "University Examination Semester Grade Card CGPA 8.5 Subject marks obtained"
    )
    assert doc_type == "marksheet"
    print(f"  Marksheet classification: type={doc_type}, conf={conf:.2f}")

    # Aadhaar
    doc_type, conf, data = classify_document(
        "Government of India Unique Identification Authority of India Aadhaar 1234 5678 9012"
    )
    assert doc_type == "aadhaar"
    print(f"  Aadhaar classification: type={doc_type}, conf={conf:.2f}")

    # Certificate
    doc_type, conf, data = classify_document(
        "Certificate of Completion. Successfully completed the course. Awarded to student."
    )
    assert doc_type == "certificate"
    print(f"  Certificate classification: type={doc_type}, conf={conf:.2f}")

    # Land record
    doc_type, conf, data = classify_document(
        "Land Record RTC Patta Survey No agricultural land area 5 acres"
    )
    assert doc_type == "land_record"
    print(f"  Land record classification: type={doc_type}, conf={conf:.2f}")

    print("  ✓ Document classification works for all types")


def test_merge_student_data():
    """Test merging parsed documents into student persona format."""
    parsed_docs = [
        {
            "document_type": "marksheet",
            "parsed_data": {
                "document_type": "marksheet",
                "cgpa": 8.2,
                "backlogs": 0,
                "branch": "Computer Science & Engineering",
                "university": "VTU",
                "usn": "2VB21CS045",
                "semesters": 6,
                "division": "First Class",
            },
        },
        {
            "document_type": "certificate",
            "parsed_data": {
                "document_type": "certificate",
                "platform": "NPTEL",
                "is_govt": False,
            },
        },
        {
            "document_type": "certificate",
            "parsed_data": {
                "document_type": "certificate",
                "platform": "Coursera",
                "is_govt": False,
            },
        },
    ]
    merged = merge_parsed_into_persona_data(parsed_docs, "student")
    assert merged["score_type"] == "cgpa"
    assert merged["score_value"] == 8.2
    assert merged["backlog_count"] == 0
    assert merged["branch_demand"] == "high"
    assert merged["institution_tier"] == 2  # VTU
    assert merged["education_level"] == "ug"
    assert len(merged["platform_certs"]) == 2
    assert merged["cert_count"] == 2
    print(f"  Merged student: CGPA={merged['score_value']}, backlogs={merged['backlog_count']}, "
          f"certs={merged['cert_count']}, branch={merged['branch_demand']}")
    print("  ✓ Student data merge works")


def test_merge_farmer_data():
    """Test merging parsed documents into farmer persona format."""
    parsed_docs = [
        {
            "document_type": "land_record",
            "parsed_data": {
                "document_type": "land_record",
                "land_acres": 5.0,
                "land_type": "irrigated",
                "owner_name": "Ramesh Patil",
            },
        },
        {
            "document_type": "utility_bill",
            "parsed_data": {
                "document_type": "utility_bill",
                "utility_type": "electricity",
                "has_electricity": True,
                "bill_amount": 1250,
                "paid": True,
            },
        },
    ]
    merged = merge_parsed_into_persona_data(parsed_docs, "farmer")
    assert merged["owns_land"] is True
    assert merged["land_acres"] == 5.0
    assert merged["has_electricity"] is True
    assert merged["on_time_pct"] == 90
    print(f"  Merged farmer: land={merged['land_acres']} acres, electricity={merged['has_electricity']}")
    print("  ✓ Farmer data merge works")


def test_merge_general_data():
    """Test merging parsed ID documents for general persona."""
    parsed_docs = [
        {
            "document_type": "aadhaar",
            "parsed_data": {
                "document_type": "aadhaar",
                "has_aadhaar": True,
                "valid": True,
                "aadhaar_masked": "XXXX XXXX 4567",
                "name": "Lakshman Naik",
            },
        },
        {
            "document_type": "voter_id",
            "parsed_data": {
                "document_type": "voter_id",
                "has_voter_id": True,
                "valid": True,
            },
        },
        {
            "document_type": "ration_card",
            "parsed_data": {
                "document_type": "ration_card",
                "has_ration_card": True,
                "valid": True,
                "card_type": "BPL",
                "family_members": 5,
            },
        },
    ]
    merged = merge_parsed_into_persona_data(parsed_docs, "general_no_bank")
    assert merged["has_aadhaar"] is True
    assert merged["has_voter_id"] is True
    assert merged["has_ration_card"] is True
    assert merged["dependents"] == 5
    print(f"  Merged general: aadhaar={merged['has_aadhaar']}, voter={merged['has_voter_id']}, "
          f"ration={merged['has_ration_card']}, dependents={merged['dependents']}")
    print("  ✓ General (no bank) data merge works")


def test_end_to_end_student():
    """End-to-end: student marksheet text → score."""
    sample_text = SAMPLE_GENERATORS["student"]()
    files = [("marksheet.txt", sample_text.encode("utf-8"))]
    analysis = analyze_documents(files, persona="student")

    assert analysis["detected_persona"] == "student"
    assert len(analysis["extracted_data"]) > 0

    # Check if structured parsing added document types
    doc_types = analysis.get("detected_document_types", [])
    print(f"  Detected doc types: {doc_types}")

    # Compute score
    result = compute_persona_score("student", analysis["extracted_data"])
    print(f"  Student score: {result['trust_score']:.0f} ({result['grade']}), "
          f"confidence={result['confidence']:.0%}")
    assert 300 <= result["trust_score"] <= 900
    print("  ✓ End-to-end student pipeline works")


def test_end_to_end_farmer():
    """End-to-end: farmer land record text → score."""
    sample_text = SAMPLE_GENERATORS["farmer"]()
    files = [("land_record.txt", sample_text.encode("utf-8"))]
    analysis = analyze_documents(files, persona="farmer")

    assert analysis["detected_persona"] == "farmer"
    result = compute_persona_score("farmer", analysis["extracted_data"])
    print(f"  Farmer score: {result['trust_score']:.0f} ({result['grade']}), "
          f"confidence={result['confidence']:.0%}")
    assert 300 <= result["trust_score"] <= 900
    print("  ✓ End-to-end farmer pipeline works")


def test_end_to_end_general():
    """End-to-end: general ID docs → score."""
    sample_text = SAMPLE_GENERATORS["general_no_bank"]()
    files = [("id_docs.txt", sample_text.encode("utf-8"))]
    analysis = analyze_documents(files, persona="general_no_bank")

    assert analysis["detected_persona"] == "general_no_bank"
    result = compute_persona_score("general_no_bank", analysis["extracted_data"])
    print(f"  General score: {result['trust_score']:.0f} ({result['grade']}), "
          f"confidence={result['confidence']:.0%}")
    assert 300 <= result["trust_score"] <= 900
    print("  ✓ End-to-end general pipeline works")


def test_multi_document_merge():
    """Test uploading multiple documents of different types together."""
    marksheet = """
    SEMESTER GRADE CARD
    University: VTU
    USN: 2VB21CS045
    Branch: Computer Science & Engineering
    Aggregate CGPA: 8.5
    Backlogs: 0
    Attendance: 90%
    Semester 6 examination
    """
    certificate = """
    CERTIFICATE OF COMPLETION
    This is to certify that the student has
    successfully completed the course on NPTEL
    Machine Learning - Elite + Gold
    Certificate in Data Science
    """
    files = [
        ("marksheet.txt", marksheet.encode("utf-8")),
        ("nptel_cert.txt", certificate.encode("utf-8")),
    ]
    analysis = analyze_documents(files, persona="student")
    data = analysis["extracted_data"]

    print(f"  Multi-doc: CGPA={data.get('score_value')}, backlogs={data.get('backlog_count')}, "
          f"certs={data.get('cert_count')}, branch_demand={data.get('branch_demand')}")

    result = compute_persona_score("student", data)
    print(f"  Score: {result['trust_score']:.0f} ({result['grade']})")
    assert 300 <= result["trust_score"] <= 900
    print("  ✓ Multi-document merge and scoring works")


def test_process_file_txt():
    """Test process_file_with_ocr with a plain text file."""
    text = "This is a student marksheet. CGPA: 8.0. University examination. Semester 4."
    result = process_file_with_ocr(text.encode("utf-8"), "test.txt")
    assert result["text"].strip() == text
    assert result["document_type"] == "marksheet"
    print(f"  Process file: type={result['document_type']}")
    print("  ✓ process_file_with_ocr works with text files")


if __name__ == "__main__":
    tests = [
        ("OCR Capabilities", test_ocr_capabilities),
        ("Aadhaar Parsing", test_aadhaar_parsing),
        ("PAN Parsing", test_pan_parsing),
        ("Voter ID Parsing", test_voter_id_parsing),
        ("Ration Card Parsing", test_ration_card_parsing),
        ("Marksheet Parsing", test_marksheet_parsing),
        ("Certificate Parsing", test_certificate_parsing),
        ("Land Record Parsing", test_land_record_parsing),
        ("Utility Bill Parsing", test_utility_bill_parsing),
        ("Document Classification", test_document_classification),
        ("Merge Student Data", test_merge_student_data),
        ("Merge Farmer Data", test_merge_farmer_data),
        ("Merge General Data", test_merge_general_data),
        ("E2E Student", test_end_to_end_student),
        ("E2E Farmer", test_end_to_end_farmer),
        ("E2E General", test_end_to_end_general),
        ("Multi-Document Merge", test_multi_document_merge),
        ("Process File TXT", test_process_file_txt),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n🧪 {name}...")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
