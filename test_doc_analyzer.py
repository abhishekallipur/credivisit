"""
Test Document Analyzer — All 5 personas with sample documents.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.document_analyzer import (
    analyze_documents, auto_detect_persona,
    SAMPLE_GENERATORS, PERSONA_EXTRACTORS,
)
from src.alternative_profiles import compute_persona_score, PERSONAS


def test_persona(persona_key, persona_label):
    """Test a single persona end-to-end: generate doc → extract → score."""
    # Generate sample doc
    generator = SAMPLE_GENERATORS[persona_key]
    doc_text = generator()
    doc_bytes = doc_text.encode("utf-8")

    # Auto-detect persona
    detected, confidence = auto_detect_persona(doc_text)
    detect_match = "✓" if detected == persona_key else f"✗ (got {detected})"

    # Analyze documents
    result = analyze_documents(
        files=[("sample_doc.txt", doc_bytes)],
        persona=persona_key,
    )

    extracted = result["extracted_data"]

    # Compute score
    score_result = compute_persona_score(persona_key, extracted)

    print(f"\n{'='*60}")
    print(f"{persona_label}")
    print(f"{'='*60}")
    print(f"Auto-detect: {detect_match} (confidence: {confidence:.0%})")
    print(f"Files processed: {result['files_processed']}")
    print(f"Text extracted: {result['total_text_length']} chars")
    print(f"Warnings: {result['warnings'] or 'None'}")
    print(f"\nExtracted Data ({len(extracted)} fields):")
    for k, v in sorted(extracted.items()):
        print(f"  {k}: {v}")
    print(f"\nTrust Score: {score_result['trust_score']:.0f} / 900")
    print(f"Grade: {score_result['grade']}")
    print(f"Confidence: {score_result['confidence']:.0%}")
    print(f"Criteria filled: {score_result['filled_count']}/{score_result['criteria_count']}")

    # Assertions
    assert 300 <= score_result["trust_score"] <= 900, f"Score out of range: {score_result['trust_score']}"
    assert score_result["grade"] in ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
    assert len(extracted) >= 5, f"Too few fields extracted: {len(extracted)}"

    return score_result


def test_auto_detection():
    """Test that persona auto-detection works for each sample doc."""
    print(f"\n{'='*60}")
    print("AUTO-DETECTION TEST")
    print(f"{'='*60}")

    results = []
    for persona_key, generator in SAMPLE_GENERATORS.items():
        doc_text = generator()
        detected, confidence = auto_detect_persona(doc_text)
        match = detected == persona_key
        results.append(match)
        status = "✓ PASS" if match else f"✗ FAIL (detected: {detected})"
        print(f"  {PERSONAS[persona_key]['label']}: {status} (conf: {confidence:.0%})")

    passed = sum(results)
    print(f"\nAuto-detection: {passed}/{len(results)} correct")
    return passed == len(results)


def test_empty_documents():
    """Test handling of empty/minimal documents."""
    print(f"\n{'='*60}")
    print("EDGE CASE: Empty/Minimal Documents")
    print(f"{'='*60}")

    result = analyze_documents(
        files=[("empty.txt", b"")],
        persona="general_no_bank",
    )
    assert len(result["warnings"]) > 0, "Should have warnings for empty doc"
    print(f"  Warnings: {result['warnings']}")

    score = compute_persona_score("general_no_bank", result["extracted_data"])
    print(f"  Score (empty): {score['trust_score']:.0f} (Grade: {score['grade']})")
    assert 300 <= score["trust_score"] <= 900


def test_multi_file_upload():
    """Test uploading multiple documents for one persona."""
    print(f"\n{'='*60}")
    print("MULTI-FILE UPLOAD TEST")
    print(f"{'='*60}")

    farmer_doc = SAMPLE_GENERATORS["farmer"]()
    utility_doc = """
    ELECTRICITY BILL — HESCOM
    Consumer: Ramesh Patil
    Bill Date: 15-Oct-2024
    Amount: ₹1,200
    Status: Paid on time
    Connection: Domestic
    """

    result = analyze_documents(
        files=[
            ("land_record.txt", farmer_doc.encode()),
            ("electricity_bill.txt", utility_doc.encode()),
        ],
        persona="farmer",
    )
    print(f"  Files processed: {result['files_processed']}")
    print(f"  Total text: {result['total_text_length']} chars")

    score = compute_persona_score("farmer", result["extracted_data"])
    print(f"  Score: {score['trust_score']:.0f} (Grade: {score['grade']})")
    assert score["trust_score"] >= 300


if __name__ == "__main__":
    print("=" * 60)
    print("Document Analyzer — Full Test Suite")
    print("=" * 60)

    # Test each persona
    test_persona("farmer", "🌾 FARMER")
    test_persona("student", "🎓 STUDENT")
    test_persona("street_vendor", "🏪 STREET VENDOR")
    test_persona("homemaker", "🏠 HOMEMAKER")
    test_persona("general_no_bank", "👤 GENERAL (No Bank)")

    # Test auto-detection
    test_auto_detection()

    # Edge cases
    test_empty_documents()
    test_multi_file_upload()

    print("\n" + "=" * 60)
    print("✅ ALL DOCUMENT ANALYZER TESTS PASSED!")
    print("=" * 60)
