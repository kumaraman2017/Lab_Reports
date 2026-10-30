import os
import sqlite3
from app import (
    init_db,
    insert_lab_results,
    fetch_all_results,
    update_lab_result,
    clean_extracted_item,
    normalize_test_name,
    get_latest_result,
    get_test_history,
    list_out_of_range_results,
    load_sample_health_data,
    clear_all_results,
    run_agent_turn,
    get_api_key,
    SAFETY_NOTICE,
    _active_has_critical,
    reset_tool_execution_logs,
)

def run_tests():
    print("=== 1. Testing Database & Schema Initialization ===")
    init_db("test_health.db")
    conn = sqlite3.connect("test_health.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(lab_results)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}
    conn.close()

    expected_cols = [
        "id", "report_date", "raw_test_name", "normalized_test_name",
        "value", "unit", "reference_range", "confidence", "is_critical", "raw_extracted_text"
    ]
    for col in expected_cols:
        assert col in columns, f"Missing expected column: {col}"
    print("[PASS] SQLite schema verified!")

    print("\n=== 2. Testing Defensive Data Cleaning Layer ===")
    # Alias tests
    assert normalize_test_name("  hba1c  ") == "A1C"
    assert normalize_test_name("Hemoglobin A1C") == "A1C"
    assert normalize_test_name("fasting blood sugar") == "Glucose, Fasting"
    assert normalize_test_name("fbs") == "Glucose, Fasting"
    assert normalize_test_name("cholesterol") == "Total Cholesterol"
    print("[PASS] Alias mapping verified!")

    # Confidence tagging tests
    item_good = {
        "test_name": "  hba1c ",
        "value": "5.6",
        "unit": "%",
        "reference_range": "4.0 - 5.6",
        "report_date": "2026-09-18",
        "confidence": "HIGH",
        "is_critical": False,
    }
    cleaned_good = clean_extracted_item(item_good)
    assert cleaned_good["normalized_test_name"] == "A1C"
    assert cleaned_good["value"] == 5.6
    assert cleaned_good["confidence"] == "HIGH"

    # Missing unit -> LOW confidence
    item_missing_unit = {
        "test_name": "fbs",
        "value": 95,
        "unit": "  ",
        "reference_range": "70-99",
        "report_date": "2026-09-18",
    }
    cleaned_missing_unit = clean_extracted_item(item_missing_unit)
    assert cleaned_missing_unit["confidence"] == "LOW"

    # Unreadable value -> LOW confidence and value None
    item_unreadable_value = {
        "test_name": "cholesterol",
        "value": "unreadable_blur",
        "unit": "mg/dL",
        "reference_range": "< 200",
        "report_date": "2026-09-18",
    }
    cleaned_unreadable = clean_extracted_item(item_unreadable_value)
    assert cleaned_unreadable["confidence"] == "LOW"
    assert cleaned_unreadable["value"] is None
    print("[PASS] Defensive cleaning & confidence tagging verified!")

    print("\n=== 3. Testing Sample Data Loading & Deterministic Tools ===")
    clear_all_results()
    load_sample_health_data()

    # Tool 1: get_latest_result
    latest_a1c = get_latest_result("A1C")
    assert latest_a1c["status"] == "success"
    assert latest_a1c["record"]["value"] == 6.1
    assert latest_a1c["record"]["report_date"] == "2026-09-18"
    print("[PASS] get_latest_result verified!")

    # Tool 2: get_test_history (Deterministic Python Math)
    history_fbs = get_test_history("FBS")
    assert history_fbs["status"] == "success"
    assert history_fbs["record_count"] == 2
    # Baseline 135 -> Recent 110: Delta = -25.0, percent = -18.52%
    assert history_fbs["overall_delta"] == -25.0
    assert history_fbs["overall_percentage_change"] == -18.52
    assert history_fbs["overall_trend"] == "decreasing"
    print("[PASS] get_test_history math (delta, %, trend) verified!")

    # Tool 3: list_out_of_range_results
    out_of_range = list_out_of_range_results("2026-09-18")
    assert len(out_of_range) > 0
    # Potassium is 6.5 (range 3.5 - 5.0) and critical
    potassium_entry = [r for r in out_of_range if "Potassium" in r["normalized_test_name"]][0]
    assert potassium_entry["is_critical"] == 1
    assert "Above reference range" in potassium_entry["out_of_range_reason"]
    print("[PASS] list_out_of_range_results verified!")

    print("\n=== 4. Testing User Correction Flow (State Management) ===")
    # Find A1C record
    all_rows = fetch_all_results()
    a1c_row = [r for r in all_rows if r["normalized_test_name"] == "A1C" and r["report_date"] == "2026-09-18"][0]
    # Simulate user correcting 6.1 to 5.9
    update_lab_result(a1c_row["id"], {"value": 5.9})
    updated_a1c = get_latest_result("A1C")
    assert updated_a1c["record"]["value"] == 5.9
    print("[PASS] User correction updates immediately reflected in database and tools!")

    print("\n=== 5. Testing Safety Tripwire ===")
    # Querying a critical test should trigger tripwire
    reset_tool_execution_logs()
    get_latest_result("Potassium")
    from app import _active_has_critical
    assert _active_has_critical is True
    print("[PASS] Safety tripwire triggered on critical result!")

    # Clean test db file
    if os.path.exists("test_health.db"):
        os.remove("test_health.db")

    print("\nALL CORE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
