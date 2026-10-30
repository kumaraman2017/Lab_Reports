from app import run_agent_turn, get_api_key, SAFETY_NOTICE, load_sample_health_data, clear_all_results
gemini_api_key = get_api_key()
def test_live_agent():
    print("=== Populating Sample Lab Data ===")
    clear_all_results()
    load_sample_health_data()

    print("\n=== Turn 1: Asking for latest A1C ===")
    reply1, tools1, crit1 = run_agent_turn("What is my latest A1C result?", api_key=gemini_api_key)
    print(f"Tools used: {[t['tool'] for t in tools1]}")
    print(f"Reply:\n{reply1}\n")
    assert any(t['tool'] == 'get_latest_result' for t in tools1), "Expected get_latest_result tool to be called"

    print("\n=== Turn 2: Asking for Glucose history and trend ===")
    reply2, tools2, crit2 = run_agent_turn("How has my fasting blood sugar changed over time?", api_key=gemini_api_key)
    print(f"Tools used: {[t['tool'] for t in tools2]}")
    print(f"Reply:\n{reply2}\n")
    assert any(t['tool'] == 'get_test_history' for t in tools2), "Expected get_test_history tool to be called"

    print("\n=== Turn 3: Asking for out-of-range results on 2026-09-18 (should trigger SAFETY NOTICE) ===")
    reply3, tools3, crit3 = run_agent_turn("Are there any out of range or critical tests on 2026-09-18?", api_key=gemini_api_key)
    print(f"Tools used: {[t['tool'] for t in tools3]}")
    print(f"Was Critical: {crit3}")
    print(f"Reply:\n{reply3}\n")
    assert crit3 is True, "Expected critical flag to be triggered"
    assert reply3.startswith(SAFETY_NOTICE), f"Expected reply to start with exact SAFETY NOTICE: {SAFETY_NOTICE}"

    print("\nALL LIVE AGENT TESTS PASSED WITH 100% ACCURACY!")

if __name__ == "__main__":
    test_live_agent()
