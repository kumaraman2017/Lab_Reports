import tomllib, pathlib
from google import genai
from google.genai import types
import sqlite3

def get_test_history(test_name: str) -> dict:
    '''Fetches all records chronologically and computes trends purely in Python.
    Args:
        test_name: The name or alias of the test.
    Returns:
        dict: Chronological records, numeric differences, percentage changes, and overall trend.
    '''
    # We will copy the exact code from app.py to see if it fails!
    ALIAS_MAP = {'hba1c': 'A1C', 'hemoglobin a1c': 'A1C'}
    norm_name = ALIAS_MAP.get(test_name.lower(), test_name)
    conn = sqlite3.connect('health_reports.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM lab_results WHERE LOWER(test_name) = ? ORDER BY report_date ASC', (norm_name.lower(),))
    rows = cursor.fetchall()
    conn.close()

    history_items = []
    prev_value = None
    for r in rows:
        val = r['value']
        entry = {
            'id': r['id'], 'report_date': r['report_date'], 'test_name': r['test_name'],
            'value': val, 'unit': r['unit'], 'reference_range': r['reference_range'],
            'is_critical': r['is_critical'], 'confidence': r['confidence'],
        }
        if val is not None and prev_value is not None:
            delta = round(val - prev_value, 4)
            pct = round((delta / prev_value) * 100, 2) if prev_value != 0 else 0.0
            entry['delta_from_previous'] = delta
            entry['percentage_change_from_previous'] = pct
            entry['step_trend'] = 'increasing' if delta > 0 else ('decreasing' if delta < 0 else 'stable')
        else:
            entry['delta_from_previous'] = 0.0
            entry['percentage_change_from_previous'] = 0.0
            entry['step_trend'] = 'baseline'
        if val is not None:
            prev_value = val
        history_items.append(entry)

    valid_vals = [h['value'] for h in history_items if h['value'] is not None]
    if len(valid_vals) >= 2:
        overall_delta = round(valid_vals[-1] - valid_vals[0], 4)
        overall_pct = round((overall_delta / valid_vals[0]) * 100, 2) if valid_vals[0] != 0 else 0.0
        trend = ('stable' if abs(overall_delta) < 0.001 or abs(overall_pct) < 1.0
                 else ('increasing' if overall_delta > 0 else 'decreasing'))
    elif len(valid_vals) == 1:
        overall_delta = 0.0; overall_pct = 0.0; trend = 'single_data_point (no trend)'
    else:
        overall_delta = 0.0; overall_pct = 0.0; trend = 'unknown'

    result = {
        'status': 'success', 'test_name': norm_name, 'record_count': len(history_items),
        'overall_delta': overall_delta, 'overall_percentage_change': overall_pct,
        'overall_trend': trend, 'history': history_items,
    }
    return result

cfg = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())
client = genai.Client(api_key=cfg['GEMINI_API_KEY'])
chat = client.chats.create(model='gemini-2.5-flash', config=types.GenerateContentConfig(tools=[get_test_history], temperature=0.2))
try:
    resp = chat.send_message('How has my A1C changed from January to March?')
    print('RESPONSE:')
    print(resp.text)
except Exception as e:
    print('ERROR:', e)
