import tomllib, pathlib
from google import genai
from google.genai import types

cfg = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())
client = genai.Client(api_key=cfg['GEMINI_API_KEY'])

def get_test_history(test_name: str) -> dict:
    '''Fetches all records chronologically and computes trends purely in Python.
    Args:
        test_name: The name or alias of the test.
    Returns:
        dict: Chronological records, numeric differences, percentage changes, and overall trend.
    '''
    return {
        'status': 'success',
        'test_name': 'A1C',
        'record_count': 5,
        'overall_delta': -0.3,
        'overall_trend': 'decreasing',
        'history': [{'report_date': '2024-01-15', 'value': 6.2}]
    }

sys_instr = '''You are a professional, informational lab-report AI agent.
Your behavioral rules:
1. INFORMATIONAL ROLE ONLY: Provide factual, clear, natural language explanations of lab test results using the data returned by your tools. NEVER output raw JSON or python dictionaries.
2. STRICT GROUNDING: Base your answers entirely on the data returned by your tools. Never hallucinate.
3. HANDLING MISSING DATA: If the user asks for a specific date range (e.g., "January to March") and data is absent for the current year but exists for previous years, explicitly inform the user about the available data (e.g., "I don't have data for 2026, but in 2024 your A1C was..."). ONLY state "Data not found in uploaded reports." if the test is completely absent from the database.'''

chat = client.chats.create(
    model='gemini-2.5-flash',
    config=types.GenerateContentConfig(
        tools=[get_test_history],
        system_instruction=sys_instr,
        temperature=0.2
    )
)
resp = chat.send_message('How has my A1C changed from January to March?')
print('MODEL OUTPUT:')
print(resp.text)
if resp.function_calls:
    print('CALLED TOOLS:', [f.name for f in resp.function_calls])
