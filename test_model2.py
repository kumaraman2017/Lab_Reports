import tomllib, pathlib
from google import genai
from google.genai import types

cfg = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())
client = genai.Client(api_key=cfg['GEMINI_API_KEY'])

def get_test_history(test_name: str) -> dict:
    return {
        'status': 'success',
        'test_name': 'A1C',
        'record_count': 5,
        'overall_delta': -0.3,
        'overall_percentage_change': -4.84,
        'overall_trend': 'decreasing',
        'history': [
            {'report_date': '2024-01-15', 'value': 6.2}, 
            {'report_date': '2024-02-16', 'value': 6.8},
            {'report_date': '2024-03-18', 'value': 8.5},
            {'report_date': '2026-06-15', 'value': 6.8}, 
            {'report_date': '2026-09-18', 'value': 5.9}
        ]
    }

sys_instr = '''You are a professional, informational lab-report AI agent.
Your behavioral rules:
1. INFORMATIONAL ROLE ONLY: Provide factual, clear, natural language explanations of lab test results using the data returned by your tools. NEVER output raw JSON or python dictionaries. Summarize the tool outputs in conversational English.
2. STRICT GROUNDING: Base your answers entirely on the data returned by your tools. If the tool returns a status indicating data is not found, state: "Data not found in uploaded reports." Never hallucinate.'''

chat = client.chats.create(
    model='gemini-3.5-flash',
    config=types.GenerateContentConfig(
        tools=[get_test_history],
        system_instruction=sys_instr,
        temperature=0.2
    )
)
resp = chat.send_message('How has my A1C changed from January to March?')
print('MODEL OUTPUT:')
print(resp.text)
