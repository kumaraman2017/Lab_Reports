import tomllib, pathlib
from google import genai
cfg = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())
client = genai.Client(api_key=cfg['GEMINI_API_KEY'])
try:
    for m in client.models.list():
        if m.name == 'models/gemini-2.5-flash':
            print("KEY IS VALID")
            break
except Exception as e:
    print(f"ERROR: {e}")
