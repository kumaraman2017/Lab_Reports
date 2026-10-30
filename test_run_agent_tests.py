import streamlit as st
import sys, time

# Mock streamlit session state
class SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
st.session_state = SessionState()
st.session_state['_resolved_model'] = 'gemini-3.6-flash'
st.session_state.chat_messages = []

import tomllib, pathlib
cfg = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())

import app

tests = [
    'What is my latest A1C?',
    'Show my A1C history.',
    'How has my A1C changed from January 2024 to March 2024?',
    'What results are outside the reference range on 2026-09-18?',
]

for t in tests:
    print('--------------------------------')
    print('PROMPT:', t)
    for retry in range(3):
        response, logs, crit = app.run_agent_turn(t, cfg['GEMINI_API_KEY'])
        if '503' in response or 'Quota' in response:
            print('Retrying due to API flakes...')
            time.sleep(3)
            continue
        print('RESPONSE:', response)
        print('LOGS:', logs)
        break
