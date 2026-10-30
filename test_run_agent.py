import streamlit as st
import sys

# Mock streamlit session state
class SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)
    def __setattr__(self, name, value):
        self[name] = value
st.session_state = SessionState()
st.session_state['_resolved_model'] = 'gemini-2.5-flash'
st.session_state.chat_messages = []

import tomllib, pathlib
cfg = tomllib.loads(pathlib.Path('.streamlit/secrets.toml').read_text())

import app
response, logs, crit = app.run_agent_turn('How has my A1C changed from January to March?', cfg['GEMINI_API_KEY'])
print('RESPONSE:', response)
print('LOGS:', logs)
