import os
import streamlit as st

# Try Streamlit Cloud secrets first, then .env file
try:
    XAI_API_KEY = st.secrets["XAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    XAI_API_KEY = os.getenv("XAI_API_KEY")

CHAT_MODEL = "grok-3"
EMBEDDING_DIMENSION = 384