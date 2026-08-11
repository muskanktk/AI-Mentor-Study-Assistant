import os
import streamlit as st
from pymongo import MongoClient


def get_database():
    # Try Streamlit secrets first (if available), then fall back to env var
    uri = None
    if hasattr(st, "secrets"):
        try:
            uri = st.secrets.get("MONGODB_URI")
        except Exception:
            try:
                uri = st.secrets["MONGODB_URI"]
            except Exception:
                uri = None

    if not uri:
        uri = os.environ.get("MONGODB_URI")

    if not uri:
        raise RuntimeError(
            'MONGODB_URI not set. Add it to Streamlit secrets (Settings → Secrets) or set the environment variable "MONGODB_URI".'
        )

    client = MongoClient(uri)
    db = client["AI_Mentor"]
    return db