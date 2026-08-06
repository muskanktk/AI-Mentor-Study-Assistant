import streamlit as st
from pymongo import MongoClient


def get_database():

    client = MongoClient(st.secrets["MONGODB_URI"])

    db = client["AI_Mentor"]

    return db