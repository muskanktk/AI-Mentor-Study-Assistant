import streamlit as st
from database import get_database

db = get_database()

with st.container(border=True, height=500):


    # to create the UI for the python you can do it in markdown 
    # using the st.markdown function and you can use html tags to create the UI
    st.markdown(
        "<h1>Welcome to the Main Page</h1>",
        # allow string as html not as plain test
        unsafe_allow_html=True,
    )

    st.write("Hello, I am Muskan, I created this website as a way to help students quickly learn concepts")
    st.write("This website is designed to help students learn concepts quickly and easily. It provides a variety of resources, including tutorials, examples, and exercises, to help students understand the material.")