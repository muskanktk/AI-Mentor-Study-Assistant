# import streamlit as st which has UI
from langchain_openai import ChatOpenAI
import streamlit as st


def Pages():

    # to create a multi-page app, we can use the st.Page class to define each page and then use the st.navigation function to create a navigation bar that allows users to switch between pages.
    # the first paramter is path to the page file, the second parameter is the title of the page, and the third parameter is the icon of the page.
    mainPage = st.Page("pages/MainPage.py", title="Main Page", icon="🏠")
    ChatPage = st.Page("pages/ChatPage.py", title="Chat Page", icon="💬")
    FileUploadPage = st.Page(
        "pages/FileUploadPage.py", title="File Upload Page", icon="📁"
    )

    Navigation = st.navigation([mainPage, ChatPage, FileUploadPage])

    Navigation.run()


if __name__ == "__main__":
    Pages()
