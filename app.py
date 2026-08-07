# import streamlit as st which has UI
from langchain_openai import ChatOpenAI
import streamlit as st


st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background: white;
        }

    /* sidebar text */
    section[data-testid="stSidebar"] * {
        color:black;
        }

    </style>
    
    """,


    # allow string as html not as plain test
    unsafe_allow_html=True,
)

def Pages():

    # to create a multi-page app, we can use the st.Page class to define each page and then use the st.navigation function to create a navigation bar that allows users to switch between pages.
    # the first paramter is path to the page file, the second parameter is the title of the page, and the third parameter is the icon of the page.
    mainPage = st.Page("pages/MainPage.py", title="Dashboard", icon = ":material/home:")
    ChatPage = st.Page("pages/ChatPage.py", title="Chatting Time", icon = ":material/chat:")
    FileUploadPage = st.Page(
        "pages/FileUploadPage.py", title=" Files", icon = ":material/folder:"
    )

    
    Navigation = st.navigation([mainPage, ChatPage, FileUploadPage])

    Navigation.run()


if __name__ == "__main__":
    Pages()
