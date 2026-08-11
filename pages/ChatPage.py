import streamlit as st
from langchain_openai import ChatOpenAI
from database import get_database

st.markdown(
    """<style>

    @import url('https://fonts.google.com/specimen/Roboto+Condensed');

    /* Chat Page Title */
    h1 {
        color: black;
        font-family: "Roboto", san-serif;
        font-weight: 700;
    }

    /* generate summary button */


    div.stButton > button {
        background-color: maroon;
        color:white;
        border-radius:20px;
    

        
    }
    /* hovering of button */
    div.stButton > button:hover {
        background-color: lightcoral;
        color:black;
        border-radius:20px;

    }
    

    </style> """,

    unsafe_allow_html = True
)

db = get_database()

st.title("Chatting...")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.7,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

def extract_text_from_files(file):
    # Function to extract text from uploaded files (PDF or DOCX)
    if file.type == "application/pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        from docx import Document
        doc = Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    else:
        st.error("Unsupported file type. Please upload a PDF or DOCX file.")
        return None

    text = extract_text_from_files(file)
    files_collection.insert_one({"Content": file.name, "Text": text})


def generate_summary():
    # Get all uploaded files from the database
    uploaded_files = list(db["uploaded_files"].find())

    if not uploaded_files:
        st.warning("No current files available")
        return 

    currentfiles = [
        file for file in uploaded_files
        if file.get("Status") == "Current"
    ]

    if not currentfiles:
        st.warning("No current files available")
        return 

    file_names = [file["Content"] for file in currentfiles]
    selected_file = st.selectbox(
        "Select File:", 
        file_names,
        key="file_selectbox",
    )


    st.session_state.selected_file = selected_file

    if st.button("Submit", key="generate_summary"):
        for file in uploaded_files:
            if file["Content"] == selected_file:
                file_text = file.get("Topics")

                if not file_text:
                    st.error("No text found in the selected file.")
                    return 
                
                    
                # Truncate long text to avoid token limits
                if len(file_text) > 4000:
                    file_text = file_text[:4000] + "..."

                
                try:
                    summary = llm.invoke(
                        f"Please summarize the following content: {file_text}")
                
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
                    return

                st.session_state.messages = [
                    msg for msg in st.session_state.messages if msg["role"] != "assistant"]
                st.session_state.messages.append(
                    {"role": "assistant", 
                    "content": summary.content}
                )
                st.session_state.show_summary_ui = False
    
                st.rerun()
                break

if "show_summary_ui" not in st.session_state:
    st.session_state.show_summary_ui = False

if st.button("Generate Summary", key="show_summary_ui_button"):
    st.session_state.show_summary_ui = True
    
if st.session_state.show_summary_ui:
    generate_summary()

# Chat box
with st.container(border=True, height=500):
    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Keep this outside so Streamlit keeps it at the bottom
prompt = st.chat_input(
    "Simply Ask....",
    accept_audio=True,
)


if prompt:
    if prompt.text:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": prompt.text})

        # Generate response
        response = llm.invoke(prompt.text)

        # Save AI response
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )

        # # Refresh UI
        # st.rerun()

    if prompt.audio:
        st.audio(prompt.audio)