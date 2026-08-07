import extractTopics
import streamlit as st
from database import get_database


st.markdown(
    """
    <style>

    /* Current tab */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: olivedrab;
        color: white;
        border-radius: 8px 8px 0px 0px;
        height: 45px;
        width: 100%;
        font-size: 16px;
        

    }
    /* Archive tab */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: tomato;
        color: white;
        border-radius: 8px 8px 0px 0px;
        height: 45px;
        width: 100%;
        font-size: 16px;

    }
    /*Upload button */
    div.stButton > button {
        background-color: maroon;
        color: white;
        border-radius:10px;
    }
    /* Hover */

    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
        /* Remove the drag-and-drop area */
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

        /* Hide upload instructions */
    div[data-testid="stFileUploader"] small {
        display: none !important;
    }


    /* Style only the Browse button */
    div[data-testid="stFileUploader"] button {
        background-color: maroon !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }


    /* Button hover */
    div[data-testid="stFileUploader"] button:hover {
        background-color: lightcoral !important;
        color: black !important;
    }


    /* Remove blue focus/hover area */
    div[data-testid="stFileUploader"] section:hover {
        border: none !important;
        background: transparent !important;
    }

    
    div[data-testid="stFileUploader"] section > div {
        display: none;
    }


    </style>
    """,
    unsafe_allow_html=True,
)

# Connect to MongoDB
db = get_database()
files_collection = db["uploaded_files"]

# Session state initialization
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = set()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Current"


@st.dialog(title="File Upload Page")
def vote(item):
    st.write("The File already exists in the database.")


# ==========================================
# Top row: Folder Tabs + Upload
# ==========================================

st.markdown('<div class = "folder-tabs">', unsafe_allow_html=True)
col_tab1, col_tab2, empty, col_upload = st.columns(
    [1.2, 1.2, 5, 2], gap= None, vertical_alignment="bottom"
)

with col_tab1:
    if st.button("Current", key="tab_current", use_container_width=True):
        st.session_state.selected_tab = "Current"
        st.rerun()

with col_tab2:
    if st.button("Archive", key="tab_archived", use_container_width=True):
        st.session_state.selected_tab = "Archived"
        st.rerun()

with col_upload:
    uploaded_file = st.file_uploader(
        "",
        type=["pdf", "docx"],
        label_visibility="collapsed",
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

if uploaded_file is not None:
    file_name = uploaded_file.name
    existing_file = files_collection.find_one({"Content": file_name})

    if existing_file:
        vote(file_name)
    else:
        file_info = {
            "Content": file_name,
            "Topics": extractTopics.extract_topics_from_pdf(uploaded_file),
            "Status": "Current",
            "Links": file_name,
        }
        files_collection.insert_one(file_info)
        st.session_state.uploader_key += 1
        st.rerun()


# Container for the table
with st.container(border=True, height=500):
    documents = list(files_collection.find({}, {"_id": 0}))
    statuses = ["Current", "Archived"]

    def display_table(status_filter):
        filtered_documents = [
            file
            for file in documents
            if file.get("Status", "Current") == status_filter
        ]

        if filtered_documents:
            col1, col2, col3, col4 = st.columns(4)
            col1.write("**Content**")
            col2.write("**Topics**")
            col3.write("**Status**")
            col4.write("**Links**")

            for file in filtered_documents:
                col1, col2, col3, col4 = st.columns(4)
                col1.write(file["Content"])



                with col2:
                    with st.popover("View Topics"):
                        st.markdown(f"### Topics for {file['Content']}")
                        st.divider()

                        topics = file["Topics"].split("\n")
                        for topic in topics:
                            st.write(topic)
                            

                with col3:
                    current_status = file.get("Status", "Current")

                    if current_status == "🟢 COMPLETE":
                        current_status = "Current"
                    elif current_status == "🟡 IN PROGRESS":
                        current_status = "Archived"

                    new_status = st.selectbox(
                        "Status",
                        statuses,
                        index=statuses.index(current_status),
                        key=f"status_{file['Content']}",
                        label_visibility="collapsed",
                    )

                    if new_status != current_status:
                        files_collection.update_one(
                            {"Content": file["Content"]},
                            {"$set": {"Status": new_status}},
                        )
                        st.rerun()

                col4.write(file["Links"])
        else:
            st.write("No files here.")

    display_table(st.session_state.selected_tab)