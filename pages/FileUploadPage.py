import requests
from urllib.parse import quote
from extractTopics import extract_topics_from_pdf
import streamlit as st
from database import get_database
# User clicks "View Topics" button in the table
# A popup window appears with:
# Title: "Topics"
# Shows: "### Topics for [filename]"
# Divider line
# Each topic on a new line


@st.dialog("Topics")
def show_topics(file):
    st.markdown(f"### Topics for {file['Content']}")
    st.divider()
    topics = file["Topics"].split("\n")
    for topic in topics:
        st.write(topic)

st.markdown(
    """
    <style>
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
        background-color: olive;
        color: white;
        border-radius: 8px 8px 0px 0px;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
        background-color: rosybrown;
        color: white;
        border-radius: 8px 8px 0px 0px;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }
    div.stButton > button {
        background-color: maroon;
        color: white;
        border-radius:10px;
    }
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    div[data-testid="stFileUploader"] small {
        display: none !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: maroon !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: lightcoral !important;
        color: black !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border: none !important;
        background: transparent !important;
    }
    div[data-testid="stFileUploader"] section > div {
        display: none;
    }
    div[class*="st-key-topics_btn_"] button {
        background-color: black !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    div[class*="st-key-topics_btn_"] button:hover {
        background-color: #333 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Connect to MongoDB
db = get_database()
files_collection = db["uploaded_files"]


def generateYoutubeLinks(topics, maxResults=5):
    youtube_api_key = st.secrets.get("YOUTUBE_API_KEY") if hasattr(st, "secrets") else None

    if isinstance(topics, list):
        topic_queries = topics
    else:
        topic_queries = []
        for line in str(topics).splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            if ":" in cleaned and cleaned.split(":", 1)[0].lower().startswith("topic"):
                cleaned = cleaned.split(":", 1)[1].strip()
            if cleaned:
                topic_queries.append(cleaned)

        if not topic_queries:
            topic_queries = [str(topics).strip()]

    youtube_links = []
    search_url = "https://www.googleapis.com/youtube/v3/search"

    for topic in topic_queries[:3]:
        topic_links = []

        if youtube_api_key:
            params = {
                "part": "snippet",
                "q": topic,
                "type": "video",
                "maxResults": maxResults,
                "key": youtube_api_key,
            }

            try:
                response = requests.get(search_url, params=params, timeout=10)
                response.raise_for_status()
                payload = response.json()

                for item in payload.get("items", []):
                    video_id = item.get("id", {}).get("videoId")
                    if not video_id:
                        continue

                    topic_links.append(
                        {
                            "topic": topic,
                            "title": item.get("snippet", {}).get("title", "YouTube video"),
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                        }
                    )
            except requests.RequestException as exc:
                st.warning(f"Could not fetch YouTube videos for '{topic}': {exc}")

        if not topic_links:
            topic_links.append(
                {
                    "topic": topic,
                    "title": f"Search YouTube: {topic}",
                    "url": f"https://www.youtube.com/results?search_query={quote(topic)}",
                }
            )

        youtube_links.extend(topic_links)

    return youtube_links


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


@st.dialog("YouTube Links")
def show_youtube_links(file):
    links = file.get("Links", [])

    if isinstance(links, dict):
        links = [links]
    elif isinstance(links, str):
        links = []
    elif not isinstance(links, list):
        links = []

    if not links:
        generated_links = generateYoutubeLinks(file.get("Topics", ""), maxResults=5)
        if generated_links:
            files_collection.update_one(
                {"Content": file["Content"]},
                {"$set": {"Links": generated_links}},
            )
            links = generated_links

    if isinstance(links, list) and links:
        for video in links:
            if isinstance(video, dict):
                title = video.get("title") or video.get("topic") or "YouTube video"
                url = video.get("url") or "https://www.youtube.com"
                st.markdown(f"[{title}]({url})")
            else:
                st.write(video)
    else:
        st.write("No YouTube links available.")

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
        "Upload PDF or DOCX files",
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
        topic = extract_topics_from_pdf(uploaded_file)

        if not topic:
            st.info("No topics were extracted from the file.")

        file_info = {
            "Content": file_name,
            "Topics": topic if topic else "No topics extracted",
            "Status": "Current",
            "Links": [],
        }
        files_collection.insert_one(file_info)

        youtube_links = generateYoutubeLinks(topic, maxResults=5)

        if not youtube_links:
            youtube_links = [
                {
                    "topic": "youtube search",
                    "title": "Search YouTube",
                    "url": "https://www.youtube.com/results?search_query=" + quote(str(topic)),
                }
            ]

        files_collection.update_one(
            {"Content": file_name},
            {"$set": {"Links": youtube_links}},
        )

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
                    with st.container(key=f"topics_btn_{file['Content']}"):
                        if st.button("View Topics", key=f"topics_{file['Content']}"):
                            show_topics(file)
                        
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

                with col4:
                    with st.container(key=f"links{file['Content']}"):
                        if st.button("View Links", key=f"links_{file['Content']}"):
                            show_youtube_links(file)

    display_table(st.session_state.selected_tab)