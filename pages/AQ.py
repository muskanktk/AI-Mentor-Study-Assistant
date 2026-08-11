import streamlit as st
from openai import OpenAI
from database import get_database
from pages.ChatPage import generate_summary

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f7;
    }
    h1 {
        color: #2f4f4f;
    }
    """,
    unsafe_allow_html=True,
)

# get the dabase
db = get_database()
st.title("Generate Quiz")


# temporarily suppress any "Submit" buttons created by generate_summary()
_orig_st_button = st.button

def _suppress_submit_button(*args, **kwargs):
    label = args[0] if args else kwargs.get("label")
    if label == "Submit":
        return False
    return _orig_st_button(*args, **kwargs)

st.button = _suppress_submit_button
SelectFile = generate_summary()
st.button = _orig_st_button

def flashcard_count():
    slider = st.slider(
        "Select number of flashcards",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )
    return slider

def Answers_Questions(count):
    selected_file = st.session_state.get("selected_file")
    if not selected_file:
        st.warning("Please select a file first.")
        return None

    file_doc = db["uploaded_files"].find_one({"Content": selected_file})
    if not file_doc:
        st.error("Selected file is not available in the database.")
        return None

    topics = file_doc.get("Topics")
    if not topics:
        st.error("No topics were found for the selected file.")
        return None

    openAIKey = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=openAIKey)

    prompt = (
        f"Create {count} flashcards from the following topics. "
        "Each flashcard should include a clear question and concise answer. "
        "Number them like 1. Q: ... A: ...\n\n"
        f"Topics:\n{topics}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful study assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    if response.choices:
        return response.choices[0].message.content.strip()
    return None

if st.session_state.get("selected_file"):
    count = flashcard_count()
    if st.button("Generate"):
        flashcards = Answers_Questions(count)
        if flashcards:
            st.session_state.flashcards = flashcards

if st.session_state.get("flashcards"):
   
    st.markdown("### PRACTICE")

    questions = []
    current_question = None

    for line in st.session_state.flashcards.splitlines():
        stripped = line.strip()
        if stripped.startswith("Q:") or ". Q:" in stripped:
            if ". Q:" in stripped:
                current_question = stripped.split(". Q:", 1)[1].strip()
            else:
                current_question = stripped
            questions.append(current_question)
        elif stripped.startswith("- Q:"):
            current_question = stripped.split("- Q:", 1)[1].strip()
            questions.append(current_question)

    if not questions:
        # fallback when flashcards are not in numbered Q/A format
        questions = [line for line in st.session_state.flashcards.splitlines() if line.strip()][:5]

    answers = []
    for idx, question in enumerate(questions, start=1):
        st.markdown(f"**{idx}. {question}**")
        answer = st.text_area(f"Your answer", key=f"flashcard_answer_{idx}", height=100)
        answers.append(answer)

    if st.button("Submit Answers"):
        if not any(answers):
            st.warning("Please write at least one answer before submitting.")
        else:
            openAIKey = st.secrets["OPENAI_API_KEY"]
            client = OpenAI(api_key=openAIKey)
            answer_text = "\n".join(
                [f"{idx}. Question: {q} Answer: {a}" for idx, (q, a) in enumerate(zip(questions, answers), start=1)]
            )
            prompt = (
                "You are a helpful study assistant. Check the student answers against the expected flashcard answers. "
                "Give feedback and indicate whether each answer is correct or needs improvement. "
                f"Here are the flashcards and submitted answers:\n{answer_text}"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful study assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )

            if response.choices:
                st.markdown("### Feedback")
                st.text_area("Feedback", response.choices[0].message.content.strip(), height=250)

        