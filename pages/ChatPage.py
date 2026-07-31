import streamlit as st
from langchain_openai import ChatOpenAI


st.title("Chatting with AI")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=st.secrets["OPENAI_API_KEY"],
    temperature=0.7,
)


if "messages" not in st.session_state:
    st.session_state.messages = []


# Chat box
with st.container(border=True, height=500):


    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


# Keep this outside so Streamlit keeps it at the bottom
prompt = st.chat_input(
    "Say or record something...",
    accept_audio=True,
)


if prompt:

    if prompt.text:

        # Save user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt.text}
        )

        # Generate response
        response = llm.invoke(prompt.text)

        # Save AI response
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )

        # Refresh UI
        st.rerun()


    if prompt.audio:
        st.audio(prompt.audio)