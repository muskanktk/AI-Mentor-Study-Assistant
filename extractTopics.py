import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

def extract_topics_from_pdf(file):

    pdf_reader = PdfReader(file)

    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    text = text[:1000]  
    
    # Use OpenAI API to extract topics from the text
    client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
            "You extract broad, high-level topics from documents."},

            {
                "role": "user",
                "content": 
                f"""Extract only the 2 to 3 main topics from the following text.:\n\n
                 In this format
                 Topic 1: Subtopics
                 Topic 2: Subtopics 
                {text}"""
            }
        ]
    )
    
    topics = response.choices[0].message.content.strip()
    return topics



