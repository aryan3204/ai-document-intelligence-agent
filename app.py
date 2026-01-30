import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import os

# --- GCP CONFIGURATION ---
PROJECT_ID = "yt-sentiment-analysis-482419" 
LOCATION = "us-central1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-flash")

# --- UI SETUP ---
st.set_page_config(page_title="AI Doc Agent", page_icon="📄")
st.title("📄 AI Document Intelligence Agent")

# --- FILE HANDLING ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_data = uploaded_file.read()
    st.success(f"Loaded: {uploaded_file.name}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            pdf_part = Part.from_data(data=pdf_data, mime_type="application/pdf")
            agent_instruction = f"Context: You are a professional data analyst. Answer: {prompt}"
            response = model.generate_content([pdf_part, agent_instruction])
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
