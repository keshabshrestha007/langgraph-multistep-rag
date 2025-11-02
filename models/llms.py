from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import streamlit as st
"""
 # loading environment variables
load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

api_key = os.getenv("GROQ_API_KEY")
"""

if st.secrets["GROQ_API_KEY"] is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")
else:
    api_key = st.secrets["GROQ_API_KEY"]

 # Initialize LLM
llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=api_key
        )