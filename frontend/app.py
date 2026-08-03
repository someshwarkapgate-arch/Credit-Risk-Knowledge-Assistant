import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Credit Risk Knowledge Assistant",
    page_icon="📊",
    layout="wide")


# Application title
st.title("📊 Credit Risk Knowledge Assistant")


st.write(
    """
    Welcome to the AI-powered Credit Risk Knowledge Assistant.

    This application will help credit analysts:
    
    - Search credit risk documents
    - Understand RBI guidelines
    - Query IFRS9 documentation
    - Retrieve information from policy documents
    
    🚧 Currently under development
    """)


# Placeholder section
st.subheader("Ask your question")

question = st.text_input(
    "Enter your credit risk question")


if question:
    st.info(
        "RAG pipeline will process this question in future phases.")