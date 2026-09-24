import os
import requests
import streamlit as st

from dotenv import load_dotenv


# ---------------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------------

load_dotenv()

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
)

QUERY_ENDPOINT = f"{API_BASE_URL}/api/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"


# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Knowledge Assistant",
    page_icon="📘",
    layout="wide"
)


# ---------------------------------------------------------
# Helper Function: Check Backend
# ---------------------------------------------------------

def check_backend():
    """
    Check whether the FastAPI backend is available.
    """

    try:

        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=5
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


# ---------------------------------------------------------
# Helper Function: Call RAG API
# ---------------------------------------------------------

def ask_rag_api(
    question,
    top_k=5
):
    """
    Send the user's question to the FastAPI RAG backend.

    Parameters
    ----------
    question : str
        User question.

    top_k : int
        Number of chunks to retrieve.

    Returns
    -------
    dict
        API response containing answer and sources.
    """

    payload = {
        "question": question,
        "top_k": top_k
    }

    try:

        response = requests.post(
            QUERY_ENDPOINT,
            json=payload,
            timeout=120
        )

        # Raises an exception for HTTP errors such as
        # 400, 422, 500, etc.
        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "The request timed out. "
            "The RAG backend took too long to respond."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to the FastAPI backend. "
            "Please make sure the backend server is running."
        )

    except requests.exceptions.HTTPError:

        try:

            error_detail = response.json().get(
                "detail",
                "Unknown API error."
            )

        except Exception:

            error_detail = response.text

        raise RuntimeError(
            f"Backend error: {error_detail}"
        )

    except requests.RequestException as error:

        raise RuntimeError(
            f"Request failed: {error}"
        )


# ---------------------------------------------------------
# Helper Function: Display Sources
# ---------------------------------------------------------

def display_sources(sources):
    """
    Display unique source documents and page numbers.
    """

    if not sources:
        return

    with st.expander(
        "View Sources",
        expanded=False
    ):

        for source in sources:

            source_file = source.get(
                "source_file",
                "Unknown"
            )

            page_number = source.get(
                "page_number",
                "Unknown"
            )

            chunk_id = source.get(
                "chunk_id"
            )

            st.markdown(
                f"**{source_file}**  \n"
                f"Page: {page_number}"
            )

            if chunk_id is not None:

                st.caption(
                    f"Chunk ID: {chunk_id}"
                )

            st.divider()


# ---------------------------------------------------------
# Initialize Chat History
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title(
    "📘 Credit Risk Knowledge Assistant"
)

st.caption(
    "Ask questions about Ind AS 109, "
    "RBI circulars and credit-risk regulatory documents."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header(
        "Settings"
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        help=(
            "Number of relevant document chunks "
            "sent to the language model."
        )
    )

    st.divider()

    if check_backend():

        st.success(
            "Backend connected"
        )

    else:

        st.error(
            "Backend unavailable"
        )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# Display Existing Chat History
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            display_sources(
                message.get(
                    "sources",
                    []
                )
            )


# ---------------------------------------------------------
# Chat Input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your credit risk documents..."
)


# ---------------------------------------------------------
# Handle New Question
# ---------------------------------------------------------

if question:

    # -----------------------------------------------------
    # Store User Message
    # -----------------------------------------------------

    user_message = {
        "role": "user",
        "content": question
    }

    st.session_state.messages.append(
        user_message
    )


    # -----------------------------------------------------
    # Display User Message
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # Generate Assistant Response
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = ask_rag_api(
                    question=question,
                    top_k=top_k
                )

                answer = result.get(
                    "answer",
                    "No answer was returned."
                )

                sources = result.get(
                    "sources",
                    []
                )

                retrieved_chunks = result.get(
                    "retrieved_chunks",
                    0
                )


                # -----------------------------------------
                # Display Answer
                # -----------------------------------------

                st.markdown(
                    answer
                )


                # -----------------------------------------
                # Display Sources
                # -----------------------------------------

                display_sources(
                    sources
                )


                # -----------------------------------------
                # Retrieval Information
                # -----------------------------------------

                st.caption(
                    f"Retrieved chunks: "
                    f"{retrieved_chunks}"
                )


                # -----------------------------------------
                # Store Assistant Message
                # -----------------------------------------

                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieved_chunks":
                        retrieved_chunks
                }

                st.session_state.messages.append(
                    assistant_message
                )


            except RuntimeError as error:

                st.error(
                    str(error)
                )