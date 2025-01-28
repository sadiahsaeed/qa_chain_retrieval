import streamlit as st
import streamlit_chat
import requests

# Define the base URL for your FastAPI service
FASTAPI_URL = "https://qa-chain-retrieval-542808340038.us-central1.run.app"

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Application Title
st.title("Conversational Document Query Application")

# File Upload Section
st.header("Upload Files")
uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, TXT, ZIP):",
    type=["pdf", "docx", "txt", "zip"],
    accept_multiple_files=True
)

if st.button("Upload"):
    if uploaded_files:
        files = [("files", (file.name, file.read(), file.type)) for file in uploaded_files]
        response = requests.post(f"{FASTAPI_URL}/upload_files/", files=files)

        if response.status_code == 200:
            data = response.json()
            st.success(data.get("message", "File processing completed successfully."))
        else:
            st.error(f"Failed to upload files. Error: {response.text}")
    else:
        st.warning("Please upload at least one file.")

# Query Section
def handle_query():
    query = st.session_state.query_input
    if query:
        # Display user query in the chat
        st.session_state.conversation_history.append({"content": query, "is_user": True})

        # Clear the input box after submission
        st.session_state.query_input = ""

        # Make the POST request to the FastAPI retrieve endpoint
        response = requests.post(f"{FASTAPI_URL}/retrieve", json={"query": query})

        if response.status_code == 200:
            result = response.json()
            answer = result.get("content", "No content found for the query.")

            # Add bot response to the conversation history
            st.session_state.conversation_history.append({"content": answer, "is_user": False})
        else:
            error_message = f"Failed to process query. Error: {response.text}"
            st.session_state.conversation_history.append({"content": error_message, "is_user": False})

# Display conversation in a chat style using streamlit_chat
for entry in st.session_state.conversation_history:
    streamlit_chat.message(entry["content"], is_user=entry["is_user"])

# Input field for queries with Enter key submission
st.text_input("Ask a question about the uploaded documents:", key="query_input", on_change=handle_query)
