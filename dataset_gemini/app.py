import os
import pandas as pd
import streamlit as st
import subprocess
import google.generativeai as genai

# Streamlit app title
st.title("Data Analysis with Gemini")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your API Key", type="password")

# Store the API key in session state
if api_key:
    st.session_state['api_key'] = api_key

# Check if API key is provided
if 'api_key' in st.session_state:
    # Configure the Gemini API with the provided key
    genai.configure(api_key=st.session_state['api_key'])

    # Create the model
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if uploaded_file is not None:
        # Read the dataset
        df = pd.read_csv(uploaded_file)

        # Display chat history
        st.subheader("Chat History")
        for question, answer in st.session_state['chat_history']:
            st.write(f"**Q:** {question}")
            st.write(f"**A:** {answer}")

        # Text input for user prompt
        user_prompt = st.text_input("Ask a question about the dataset:", key="user_input")

        # Function to handle search
        def handle_search():
            if user_prompt:
                try:
                    # Prepare the data and prompt for the model
                    prompt = f"Here is the dataset: {df.head().to_dict()}. Based on this data, {user_prompt}"

                    # Interact with the Gemini API
                    chat_session = model.start_chat()
                    response = chat_session.send_message(prompt)
                    suggestions = response.text

                    # Append the question and response to chat history
                    st.session_state['chat_history'].append((user_prompt, suggestions))

                    st.success("Question processed successfully.")
                    st.write("Suggestions:")
                    st.write(suggestions)

                except Exception as e:
                    st.error(f"Error processing question: {e}")
            else:
                st.warning("Please enter a question.")

        # Button to trigger search
        if st.button("Search"):
            handle_search()

        # Allow Enter key to trigger search
        if user_prompt and st.session_state.get("user_input") is not None:
            handle_search()

    else:
        st.warning("Please upload a file to ask questions about.")
else:
    st.warning("Please enter your API key in the sidebar.")
