import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/cvd-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the cardiovascular system, patients,
        social media posts, and anything related to the conversation around
        cardiovascular diseases.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("Which is the Entity with the highest sum of earned metrics?")
    st.markdown("What do people think about IL-6 inhibitors?")
 


st.title("CVD System Chatbot")
st.info(
    """Ask me questions about patients, social media posts or the entities and relationships
    in the graph database!"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            This usually means the chatbot failed at generating a query to
            answer your question. Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated?", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )
