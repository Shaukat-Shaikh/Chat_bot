import streamlit as st
import requests
import json
import os
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# --- LangGraph Setup ---

class SummarizationState(TypedDict):
    input_text: str
    summarization_style: str
    messages: Annotated[list[AnyMessage], add_messages]

def summarizer_llm_node(state: SummarizationState):
    text_to_summarize = state['input_text']
    summarization_style = state['summarization_style']

    # Construct prompt
    if summarization_style == 'brief':
        prompt = f"Summarize briefly:\n\n{text_to_summarize}"
    elif summarization_style == 'detailed':
        prompt = f"Provide a detailed summary:\n\n{text_to_summarize}"
    elif summarization_style == 'bullet_points':
        prompt = f"Summarize in bullet points:\n\n{text_to_summarize}"
    else:
        prompt = f"Summarize:\n\n{text_to_summarize}"

    # Groq-compatible message format
    chat_history = [{"role": "user", "content": prompt}]

    payload = {
        "messages": chat_history,
        "model": "llama3-8b-8192",
        "temperature": 0.7
    }

    # Load API key (recommend using environment variable)
    groq_api_key = os.getenv("GROQ_API_KEY", "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")

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {groq_api_key}'
    }

    api_url = "https://api.groq.com/openai/v1/chat/completions"

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content']
        return {"messages": AIMessage(content=content)}

    except requests.exceptions.HTTPError as e:
        return {"messages": AIMessage(content=f"HTTPError: {e}\n\nResponse: {response.text}")}
    except Exception as e:
        return {"messages": AIMessage(content=f"Unexpected error: {e}")}

# Build LangGraph
builder = StateGraph(SummarizationState)
builder.add_node("summarize_text", summarizer_llm_node)
builder.add_edge(START, "summarize_text")
builder.add_edge("summarize_text", END)
graph = builder.compile()

# --- Streamlit UI ---

st.set_page_config(page_title="Document Summarizer", layout="centered")

st.title("📄 Document Summarizer (LangGraph + Groq)")
st.markdown("Paste or upload a document and choose how you'd like it summarized.")

# Input section
st.header("1. Input Document")
input_area_text = st.text_area("Paste your text here...", height=200, key="input_text_area")

uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

processed_input_text = input_area_text
if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8")
        processed_input_text = file_content
        st.info("File uploaded and content loaded.")
        st.text_area("Uploaded Content (read-only)", value=file_content, height=150, disabled=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        processed_input_text = ""

# Style selection
st.header("2. Choose Summarization Style")
summarization_style_option = st.radio(
    "Select summarization style:",
    ('brief', 'detailed', 'bullet_points'),
    horizontal=True
)

# Summarize button
st.markdown("---")
if st.button("Summarize Document", use_container_width=True, type="primary"):
    if not processed_input_text.strip():
        st.error("Please enter or upload some text first.")
    else:
        with st.spinner("Generating summary..."):
            try:
                inputs = {
                    "input_text": processed_input_text,
                    "summarization_style": summarization_style_option,
                    "messages": []
                }

                final_state = graph.invoke(inputs)

                if final_state and final_state.get('messages'):
                    ai_response_messages = [msg for msg in final_state['messages'] if isinstance(msg, AIMessage)]
                    if ai_response_messages:
                        summary_message = ai_response_messages[-1]
                        st.subheader("3. Summary Output")
                        st.markdown(summary_message.content)
                    else:
                        st.error("No valid summary found in the response.")
                else:
                    st.error("No messages found. Something went wrong.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Try a shorter input or check your API connection.")

# Sidebar instructions
st.sidebar.markdown("### How to Run:")
st.sidebar.code("pip install streamlit langchain-core langgraph requests")
st.sidebar.code("streamlit run app.py")
st.sidebar.markdown("---")
st.sidebar.markdown("🛡️ Tip: Set your API key securely via:")
st.sidebar.code("export GROQ_API_KEY='your_api_key_here'")

