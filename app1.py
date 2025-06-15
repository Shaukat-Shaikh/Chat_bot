import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import Runnable

# --- Load API key from .env ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Initialize LLM ---
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    api_key=groq_api_key
)

# --- Prompt Templates ---
input_prompt = PromptTemplate.from_template(
    "You are a summarization assistant.\nDocument: {raw_text}\nStyle: {style}\nFormat it properly for summarization."
)
summary_prompt = PromptTemplate.from_template(
    "Summarize the following text in {style} style:\n\n{cleaned_text}"
)
output_prompt = PromptTemplate.from_template(
    "Format this summary nicely for display:\n\n{summary}"
)

# --- Define LLM Chains ---
input_chain = LLMChain(
    llm=llm,
    prompt=input_prompt,
    output_key="cleaned_text"
)

llm_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    output_key="summary"
)

output_chain = LLMChain(
    llm=llm,
    prompt=output_prompt,
    output_key="final_output"
)

# --- Pipe the Chains Together ---
pipe_chain: Runnable = input_chain | llm_chain | output_chain

# --- Streamlit UI ---
st.set_page_config(page_title="📄 Chained Summarizer", layout="centered")
st.title("📄 Document Summarizer (LangChain + Groq)")

st.header("1. Input Document")
user_input_text = st.text_area("Paste your text here...", height=200)
uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])

if uploaded_file is not None:
    try:
        uploaded_text = uploaded_file.read().decode("utf-8")
        user_input_text = uploaded_text
        st.info("✅ File uploaded.")
        st.text_area("Uploaded Content", uploaded_text, height=150, disabled=True)
    except Exception as e:
        st.error(f"Error reading file: {e}")

st.header("2. Choose Summarization Style")
style = st.radio("Style:", ("brief", "detailed", "bullet_points"), horizontal=True)

# --- Summarize Button ---
st.markdown("---")
if st.button("Summarize Document", type="primary", use_container_width=True):
    if not user_input_text.strip():
        st.error("Please input or upload some text.")
    else:
        with st.spinner("⏳ Generating summary..."):
            try:
                result = pipe_chain.invoke({
                    "raw_text": user_input_text,
                    "style": style
                })
                st.subheader("3. Summary Output")
                st.markdown(result["final_output"])

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Check your input or API key.")

# --- Sidebar ---
st.sidebar.markdown("### 💡 How to Run")
st.sidebar.code("streamlit run app.py")
st.sidebar.markdown("🔑 Add to `.env`:")
st.sidebar.code("GROQ_API_KEY=your_groq_api_key")
