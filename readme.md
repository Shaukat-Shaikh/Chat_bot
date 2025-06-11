# 🧠 Document Summarization Service

![Document Summarization](images.png)

A simple, intelligent web app that uses **Gemini 2.0 Flash LLM** and **LangGraph orchestration** to summarize documents. Users can upload `.txt` files or enter text directly, and choose the summarization style.

---

## 🌟 Features

- **📝 Text Input**  
  Paste text into a text area for instant summarization.

- **📁 File Upload**  
  Upload plain `.txt` files for document input.

- **🎯 Summarization Styles**
  - **Brief**: Short, concise summary  
  - **Detailed**: Rich, comprehensive summary  
  - **Bullet Points**: Key takeaways in bullet format

- **🤖 AI-Powered Engine**  
  Powered by **Gemini 2.0 Flash** for high-quality summaries.

- **🧩 LangGraph Orchestration**  
  Efficient workflow control with LangGraph.

- **💻 Streamlit Interface**  
  Clean, user-friendly, and responsive UI.

- **🛡️ Basic Input Validation**  
  Ensures no empty input is sent to the LLM.

- **🚨 Graceful Error Handling**  
  Detects and reports API issues, invalid files, and missing input.

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.7+
- `pip` (Python package manager)

### 🛠 Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
