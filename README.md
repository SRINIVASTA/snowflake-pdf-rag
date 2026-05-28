# 🤖 Dynamic Multi-LLM Conversational RAG Engine

A production-grade, conversational Retrieval-Augmented Generation (RAG) platform built with **Streamlit**, **Snowflake Snowpark**, and the official **Google GenAI SDK**. This application securely connects to a centralized Snowflake knowledge base, runs hybrid vector cosine similarity matching over tokenised textbook chunks, and pipes contextual groundings to state-of-the-art LLMs using an interactive chat interface with historical runtime memory.

---

## 🏗️ Architecture Overview

```text
 ┌────────────────┐       🔍 User Query      ┌────────────────┐
 │                ├─────────────────────────►│                │
 │                │                          │                │
 │  Streamlit UI  │  📚 Contextual Grounding  │ Google Gemini  │
 │  (Chat Layout) │◄─────────────────────────┤ (Official SDK) │
 │                │                          └────────────────┘
 └───────┬────────┘                                  ▲
         │                                           │
         │ ⚡ Vector Similarity SQL Query             │
         ▼                                           │ Text
 ┌───────────────────────────────────────────┐       │ Fragments
 │            Snowflake Data Cloud           ├───────┘
 │                                           │
 │  • Table: pdf_document_chunks             │
 │  • Function: local_python_embed()         │
 │  • Metric: VECTOR_COSINE_SIMILARITY()     │
 └───────────────────────────────────────────┘
```

---

## 🚀 Core Features

- **Conversational Chat Interface:** Replaces traditional static inputs with fluid `st.chat_input` and `st.chat_message` streams.
- **Runtime State Memory:** Tracks and feeds preceding message arrays (`st.session_state`) back to the AI engine for seamless multi-turn context awareness.
- **Enterprise Vector Routing:** Leverages Snowflake native `VECTOR_COSINE_SIMILARITY` metrics combined with keyword bonus scoring (`LIKE` clauses) for precision document lookups.
- **Zero-Friction SDK Migration:** Powered by the official modern `google-genai` framework, avoiding fragile manual HTTP URL constructions and dictionary indexing bugs.
- **Transparent Provenance Tracking:** Embeds interactive reference drawers beneath assistant answers, preserving data lineage as you review conversation history.

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── secrets.toml      # Local credential storage (Never commit to Git!)
├── .gitignore            # Excludes sensitive environment files
├── app.py                # Primary Streamlit platform application script
└── requirements.txt      # Automated server assembly dependencies list
```

---

## ⚙️ Local Setup Instructions

### 1. Clone the Workspace Environment
```bash
git clone https://github.com
cd snowflake-pdf-rag
```

### 2. Install Required Dependencies
Ensure your local system or runtime virtual environment has all necessary modules installed:
```bash
pip install -r requirements.txt
```

### 3. Configure Local Credentials
Create a folder named `.streamlit/` containing a `secrets.toml` file to safely reference database routes and keys:
```toml
# Google Gemini Key Fallback
GOOGLE_API_KEY = "AIzaSy..."

# Hugging Face Token Fallback
HUGGINGFACE_API_KEY = "hf_..."

# Snowflake Connection Matrix
[snowflake]
account = "your_snowflake_account_locator"
user = "your_username"
password = "your_secure_password"
role = "ACCOUNTADMIN"
warehouse = "COMPUTE_WH"
database = "RAG_DEMO_DB"
schema = "DATA"
```

### 4. Execute the Application Terminal Connection
```bash
streamlit run app.py
```

---

## 🌐 Deploying to Streamlit Cloud

1. Commit your codebase changes and push them safely to your remote **GitHub Repository** (ensuring `.streamlit/secrets.toml` remains hidden via your `.gitignore` configuration).
2. Open your [Streamlit Community Cloud Dashboard](https://app-pdf-rag-egximnthuxjnaiqwygkk9l.streamlit.app/) and initialize a new application deployment pointing to your branch repository target.
3. Access **Advanced Settings** ──► **Secrets** panel, copy-paste the entire TOML parameter block from your local configuration, and hit save.
4. Your application will build automatically and change to live mode immediately!

---

## 🛠️ Tech Stack & Versioning

- **Frontend:** Streamlit Core (Layout Canvas)
- **Database Engine:** Snowflake Snowpark Data Cloud (Enterprise Processing Node)
- **AI Processing Layer:** Google Gemini Flash (`gemini-2.5-flash`) via the modern `google-genai` framework ecosystem.
