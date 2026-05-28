import streamlit as st
import requests
import time
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from google import genai  # Official Google SDK

# App Layout Configuration
st.set_page_config(page_title="Multi-LLM Conversational RAG Engine", page_icon="🤖", layout="wide")

st.title("🤖 Dynamic AI Document RAG")
st.caption("Powered by Snowflake, Streamlit & Multi-LLM Routing (Free Tier APIs)")
st.markdown("---")

# DYNAMIC CONNECTION HANDLER FOR SNOWFLAKE
@st.cache_resource
def get_snowflake_session():
    try:
        # 1. Try connecting natively inside Snowflake
        return get_active_session()
    except Exception:
        # 2. Fallback: Connect using Streamlit Cloud Secrets dashboard
        if "snowflake" in st.secrets:
            return Session.builder.configs(st.secrets["snowflake"]).create()
        else:
            st.error("🔒 Missing Snowflake configuration secrets!")
            st.stop()

# HUGGING FACE INFERENCE ENGINE (MISTRAL)
def ask_free_hf(query, textbook_context, api_key):
    api_url = "https://huggingface.co"
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    prompt = f"<s>[INST] You are a professor teaching data science. Answer the student question using ONLY the textbook facts provided below. If the text does not contain the answer, say 'I cannot find that in the textbook.' Keep your response concise, clear, and accurate.\n\nTextbook Context:\n{textbook_context}\n\nQuestion: {query} [/INST]"

    for attempt in range(3):
        try:
            response = requests.post(api_url, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.2}}, timeout=15)
            if response.status_code == 401:
                return "❌ *Hugging Face API Error: Unauthorized Token.*"
            if response.status_code == 503:
                time.sleep(5)
                continue
            if response.status_code != 200:
                return f"⚠️ *Hugging Face server error code: {response.status_code}*"
            output = response.json()
            if isinstance(output, list) and len(output) > 0 and "generated_text" in output:
                raw_text = output["generated_text"]
                return raw_text.split("[/INST]")[-1].strip() if "[/INST]" in raw_text else raw_text
        except Exception as e:
            return f"⚠️ *HF Connection Error: {str(e)}*"
    return "❌ *The Hugging Face model cluster timed out.*"

# GOOGLE GEMINI INFERENCE ENGINE (OFFICIAL SDK IMPLEMENTATION WITH OPTIMIZED PROMPT)
def ask_gemini(query, textbook_context, api_key, model_option="gemini-2.5-flash", chat_history=""):
    try:
        client = genai.Client(api_key=api_key.strip())
        
        prompt = f"""You are a helpful and knowledgeable data science professor. 
Answer the student's question using the textbook context provided below.

INSTRUCTIONS:
1. Provide a comprehensive, clear, and technical explanation based on the context.
2. If the context does not explicitly define the term but discusses relevant properties, equations, or examples, use that information to explain the concept.
3. Keep recent conversation history in mind to give coherent answers.
4. Only say "I cannot find that in the textbook" if the provided context is completely unrelated to the topic of the question.

RECENT CONVERSATION HISTORY:
{chat_history}

TEXTBOOK CONTEXT:
{textbook_context}

STUDENT QUESTION:
{query}"""

        response = client.models.generate_content(
            model=model_option,
            contents=prompt,
        )
        
        if response.text:
            return response.text.strip()
        return "⚠️ *Google Gemini returned an empty response.*"
        
    except Exception as e:
        return f"⚠️ *Google Gemini SDK Error: {str(e)}*"

# Initialize Snowflake Session
session = get_snowflake_session()
try:
    session.sql("USE DATABASE RAG_DEMO_DB;").collect()
    session.sql("USE SCHEMA DATA;").collect()
except Exception:
    st.error("❌ Failed to set Snowflake Database Context!")
    st.stop()

# INITIALIZE CHAT HISTORY RUNTIME STATES
if "messages" not in st.session_state:
    st.session_state.messages = []

# SIDEBAR PANEL CONFIGURATION
with st.sidebar:
    st.header("🔑 Authentication Mode")
    provider = st.radio("Select AI Engine Provider:", ["Google Gemini", "Hugging Face", "Pure Offline (No AI)"], index=0)
    active_api_key = ""
    gemini_model = "gemini-2.5-flash"
    
    if provider == "Google Gemini":
        secret_gemini = st.secrets.get("GOOGLE_API_KEY", "")
        manual_gemini = st.text_input("Enter Google API Key manually:", type="password", placeholder="AIzaSy...")
        active_api_key = manual_gemini if manual_gemini else secret_gemini
        gemini_model = st.selectbox("Select Gemini Model Version:", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"])
        
        if active_api_key: st.success(f"🟢 **Gemini Live ({gemini_model})**")
        else: st.warning("⚠️ **Missing Gemini API Key**")
            
    elif provider == "Hugging Face":
        secret_hf = st.secrets.get("HUGGINGFACE_API_KEY", "")
        manual_hf = st.text_input("Enter Hugging Face Token manually:", type="password", placeholder="hf_...")
        active_api_key = manual_hf if manual_hf else secret_hf
        
        if active_api_key: st.success("🟢 **Mistral AI Live**")
        else: st.warning("⚠️ **Missing HF Token**")
    else:
        st.info("🔵 **Pure Offline Mode Selected**")
        
    st.markdown("---")
    st.header("📤 Document Ingestion")
    uploaded_file = st.file_uploader("Upload new textbook PDF:", type=["pdf"])
    if uploaded_file is not None:
        st.info("🔄 Document received. Hook this up to your chunking/embedding stage.")
        
    st.markdown("---")
    st.header("📊 Database Status")
    try:
        stats_df = session.sql("SELECT COUNT(*) as total_chunks, COUNT(DISTINCT file_name) as total_files FROM pdf_document_chunks;").to_pandas()
        total_chunks = int(stats_df.iloc[0, 0])
        total_files = int(stats_df.iloc[0, 1])
        col_files, col_chunks = st.columns(2)
        col_files.metric("Total Files", f"📁 {total_files}")
        col_chunks.metric("Total Chunks", f"🧩 {total_chunks}")
    except Exception:
        st.warning("Could not read real-time database rows.")

# DISPLAY HISTORICAL CHAT LOGS NATIVELY ON REFRESH
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "references" in msg:
            with st.expander("📚 View Snowflake References Used"):
                for ref in msg["references"]:
                    st.markdown(f"**File:** {ref['file']} (Chunk {ref['id']}) — **Match:** {ref['score']:.1f}%")
                    st.info(ref['text'])

# INTERACTIVE CHAT PROCESSING LOOP
if prompt_input := st.chat_input("Ask a question about your textbook (e.g., What is a random graph?)"):
    
    # 1. Render user text immediately 
    with st.chat_message("user"):
        st.write(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    # 2. Vector Context Retrieval Block
    with st.spinner("Searching Snowflake Knowledge Base..."):
        try:
            safe_query = prompt_input.replace("'", "''")
            words = [w.strip().lower() for w in safe_query.split() if len(w.strip()) >= 2]
            like_clauses = " + ".join([f"(CASE WHEN LOWER(chunk_text) LIKE '%{w}%' THEN 0.15 ELSE 0.0 END)" for w in words])
            if not like_clauses: like_clauses = "0.0"

            search_sql = f"""
                WITH search_query AS (
                    SELECT CAST(local_python_embed('{safe_query}') AS VECTOR(FLOAT, 384)) AS q_vec
                )
                SELECT 
                    file_name, chunk_id, chunk_text,
                    ((VECTOR_COSINE_SIMILARITY(CAST(chunk_vector AS VECTOR(FLOAT, 384)), q.q_vec) + 1.0) / 2.0 * 0.5) + 
                    (CASE WHEN ({like_clauses}) > 0.5 THEN 0.5 ELSE ({like_clauses}) END) AS similarity
                FROM pdf_document_chunks, search_query q
                ORDER BY similarity DESC LIMIT 3;
            """
            raw_results_df = session.sql(search_sql).to_pandas()
            raw_results_df.columns = [c.upper() for c in raw_results_df.columns]
            results_df = raw_results_df.head(3)
            
            # Format references metadata dictionary list
            current_references = []
            context_block = ""
            if not results_df.empty:
                context_block = "\n".join([row['CHUNK_TEXT'] for _, row in results_df.iterrows()])
                for _, row in results_df.iterrows():
                    current_references.append({
                        "file": row['FILE_NAME'],
                        "id": int(row['CHUNK_ID']),
                        "score": float(row['SIMILARITY']) * 100,
                        "text": row['CHUNK_TEXT']
                    })
            
            # 3. LLM Synthesis Generation Block
            with st.chat_message("assistant"):
                # Compile preceding interaction message block history
                history_summary = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-5:]])
                
                if provider == "Google Gemini" and active_api_key:
                    answer = ask_gemini(prompt_input, context_block, active_api_key, model_option=gemini_model, chat_history=history_summary)
                elif provider == "Hugging Face" and active_api_key:
                    answer = ask_free_hf(prompt_input, context_block, active_api_key)
                else:
                    answer = "📄 **Offline Retrieval Output:** Text fragments matched successfully. Select an AI engine provider to synthesize answers."
                
                st.write(answer)
                # Render the expandable reference drawer under the assistant text block
                if current_references:
                    with st.expander("📚 View Snowflake References Used"):
                        for ref in current_references:
                            st.markdown(f"**File:** {ref['file']} (Chunk {ref['id']}) — **Match:** {ref['score']:.1f}%")
                            st.info(ref['text'])
            
            # Append complete execution package to app state records
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "references": current_references
            })
            
        except Exception as e:
            st.error("🚨 System Execution Error")
            st.exception(e)
