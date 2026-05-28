import streamlit as st
import requests
import time
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session
from google import genai  # Official Google SDK

# App Layout Configuration
st.set_page_config(page_title="Multi-LLM Conversational RAG Engine", page_icon="🤖", layout="centered")

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
        # 2. Fallback: Connect from GitHub/Streamlit Cloud using secrets
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

# GOOGLE GEMINI INFERENCE ENGINE (OFFICIAL SDK IMPLEMENTATION)
def ask_gemini(query, textbook_context, api_key, model_option="gemini-2.5-flash"):
    try:
        # Initialize official GenAI client using SDK standards
        client = genai.Client(api_key=api_key.strip())
        
        prompt = f"""You are a professor teaching data science. 
Answer the student question using ONLY the textbook facts provided below. 
If the text does not contain the answer, say "I cannot find that in the textbook." 
Keep your response concise, clear, and accurate.

Textbook Context:
{textbook_context}

Question: {query}"""

        # Native SDK execution path
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

# SIDEBAR: DYNAMIC MULTI-LLM API CREDENTIAL CONTROLS
with st.sidebar:
    st.header("🔑 Authentication Mode")
    
    # Provider Choice Selection
    provider = st.radio(
        "Select AI Engine Provider:", 
        ["Google Gemini", "Hugging Face", "Pure Offline (No AI)"],
        index=0
    )
    active_api_key = ""
    gemini_model = "gemini-2.5-flash"  # Default fallback assignment definition
    
    if provider == "Google Gemini":
        secret_gemini = st.secrets.get("GOOGLE_API_KEY", "")
        manual_gemini = st.text_input("Enter Google API Key manually:", type="password", placeholder="AIzaSy...")
        active_api_key = manual_gemini if manual_gemini else secret_gemini
        
        # Dynamic dropdown selector for model versions
        gemini_model = st.selectbox(
            "Select Gemini Model Version:",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
        )
        
        if active_api_key:
            st.success(f"🟢 **Gemini Engine Live ({gemini_model})**")
        else:
            st.warning("⚠️ **Missing Gemini API Key**")
            st.info("💡 Paste your key above or add `GOOGLE_API_KEY` to your secrets configuration file.")
            
    elif provider == "Hugging Face":
        secret_hf = st.secrets.get("HUGGINGFACE_API_KEY", "")
        manual_hf = st.text_input("Enter Hugging Face Token manually:", type="password", placeholder="hf_...")
        active_api_key = manual_hf if manual_hf else secret_hf
        
        if active_api_key:
            st.success("🟢 **Mistral AI Engine Live**")
        else:
            st.warning("⚠️ **Missing Hugging Face Token**")
            st.info("💡 Paste a free Read Token above or add `HUGGINGFACE_API_KEY` to secrets.")
            
    else:
        st.info("🔵 **Pure Offline Mode Selected**\n\nDocument chunks will be parsed cleanly from Snowflake without passing through third-party LLM systems.")
        
    st.markdown("---")
    st.header("📊 Database Status")
    try:
        stats_df = session.sql("""
            SELECT 
                COUNT(*) as total_chunks, 
                COUNT(DISTINCT file_name) as total_files 
            FROM pdf_document_chunks;
        """).to_pandas()
        
        # Safe structural integer indexing prevents column case mismatch bugs
        total_chunks = int(stats_df.iloc[0, 0])
        total_files = int(stats_df.iloc[0, 1])
        
        col_files, col_chunks = st.columns(2)
        col_files.metric("Total Files", f"📁 {total_files}")
        col_chunks.metric("Total Chunks", f"🧩 {total_chunks}")
    except Exception:
        st.warning("Could not read real-time database rows.")
        
    st.markdown("---")
    min_relevance = st.slider("Minimum Relevance Cutoff (%)", min_value=0, max_value=100, value=20)


# User Query Box
query_text = st.text_input("🔍 Ask a question about your textbook:", placeholder="e.g., What is a random graph?")

if query_text:
    with st.spinner("Processing framework vector alignments..."):
        try:
            safe_query = query_text.replace("'", "''")
            
            # ✅ FIX 1: Allow 2-letter words so technical acronyms/short queries don't break
            words = [w.strip().lower() for w in safe_query.split() if len(w.strip()) >= 2]
            like_clauses = " + ".join([f"(CASE WHEN LOWER(chunk_text) LIKE '%{w}%' THEN 0.15 ELSE 0.0 END)" for w in words])
            if not like_clauses: like_clauses = "0.0"

            search_sql = f"""
                WITH search_query AS (
                    SELECT CAST(local_python_embed('{safe_query}') AS VECTOR(FLOAT, 384)) AS q_vec
                )
                SELECT 
                    file_name,
                    chunk_id,
                    chunk_text,
                    ((VECTOR_COSINE_SIMILARITY(CAST(chunk_vector AS VECTOR(FLOAT, 384)), q.q_vec) + 1.0) / 2.0 * 0.5) + 
                    (CASE WHEN ({like_clauses}) > 0.5 THEN 0.5 ELSE ({like_clauses}) END) AS similarity
                FROM pdf_document_chunks, search_query q
                ORDER BY similarity DESC
                LIMIT 3;
            """
            
            raw_results_df = session.sql(search_sql).to_pandas()
            raw_results_df.columns = [c.upper() for c in raw_results_df.columns]
            
            # ✅ FIX 2: Completely bypass the percentage filter so the AI always gets data
            results_df = raw_results_df.head(3) 
            
            if results_df.empty:
                st.warning("⚠️ No matching content found in your database table at all.")
            else:
                context_block = "\n".join([row['CHUNK_TEXT'] for idx, row in results_df.iterrows()])
                
                # ROUTE SYNTHESIS REQUEST BASED ON ACTIVE PROVIDER
                if provider == "Google Gemini" and active_api_key:
                    answer = ask_gemini(query_text, context_block, active_api_key, model_option=gemini_model)
                    st.subheader("💡 AI Professor Answer (Gemini)")
                    st.write(answer)
                    st.markdown("---")
                elif provider == "Hugging Face" and active_api_key:
                    answer = ask_free_hf(query_text, context_block, active_api_key)
                    st.subheader("💡 AI Professor Answer (Mistral)")
                    st.write(answer)
                    st.markdown("---")
                
                # ALWAYS RUN NATIVE SOURCE RETRIEVAL VIEW
                st.subheader("📚 Verified References Used From Snowflake")
                for idx, row in results_df.iterrows():
                    with st.expander(f"📄 {row['FILE_NAME']} (Chunk {int(row['CHUNK_ID'])}) - Match: {float(row['SIMILARITY'])*100:.1f}%"):
                        st.info(row['CHUNK_TEXT'])
                        
                        clean_filename = f"chunk_{int(row['CHUNK_ID'])}.txt"
                        st.download_button(
                            label="💾 Save chunk text",
                            data=row['CHUNK_TEXT'],
                            file_name=clean_filename,
                            mime="text/plain",
                            key=f"dl_{int(row['CHUNK_ID'])}_{idx}"
                        )
                        
        except Exception as e:
            st.error("🚨 System Execution Error")
            st.exception(e)
