import streamlit as st
import requests
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# App Layout Configuration
st.set_page_config(page_title="AI Conversational RAG Engine", page_icon="🤖", layout="centered")

st.title("🤖 Dynamic AI Document RAG")
st.caption("Powered by Snowflake, Streamlit & Hugging Face (Hybrid API & Offline Modes)")
st.markdown("---")

# DYNAMIC CONNECTION HANDLER FOR SNOWFLAKE
@st.cache_resource
def get_snowflake_session():
    try:
        return get_active_session()
    except Exception:
        if "snowflake" in st.secrets:
            return Session.builder.configs(st.secrets["snowflake"]).create()
        else:
            st.error("🔒 Missing Snowflake configuration secrets!")
            st.stop()

# HUGGING FACE INFERENCE ENGINE
def ask_free_llm(query, textbook_context, api_key):
    api_url = "https://huggingface.co"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professor teaching data science. Answer the student question using ONLY the textbook facts provided. 
If the text does not contain the answer, say "I cannot find that in the textbook." Keep it concise and accurate.

Textbook Context:
{textbook_context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        response = requests.post(api_url, headers=headers, json={"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.2}})
        output = response.json()
        
        if isinstance(output, list) and len(output) > 0 and "generated_text" in output:
            raw_text = output["generated_text"]
            if "<|start_header_id|>assistant<|end_header_id|>" in raw_text:
                return raw_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            return raw_text
        return "⚠️ *The free AI cluster is temporarily busy. Please re-submit your query in a few moments.*"
    except Exception as e:
        return f"⚠️ *API Connection Error: {str(e)}*"

# Initialize Snowflake Session
session = get_snowflake_session()
try:
    session.sql("USE DATABASE RAG_DEMO_DB;").collect()
    session.sql("USE SCHEMA DATA;").collect()
except Exception:
    st.error("❌ Failed to set Snowflake Database Context!")
    st.stop()

# SIDEBAR: DYNAMIC API CREDENTIAL CONTROLS
with st.sidebar:
    st.header("🔑 Authentication Mode")
    
    # 1. Look for pre-configured secret token
    secret_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
    
    # 2. Provide a manual manual input field on-screen
    manual_key = st.text_input(
        "Enter Hugging Face Token manually:", 
        type="password", 
        placeholder="hf_...",
        help="Paste a free Read token from huggingface.co/settings/tokens to enable Conversational summaries."
    )
    
    # Determine which key to activate
    active_api_key = manual_key if manual_key else secret_key
    
    if active_api_key:
        st.success("🤖 **Conversational Mode Active** (AI summary enabled)")
    else:
        st.info("📄 **Pure Search Mode Active** (Displaying source chunks directly without API)")
        
    st.markdown("---")
    st.header("📊 Database Status")
    try:
        stats_df = session.sql("SELECT COUNT(*) as total FROM pdf_document_chunks;").to_pandas()
        st.success(f"🧩 Indexed Textbook Chunks: **{int(stats_df['TOTAL'].iloc[0])}**")
    except Exception:
        st.warning("Could not read real-time database rows.")
        
    st.markdown("---")
    min_relevance = st.slider("Minimum Relevance Cutoff (%)", min_value=0, max_value=100, value=20)

# User Query Box
query_text = st.text_input("🔍 Ask a question about your textbook:", placeholder="e.g., What is a random graph?")

if query_text:
    with st.spinner("Processing request..."):
        try:
            safe_query = query_text.replace("'", "''")
            
            # Word-by-word token split scoring
            words = [w.strip().lower() for w in safe_query.split() if len(w.strip()) > 2]
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
            results_df = raw_results_df[raw_results_df["SIMILARITY"] * 100 >= min_relevance]
            
            if results_df.empty:
                st.warning("⚠️ No matching content found above your cutoff. Try lowering the sidebar slider.")
            else:
                # OPTION 1: CONVERSATIONAL ANSWER GENERATION (If API Key Present)
                if active_api_key:
                    context_block = "\n".join([row['CHUNK_TEXT'] for idx, row in results_df.iterrows()])
                    answer = ask_free_llm(query_text, context_block, active_api_key)
                    
                    st.subheader("💡 AI Professor Answer")
                    st.write(answer)
                    st.markdown("---")
                
                # OPTION 2: SOURCE DATA RETRIEVAL (Always Runs)
                st.subheader("📚 Verified References Used From Snowflake")
                for idx, row in results_df.iterrows():
                    with st.expander(f"📄 {row['FILE_NAME']} (Chunk {int(row['CHUNK_ID'])}) - Match: {float(row['SIMILARITY'])*100:.1f}%"):
                        st.info(row['CHUNK_TEXT'])
                        
        except Exception as e:
            st.error("🚨 System Execution Error")
            st.exception(e)
