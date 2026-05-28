import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# App Layout Configuration
st.set_page_config(page_title="Offline PDF RAG Search", page_icon="📄", layout="centered")

st.title("📄 Offline Document Search (RAG)")
st.caption("Powered by Snowflake, Streamlit & GitHub")
st.markdown("---")

# DYNAMIC CONNECTION HANDLER
@st.cache_resource
def get_snowflake_session():
    try:
        # 1. Try connecting natively inside Snowflake (Streamlit in Snowflake)
        return get_active_session()
    except Exception:
        # 2. Fallback: Connect from GitHub/Streamlit Cloud using secrets
        if "snowflake" in st.secrets:
            return Session.builder.configs(st.secrets["snowflake"]).create()
        else:
            st.error("🔒 Missing Snowflake configuration secrets in Streamlit Cloud Dashboard!")
            st.info("Please verify your `.streamlit/secrets.toml` file or Streamlit Cloud Advanced Settings.")
            st.stop()

# Initialize session
session = get_snowflake_session()

# FORCE THE DATABASE AND SCHEMA CONTEXT EVERY TIME THE APP LOADS
try:
    session.sql("USE DATABASE RAG_DEMO_DB;").collect()
    session.sql("USE SCHEMA DATA;").collect()
except Exception as context_error:
    st.error("❌ Failed to set Snowflake Database Context!")
    st.warning("Make sure RAG_DEMO_DB and SCHEMA DATA exist in your account and your role has access.")
    st.exception(context_error)
    st.stop()

# User Input Box
query_text = st.text_input("🔍 Enter your search query:", placeholder="Type what you want to find inside your PDFs...")

if query_text:
    with st.spinner("Searching document vector database..."):
        try:
            # Escape single quotes in the user input to prevent SQL syntax errors
            safe_query = query_text.replace("'", "''")
            
            # FIXED: Explicitly casting both the query array and column array into VECTOR(FLOAT, 384)
            search_sql = f"""
                WITH search_query AS (
                    SELECT CAST(local_python_embed('{safe_query}') AS VECTOR(FLOAT, 384)) AS q_vec
                )
                SELECT 
                    file_name,
                    chunk_id,
                    chunk_text,
                    VECTOR_COSINE_SIMILARITY(CAST(chunk_vector AS VECTOR(FLOAT, 384)), q.q_vec) AS similarity
                FROM pdf_document_chunks, search_query q
                ORDER BY similarity DESC
                LIMIT 3;
            """
            
            # Execute query and convert the dataset straight into a Pandas DataFrame
            results_df = session.sql(search_sql).to_pandas()
            
            # Render the Results UI
            st.subheader("📚 Top Matching Document Segments")
            
            if results_df.empty:
                st.warning("⚠️ No matches found. Ensure your `pdf_document_chunks` database table contains data rows.")
            else:
                for idx, row in results_df.iterrows():
                    with st.container():
                        st.markdown(f"📁 **Source File:** `{row['FILE_NAME']}` | **Chunk:** `{int(row['CHUNK_ID'])}`")
                        st.info(row['CHUNK_TEXT'])
                        
                        # Format the floating-point vector similarity match percentage
                        match_percentage = float(row['SIMILARITY']) * 100
                        st.caption(f"🎯 Relevance Score: **{match_percentage:.2f}%**")
                        st.divider()
                        
        except Exception as sql_error:
            st.error("🚨 An error occurred during database vector calculation!")
            st.info("This usually means your `local_python_embed` function or `pdf_document_chunks` table needs to be checked in Snowflake.")
            # Print explicit raw traceback details directly to the developer canvas interface
            st.exception(sql_error)
