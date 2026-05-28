import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# App Heading
st.title("📄 Offline Document Search (RAG)")
st.caption("Powered by Snowflake, Streamlit & GitHub")

# DYNAMIC CONNECTION HANDLER
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
            st.error("Missing Snowflake configuration secrets!")
            st.stop()

# Initialize session
session = get_snowflake_session()

# User Input Box
query_text = st.text_input("Enter your search query:", placeholder="What are you looking for?")

if query_text:
    with st.spinner("Searching document vectors..."):
        # Fixed query syntax using your database identifiers
        search_sql = f"""
            WITH search_query AS (
                SELECT RAG_DEMO_DB.DATA.local_python_embed('{query_text.replace("'", "''")}') AS q_vec
            )
            SELECT 
                file_name,
                chunk_id,
                chunk_text,
                VECTOR_COSINE_SIMILARITY(chunk_vector, q.q_vec) AS similarity
            FROM RAG_DEMO_DB.DATA.pdf_document_chunks, search_query q
            ORDER BY similarity DESC
            LIMIT 3;
        """
        
        # Execute the query
        results_df = session.sql(search_sql).to_pandas()
        
        # Render the Results UI
        st.subheader("📚 Top matching text segments:")
        if results_df.empty:
            st.warning("No matches found. Make sure your database table contains data.")
        for idx, row in results_df.iterrows():
            with st.container():
                st.markdown(f"**Source File:** `{row['FILE_NAME']}` (Chunk {int(row['CHUNK_ID'])})")
                st.info(row['CHUNK_TEXT'])
                st.caption(f"Confidence Match Score: {row['SIMILARITY']:.4f}")
                st.divider()
