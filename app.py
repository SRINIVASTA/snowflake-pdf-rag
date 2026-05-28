import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

# App Layout Configuration
st.set_page_config(page_title="Advanced PDF RAG Search", page_icon="📄", layout="centered")

st.title("📄 Advanced Document Search Engine")
st.caption("Powered by Snowflake, Streamlit & GitHub (Zero External APIs)")
st.markdown("---")

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
            st.error("🔒 Missing Snowflake configuration secrets in Streamlit Cloud Dashboard!")
            st.stop()

# Initialize session
session = get_snowflake_session()

# FORCE THE DATABASE AND SCHEMA CONTEXT EVERY TIME THE APP LOADS
try:
    session.sql("USE DATABASE RAG_DEMO_DB;").collect()
    session.sql("USE SCHEMA DATA;").collect()
except Exception as context_error:
    st.error("❌ Failed to set Snowflake Database Context!")
    st.stop()

# SIDEBAR: Stats, Filters, and Advanced Relevance Controls
with st.sidebar:
    st.header("📊 Database Statistics")
    try:
        stats_df = session.sql("""
            SELECT 
                COUNT(*) as total_chunks, 
                COUNT(DISTINCT file_name) as total_files 
            FROM pdf_document_chunks;
        """).to_pandas()
        
        total_chunks = int(stats_df["TOTAL_CHUNKS"].iloc[0])
        total_files = int(stats_df["TOTAL_FILES"].iloc[0])
        
        col_files, col_chunks = st.columns(2)
        col_files.metric("Total Files", f"📁 {total_files}")
        col_chunks.metric("Total Chunks", f"🧩 {total_chunks}")
    except Exception:
        st.warning("⚠️ Could not read real-time table statistics from Snowflake.")
    
    st.markdown("---")
    st.header("🎛️ Search Adjustments")
    
    # NEW: Relevance Threshold Slider to filter out junk matches
    min_relevance = st.slider(
        "Minimum Relevance Cutoff (%)", 
        min_value=0, 
        max_value=100, 
        value=35,
        help="Any document fragment scoring below this percentage will be automatically filtered out."
    )
    
    st.markdown("---")
    st.header("🔍 Document Filters")
    try:
        distinct_files_df = session.sql("SELECT DISTINCT file_name FROM pdf_document_chunks ORDER BY file_name;").to_pandas()
        file_options = ["All Documents"] + distinct_files_df["FILE_NAME"].tolist()
    except Exception:
        file_options = ["All Documents"]

    selected_file = st.selectbox("Select document to search within:", options=file_options)
    st.divider()

# User Input Box
query_text = st.text_input("🔍 Enter your textbook search query:", placeholder="Type what you want to find inside your files...")

if query_text:
    with st.spinner("Searching document database chunks..."):
        try:
            # Escape single quotes in user input
            safe_query = query_text.replace("'", "''")
            
            # Dynamically build the optional WHERE clause for file filtering
            file_filter_clause = ""
            if selected_file != "All Documents":
                file_filter_clause = f"AND file_name = '{selected_file.replace("'", "''")}'"

            # Re-calibrated scoring logic designed to hit 100% on high-quality keyword intersections
            search_sql = f"""
                WITH search_query AS (
                    SELECT CAST(local_python_embed('{safe_query}') AS VECTOR(FLOAT, 384)) AS q_vec
                )
                SELECT 
                    file_name,
                    chunk_id,
                    chunk_text,
                    -- Boosted Equation: Normalizes vector math and awards a large bonus for keyword alignment
                    (
                        (VECTOR_COSINE_SIMILARITY(CAST(chunk_vector AS VECTOR(FLOAT, 384)), q.q_vec) + 1.0) / 2.0 * 0.4
                    ) + 
                    (
                        CASE WHEN LOWER(chunk_text) LIKE '%{safe_query.lower()}%' THEN 0.6 ELSE 0.0 END
                    ) AS similarity
                FROM pdf_document_chunks, search_query q
                WHERE 1=1 {file_filter_clause}
                ORDER BY similarity DESC
                LIMIT 5;
            """
            
            # Execute query and convert to a Pandas DataFrame
            raw_results_df = session.sql(search_sql).to_pandas()
            
            # Filter the Pandas dataframe based on the slider state selection
            results_df = raw_results_df[raw_results_df["SIMILARITY"] * 100 >= min_relevance]
            
            # Render the Results UI
            st.subheader("📚 Top Matching Document Segments")
            
            if results_df.empty:
                st.warning(f"⚠️ No matches met your {min_relevance}% relevance cutoff. Try lowering the sidebar slider or expanding your search query terms.")
            else:
                for idx, row in results_df.iterrows():
                    with st.container():
                        st.markdown(f"📁 **Source File:** `{row['FILE_NAME']}` | **Chunk:** `{int(row['CHUNK_ID'])}`")
                        st.info(row['CHUNK_TEXT'])
                        
                        # Format the floating-point vector similarity match percentage
                        match_percentage = float(row['SIMILARITY']) * 100
                        # Cap values at 100% for aesthetic UI uniformity
                        if match_percentage > 100.0:
                            match_percentage = 100.0
                            
                        st.caption(f"🎯 Relevance Score: **{match_percentage:.2f}%**")
                        st.divider()
                        
        except Exception as sql_error:
            st.error("🚨 An error occurred during database vector calculation!")
            st.exception(sql_error)
