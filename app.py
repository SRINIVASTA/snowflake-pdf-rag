import streamlit as st
from snowflake.snowpark.context import get_active_session

# App Heading
st.title("📄 Offline Document Search (RAG)")
st.caption("Powered by Snowflake, Streamlit & GitHub")

# Connect directly to your existing Snowflake session context
session = get_active_session()

# User Input Box
query_text = st.text_input("Enter your search query:", placeholder="What are you looking for?")

if query_text:
    with st.spinner("Searching document vectors..."):
        # Querying the database chunks table using the offline vector math function
        search_sql = f"""
            WITH search_query AS (
                SELECT local_python_embed('{query_text.replace("'", "''")}') AS q_vec
            )
            SELECT 
                file_name,
                chunk_id,
                chunk_text,
                VECTOR_COSINE_SIMILARITY(chunk_vector, q.q_vec) AS similarity
            FROM rag_demo_db.data.pdf_document_chunks, search_query q
            ORDER BY similarity DESC
            LIMIT 3;
        """
        
        # Execute the query and convert to a Pandas DataFrame
        results_df = session.sql(search_sql).to_pandas()
        
        # Render the Results UI
        st.subheader("📚 Top matching text segments:")
        for idx, row in results_df.iterrows():
            with st.container():
                st.markdown(f"**Source File:** `{row['FILE_NAME']}` (Chunk {int(row['CHUNK_ID'])})")
                st.info(row['CHUNK_TEXT'])
                st.caption(f"Confidence Match Score: {row['SIMILARITY']:.4f}")
                st.divider()
