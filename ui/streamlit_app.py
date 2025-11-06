# ui/streamlit_app.py
import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="AI Market Analyst", page_icon="📊", layout="wide")

st.title("🤖 AI Market Analyst")
st.markdown("Analyze market research documents with AI-powered insights")

# Sidebar for task selection
task = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Autonomous Analysis", "General Q&A", "Research Summary", "Data Extraction"]
)

if task == "Autonomous Analysis":
    st.header("Autonomous Analysis")
    query = st.text_area("Enter your query:", "What are the main growth opportunities?")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing your query..."):
            try:
                response = requests.post(f"{API_BASE}/autonomous", 
                                       json={"query": query})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Task Type: {result['task_type']}")
                    st.info(f"Reasoning: {result['reasoning']}")
                    st.subheader("Response:")
                    st.write(result['response'])
                else:
                    st.error("Error processing request")
            except Exception as e:
                st.error(f"Connection error: {e}")

elif task == "General Q&A":
    st.header("General Q&A")
    question = st.text_input("Ask a question about the market research:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Finding answer..."):
            try:
                response = requests.post(f"{API_BASE}/qa", 
                                       json={"question": question})
                if response.status_code == 200:
                    result = response.json()
                    st.success("Answer:")
                    st.write(result['answer'])
                    with st.expander("View Sources"):
                        for source in result['sources']:
                            st.text(source[:200] + "...")
                else:
                    st.error("Error processing question")
            except Exception as e:
                st.error(f"Connection error: {e}")

elif task == "Research Summary":
    st.header("Market Research Summary")
    
    if st.button("Generate Summary"):
        with st.spinner("Generating comprehensive summary..."):
            try:
                response = requests.post(f"{API_BASE}/summary")
                if response.status_code == 200:
                    result = response.json()
                    st.success("Executive Summary:")
                    st.write(result['summary'])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Key Findings")
                        for finding in result['key_findings']:
                            st.write(f"• {finding}")
                    
                    with col2:
                        st.subheader("Recommendations") 
                        for recommendation in result['recommendations']:
                            st.write(f"• {recommendation}")
                else:
                    st.error("Error generating summary")
            except Exception as e:
                st.error(f"Connection error: {e}")

elif task == "Data Extraction":
    st.header("Structured Data Extraction")
    
    if st.button("Extract Market Data"):
        with st.spinner("Extracting structured data..."):
            try:
                response = requests.post(f"{API_BASE}/extract")
                if response.status_code == 200:
                    result = response.json()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Market Overview")
                        data = result['extracted_data']
                        st.metric("Market Size", data.get('market_size', 'N/A'))
                        st.metric("Growth Rate", data.get('growth_rate', 'N/A'))
                        
                        st.subheader("Market Share")
                        for company, share in data.get('market_share', {}).items():
                            st.write(f"**{company}**: {share}")
                    
                    with col2:
                        st.subheader("SWOT Analysis")
                        swot = data.get('swot_analysis', {})
                        
                        for category, items in swot.items():
                            with st.expander(f"{category.title()}"):
                                for item in items:
                                    st.write(f"• {item}")
                else:
                    st.error("Error extracting data")
            except Exception as e:
                st.error(f"Connection error: {e}")

st.sidebar.markdown("---")
st.sidebar.info(
    "This AI Market Analyst can process market research documents and provide "
    "insights through Q&A, summaries, and structured data extraction."
)