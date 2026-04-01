import streamlit as st
import time
from modules.rag import RAGPipeline

# ═══════════════════════════════════════════════
# CHANGE THESE 5 THINGS FOR EACH JOB APPLICATION
# ═══════════════════════════════════════════════

COMPANY = "Cartier"
ROLE = "AI Lead — APAC"
ACCENT_COLOR = "#8B1A2B"
CORPUS_FILE = "data/corpus.md"

SYSTEM_PROMPT = f"""You are Cartier APAC's internal Decision Support assistant.
You help Sales Associates, Boutique Managers, and Regional Leaders make 
informed decisions based on Cartier's internal knowledge base.

Rules:
1. ONLY answer from the provided context
2. Use Cartier terminology: "creation" not "product", "Maison" not "brand", "boutique" not "store"
3. Cite sources using [Source: section name]
4. Never mention discounts, sales, or promotional language
5. If unsure, say so honestly
6. End with an actionable recommendation when relevant"""

SAMPLE_QUESTIONS = [
    "What is the price of Love bracelet in Hong Kong?",
    "What are the key market trends in Korea?",
    "How should I engage a VIC client who hasn't visited in 6 months?",
    "Compare Tank and Santos for a male client",
    "What gifting pieces work for a 30th birthday?",
    "What is our pricing strategy across APAC?",
    "How does stacking culture drive repeat purchases?",
    "What are the brand voice guidelines for WhatsApp?",

]

# ═══════════════════════════════════════════════
# EVERYTHING BELOW STAYS THE SAME — DON'T EDIT
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title=f"{COMPANY} — AI Decision Support",
    page_icon="💎",
    layout="wide"
)

st.markdown(f"""
<style>
    .main-header {{
        font-family: Georgia, serif;
        font-size: 32px;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: 4px;
        text-align: center;
        text-transform: uppercase;
        padding-top: 20px;
    }}
    .sub-header {{
        font-size: 13px;
        color: {ACCENT_COLOR};
        text-align: center;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 30px;
    }}
    .source-badge {{
        background: #F5F0EB;
        border-left: 3px solid {ACCENT_COLOR};
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12px;
        border-radius: 0 4px 4px 0;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_pipeline():
    rag = RAGPipeline(system_prompt=SYSTEM_PROMPT)
    num_chunks = rag.build_from_markdown(CORPUS_FILE)
    return rag, num_chunks

rag, num_chunks = init_pipeline()

st.markdown(f'<div class="main-header">{COMPANY}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">AI Decision Support System — Built for {ROLE}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Knowledge Base")
    st.metric("Chunks Indexed", num_chunks)
    
    st.markdown("---")
    st.markdown("### 💡 Try These Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=q[:30]):
            st.session_state['pending_query'] = q
    
    st.markdown("---")
    st.markdown("### 🧠 AI Concepts")
    st.markdown("""
    - ✅ RAG (Retrieval-Augmented Generation)
    - ✅ Document Chunking (header-aware)
    - ✅ Vector Embeddings & Similarity Search
    - ✅ Context-Grounded Generation
    - ✅ Source Citations
    - ✅ System Prompt Engineering
    """)
    
    st.markdown("---")
    st.caption("Built by Alexandre Lee")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input(f"Ask anything about {COMPANY}...")

if 'pending_query' in st.session_state:
    query = st.session_state.pop('pending_query')

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            start = time.time()
            result = rag.query(query, top_k=5)
            elapsed = time.time() - start
        
        st.markdown(result['answer'])
        
        if result['sources']:
            st.markdown("---")
            st.markdown("**📄 Sources:**")
            for source, sim in zip(result['sources'], result['similarities']):
                if source:
                    st.markdown(
                        f'<div class="source-badge">📌 {source} '
                        f'(relevance: {sim:.2f})</div>',
                        unsafe_allow_html=True
                    )
        
        st.caption(f"⏱️ {elapsed:.2f}s")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result['answer']
    })