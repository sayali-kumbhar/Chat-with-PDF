# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.prompts import PromptTemplate
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import PromptTemplate
# import os

import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ─── Page Config ───
st.set_page_config(
    page_title="📄 Chat with PDF — AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# ─── Custom CSS for Professional Look ───
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
    }
    .stats-box {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ─── API Key ───
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# ─── Helper Functions ───
@st.cache_data
def extract_pdf_text(pdf_files):
    """Extract text from uploaded PDFs"""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

@st.cache_data
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

@st.cache_resource
def create_vector_store(_chunks):
    """Create FAISS vector store from text chunks"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(_chunks, embedding=embeddings)
    return vector_store

def process_question(question, vector_store):
    """Retrieve context + call Gemini directly — no chains needed!"""

    # Step 1: Semantic search → get relevant chunks
    docs = vector_store.similarity_search(question, k=4)

    # Step 2: Merge chunks into one context string
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 3: Build the prompt manually
    prompt = f"""
    You are an expert AI assistant specialized in analyzing documents.
    Answer the question in detail based ONLY on the context below.
    If the answer is not in the context, say:
    "⚠️ This information is not available in the uploaded document."
    Use bullet points or numbered lists where appropriate.

    Context:
    {context}

    Question:
    {question}

    Detailed Answer:
    """

    # Step 4: Call Gemini directly
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=1.0
    )

    response = model.invoke([HumanMessage(content=prompt)])
    return response.content
    
# ─── Main App UI ───
st.markdown('<p class="main-header">📄 Chat with Your PDF</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload any PDF and ask questions — Powered by Google Gemini + RAG Pipeline</p>', unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/pdf.png", width=100)
    st.title("📁 Document Upload")
    
    pdf_files = st.file_uploader(
        "Upload your PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to chat with"
    )
    
    if pdf_files:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("🔄 Processing your documents..."):
                # Step 1: Extract text
                raw_text = extract_pdf_text(pdf_files)
                
                if not raw_text.strip():
                    st.error("❌ No text found in the PDF(s). Please upload a text-based PDF.")
                else:
                    # Step 2: Split into chunks
                    chunks = split_text_into_chunks(raw_text)
                    
                    # Step 3: Create vector store
                    vector_store = create_vector_store(tuple(chunks))
                    st.session_state["vector_store"] = vector_store
                    st.session_state["doc_processed"] = True
                    st.session_state["num_chunks"] = len(chunks)
                    st.session_state["num_pages"] = sum(len(PdfReader(pdf).pages) for pdf in pdf_files)
                    
                    st.success("✅ Documents processed successfully!")
    
    # Stats
    if st.session_state.get("doc_processed"):
        st.markdown("---")
        st.markdown("### 📊 Document Stats")
        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.get("num_pages", 0))
        col2.metric("Chunks", st.session_state.get("num_chunks", 0))
    
    st.markdown("---")
    st.markdown("### 🛠️ Built With")
    st.markdown("""
    - 🧠 Google Gemini 1.5 Flash
    - 🔗 LangChain
    - 📦 FAISS Vector Store
    - 🤗 HuggingFace Embeddings
    - 🎈 Streamlit
    """)
    
    st.markdown("---")
    st.markdown(
        "Made with ❤️ by [Sayali Kumbhar](https://linkedin.com/in/sayali-kumbhar)"
    )

# ─── Chat Interface ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show example questions if no chat yet
if not st.session_state.messages and st.session_state.get("doc_processed"):
    st.markdown("### 💡 Try asking:")
    cols = st.columns(3)
    example_qs = [
        "📝 Summarize this document",
        "🔑 What are the key points?",
        "📊 List all important data/numbers"
    ]
    for col, q in zip(cols, example_qs):
        if col.button(q, use_container_width=True):
            st.session_state["auto_question"] = q

# Chat input
question = st.chat_input("Ask anything about your PDF...")

# Handle auto-question from example buttons
if st.session_state.get("auto_question"):
    question = st.session_state.pop("auto_question")

if question:
    if not st.session_state.get("doc_processed"):
        st.warning("⚠️ Please upload and process a PDF first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = process_question(
                    question, 
                    st.session_state["vector_store"]
                )
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

# ─── Landing State ───
if not pdf_files:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📤 Step 1")
        st.markdown("Upload your PDF document(s) using the sidebar")
    
    with col2:
        st.markdown("### ⚡ Step 2")
        st.markdown("Click 'Process Documents' to analyze with AI")
    
    with col3:
        st.markdown("### 💬 Step 3")
        st.markdown("Ask any question and get instant answers")
    
    st.markdown("---")
    st.markdown("#### 🏆 Use Cases")
    st.markdown("""
    | Industry | Use Case |
    |---|---|
    | 📚 Education | Chat with textbooks, research papers |
    | ⚖️ Legal | Analyze contracts, legal documents |
    | 🏥 Healthcare | Review medical reports |
    | 💼 Business | Extract insights from reports |
    | 💻 Tech | Query API docs, technical manuals |
    """)
