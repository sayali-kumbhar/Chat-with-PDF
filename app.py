import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

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
    """Split text into manageable chunks for embedding"""
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

def get_conversational_chain():
    """Create the QA chain with a custom prompt"""
    prompt_template = """
    You are an expert AI assistant specialized in analyzing documents. 
    Answer the question in detail based on the provided context. 
    If the answer is not in the context, say: "⚠️ This information is not available in the uploaded document."
    
    Always be helpful, structured, and use bullet points or numbered lists when appropriate.
    
    Context:\n{context}\n
    Question:\n{question}\n

    Detailed Answer:
    """
    
    # ✅ NEW - WORKING MODEL
    model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=1.0
)
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_question(question, vector_store):
    """Process user question and return AI response"""
    # Semantic search — find relevant chunks
    docs = vector_store.similarity_search(question, k=4)
    
    # Run QA chain
    chain = get_conversational_chain()
    response = chain.invoke(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    
    return response["output_text"]

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