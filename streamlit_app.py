import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

def load_document(uploaded_file) -> List[Document]:
    """Load a document from an uploaded file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.name.endswith('.txt'):
            loader = TextLoader(tmp_file_path)
        else:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            return []
        
        documents = loader.load()
        return documents
    finally:
        os.unlink(tmp_file_path)

def create_vectorstore(documents: List[Document], use_openai: bool = False):
    """Create a vector store from documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings
    if use_openai and os.getenv("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    else:
        # Use free local embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_collection"
    )
    
    return vectorstore

def create_qa_chain(vectorstore, use_openai: bool = False):
    """Create a QA chain with the vector store."""
    # Custom prompt template
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}

Question: {question}

Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    # Initialize LLM
    if use_openai and os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        st.warning("⚠️ No OpenAI API key found. Please configure an LLM provider in the sidebar.")
        return None
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# Main UI
st.title("🤖 LLM RAG System")
st.markdown("### Retrieval-Augmented Generation with Document Q&A")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Provider selection
    st.subheader("LLM Settings")
    use_openai = st.checkbox("Use OpenAI", value=True)
    
    if use_openai:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
    
    st.divider()
    
    # Document upload section
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files to build your knowledge base"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    docs = load_document(uploaded_file)
                    all_documents.extend(docs)
                
                if all_documents:
                    st.session_state.vectorstore = create_vectorstore(
                        all_documents, 
                        use_openai=use_openai
                    )
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Processed {len(uploaded_files)} document(s) with {len(all_documents)} page(s)")
                else:
                    st.error("No documents could be loaded")
    
    if st.session_state.documents_loaded:
        st.info(f"📚 Knowledge base is ready!")
        if st.button("Clear Knowledge Base"):
            st.session_state.vectorstore = None
            st.session_state.documents_loaded = False
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    st.markdown("### 💡 How to use:")
    st.markdown("""
1. Enter your OpenAI API key (or configure another LLM)
2. Upload your documents (PDF or TXT)
3. Click 'Process Documents'
4. Ask questions about your documents!
    """)

# Main chat interface
if not st.session_state.documents_loaded:
    st.info("👈 Please upload and process documents in the sidebar to get started!")
    
    # Example use cases
    st.markdown("### 🎯 What you can do with this RAG system:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📚 Document Analysis**
        - Upload research papers, reports, or books
        - Ask questions about the content
        - Get accurate answers with sources
        """)
        
    with col2:
        st.markdown("""
        **💼 Knowledge Management**
        - Build a searchable knowledge base
        - Query multiple documents at once
        - Extract insights efficiently
        """)
else:
    # Chat interface
    st.markdown("### 💬 Ask questions about your documents")
    
    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if "sources" in chat and chat["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(chat["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source.page_content[:300] + "...")
                        st.markdown("---")
    
    # Query input
    query = st.chat_input("Ask a question about your documents...")
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = create_qa_chain(st.session_state.vectorstore, use_openai=use_openai)
                
                if qa_chain:
                    try:
                        response = qa_chain({"query": query})
                        answer = response["result"]
                        source_docs = response.get("source_documents", [])
                        
                        st.write(answer)
                        
                        if source_docs:
                            with st.expander("📚 View Sources"):
                                for i, doc in enumerate(source_docs, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(doc.page_content[:300] + "...")
                                    st.markdown("---")
                        
                        # Save to chat history
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer,
                            "sources": source_docs
                        })
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                else:
                    st.error("Please configure an LLM provider to generate answers.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Built with LangChain, Streamlit, and ChromaDB | RAG System v1.0</p>
</div>
""", unsafe_allow_html=True)
