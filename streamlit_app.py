import streamlit as st
import os
from pathlib import Path
import tempfile
from typing import List, Optional
from dotenv import load_dotenv
import torch
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Hugging Face imports
from huggingface_hub import login
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Evaluation imports
from evaluation_metrics import RAGEvaluator, format_evaluation_summary

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System - Hugging Face",
    page_icon="🤖",
    layout="wide"
)

# Available models configuration
AVAILABLE_MODELS = {
    "Qwen 2.5 - 3B (Recommended for 8GB VRAM)": "Qwen/Qwen2.5-3B-Instruct",
    "Mistral 7B - Instruct v0.1 (Requires 12GB VRAM)": "mistralai/Mistral-7B-Instruct-v0.1",
    "Llama 3.2 - 3B Instruct (Recommended for 8GB VRAM)": "meta-llama/Llama-3.2-3B-Instruct",
}

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'llm_loaded' not in st.session_state:
    st.session_state.llm_loaded = False
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'evaluations' not in st.session_state:
    st.session_state.evaluations = []
if 'trial_counter' not in st.session_state:
    st.session_state.trial_counter = 0
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'enable_evaluation' not in st.session_state:
    st.session_state.enable_evaluation = True

@st.cache_resource
def load_llm_model(model_name: str, hf_token: str):
    """Load the Hugging Face model with quantization."""
    try:
        # Login to Hugging Face
        login(hf_token)
        
        # BitsAndBytes configuration for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        
        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return llm, True, "✅ Model loaded successfully!"
    
    except Exception as e:
        return None, False, f"❌ Error loading model: {str(e)}"

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

def create_vectorstore(documents: List[Document]):
    """Create a vector store from documents."""
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    # Use local embeddings
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

def create_qa_chain(vectorstore, llm):
    """Create a QA chain with the vector store."""
    # Custom prompt template optimized for instruction-following models
    template = """You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say "I cannot find this information in the provided documents."
Keep your answer concise and accurate.

Context: {context}

Question: {question}

Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
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
st.markdown("### Retrieval-Augmented Generation with Hugging Face Models")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model configuration
    st.subheader("🤗 Hugging Face Model Settings")
    
    hf_token = st.text_input(
        "Hugging Face Token",
        type="password",
        value=os.getenv("HF_TOKEN", ""),
        help="Enter your Hugging Face API token"
    )
    
    # Model selection dropdown
    selected_model_display = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        help="Choose from pre-configured models"
    )
    
    selected_model = AVAILABLE_MODELS[selected_model_display]
    
    # Show current loaded model
    if st.session_state.llm_loaded:
        st.info(f"✅ **Current Model**: {st.session_state.current_model}")
    
    if st.button("Load Model", type="primary", disabled=st.session_state.llm_loaded):
        with st.spinner(f"Loading {selected_model_display}... This may take a few minutes..."):
            llm, success, message = load_llm_model(selected_model, hf_token)
            if success:
                st.session_state.llm = llm
                st.session_state.llm_loaded = True
                st.session_state.current_model = selected_model_display
                st.success(message)
            else:
                st.error(message)
    
    if st.session_state.llm_loaded:
        if st.button("Unload Model"):
            st.session_state.llm = None
            st.session_state.llm_loaded = False
            st.session_state.current_model = None
            st.cache_resource.clear()
            st.rerun()
    
    st.divider()
    
    # Document upload section
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload PDF or TXT files to build your knowledge base"
    )
    
    if uploaded_files and st.session_state.llm_loaded:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    docs = load_document(uploaded_file)
                    all_documents.extend(docs)
                
                if all_documents:
                    st.session_state.vectorstore = create_vectorstore(all_documents)
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Processed {len(uploaded_files)} document(s) with {len(all_documents)} page(s)")
                else:
                    st.error("No documents could be loaded")
    elif uploaded_files and not st.session_state.llm_loaded:
        st.warning("⚠️ Please load the model first before processing documents")
    
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
1. Enter your Hugging Face token
2. Select a model from the dropdown
3. Click 'Load Model' (first time: 5-10 min)
4. Upload your documents (PDF or TXT)
5. Click 'Process Documents'
6. Ask questions about your documents!
    """)
    
    # Model info
    with st.expander("ℹ️ Model Info & Requirements"):
        st.markdown("""
        **Available Models:**
        
        1. **Qwen 2.5 - 3B Instruct**
           - VRAM: ~6GB (with 4-bit quantization)
           - Speed: Fast
           - Quality: Excellent for 3B size
        
        2. **Mistral 7B - Instruct v0.1**
           - VRAM: ~10GB (with 4-bit quantization)
           - Speed: Moderate
           - Quality: Very high quality
        
        3. **Llama 3.2 - 3B Instruct**
           - VRAM: ~6GB (with 4-bit quantization)
           - Speed: Fast
           - Quality: Excellent for 3B size
        
        **Quantization Settings:**
        - Type: 4-bit NF4
        - Compute: bfloat16
        - Memory savings: ~75%
        """)

# Main chat interface
if not st.session_state.llm_loaded:
    st.info("👈 Please load the model in the sidebar to get started!")
    
    # Model information
    st.markdown("### 🎯 About This RAG System")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤗 Powered by Hugging Face**
        - Choose from 3 models:
          - Qwen 2.5 - 3B (Fast, 6GB VRAM)
          - Mistral 7B (High quality, 10GB VRAM)
          - Llama 3.2 - 3B (Fast, 6GB VRAM)
        - 4-bit quantization for efficiency
        - Runs locally - no API costs!
        """)
        
    with col2:
        st.markdown("""
        **📚 Features**
        - Upload multiple documents
        - Semantic search with embeddings
        - Source citations
        - Chat history
        """)

elif not st.session_state.documents_loaded:
    st.info("👈 Please upload and process documents in the sidebar!")
    
    st.markdown("### 📚 What you can do:")
    st.markdown("""
    - Upload research papers, reports, or books
    - Ask questions about the content
    - Get accurate answers with source citations
    - Build a searchable knowledge base
    """)
    
else:
    # Create tabs for Chat and Evaluation
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation Dashboard"])
    
    with tab1:
        # Chat interface
        st.markdown("### 💬 Ask questions about your documents")
        
        # Evaluation toggle
        col1, col2 = st.columns([3, 1])
        with col2:
            st.session_state.enable_evaluation = st.checkbox(
                "Enable Evaluation", 
                value=st.session_state.enable_evaluation,
                help="Track metrics for each response"
            )
        
        # Display chat history
        for idx, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    if "sources" in chat and chat["sources"]:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(chat["sources"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source.page_content[:300] + "...")
                                st.markdown("---")
                
                with col2:
                    if "evaluation" in chat:
                        with st.expander("📊 Metrics"):
                            eval_data = chat["evaluation"]
                            st.metric("Trial", f"#{eval_data.get('trial_number', 'N/A')}")
                            st.metric("Latency", f"{eval_data.get('latency', 0):.3f}s")
                            st.metric("Relevance", f"{eval_data.get('answer_relevance', 0):.3f}")
                            st.metric("Trial Score", f"{eval_data.get('trial_score', 0):.1f}")
        
        # Query input
        query = st.chat_input("Ask a question about your documents...")
        
        if query:
            # Initialize evaluator if needed
            if st.session_state.enable_evaluation and st.session_state.evaluator is None:
                with st.spinner("Initializing evaluator..."):
                    st.session_state.evaluator = RAGEvaluator()
            
            # Display user message
            with st.chat_message("user"):
                st.write(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Start timing
                        start_time = time.time()
                        
                        qa_chain = create_qa_chain(
                            st.session_state.vectorstore, 
                            st.session_state.llm
                        )
                        
                        response = qa_chain({"query": query})
                        answer = response["result"]
                        source_docs = response.get("source_documents", [])
                        
                        # End timing
                        end_time = time.time()
                        latency = end_time - start_time
                        
                        st.write(answer)
                        
                        # Evaluation
                        evaluation_results = None
                        if st.session_state.enable_evaluation and st.session_state.evaluator:
                            with st.spinner("Evaluating response..."):
                                st.session_state.trial_counter += 1
                                
                                # Extract context from source docs
                                contexts = [doc.page_content for doc in source_docs]
                                
                                # Run comprehensive evaluation
                                evaluation_results = st.session_state.evaluator.comprehensive_evaluation(
                                    question=query,
                                    answer=answer,
                                    contexts=contexts,
                                    latency=latency
                                )
                                
                                # Calculate trial score
                                trial_score = st.session_state.evaluator.calculate_trial_score(
                                    evaluation_results,
                                    st.session_state.trial_counter
                                )
                                
                                evaluation_results["trial_number"] = st.session_state.trial_counter
                                evaluation_results["trial_score"] = trial_score
                                evaluation_results["model"] = st.session_state.current_model
                                
                                # Store evaluation
                                st.session_state.evaluations.append(evaluation_results)
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if source_docs:
                                with st.expander("📚 View Sources"):
                                    for i, doc in enumerate(source_docs, 1):
                                        st.markdown(f"**Source {i}:**")
                                        st.text(doc.page_content[:300] + "...")
                                        st.markdown("---")
                        
                        with col2:
                            if evaluation_results:
                                with st.expander("📊 Metrics"):
                                    st.metric("Trial", f"#{evaluation_results['trial_number']}")
                                    st.metric("Latency", f"{latency:.3f}s")
                                    st.metric("Relevance", f"{evaluation_results.get('answer_relevance', 0):.3f}")
                                    st.metric("Trial Score", f"{trial_score:.1f}")
                        
                        # Save to chat history
                        chat_entry = {
                            "question": query,
                            "answer": answer,
                            "sources": source_docs
                        }
                        if evaluation_results:
                            chat_entry["evaluation"] = evaluation_results
                        
                        st.session_state.chat_history.append(chat_entry)
                        
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
    
    with tab2:
        # Evaluation Dashboard
        st.markdown("### 📊 Evaluation Dashboard")
        
        if not st.session_state.evaluations:
            st.info("💡 Enable evaluation in the Chat tab and ask questions to see metrics here!")
            
            st.markdown("""
            **Available Metrics:**
            - **Trial Score**: Overall performance (0-100)
            - **Latency**: Response time in seconds
            - **Cosine Similarity**: Semantic similarity scores
            - **BERTScore F1**: Semantic similarity using BERT
            - **Completeness**: How complete the answer is
            - **Hallucination Risk**: Likelihood of fabricated information
            - **Relevance**: How relevant the answer is to the question
            - **BLEU Score**: N-gram overlap with reference
            - **METEOR Score**: Unigram precision/recall
            """)
        else:
            # Create DataFrame from evaluations
            df = pd.DataFrame(st.session_state.evaluations)
            
            # Summary statistics
            st.markdown("#### 📈 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trials", len(st.session_state.evaluations))
                st.metric("Avg Latency", f"{df['latency'].mean():.3f}s")
            
            with col2:
                avg_score = df['trial_score'].mean() if 'trial_score' in df else 0
                st.metric("Avg Trial Score", f"{avg_score:.1f}")
                st.metric("Avg Relevance", f"{df['answer_relevance'].mean():.3f}")
            
            with col3:
                st.metric("Avg Completeness", f"{df['completeness_score'].mean():.3f}")
                st.metric("Avg Hallucination", f"{df['hallucination_score'].mean():.3f}")
            
            with col4:
                st.metric("Avg Context Sim", f"{df['context_similarity'].mean():.3f}")
                st.metric("Avg Irrelevance", f"{df['irrelevance_score'].mean():.3f}")
            
            st.divider()
            
            # Visualizations
            st.markdown("#### 📊 Metric Comparisons")
            
            # Trial Scores comparison
            fig_trials = go.Figure()
            fig_trials.add_trace(go.Bar(
                x=[f"Trial {i}" for i in df['trial_number']],
                y=df['trial_score'] if 'trial_score' in df else [0] * len(df),
                name='Trial Score',
                marker_color='lightblue'
            ))
            fig_trials.update_layout(
                title="Trial Scores Comparison",
                xaxis_title="Trial",
                yaxis_title="Score (0-100)",
                height=400
            )
            st.plotly_chart(fig_trials, use_container_width=True)
            
            # Latency comparison
            fig_latency = go.Figure()
            fig_latency.add_trace(go.Bar(
                x=[f"Trial {i}" for i in df['trial_number']],
                y=df['latency'],
                name='Latency',
                marker_color='coral'
            ))
            fig_latency.update_layout(
                title="Latency Comparison (Lower is Better)",
                xaxis_title="Trial",
                yaxis_title="Latency (seconds)",
                height=400
            )
            st.plotly_chart(fig_latency, use_container_width=True)
            
            # Multi-metric comparison
            metrics_to_plot = ['answer_relevance', 'completeness_score', 'context_similarity']
            fig_multi = go.Figure()
            
            for metric in metrics_to_plot:
                if metric in df.columns:
                    fig_multi.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=[f"Trial {i}" for i in df['trial_number']],
                        y=df[metric]
                    ))
            
            fig_multi.update_layout(
                title="Quality Metrics Comparison (Higher is Better)",
                xaxis_title="Trial",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # Hallucination and Irrelevance (lower is better)
            fig_risk = go.Figure()
            fig_risk.add_trace(go.Bar(
                name='Hallucination Score',
                x=[f"Trial {i}" for i in df['trial_number']],
                y=df['hallucination_score'],
                marker_color='red'
            ))
            fig_risk.add_trace(go.Bar(
                name='Irrelevance Score',
                x=[f"Trial {i}" for i in df['trial_number']],
                y=df['irrelevance_score'],
                marker_color='orange'
            ))
            fig_risk.update_layout(
                title="Risk Metrics (Lower is Better)",
                xaxis_title="Trial",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # BERTScore, BLEU, METEOR (if available)
            if 'bertscore_f1' in df.columns:
                fig_nlp = go.Figure()
                
                if 'bertscore_f1' in df.columns:
                    fig_nlp.add_trace(go.Bar(
                        name='BERTScore F1',
                        x=[f"Trial {i}" for i in df['trial_number']],
                        y=df['bertscore_f1']
                    ))
                if 'bleu_score' in df.columns:
                    fig_nlp.add_trace(go.Bar(
                        name='BLEU Score',
                        x=[f"Trial {i}" for i in df['trial_number']],
                        y=df['bleu_score']
                    ))
                if 'meteor_score' in df.columns:
                    fig_nlp.add_trace(go.Bar(
                        name='METEOR Score',
                        x=[f"Trial {i}" for i in df['trial_number']],
                        y=df['meteor_score']
                    ))
                
                fig_nlp.update_layout(
                    title="NLP Metrics (Requires Reference Answer)",
                    xaxis_title="Trial",
                    yaxis_title="Score",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig_nlp, use_container_width=True)
            
            st.divider()
            
            # Detailed evaluation table
            st.markdown("#### 📋 Detailed Evaluation Data")
            
            # Select columns to display
            display_cols = ['trial_number', 'latency', 'trial_score', 'answer_relevance', 
                          'completeness_score', 'hallucination_score', 'irrelevance_score',
                          'context_similarity']
            
            available_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[available_cols].round(4), use_container_width=True)
            
            # Export functionality
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Clear Evaluations"):
                    st.session_state.evaluations = []
                    st.session_state.trial_counter = 0
                    st.rerun()
            
            with col2:
                # Download evaluations as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Evaluation Data (CSV)",
                    data=csv,
                    file_name="rag_evaluations.csv",
                    mime="text/csv"
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Built with 🤗 Hugging Face, LangChain, Streamlit, and ChromaDB | RAG System v2.0</p>
</div>
""", unsafe_allow_html=True)
