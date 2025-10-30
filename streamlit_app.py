import streamlit as st
import os
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Page configuration
st.set_page_config(
    page_title="📚 Multi-Model RAG Evaluation System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

class RAGEvaluator:
    """Comprehensive RAG evaluation system"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction()
        
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        return sentence_bleu([reference_tokens], candidate_tokens, 
                            smoothing_function=self.smoothing.method1)
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score"""
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        return meteor_score([reference_tokens], candidate_tokens)
    
    def calculate_bertscore(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BERTScore"""
        P, R, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
        return {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        }
    
    def calculate_cosine_similarity(self, reference: str, candidate: str) -> float:
        """Calculate cosine similarity using TF-IDF"""
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([reference, candidate])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            return 0.0
    
    def calculate_completeness(self, reference: str, candidate: str) -> float:
        """Calculate completeness score (how much of reference is covered)"""
        reference_words = set(nltk.word_tokenize(reference.lower()))
        candidate_words = set(nltk.word_tokenize(candidate.lower()))
        if len(reference_words) == 0:
            return 0.0
        overlap = len(reference_words.intersection(candidate_words))
        return overlap / len(reference_words)
    
    def calculate_hallucination(self, context: str, answer: str) -> float:
        """Estimate hallucination (information not in context)"""
        context_words = set(nltk.word_tokenize(context.lower()))
        answer_words = set(nltk.word_tokenize(answer.lower()))
        # Remove common stop words for better estimation
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        answer_words = answer_words - stop_words
        if len(answer_words) == 0:
            return 0.0
        hallucinated = len(answer_words - context_words)
        return hallucinated / len(answer_words)
    
    def calculate_irrelevance(self, question: str, answer: str) -> float:
        """Calculate irrelevance score"""
        # Simple heuristic: measure how different the answer is from the question topic
        question_words = set(nltk.word_tokenize(question.lower()))
        answer_words = set(nltk.word_tokenize(answer.lower()))
        stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'the', 'a', 'an'}
        question_words = question_words - stop_words
        if len(question_words) == 0:
            return 0.0
        relevant = len(question_words.intersection(answer_words))
        return 1.0 - (relevant / len(question_words))

def load_pdfs(uploaded_files):
    """Load and process PDF files"""
    documents = []
    temp_dir = Path("temp_pdfs")
    temp_dir.mkdir(exist_ok=True)
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load PDF
        loader = PyPDFLoader(str(temp_path))
        docs = loader.load()
        documents.extend(docs)
    
    return documents

def create_vectorstore(documents):
    """Create vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    return vectorstore

def get_llm_model(model_name: str, api_key: str):
    """Initialize LLM model"""
    model_mapping = {
        "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
        "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
        "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct"
    }
    
    repo_id = model_mapping.get(model_name, model_mapping["Mistral-7B"])
    
    llm = HuggingFaceHub(
        repo_id=repo_id,
        huggingfacehub_api_token=api_key,
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512
        }
    )
    return llm

def create_qa_chain(vectorstore, llm):
    """Create QA chain"""
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context provided, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer based only on the context provided:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

def run_evaluation_trial(qa_chain, question: str, reference_answer: str, trial_num: int, evaluator: RAGEvaluator):
    """Run a single evaluation trial"""
    start_time = time.time()
    
    try:
        result = qa_chain.invoke({"query": question})
        answer = result['result']
        source_docs = result.get('source_documents', [])
        context = " ".join([doc.page_content for doc in source_docs])
    except Exception as e:
        st.error(f"Error in trial {trial_num}: {str(e)}")
        return None
    
    latency = time.time() - start_time
    
    # Calculate all metrics
    metrics = {
        'trial': trial_num,
        'question': question,
        'answer': answer,
        'latency': latency,
        'bleu': evaluator.calculate_bleu(reference_answer, answer),
        'meteor': evaluator.calculate_meteor(reference_answer, answer),
        'cosine_similarity': evaluator.calculate_cosine_similarity(reference_answer, answer),
        'completeness': evaluator.calculate_completeness(reference_answer, answer),
        'hallucination': evaluator.calculate_hallucination(context, answer),
        'irrelevance': evaluator.calculate_irrelevance(question, answer),
    }
    
    # Calculate BERTScore
    bert_scores = evaluator.calculate_bertscore(reference_answer, answer)
    metrics.update({
        'bert_precision': bert_scores['precision'],
        'bert_recall': bert_scores['recall'],
        'bert_f1': bert_scores['f1']
    })
    
    return metrics

# Main UI
st.markdown('<div class="main-header">📚 Multi-Model RAG Evaluation System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("HuggingFace API Token", type="password", 
                            help="Get your token from https://huggingface.co/settings/tokens")
    
    st.markdown("---")
    
    selected_models = st.multiselect(
        "Select Models to Evaluate",
        ["Mistral-7B", "Qwen2.5-7B", "Llama-3.2-3B"],
        default=["Mistral-7B"]
    )
    
    num_trials = st.slider("Number of Trials per Model", 1, 5, 3)
    
    st.markdown("---")
    st.markdown("### 📊 Evaluation Metrics")
    st.markdown("""
    - **BLEU**: Translation quality
    - **METEOR**: Semantic similarity
    - **BERTScore**: Contextual embedding similarity
    - **Cosine Similarity**: Vector similarity
    - **Completeness**: Coverage of reference
    - **Hallucination**: Info not in context
    - **Irrelevance**: Off-topic responses
    - **Latency**: Response time
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "🔍 Evaluation", "📊 Results & Visualizations"])

with tab1:
    st.header("Upload PDF Documents")
    st.write("Upload your PDF documents to create the knowledge base for RAG.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process PDFs and Create Vector Store"):
            with st.spinner("Processing PDFs..."):
                documents = load_pdfs(uploaded_files)
                st.success(f"Loaded {len(documents)} pages from PDFs")
                
                with st.spinner("Creating vector store..."):
                    st.session_state.vectorstore = create_vectorstore(documents)
                    st.session_state.documents_loaded = True
                    st.success("✅ Vector store created successfully!")

with tab2:
    st.header("Run Evaluation")
    
    if not st.session_state.documents_loaded:
        st.warning("⚠️ Please upload and process PDFs first in the 'Document Upload' tab.")
    elif not api_key:
        st.warning("⚠️ Please provide your HuggingFace API token in the sidebar.")
    else:
        st.write("Enter test questions and reference answers for evaluation.")
        
        col1, col2 = st.columns(2)
        with col1:
            test_question = st.text_area("Test Question", 
                                        placeholder="e.g., What is the main topic of the document?")
        with col2:
            reference_answer = st.text_area("Reference Answer (Ground Truth)",
                                          placeholder="Expected answer based on the documents")
        
        if st.button("🚀 Run Evaluation on Selected Models", type="primary"):
            if not test_question or not reference_answer:
                st.error("Please provide both a question and reference answer.")
            else:
                evaluator = RAGEvaluator()
                all_results = []
                
                for model_name in selected_models:
                    st.subheader(f"Evaluating {model_name}")
                    progress_bar = st.progress(0)
                    
                    try:
                        llm = get_llm_model(model_name, api_key)
                        qa_chain = create_qa_chain(st.session_state.vectorstore, llm)
                        
                        model_results = []
                        for trial in range(1, num_trials + 1):
                            with st.spinner(f"Running trial {trial}/{num_trials}..."):
                                metrics = run_evaluation_trial(
                                    qa_chain, test_question, reference_answer, trial, evaluator
                                )
                                if metrics:
                                    metrics['model'] = model_name
                                    model_results.append(metrics)
                                    all_results.append(metrics)
                            
                            progress_bar.progress(trial / num_trials)
                        
                        # Show trial results
                        if model_results:
                            st.success(f"✅ Completed {len(model_results)} trials for {model_name}")
                            avg_latency = np.mean([r['latency'] for r in model_results])
                            st.metric(f"Average Latency", f"{avg_latency:.3f}s")
                    
                    except Exception as e:
                        st.error(f"Error evaluating {model_name}: {str(e)}")
                
                if all_results:
                    st.session_state.evaluation_results.extend(all_results)
                    st.success(f"🎉 Evaluation complete! Check the 'Results & Visualizations' tab.")

with tab3:
    st.header("Results & Visualizations")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation results yet. Run an evaluation in the 'Evaluation' tab.")
    else:
        df = pd.DataFrame(st.session_state.evaluation_results)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        metrics_to_show = ['latency', 'bleu', 'meteor', 'bert_f1', 'cosine_similarity', 
                          'completeness', 'hallucination', 'irrelevance']
        
        summary_stats = df.groupby('model')[metrics_to_show].agg(['mean', 'std'])
        st.dataframe(summary_stats, use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Comparative Visualizations")
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Latency", "Quality Metrics", "Error Metrics", "All Metrics"])
        
        with viz_tabs[0]:
            # Latency comparison
            fig_latency = px.bar(
                df.groupby('model')['latency'].mean().reset_index(),
                x='model', y='latency',
                title='Average Response Time by Model',
                labels={'latency': 'Latency (seconds)', 'model': 'Model'},
                color='model'
            )
            st.plotly_chart(fig_latency, use_container_width=True)
        
        with viz_tabs[1]:
            # Quality metrics
            quality_metrics = ['bleu', 'meteor', 'bert_f1', 'cosine_similarity', 'completeness']
            quality_data = df.groupby('model')[quality_metrics].mean().reset_index()
            quality_data_melted = quality_data.melt(id_vars='model', 
                                                    var_name='metric', 
                                                    value_name='score')
            
            fig_quality = px.bar(
                quality_data_melted,
                x='metric', y='score', color='model',
                barmode='group',
                title='Quality Metrics Comparison',
                labels={'score': 'Score', 'metric': 'Metric'}
            )
            st.plotly_chart(fig_quality, use_container_width=True)
        
        with viz_tabs[2]:
            # Error metrics (lower is better)
            error_metrics = ['hallucination', 'irrelevance']
            error_data = df.groupby('model')[error_metrics].mean().reset_index()
            error_data_melted = error_data.melt(id_vars='model', 
                                                var_name='metric', 
                                                value_name='score')
            
            fig_errors = px.bar(
                error_data_melted,
                x='metric', y='score', color='model',
                barmode='group',
                title='Error Metrics Comparison (Lower is Better)',
                labels={'score': 'Score', 'metric': 'Metric'}
            )
            st.plotly_chart(fig_errors, use_container_width=True)
        
        with viz_tabs[3]:
            # Radar chart for all metrics
            all_metrics = metrics_to_show
            radar_data = df.groupby('model')[all_metrics].mean()
            
            fig_radar = go.Figure()
            
            for model in radar_data.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_data.loc[model].values,
                    theta=all_metrics,
                    fill='toself',
                    name=model
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title='Comprehensive Model Comparison (Radar Chart)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Trial-by-trial results
        st.subheader("🔍 Detailed Trial Results")
        
        for model in df['model'].unique():
            with st.expander(f"📋 {model} - Trial Details"):
                model_df = df[df['model'] == model]
                st.dataframe(
                    model_df[['trial', 'latency', 'bleu', 'meteor', 'bert_f1', 
                             'cosine_similarity', 'hallucination']],
                    use_container_width=True
                )
                
                # Show sample answer
                if len(model_df) > 0:
                    st.write("**Sample Answer (Trial 1):**")
                    st.write(model_df.iloc[0]['answer'])
        
        # Download results
        st.subheader("💾 Export Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="rag_evaluation_results.csv",
            mime="text/csv"
        )
        
        # Clear results button
        if st.button("🗑️ Clear All Results"):
            st.session_state.evaluation_results = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Multi-Model RAG Evaluation System | Built with Streamlit & LangChain</p>
    <p>Models: Mistral AI • Qwen • LLaMa | Metrics: BLEU • METEOR • BERTScore • Cosine Similarity • and more</p>
</div>
""", unsafe_allow_html=True)
