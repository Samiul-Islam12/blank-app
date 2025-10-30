import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
import json
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# Evaluation imports
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score
import evaluate
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Page config
st.set_page_config(page_title="PDF RAG Evaluation System", page_icon="📚", layout="wide")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Model configurations
MODEL_CONFIGS = {
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct"
}

class PDFRAGEvaluator:
    """Comprehensive evaluation system for PDF RAG with multiple LLMs"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.rouge = evaluate.load('rouge')
        
    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings"""
        try:
            emb1 = self.embeddings.embed_query(text1)
            emb2 = self.embeddings.embed_query(text2)
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def calculate_bert_score(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate BERTScore F1"""
        try:
            P, R, F1 = bert_score([prediction], [reference], lang='en', verbose=False)
            return {
                'precision': float(P.mean()),
                'recall': float(R.mean()),
                'f1': float(F1.mean())
            }
        except:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score"""
        try:
            pred_tokens = nltk.word_tokenize(prediction.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            smoothie = SmoothingFunction().method4
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
            return float(score)
        except:
            return 0.0
    
    def calculate_meteor(self, prediction: str, reference: str) -> float:
        """Calculate METEOR score"""
        try:
            pred_tokens = nltk.word_tokenize(prediction.lower())
            ref_tokens = nltk.word_tokenize(reference.lower())
            score = meteor_score([ref_tokens], pred_tokens)
            return float(score)
        except:
            return 0.0
    
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            results = self.rouge.compute(predictions=[prediction], references=[reference])
            return {
                'rouge1': float(results['rouge1']),
                'rouge2': float(results['rouge2']),
                'rougeL': float(results['rougeL'])
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def detect_hallucination(self, response: str, context: str) -> float:
        """
        Detect hallucination by measuring semantic divergence from context
        Returns score between 0 (no hallucination) and 1 (high hallucination)
        """
        try:
            # Calculate inverse cosine similarity as hallucination indicator
            similarity = self.calculate_cosine_similarity(response, context)
            hallucination_score = max(0.0, 1.0 - similarity)
            return float(hallucination_score)
        except:
            return 0.5
    
    def calculate_completeness(self, response: str, reference: str) -> float:
        """
        Measure completeness of response compared to reference
        Returns score between 0 (incomplete) and 1 (complete)
        """
        try:
            # Use ROUGE-L as completeness indicator
            rouge_scores = self.calculate_rouge(response, reference)
            return float(rouge_scores['rougeL'])
        except:
            return 0.0
    
    def calculate_irrelevance(self, response: str, query: str) -> float:
        """
        Detect irrelevance to the query
        Returns score between 0 (relevant) and 1 (irrelevant)
        """
        try:
            # Calculate inverse cosine similarity as irrelevance indicator
            similarity = self.calculate_cosine_similarity(response, query)
            irrelevance_score = max(0.0, 1.0 - similarity)
            return float(irrelevance_score)
        except:
            return 0.5
    
    def evaluate_response(self, query: str, response: str, reference: str, 
                         context: str, latency: float) -> Dict[str, Any]:
        """Comprehensive evaluation of a model's response"""
        
        # Calculate all metrics
        cosine_sim = self.calculate_cosine_similarity(response, reference)
        bert_scores = self.calculate_bert_score(response, reference)
        bleu = self.calculate_bleu(response, reference)
        meteor = self.calculate_meteor(response, reference)
        rouge_scores = self.calculate_rouge(response, reference)
        hallucination = self.detect_hallucination(response, context)
        completeness = self.calculate_completeness(response, reference)
        irrelevance = self.calculate_irrelevance(response, query)
        
        return {
            'latency': latency,
            'cosine_similarity': cosine_sim,
            'bert_f1': bert_scores['f1'],
            'bert_precision': bert_scores['precision'],
            'bert_recall': bert_scores['recall'],
            'bleu': bleu,
            'meteor': meteor,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'hallucination': hallucination,
            'completeness': completeness,
            'irrelevance': irrelevance
        }


def load_pdfs(uploaded_files) -> List[Any]:
    """Load and process PDF files"""
    documents = []
    
    with st.spinner("Loading PDFs..."):
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load PDF
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
            
            # Clean up
            os.remove(temp_path)
    
    return documents


def create_vectorstore(documents, embeddings):
    """Create FAISS vectorstore from documents"""
    with st.spinner("Creating vector store..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore


def query_model(model_name: str, model_id: str, vectorstore, query: str, 
                api_token: str) -> tuple:
    """Query a specific model and measure latency"""
    
    try:
        # Initialize LLM
        llm = HuggingFaceHub(
            repo_id=model_id,
            huggingfacehub_api_token=api_token,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Measure latency
        start_time = time.time()
        result = qa_chain.invoke({"query": query})
        latency = time.time() - start_time
        
        response = result['result']
        context = " ".join([doc.page_content for doc in result['source_documents']])
        
        return response, context, latency
    
    except Exception as e:
        st.error(f"Error querying {model_name}: {str(e)}")
        return f"Error: {str(e)}", "", 0.0


def create_comparison_charts(results_df: pd.DataFrame):
    """Create comprehensive comparison visualizations"""
    
    # Define metric groups
    quality_metrics = ['cosine_similarity', 'bert_f1', 'bleu', 'meteor', 'completeness']
    reliability_metrics = ['hallucination', 'irrelevance']
    performance_metrics = ['avg_latency']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Quality Metrics Comparison
        st.subheader("📊 Quality Metrics Comparison")
        fig_quality = go.Figure()
        
        for metric in quality_metrics:
            if metric in results_df.columns:
                fig_quality.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=results_df['model'],
                    y=results_df[metric],
                    text=results_df[metric].round(3),
                    textposition='auto',
                ))
        
        fig_quality.update_layout(
            barmode='group',
            title='Quality Metrics by Model',
            xaxis_title='Model',
            yaxis_title='Score',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # Reliability Metrics (Lower is Better)
        st.subheader("🔍 Reliability Metrics (Lower is Better)")
        fig_reliability = go.Figure()
        
        for metric in reliability_metrics:
            if metric in results_df.columns:
                fig_reliability.add_trace(go.Bar(
                    name=metric.replace('_', ' ').title(),
                    x=results_df['model'],
                    y=results_df[metric],
                    text=results_df[metric].round(3),
                    textposition='auto',
                ))
        
        fig_reliability.update_layout(
            barmode='group',
            title='Reliability Metrics by Model',
            xaxis_title='Model',
            yaxis_title='Score',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_reliability, use_container_width=True)
    
    # Latency Comparison
    st.subheader("⚡ Average Response Time Comparison")
    fig_latency = px.bar(
        results_df,
        x='model',
        y='avg_latency',
        text=results_df['avg_latency'].round(2),
        title='Average Response Time (seconds)',
        labels={'avg_latency': 'Latency (s)', 'model': 'Model'},
        color='avg_latency',
        color_continuous_scale='RdYlGn_r'
    )
    fig_latency.update_traces(textposition='outside')
    fig_latency.update_layout(height=400)
    st.plotly_chart(fig_latency, use_container_width=True)
    
    # Overall Comparison Radar Chart
    st.subheader("🎯 Overall Performance Radar")
    
    # Normalize metrics for radar chart (invert hallucination and irrelevance)
    radar_metrics = ['cosine_similarity', 'bert_f1', 'completeness', 'bleu', 'meteor']
    
    fig_radar = go.Figure()
    
    for idx, row in results_df.iterrows():
        values = [row[metric] for metric in radar_metrics if metric in results_df.columns]
        values.append(values[0])  # Close the radar
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_metrics + [radar_metrics[0]],
            fill='toself',
            name=row['model']
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)


def main():
    st.title("📚 PDF RAG Evaluation System")
    st.markdown("**Evaluate Multiple LLMs on PDF Question Answering with Comprehensive Metrics**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Token
        api_token = st.text_input("HuggingFace API Token", type="password")
        
        if not api_token:
            st.warning("⚠️ Please enter your HuggingFace API token to proceed.")
            st.info("Get your token from: https://huggingface.co/settings/tokens")
        
        st.divider()
        
        # PDF Upload
        st.header("📄 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🔄 Process PDFs", use_container_width=True):
                documents = load_pdfs(uploaded_files)
                st.session_state.documents = documents
                
                # Create embeddings and vectorstore
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore = create_vectorstore(documents, embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.embeddings = embeddings
                
                st.success(f"✅ Processed {len(documents)} document chunks")
        
        st.divider()
        
        # Model Selection
        st.header("🤖 Models")
        selected_models = st.multiselect(
            "Select models to evaluate",
            list(MODEL_CONFIGS.keys()),
            default=list(MODEL_CONFIGS.keys())
        )
        
        # Number of trials
        num_trials = st.slider("Number of trials per model", 1, 5, 3)
    
    # Main content area
    if st.session_state.vectorstore is None:
        st.info("👈 Please upload PDFs and process them to begin evaluation.")
        return
    
    if not api_token:
        st.warning("⚠️ Please enter your HuggingFace API token in the sidebar.")
        return
    
    # Evaluation section
    st.header("🎯 Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["📝 Query & Evaluate", "📊 Results", "📈 Analytics"])
    
    with tab1:
        st.subheader("Enter Evaluation Query")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Question:", placeholder="Ask a question about your PDFs...")
        
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            evaluate_btn = st.button("🚀 Evaluate All Models", use_container_width=True)
        
        reference_answer = st.text_area(
            "Reference Answer (optional):",
            placeholder="Provide a reference answer for better evaluation metrics...",
            height=100
        )
        
        if evaluate_btn and query and selected_models:
            if not reference_answer:
                st.warning("⚠️ No reference answer provided. Some metrics may be less accurate.")
                reference_answer = query  # Use query as fallback
            
            # Initialize evaluator
            evaluator = PDFRAGEvaluator(st.session_state.embeddings)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_results = []
            total_iterations = len(selected_models) * num_trials
            current_iteration = 0
            
            # Evaluate each model
            for model_name in selected_models:
                model_id = MODEL_CONFIGS[model_name]
                
                st.subheader(f"🤖 Evaluating {model_name}")
                
                trial_results = []
                
                # Run multiple trials
                for trial in range(num_trials):
                    current_iteration += 1
                    status_text.text(f"Running Trial {trial + 1}/{num_trials} for {model_name}...")
                    progress_bar.progress(current_iteration / total_iterations)
                    
                    # Query model
                    response, context, latency = query_model(
                        model_name, model_id, st.session_state.vectorstore,
                        query, api_token
                    )
                    
                    if response and not response.startswith("Error:"):
                        # Evaluate response
                        metrics = evaluator.evaluate_response(
                            query, response, reference_answer, context, latency
                        )
                        
                        trial_results.append({
                            'trial': trial + 1,
                            'response': response,
                            'context': context[:200] + "...",
                            **metrics
                        })
                
                # Average metrics across trials
                if trial_results:
                    avg_metrics = {
                        'model': model_name,
                        'num_trials': len(trial_results),
                        'avg_latency': np.mean([r['latency'] for r in trial_results]),
                        'cosine_similarity': np.mean([r['cosine_similarity'] for r in trial_results]),
                        'bert_f1': np.mean([r['bert_f1'] for r in trial_results]),
                        'bleu': np.mean([r['bleu'] for r in trial_results]),
                        'meteor': np.mean([r['meteor'] for r in trial_results]),
                        'hallucination': np.mean([r['hallucination'] for r in trial_results]),
                        'completeness': np.mean([r['completeness'] for r in trial_results]),
                        'irrelevance': np.mean([r['irrelevance'] for r in trial_results]),
                        'trials': trial_results
                    }
                    
                    all_results.append(avg_metrics)
                    
                    # Display trial results
                    with st.expander(f"View {model_name} Trial Details"):
                        for i, trial in enumerate(trial_results):
                            st.markdown(f"**Trial {i+1} Response:**")
                            st.write(trial['response'])
                            st.caption(f"Latency: {trial['latency']:.2f}s | BERTScore F1: {trial['bert_f1']:.3f}")
                            st.divider()
            
            # Save results
            if all_results:
                st.session_state.evaluation_results = all_results
                progress_bar.progress(1.0)
                status_text.text("✅ Evaluation Complete!")
                st.success("🎉 Evaluation completed successfully! Check the Results and Analytics tabs.")
    
    with tab2:
        st.subheader("📊 Evaluation Results")
        
        if st.session_state.evaluation_results:
            results_df = pd.DataFrame([
                {k: v for k, v in result.items() if k != 'trials'}
                for result in st.session_state.evaluation_results
            ])
            
            # Display summary table
            st.dataframe(
                results_df.style.highlight_max(
                    subset=['cosine_similarity', 'bert_f1', 'bleu', 'meteor', 'completeness'],
                    color='lightgreen'
                ).highlight_min(
                    subset=['avg_latency', 'hallucination', 'irrelevance'],
                    color='lightgreen'
                ).format({
                    'avg_latency': '{:.3f}',
                    'cosine_similarity': '{:.3f}',
                    'bert_f1': '{:.3f}',
                    'bleu': '{:.3f}',
                    'meteor': '{:.3f}',
                    'hallucination': '{:.3f}',
                    'completeness': '{:.3f}',
                    'irrelevance': '{:.3f}'
                }),
                use_container_width=True
            )
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No evaluation results yet. Run an evaluation in the Query & Evaluate tab.")
    
    with tab3:
        st.subheader("📈 Analytics Dashboard")
        
        if st.session_state.evaluation_results:
            results_df = pd.DataFrame([
                {k: v for k, v in result.items() if k != 'trials'}
                for result in st.session_state.evaluation_results
            ])
            
            # Create comparison charts
            create_comparison_charts(results_df)
            
            # Detailed metrics breakdown
            st.subheader("📋 Detailed Metrics Breakdown")
            
            for result in st.session_state.evaluation_results:
                with st.expander(f"📊 {result['model']} - Detailed Metrics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Latency", f"{result['avg_latency']:.3f}s")
                        st.metric("Cosine Similarity", f"{result['cosine_similarity']:.3f}")
                        st.metric("BERTScore F1", f"{result['bert_f1']:.3f}")
                    
                    with col2:
                        st.metric("BLEU Score", f"{result['bleu']:.3f}")
                        st.metric("METEOR Score", f"{result['meteor']:.3f}")
                        st.metric("Completeness", f"{result['completeness']:.3f}")
                    
                    with col3:
                        st.metric("Hallucination", f"{result['hallucination']:.3f}")
                        st.metric("Irrelevance", f"{result['irrelevance']:.3f}")
                        st.metric("Trials", result['num_trials'])
                    
                    # Trial-by-trial breakdown
                    if 'trials' in result:
                        st.markdown("**Trial-by-Trial Performance:**")
                        trial_df = pd.DataFrame([
                            {k: v for k, v in trial.items() if k not in ['response', 'context']}
                            for trial in result['trials']
                        ])
                        st.dataframe(trial_df, use_container_width=True)
        else:
            st.info("No analytics data yet. Run an evaluation first.")


if __name__ == "__main__":
    main()
