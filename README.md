# 📚 Multi-Model RAG Evaluation System

A comprehensive LLM RAG (Retrieval Augmented Generation) application that answers questions based only on uploaded PDFs, with extensive evaluation metrics and model comparison capabilities.

## 🌟 Features

### Multi-Model Support
- **Mistral AI 7B** - Mistral-7B-Instruct-v0.2
- **Qwen 2.5 7B** - Qwen2.5-7B-Instruct  
- **LLaMa 3.2 3B** - Llama-3.2-3B-Instruct

### PDF-Based RAG System
- Upload multiple PDF documents
- Automatic text chunking and vectorization
- Semantic search with embeddings
- Context-aware question answering

### Comprehensive Evaluation Metrics

#### Quality Metrics (Higher is Better)
- **BLEU Score** - Translation/generation quality
- **METEOR Score** - Semantic similarity with alignment
- **BERTScore (F1)** - Contextual embedding similarity
- **Cosine Similarity** - TF-IDF vector similarity
- **Completeness** - Coverage of reference answer

#### Error Metrics (Lower is Better)
- **Hallucination Score** - Information not present in context
- **Irrelevance Score** - Off-topic responses

#### Performance Metrics
- **Latency** - Response time in seconds
- **Trial Scores** - Multiple runs (1, 2, 3, etc.) for consistency

### Rich Visualizations
- 📊 Bar charts for latency comparison
- 📈 Quality metrics comparison across models
- 🎯 Error metrics visualization
- 🕸️ Radar charts for comprehensive model comparison
- 📋 Detailed trial-by-trial results
- 💾 CSV export for further analysis

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- HuggingFace API Token ([Get one here](https://huggingface.co/settings/tokens))

### Installation

1. Clone this repository or download the files

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

4. Open your browser to `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Configuration
1. Enter your HuggingFace API token in the sidebar
2. Select which models you want to evaluate (Mistral, Qwen, LLaMa)
3. Choose the number of trials per model (1-5)

### Step 2: Upload Documents
1. Go to the "Document Upload" tab
2. Upload one or more PDF files
3. Click "Process PDFs and Create Vector Store"
4. Wait for the vector store to be created

### Step 3: Run Evaluation
1. Go to the "Evaluation" tab
2. Enter a test question
3. Provide a reference answer (ground truth)
4. Click "Run Evaluation on Selected Models"
5. Watch as each model is evaluated across multiple trials

### Step 4: Analyze Results
1. Go to the "Results & Visualizations" tab
2. View summary statistics
3. Explore interactive charts:
   - Latency comparison
   - Quality metrics
   - Error metrics
   - Comprehensive radar chart
4. Check detailed trial results per model
5. Download results as CSV for external analysis

## 📊 Evaluation Metrics Explained

### BLEU (Bilingual Evaluation Understudy)
Measures n-gram overlap between generated and reference text. Score: 0-1 (higher is better).

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
Considers synonyms and word stems for better semantic matching. Score: 0-1 (higher is better).

### BERTScore
Uses BERT embeddings to compute similarity between tokens. Provides precision, recall, and F1. Score: 0-1 (higher is better).

### Cosine Similarity
Measures the cosine of the angle between TF-IDF vectors. Score: 0-1 (higher is better).

### Completeness
Percentage of reference answer concepts covered in the response. Score: 0-1 (higher is better).

### Hallucination Score
Estimates how much information in the answer is NOT found in the source context. Score: 0-1 (lower is better).

### Irrelevance Score
Measures how much the answer deviates from the question topic. Score: 0-1 (lower is better).

### Latency
Time taken to generate the response in seconds.

## 🛠️ Technical Stack

- **Streamlit** - Web interface
- **LangChain** - RAG orchestration
- **HuggingFace** - LLM models and embeddings
- **ChromaDB** - Vector database
- **NLTK** - NLP metrics (BLEU, METEOR)
- **BERTScore** - Contextual similarity
- **Plotly** - Interactive visualizations
- **PyPDF** - PDF processing

## 🎯 Use Cases

- Research paper Q&A
- Document analysis and comparison
- Model benchmarking for RAG systems
- Educational content verification
- Legal document review
- Technical documentation assistant

## 🔒 Privacy & Security

- All processing happens locally or through your HuggingFace API
- PDFs are temporarily stored only during processing
- No data is sent to third parties (except HuggingFace for model inference)
- Vector store is stored locally in `./chroma_db`

## 📝 Notes

- First run will download NLTK data and model weights
- Larger models (7B parameters) require good API tier on HuggingFace
- Vector store persists between sessions
- Clear results to start fresh evaluations

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

See LICENSE file for details.

---

**Built with ❤️ using Streamlit, LangChain, and HuggingFace Transformers**
