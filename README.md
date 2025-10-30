# 📚 PDF RAG Evaluation System

A comprehensive evaluation framework for comparing multiple Large Language Models (LLMs) on PDF-based Retrieval-Augmented Generation (RAG) tasks.

## 🌟 Features

### Supported Models
- **Mistral-7B-Instruct** - High-performance instruction-tuned model
- **Qwen2.5-7B-Instruct** - Advanced multilingual model
- **Llama-3.2-3B-Instruct** - Efficient Meta's Llama model

### Comprehensive Evaluation Metrics

#### Quality Metrics (Higher is Better)
- **Cosine Similarity**: Semantic similarity between response and reference using embeddings
- **BERTScore F1**: Contextual similarity using BERT embeddings (precision, recall, F1)
- **BLEU Score**: N-gram overlap measure for translation quality
- **METEOR Score**: Advanced metric considering synonyms and paraphrases
- **Completeness**: Measures how complete the response is compared to reference

#### Reliability Metrics (Lower is Better)
- **Hallucination Score**: Detects when model generates information not in the context
- **Irrelevance Score**: Measures how off-topic the response is from the query

#### Performance Metrics
- **Latency**: Response time for each query (in seconds)
- **Average Response Time**: Mean latency across multiple trials

### Advanced Features
- ✅ **Multi-Trial Evaluation**: Run 1-5 trials per model for statistical reliability
- ✅ **Interactive Visualizations**: Bar charts, radar charts, and comparison dashboards
- ✅ **PDF Upload & Processing**: Automatic chunking and vector store creation
- ✅ **Real-time Progress Tracking**: Monitor evaluation progress across models
- ✅ **Export Results**: Download evaluation data as CSV for further analysis
- ✅ **Trial-by-Trial Breakdown**: View individual trial metrics and responses

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- HuggingFace API Token ([Get one here](https://huggingface.co/settings/tokens))

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Configuration
1. Enter your **HuggingFace API Token** in the sidebar
2. Upload one or more **PDF files**
3. Click **"Process PDFs"** to create the vector store

### Step 2: Model Selection
- Select which models to evaluate (Mistral, Qwen3, Llama)
- Choose the number of trials (1-5) for statistical reliability

### Step 3: Evaluation
1. Go to the **"Query & Evaluate"** tab
2. Enter your question about the PDFs
3. Provide a reference answer (recommended for accurate metrics)
4. Click **"Evaluate All Models"**
5. Wait for evaluation to complete across all selected models

### Step 4: View Results
- **Results Tab**: View summary table with all metrics
  - Green highlights indicate best performance
  - Download results as CSV
- **Analytics Tab**: Interactive visualizations
  - Quality metrics comparison
  - Reliability metrics (hallucination, irrelevance)
  - Latency comparison
  - Overall performance radar chart
  - Detailed metrics breakdown per model

## 📊 Understanding the Metrics

### Quality Metrics
| Metric | Range | Best | Description |
|--------|-------|------|-------------|
| Cosine Similarity | 0-1 | Higher | Semantic similarity using embeddings |
| BERTScore F1 | 0-1 | Higher | Contextual similarity with BERT |
| BLEU | 0-1 | Higher | N-gram overlap precision |
| METEOR | 0-1 | Higher | Advanced similarity with synonyms |
| Completeness | 0-1 | Higher | Coverage of reference content |

### Reliability Metrics
| Metric | Range | Best | Description |
|--------|-------|------|-------------|
| Hallucination | 0-1 | Lower | Semantic divergence from context |
| Irrelevance | 0-1 | Lower | Off-topic response detection |

### Performance Metrics
| Metric | Unit | Best | Description |
|--------|------|------|-------------|
| Latency | seconds | Lower | Response generation time |
| Avg Response Time | seconds | Lower | Mean across all trials |

## 🎯 Use Cases

1. **Model Selection**: Compare models to choose the best for your use case
2. **Performance Benchmarking**: Track improvements across model versions
3. **Quality Assurance**: Ensure responses meet quality standards
4. **Research & Development**: Analyze model behavior on domain-specific documents
5. **Production Monitoring**: Evaluate model performance before deployment

## 🔧 Technical Architecture

### Components
- **PDF Processing**: PyPDF2 for document loading
- **Embeddings**: HuggingFace sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for efficient similarity search
- **LLM Integration**: HuggingFace Hub API for model inference
- **Evaluation**: Multiple metrics libraries (bert-score, evaluate, nltk, scikit-learn)
- **Visualization**: Plotly for interactive charts

### Workflow
```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Store
                                                          ↓
Query → Retrieval → Context + Query → LLM → Response → Evaluation
                                                          ↓
        Metrics Calculation → Trial Aggregation → Visualization
```

## 📦 Dependencies

Key packages:
- `streamlit` - Web application framework
- `langchain` - LLM orchestration and RAG pipeline
- `transformers` - HuggingFace models and tokenizers
- `faiss-cpu` - Vector similarity search
- `bert-score` - BERTScore evaluation
- `nltk` - BLEU and METEOR scores
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `scikit-learn` - Cosine similarity

See `requirements.txt` for complete list.

## 🎨 Visualization Examples

The system provides multiple visualization types:

1. **Bar Charts**: Compare metrics across models side-by-side
2. **Radar Charts**: Overall performance profile for each model
3. **Latency Charts**: Response time comparison with color coding
4. **Trial Breakdown**: Individual trial performance tables

## 🔐 Security Notes

- API tokens are handled securely (input type: password)
- Temporary PDF files are deleted after processing
- No data is stored permanently on the server
- All processing happens in your local environment

## 🐛 Troubleshooting

### Common Issues

**Error: "API token is invalid"**
- Verify your HuggingFace token is correct
- Ensure you have access to the selected models
- Some models may require acceptance of terms on HuggingFace

**Error: "Model loading failed"**
- Check your internet connection
- Verify model names are correct
- Some models may be rate-limited

**Slow performance**
- Reduce number of trials
- Use smaller PDF files for testing
- Consider using models with fewer parameters

**NLTK download errors**
- The app automatically downloads required NLTK data
- If it fails, manually run: `python -m nltk.downloader wordnet punkt omw-1.4`

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM support (GPT, Claude, etc.)
- More evaluation metrics
- Custom metric weights
- Batch evaluation from CSV
- Model fine-tuning integration

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace for model hosting and APIs
- LangChain for RAG framework
- Streamlit for the web framework
- The open-source community for evaluation metrics

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review HuggingFace model pages for model-specific help

---

**Built with ❤️ for the AI research community**

Happy evaluating! 🚀
