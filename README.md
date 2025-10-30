# 🤖 LLM RAG System with Hugging Face

A powerful Retrieval-Augmented Generation (RAG) system built with LangChain, Streamlit, ChromaDB, and Hugging Face models. This application runs completely locally with quantized models, allowing you to upload documents and ask questions about them without any API costs!

## ✨ Features

- **📄 Document Processing**: Upload and process PDF and TXT files
- **🔍 Intelligent Retrieval**: Uses vector embeddings for semantic search
- **💬 Interactive Chat**: Ask questions and get answers based on your documents
- **📚 Source Citations**: View the exact sources used to generate each answer
- **🤗 Local LLM**: Uses Hugging Face models with 4-bit quantization
- **💰 No API Costs**: Runs completely locally on your hardware
- **🔒 Privacy First**: All data stays on your machine
- **📊 Comprehensive Evaluation**: Track 10+ metrics including trial scores, latency, BERTScore, BLEU, METEOR, hallucination detection, and more
- **📈 Visual Analytics**: Interactive bar charts and dashboards to compare performance across trials
- **💾 Export Results**: Download evaluation data as CSV for further analysis

## 🏗️ Architecture

The system consists of four main components:

1. **Document Ingestion**: Loads and processes uploaded documents
2. **Text Chunking**: Splits documents into manageable chunks with overlap
3. **Vector Storage**: Creates embeddings and stores them in ChromaDB
4. **Retrieval & Generation**: Retrieves relevant chunks and generates answers using a quantized LLM

### Quantization Details

- **Method**: 4-bit quantization using BitsAndBytes (NF4)
- **Compute Type**: bfloat16
- **Benefits**: 
  - ~75% reduction in memory usage
  - Faster inference on consumer GPUs
  - Minimal accuracy degradation
  - Enables running larger models on smaller hardware

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)
- 8GB+ RAM (16GB+ recommended)
- Hugging Face account and token

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Get your Hugging Face token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" access
   - Copy the token

3. (Optional) Create a `.env` file in the project root:

```
HF_TOKEN=your_huggingface_token_here
```

### Running the Application

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage Guide

### Step 1: Load the Model

- Enter your Hugging Face token in the sidebar
- Select one of the 3 available models from the dropdown:
  - **Qwen 2.5 - 3B**: Fast, excellent quality, 6GB VRAM
  - **Mistral 7B**: High quality, 10GB VRAM
  - **Llama 3.2 - 3B**: Fast, excellent quality, 6GB VRAM
- Click "Load Model" (first load may take 5-10 minutes to download)
- Wait for the model to load into memory with quantization

### Step 2: Upload Documents

- Click "Browse files" in the sidebar
- Select one or more PDF or TXT files
- Click "Process Documents" to build your knowledge base

### Step 3: Ask Questions

- Once documents are processed, type your question in the chat input
- The system will retrieve relevant information and generate an answer
- Click "View Sources" to see which document sections were used

### Step 4: View Evaluations (Optional)

- Enable "Enable Evaluation" checkbox in the Chat tab
- Each response will be automatically evaluated
- Switch to the "Evaluation Dashboard" tab to see:
  - Trial scores (1, 2, 3, etc.)
  - Latency measurements
  - Cosine similarity scores
  - BERTScore F1, Precision, Recall
  - BLEU and METEOR scores
  - Hallucination risk detection
  - Relevance and completeness metrics
  - Interactive bar charts comparing all metrics
  - Downloadable CSV data

### Step 5: Manage System

- Upload additional documents to expand your knowledge base
- Click "Clear Knowledge Base" to start fresh
- Click "Unload Model" to free up GPU memory
- Download evaluation data for offline analysis

## 🔧 Configuration Options

### Model Selection

The app comes with 3 pre-configured models available in a dropdown menu:

1. **Qwen/Qwen2.5-3B-Instruct** ⭐ Recommended for most users
   - Size: 3 billion parameters
   - VRAM: ~6GB with 4-bit quantization
   - Speed: Fast
   - Quality: Excellent for its size
   - Best for: General purpose RAG tasks

2. **mistralai/Mistral-7B-Instruct-v0.1**
   - Size: 7 billion parameters
   - VRAM: ~10GB with 4-bit quantization
   - Speed: Moderate
   - Quality: Very high quality
   - Best for: Complex queries requiring better reasoning

3. **meta-llama/Llama-3.2-3B-Instruct**
   - Size: 3 billion parameters
   - VRAM: ~6GB with 4-bit quantization
   - Speed: Fast
   - Quality: Excellent for its size
   - Best for: Alternative to Qwen with Meta's LLama architecture

### Quantization Settings

Current configuration (in code):
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

### Embedding Model

Uses `sentence-transformers/all-MiniLM-L6-v2` (local, free)

### Chunking Parameters

- `chunk_size`: 1000 characters
- `chunk_overlap`: 200 characters
- `k`: Number of relevant chunks to retrieve (3 by default)

## 📦 Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM orchestration framework
- **langchain-community**: Community integrations
- **langchain-huggingface**: Hugging Face integration
- **chromadb**: Vector database
- **sentence-transformers**: Local embedding models
- **pypdf**: PDF processing
- **python-dotenv**: Environment variable management
- **transformers**: Hugging Face transformers library
- **huggingface_hub**: HF model hub access
- **bitsandbytes**: Quantization library
- **accelerate**: Model loading optimization
- **torch**: PyTorch deep learning framework
- **ragas**: RAG evaluation framework
- **bert-score**: BERTScore metric
- **nltk**: Natural language processing (BLEU, METEOR)
- **rouge-score**: ROUGE metric
- **scikit-learn**: Cosine similarity
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Additional plotting

## 📊 Evaluation Metrics

The system includes comprehensive evaluation capabilities to measure RAG performance:

### Core Metrics

1. **Trial Score (0-100)**: Overall performance score combining multiple metrics
2. **Latency**: Response time in seconds
3. **Cosine Similarity**: Semantic similarity between question, answer, and context
4. **Answer Relevance**: How well the answer addresses the question
5. **Completeness Score**: How complete and thorough the answer is

### NLP Quality Metrics

6. **BERTScore F1**: Semantic similarity using BERT embeddings
   - Precision: How much of the answer is relevant
   - Recall: How much of the expected content is covered
7. **BLEU Score**: N-gram overlap (standard machine translation metric)
8. **METEOR Score**: Unigram precision/recall with synonyms

### Reliability Metrics

9. **Hallucination Score**: Risk of fabricated information (lower is better)
10. **Irrelevance Score**: Measure of off-topic content (lower is better)
11. **Context Similarity**: How grounded the answer is in source documents

### Visualization Features

- **Bar Charts**: Compare metrics across multiple trials
- **Summary Statistics**: Average scores across all evaluations
- **Detailed Tables**: View raw evaluation data
- **CSV Export**: Download data for external analysis

## 🎯 Use Cases

- **Research**: Query multiple research papers simultaneously + track answer quality
- **Documentation**: Build a searchable knowledge base from documentation
- **Legal/Compliance**: Analyze contracts and legal documents with hallucination detection
- **Education**: Create interactive study materials with performance tracking
- **Business Intelligence**: Query reports and business documents
- **Personal Knowledge Base**: Search through your personal document collection
- **Model Comparison**: Evaluate different LLMs on the same questions
- **Performance Benchmarking**: Track improvements over time

## 🔒 Privacy & Security

- ✅ Everything runs locally on your machine
- ✅ Documents never leave your computer
- ✅ No API calls to external services (except model download)
- ✅ No usage tracking or telemetry
- ✅ Complete data privacy
- ✅ No per-query costs

## 💻 Hardware Requirements

### Minimum
- **GPU**: 6GB VRAM (for 3B models with 4-bit quantization)
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space (for model cache)

### Recommended
- **GPU**: 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16GB system RAM
- **Storage**: 20GB+ free space

### For Larger Models (7B)
- **GPU**: 12GB+ VRAM (RTX 3060 12GB or better)
- **RAM**: 16GB+ system RAM

## 🛠️ Extending the System

### Add More Document Types

Extend the `load_document()` function:

```python
from langchain_community.document_loaders import Docx2txtLoader

elif uploaded_file.name.endswith('.docx'):
    loader = Docx2txtLoader(tmp_file_path)
```

### Add More Models

To add more models, edit the `AVAILABLE_MODELS` dictionary in `streamlit_app.py`:

```python
AVAILABLE_MODELS = {
    "Model Display Name": "huggingface/model-name",
    "Your Custom Model": "your-username/your-model-name",
}
```

Any instruction-tuned model from Hugging Face that supports text generation will work!

### Adjust Generation Parameters

Modify the pipeline settings in `load_llm_model()`:

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  # Increase for longer responses
    temperature=0.7,      # Higher = more creative
    top_p=0.95,
    repetition_penalty=1.15,
)
```

### Persist Vector Store

Add persistence to ChromaDB:

```python
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

## 📝 Technical Details

### How RAG Works

1. **Indexing Phase**:
   - Documents are split into chunks (1000 chars with 200 overlap)
   - Each chunk is converted to a 384-dim vector embedding
   - Embeddings are stored in ChromaDB with metadata

2. **Retrieval Phase**:
   - User query is converted to a vector embedding
   - Top-k most similar document chunks are retrieved (k=3)
   - Retrieved chunks provide context for the LLM

3. **Generation Phase**:
   - LLM receives the query and relevant context
   - Generates an answer based on provided information
   - Returns answer with source citations

### Model Quantization

The system uses **4-bit NormalFloat (NF4)** quantization:
- Reduces model size by ~75%
- Maintains ~99% of original model quality
- Enables running 7B models on 8GB GPUs
- Uses bfloat16 for compute precision

### Vector Similarity Search

Uses **cosine similarity** to find relevant chunks:
- Embeddings are normalized to unit vectors
- Fast approximate nearest neighbor search
- Handles thousands of documents efficiently

## 🚀 Performance Tips

1. **First Load**: Model download and quantization take time (5-10 min)
2. **Subsequent Loads**: Models are cached locally (~10-30 seconds)
3. **GPU Memory**: Close other GPU-intensive applications
4. **Chunk Size**: Smaller chunks = more precise, larger = more context
5. **Temperature**: Lower (0.1-0.3) = more focused, higher (0.7-0.9) = more creative

## 🐛 Troubleshooting

### Out of Memory Error
- Try a smaller model (3B instead of 7B)
- Reduce `max_new_tokens` in the pipeline
- Close other applications using GPU

### Model Download Fails
- Check your internet connection
- Verify Hugging Face token has "Read" access
- Try a different model

### Slow Generation
- Normal for first query (model warmup)
- Subsequent queries should be faster
- Consider using a smaller model for speed

## 🤝 Contributing

Feel free to extend this system with:
- Additional document loaders (DOCX, HTML, Markdown, etc.)
- Different embedding models
- Alternative quantization methods (8-bit, GPTQ, AWQ)
- Advanced retrieval strategies (hybrid search, reranking)
- Multi-turn conversation support
- Document summarization features

## 📄 License

See LICENSE file for details.

## 🙋 Support

For issues:
1. Check error messages in the Streamlit interface
2. Verify GPU/CUDA is properly configured
3. Ensure sufficient GPU memory for your chosen model
4. Check Hugging Face token permissions

---

**Built with ❤️ using 🤗 Hugging Face, LangChain, Streamlit, and ChromaDB**

*No API costs. Complete privacy. Runs locally.*
