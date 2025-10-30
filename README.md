# 🤖 LLM RAG System

A powerful Retrieval-Augmented Generation (RAG) system built with LangChain, Streamlit, and ChromaDB. This application allows you to upload documents and ask questions about them using Large Language Models.

## ✨ Features

- **📄 Document Processing**: Upload and process PDF and TXT files
- **🔍 Intelligent Retrieval**: Uses vector embeddings for semantic search
- **💬 Interactive Chat**: Ask questions and get answers based on your documents
- **📚 Source Citations**: View the exact sources used to generate each answer
- **🎯 Multiple Embedding Options**: Support for both local (free) and OpenAI embeddings
- **🔧 Flexible LLM Integration**: Compatible with OpenAI GPT models

## 🏗️ Architecture

The system consists of four main components:

1. **Document Ingestion**: Loads and processes uploaded documents
2. **Text Chunking**: Splits documents into manageable chunks with overlap
3. **Vector Storage**: Creates embeddings and stores them in ChromaDB
4. **Retrieval & Generation**: Retrieves relevant chunks and generates answers using LLM

## 🚀 How to Run

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for LLM inference)

### Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Create a `.env` file in the project root with your API key:

```
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## 📖 Usage Guide

### Step 1: Configure API Key

- Enter your OpenAI API key in the sidebar
- Alternatively, set it in the `.env` file

### Step 2: Upload Documents

- Click "Browse files" in the sidebar
- Select one or more PDF or TXT files
- Click "Process Documents" to build your knowledge base

### Step 3: Ask Questions

- Once documents are processed, type your question in the chat input
- The system will retrieve relevant information and generate an answer
- Click "View Sources" to see which document sections were used

### Step 4: Manage Knowledge Base

- Upload additional documents to expand your knowledge base
- Click "Clear Knowledge Base" to start fresh

## 🔧 Configuration Options

### Embedding Models

- **Local (Default)**: Uses `sentence-transformers/all-MiniLM-L6-v2` - Free and runs locally
- **OpenAI**: Uses OpenAI's embedding models - Requires API key

### LLM Models

- Currently supports OpenAI's GPT-3.5-turbo
- Easy to extend for other LLM providers (Anthropic, Cohere, local models, etc.)

### Chunking Parameters

Modify in the code if needed:
- `chunk_size`: 1000 characters (default)
- `chunk_overlap`: 200 characters (default)
- `k`: Number of relevant chunks to retrieve (3 by default)

## 📦 Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM orchestration framework
- **langchain-community**: Community integrations
- **langchain-openai**: OpenAI integration
- **chromadb**: Vector database
- **sentence-transformers**: Local embedding models
- **pypdf**: PDF processing
- **python-dotenv**: Environment variable management
- **openai**: OpenAI API client
- **tiktoken**: Token counting for OpenAI models

## 🎯 Use Cases

- **Research**: Query multiple research papers simultaneously
- **Documentation**: Build a searchable knowledge base from documentation
- **Legal/Compliance**: Analyze contracts and legal documents
- **Education**: Create interactive study materials
- **Business Intelligence**: Query reports and business documents

## 🔒 Privacy & Security

- Documents are processed locally and stored in a local ChromaDB instance
- No data is persisted between sessions (unless you modify the code)
- Only the query and retrieved chunks are sent to the LLM provider
- API keys are handled securely through environment variables

## 🛠️ Extending the System

### Add More Document Types

Extend the `load_document()` function to support additional formats:

```python
elif uploaded_file.name.endswith('.docx'):
    loader = Docx2txtLoader(tmp_file_path)
```

### Use Different LLM Providers

Modify the `create_qa_chain()` function to use other providers:

```python
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-sonnet-20240229")
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
   - Documents are split into chunks
   - Each chunk is converted to a vector embedding
   - Embeddings are stored in a vector database

2. **Retrieval Phase**:
   - User query is converted to a vector embedding
   - Most similar document chunks are retrieved using semantic search
   - Retrieved chunks provide context for the LLM

3. **Generation Phase**:
   - LLM receives the query and relevant context
   - Generates an answer based on the provided information
   - Returns answer with source citations

### Vector Similarity Search

The system uses cosine similarity to find the most relevant document chunks. The ChromaDB vector store handles this efficiently even with large document collections.

## 🤝 Contributing

Feel free to extend this system with:
- Additional document loaders
- Different embedding models
- Alternative LLM providers
- Persistent storage options
- Advanced retrieval strategies (hybrid search, reranking, etc.)

## 📄 License

See LICENSE file for details.

## 🙋 Support

For issues or questions:
1. Check the error messages in the app
2. Verify your API keys are configured correctly
3. Ensure your documents are in supported formats (PDF, TXT)

---

**Built with ❤️ using LangChain, Streamlit, and ChromaDB**
