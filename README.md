# ⚛️ Quantum RAG Assistant V3
### An Intelligent Quantum Computing Tutor Powered by Retrieval-Augmented Generation, Knowledge Graphs, and Large Language Models


**An AI-powered learning assistant that makes quantum computing easier to understand through document-grounded answers, semantic retrieval, knowledge graph visualization, and interactive quantum circuit explanations.**

</p>

---

## 📌 Overview

Quantum RAG Assistant V3 is an intelligent, document-aware conversational AI application designed to help students, researchers, and quantum computing enthusiasts understand complex quantum computing concepts.

The application combines **Retrieval-Augmented Generation (RAG)**, **Knowledge Graphs**, **Vector Embeddings**, and **Large Language Models (LLMs)** to provide context-aware answers based on uploaded educational documents.

Unlike conventional AI chatbots, this system can retrieve information from a user's own knowledge base, identify relevant concepts, establish relationships between documents, and generate answers supported by retrieved sources.

The application provides a modern browser-based interface powered by FastAPI. The frontend is served directly by the backend, so a separate frontend development server is not required.

## ✨ Key Features

### 1. Intelligent RAG-Based Chat

- Ask questions about quantum computing in natural language.
- Retrieve relevant information from indexed documents.
- Generate context-aware answers using Groq-hosted LLMs.
- Support conversational questions and follow-up queries.
- Stream generated answers progressively.
- Provide source references for retrieved knowledge.
- Generate detailed explanations with examples and mathematical expressions.

### 2. Multimodal Document Processing

Upload educational and research documents to build your own knowledge base.

**Supported formats include:**

| Format | Processing |
|---|---|
| PDF | Text extraction and page-aware processing |
| DOCX | Paragraph and document text extraction |
| PPTX | Slide text and embedded-image processing |
| TXT | Plain-text extraction |
| Markdown | Structured text extraction |
| PNG | Image text extraction through OCR |
| JPG / JPEG | Image text extraction through OCR |

Document ingestion includes:

- Text extraction.
- Image and OCR processing where supported.
- Text preprocessing.
- Document chunking.
- Vector embedding generation.
- Knowledge graph entity extraction.
- Document indexing.

### 3. Knowledge Graph Visualization

The application maintains a graph-based representation of the indexed knowledge.

The graph contains three primary types of nodes:

- **Document Nodes:** Represent uploaded knowledge documents.
- **Chunk Nodes:** Represent individual text segments extracted from documents.
- **Concept Nodes:** Represent important concepts identified from document content.

Relationships connect documents, chunks, and concepts.

The knowledge graph helps users explore relationships between quantum computing topics and understand how information is distributed across their uploaded documents.

### 4. Document Vault

Manage your knowledge base through the Document Vault.

Features include:

- Upload multiple documents.
- View indexed documents.
- View document metadata.
- Monitor indexed chunk counts.
- Delete documents.
- Remove associated vector chunks and document relationships.
- Maintain the knowledge base as documents are added or removed.

### 5. Quantum Computing Visualizations

The application includes quantum visualization capabilities for explaining important concepts.

Supported visualization categories include:

- Quantum gates.
- Quantum circuits.
- Bloch sphere.
- Quantum superposition.
- Quantum entanglement.
- Quantum teleportation.
- Quantum algorithms.
- Quantum measurement.

The visualization engine includes support for concepts and algorithms such as Hadamard gates, Pauli gates, CNOT, SWAP, Grover's algorithm, Shor's algorithm, Quantum Fourier Transform, and quantum teleportation.

### 6. Conversational Intelligence

The system distinguishes basic conversational messages from knowledge-intensive questions.

Examples:

- Hello
- Hi
- How are you?
- Thank you
- Who are you?
- What can you do?

Basic conversational messages can receive natural responses without unnecessary knowledge-base retrieval.

### 7. Streaming Responses

The chat interface supports progressive answer generation.

Instead of waiting for the entire response, users can see generated text appearing progressively as the LLM produces it.

### 8. Conversation History

- Maintain previous conversations.
- Access earlier questions and answers.
- Start new conversations.
- Continue existing conversations.

### 9. Configurable RAG Parameters

The application provides configurable retrieval settings, including:

- Top-K retrieval.
- Chunk configuration.
- Similarity threshold.
- Retrieval-related parameters.

These settings help control how relevant document content is selected during question answering.

---

## 🏗️ System Architecture

The application follows a modular architecture that separates the frontend, API layer, document ingestion, retrieval, language generation, and knowledge graph components.

### Architecture Overview

```text
                 USER
                   |
                   v
        +----------------------+
        |   Browser Frontend   |
        |  HTML / CSS / JS     |
        +----------------------+
                   |
                   v
        +----------------------+
        |      FastAPI         |
        |    REST API Layer    |
        +----------------------+
                   |
          +--------+---------+
          |                  |
          v                  v
 +----------------+  +----------------+
 | Document       |  | Conversational |
 | Ingestion      |  | Chat Interface |
 +----------------+  +----------------+
          |                  |
          v                  v
 +----------------+  +----------------+
 | Text Extraction|  | Query Processing|
 | OCR Processing |  | Intent Detection|
 +----------------+  +----------------+
          |                  |
          v                  v
 +----------------+  +----------------+
 | Text Chunking  |  | Vector Retrieval|
 +----------------+  +----------------+
          |                  |
          v                  |
 +----------------+          |
 | Embedding Model|          |
 +----------------+          |
          |                  |
          v                  v
 +-----------------------------------+
 |       Knowledge Graph / RAG       |
 | Documents - Chunks - Concepts     |
 +-----------------------------------+
                   |
                   v
        +----------------------+
        | Context Construction |
        +----------------------+
                   |
                   v
        +----------------------+
        |     Groq LLM         |
        | Answer Generation    |
        +----------------------+
                   |
                   v
        +----------------------+
        | Streaming Response   |
        +----------------------+
                   |
                   v
                  USER
```

### RAG Workflow

1. The user submits a question.
2. The application processes the query.
3. Relevant document chunks are retrieved from the indexed knowledge base.
4. Retrieved context is prepared for the language model.
5. The LLM generates an answer using the available context.
6. The response is streamed to the frontend.
7. Relevant source references are displayed with the answer.

### Document Ingestion Workflow

1. The user uploads a document.
2. The document loader identifies its format.
3. Text and supported image content are extracted.
4. OCR is applied where required.
5. Extracted content is divided into manageable chunks.
6. Chunks are converted into vector embeddings.
7. Concepts and relationships are extracted.
8. The knowledge graph is updated.
9. The document becomes available for RAG-based question answering.

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Backend | Python |
| API Framework | FastAPI |
| ASGI Server | Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Language Model | Groq API |
| Embedding Model | Sentence Transformers |
| Default Embedding Model | all-MiniLM-L6-v2 |
| Vector Representation | Numerical embeddings |
| Knowledge Graph | NetworkX |
| PDF Processing | PyMuPDF / PDF extraction dependencies |
| Word Processing | python-docx |
| PowerPoint Processing | python-pptx |
| Image Processing | Pillow |
| OCR | Tesseract OCR / pytesseract |
| Data Processing | NumPy and supporting Python libraries |
| Testing | pytest |

*The exact libraries and versions are defined in the project's requirements files.*

---

## 📂 Project Structure

```text
quantum-rag-assistant-knowledge-graph/
│
├── app/
│   ├── routes/
│   │   ├── chat.py
│   │   ├── documents.py
│   │   ├── graph.py
│   │   ├── health.py
│   │   ├── history.py
│   │   └── v3.py
│   │
│   ├── static/
│   │   └── index.html
│   │
│   ├── main.py
│   ├── config.py
│   ├── logger.py
│   ├── state.py
│   └── quantum_visualizer.py
│
├── configs/
│   └── settings.yaml
│
├── data/
│   └── raw/
│       ├── uploads/
│       └── sample documents
│
├── docs/
│   └── architecture/
│
├── scripts/
│   ├── ingest.py
│   └── build_index.py
│
├── src/
│   ├── embeddings/
│   │   └── embed_documents.py
│   │
│   ├── ingestion/
│   │   ├── load_documents.py
│   │   └── chunker.py
│   │
│   ├── knowledge_graph/
│   │   └── graph_store.py
│   │
│   ├── llm/
│   │   ├── qa_chain.py
│   │   └── prompt_templates.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── .python-version
├── requirements.txt
├── requirements-dev.txt
├── requirements-quantum.txt
├── README.md
└── view_graph.py
```

---

## ⚙️ Installation and Setup

### Prerequisites

Before installing the project, make sure you have:

- Python 3.11 recommended.
- pip package manager.
- Git.
- A Groq API key.
- Internet access for downloading dependencies and the embedding model.

For image OCR functionality, install the native Tesseract OCR application.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Nitish-Goregaonkar/QuantumVerse_AI.git
```

Navigate to the project directory:

```bash
cd quantum-rag-assistant-knowledge-graph
```

### Step 2: Create a Virtual Environment

Windows:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the main dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

### Step 4: Configure Environment Variables

Create your local environment file.

Windows:

```powershell
Copy-Item .env.example .env
```

Linux / macOS:

```bash
cp .env.example .env
```

Open `.env` and configure your API key and application settings.

Example:

```env
GROQ_API_KEY=your_groq_api_key
```

Use the variable names supported by the project's configuration module. Do not add API keys directly to source code.

### Step 5: Obtain a Groq API Key

1. Visit https://console.groq.com/
2. Create or sign in to your account.
3. Generate an API key.
4. Add it to your `.env` file.

Keep your API key private.

### Step 6: Install Tesseract OCR

Tesseract is required for OCR-based extraction of text from supported images and scanned documents.

For Windows:

1. Install Tesseract OCR.
2. Add its installation directory to the Windows PATH.
3. Open a new terminal.
4. Verify the installation:

```powershell
tesseract --version
```

The Python OCR package can also be installed if it is not already included:

```bash
pip install pytesseract
```

Note that installing `pytesseract` alone does not install the native Tesseract application.

---

## 🚀 Running the Application

Start the backend from the project root:

```bash
uvicorn app.main:app --reload --port 8000
```

After successful startup, open:

**Application:**

http://localhost:8000/

**FastAPI API Documentation:**

http://localhost:8000/docs

**Alternative API Documentation:**

http://localhost:8000/redoc

**Health Endpoint:**

http://localhost:8000/api/v1/health

The frontend is served directly through FastAPI.

You do not need:

- Streamlit.
- A separate React development server.
- A separate Node.js frontend server.

---

## 📖 How to Use the Application

### 1. Upload Documents

1. Open the application.
2. Navigate to Document Vault.
3. Upload PDF, DOCX, PPTX, TXT, Markdown, or supported image files.
4. Wait for document extraction and indexing.
5. Check the indexed document and chunk information.

### 2. Ask Questions

1. Navigate to RAG Chat AI.
2. Enter a question in the chat input.
3. Submit your question.
4. Wait for the progressively streamed answer.
5. Review the generated response and source references.

Example questions:

- What is quantum computing?
- Explain quantum superposition with an example.
- What is the difference between classical and quantum computing?
- Explain Grover's algorithm step by step.
- How does quantum teleportation work?

### 3. Explore the Knowledge Graph

1. Navigate to Knowledge Graph.
2. Explore document, chunk, and concept nodes.
3. Select a node to inspect its information.
4. Explore relationships between concepts and documents.

### 4. Manage Conversations

Use the conversation sidebar to create new conversations and access previous chat sessions.

---

## 🧪 Testing

The project includes automated regression tests for important application components.

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run all tests:

```bash
python -m pytest -v
```

Run tests with a concise summary:

```bash
python -m pytest -q
```

Run a specific test module:

```bash
python -m pytest tests/test_document_deletion.py -v
```

The test suite covers areas including:

- RAG intelligence.
- Document ingestion.
- PowerPoint image processing.
- Quantum visualizations.
- Chat streaming and conversational responses.
- Document deletion.
- Knowledge graph document connectivity.

---

## 🔌 API Endpoints

The FastAPI backend exposes endpoints for interacting with the application.

The primary API route groups include:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Check application health |
| GET | `/api/v1/documents` | Retrieve indexed documents |
| GET | `/api/v1/documents/status` | Retrieve indexing status |
| POST | `/api/v1/documents/upload` | Upload and process documents |
| DELETE | `/api/v1/documents/{document_id}` | Delete a document |
| GET | `/api/v1/graph/nodes` | Retrieve knowledge graph nodes |
| GET | `/api/v1/history` | Retrieve conversation history |

The chat and additional V3 endpoints are defined in the application route modules.

For the complete, live endpoint schemas and request parameters, open:

http://localhost:8000/docs

---

## 🔧 Configuration

Application configuration is maintained through environment variables and configuration files.

Important configurable areas include:

| Configuration | Purpose |
|---|---|
| Groq API key | Authentication for LLM requests |
| LLM model | Select the supported generation model |
| Embedding model | Select the sentence embedding model |
| Top-K retrieval | Number of relevant chunks to retrieve |
| Similarity threshold | Control retrieval relevance |
| Chunk size | Configure text segmentation |
| Chunk overlap | Preserve context between chunks |
| Knowledge graph path | Configure graph persistence |

Check `.env.example`, `configs/settings.yaml`, and the configuration modules for the exact supported variable names and defaults.

---

## ⚠️ Troubleshooting

### 1. Groq API Error

Check that:

- Your API key is correct.
- Your `.env` file is present.
- Your account has access to the selected model.
- You have not exceeded the applicable API limits.

Restart the application after changing environment variables.

### 2. Embedding Model Download Failure

The first application startup may require downloading the embedding model.

Check your internet connection and retry.

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 3. Tesseract OCR Not Working

Verify:

```powershell
tesseract --version
```

If the command is not recognized, add the Tesseract installation directory to PATH and restart the terminal.

### 4. Documents Are Not Indexed

Check:

- File format support.
- Document extraction logs.
- File permissions.
- Available disk space.
- Embedding model availability.
- Knowledge graph persistence configuration.

### 5. Knowledge Graph Appears Empty

Verify that:

- Documents have completed indexing.
- Concepts were extracted successfully.
- The graph persistence file is available.
- The backend has loaded the correct graph.

### 6. Port 8000 Is Already in Use

Run the application on another port:

```bash
uvicorn app.main:app --reload --port 8001
```

Then open:

http://localhost:8001/

---

## 🔐 Security and Data Privacy

- Never commit `.env` files.
- Never publish Groq API keys.
- Avoid uploading confidential documents to public repositories.
- Review `.gitignore` before pushing the project.
- Keep generated knowledge graph data and private uploaded files out of public repositories.
- Use environment variables for API credentials.
- Review external API usage and data handling requirements before deploying the application.

The application uses a local backend and local document/graph storage, while LLM generation uses the configured external Groq API.

---

## 🚧 Future Enhancements

Potential future improvements include:

- More advanced multimodal document understanding.
- Improved extraction of diagrams, tables, and mathematical equations.
- More sophisticated semantic knowledge graph construction.
- Automatic concept clustering.
- Graph-based multi-hop retrieval.
- Improved answer evaluation and citation verification.
- Support for additional quantum algorithms and circuit visualizations.
- Exportable learning notes and summaries.
- Advanced document comparison.
- Authentication and multi-user knowledge bases.
- Production deployment and monitoring.

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome.

To contribute:

1. Fork this repository.
2. Create a feature branch.
3. Implement your changes.
4. Run the automated tests.
5. Commit your changes.
6. Open a Pull Request.

Example:

```bash
git checkout -b feature/your-feature-name

git add .

git commit -m "Add your feature"

git push origin feature/your-feature-name
```

---

## 📜 License

Choose and add an appropriate open-source license before publishing this project.

For example, if you choose MIT, include a `LICENSE` file containing the MIT License text.

Do not assume that dependencies, sample documents, or third-party assets automatically share your project's license.

---

## 👨‍💻 Author

**Project:** Quantum RAG Assistant V3

**Domain:** Artificial Intelligence, Quantum Computing, Retrieval-Augmented Generation, and Knowledge Graphs.

**GitHub:** (https://github.com/Nitish-Goregaonkar/QuantumVerse_AI.git)

---

## ⭐ Acknowledgements

This project builds upon the open-source Python ecosystem and technologies for:

- FastAPI
- Groq
- Sentence Transformers
- NetworkX
- Hugging Face
- PyTorch
- Pillow
- Tesseract OCR

Special thanks to the developers and research communities working on artificial intelligence, quantum computing, and open-source software.

---

Team Members:- 1. Nitish Goregaonkar (2503033111245507), 2. Dnayneshwar Yashwant Kamblekar (2503033111245508)
**Built to make Quantum Computing more accessible through AI. ⚛️**

If you find this project useful, consider giving the repository a ⭐ on GitHub.

</p>