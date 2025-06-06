# 🔬 VisioTextual Insight Engine for Research Papers (VTIERP)

VTIERP is an advanced AI-powered system engineered to revolutionize how researchers and professionals interact with academic literature and complex documents. Unlike traditional RAG systems, VTIERP doesn't just process text; it intelligently **understands and synthesizes information from both textual and visual modalities**, including figures, tables, and complex layouts. This project pushes the boundaries of multimodal RAG by emphasizing human-like document understanding and contextual reasoning.

## 🚀 Why VTIERP Stands Out: An Innovative Edge

Many LLM applications for document Q&A struggle with the inherent complexity of PDFs: jumbled text from multi-column layouts, inaccessible data in tables, and uninterpreted visual content. VTIERP addresses these critical challenges, offering several distinct advantages:

1.  **True Multimodal Context Understanding (Beyond Text-Only RAG):**
    *   Leveraging insights from, VTIERP goes beyond basic textual RAG. It actively extracts and integrates content from both text and visual elements (figures, tables).
    *   **Intelligent Visual Element Extraction:** It identifies figures and tables, generates rich VLM (Vision-Language Model) descriptions for them, allowing the LLM to "see" and interpret visual data. This is particularly impactful for scientific papers where figures and tables convey crucial information.
    *   **Robust Table Handling:** Recognizes tables and attempts to extract their raw textual content and even generate Markdown representations. Critically, if text or Markdown extraction is insufficient, it intelligently creates a visual image of the table region and generates a VLM description, ensuring no table information is lost. This addresses a major pain point in document AI.

2.  **Advanced Layout Awareness & Contextual Grounding:**
    *   Inspired by research in Document Understanding, VTIERP's parsing is heuristically aware of document layout. It attempts to:
        *   Preserve line and block structures.
        *   Handle text from HTML extraction (e.g., for basic subscript/superscript preservation) for digital PDFs.
        *   Accurately define bounding boxes for extracted elements to ensure visual context for LLM.
    *   **Precise Source Attribution:** Answers are meticulously grounded with explicit citations to the source document (e.g., 'AAG.pdf') and page number (e.g., 'Page 13'), mirroring human research practices and building trust in LLM outputs.

3.  **Sophisticated LLM Agent Orchestration with LangGraph:**
    *   Unlike simple RAG chains, VTIERP employs a LangGraph-powered agent. This agent orchestrates a multi-step reasoning process:
        *   **Query Transformation:** Prepares the user's query for optimal retrieval.
        *   **Multimodal Retrieval:** Simultaneously fetches relevant text chunks and image descriptions from dedicated vector stores.
        *   **Intelligent Reranking & Selection:** Prioritizes critical information like document titles, abstracts, and overall corpus summaries. It ensures the most relevant context, including structured table data (Markdown, raw text) and VLM descriptions, is presented to the LLM.
        *   **Context-Aware Generation:** The LLM is meticulously prompted to synthesize information from various modalities, handle follow-up questions using chat history, and provide precise, hallucination-free answers.

4.  **Scalable & Deployable Architecture (12-Factor Compliant):**
    *   Built with FastAPI for a robust API backend and Streamlit for an interactive UI.
    *   Fully **Dockerized** using `docker-compose`, adhering to 12-Factor App principles for consistent development, testing, and deployment across environments.
    *   **CI/CD Pipeline** with GitHub Actions ensures code quality (linting, testing) and automated image builds, facilitating continuous integration and reliable releases.

## 🛠️ Tech Stack

*   **LLM/VLM:** Google Gemini (e.g., `gemini-1.5-flash-preview-05-20`, `models/text-embedding-004` via API)
*   **Orchestration:** Langchain, LangGraph
*   **PDF Processing:** PyMuPDF (Fitz) for extraction and rendering, augmented with heuristics for layout.
*   **Text Cleaning/OCR:** Unstructured.io (for robust text cleaning and optional OCR fallback for scanned documents).
*   **Vector Store:** ChromaDB (for efficient semantic search of textual and image descriptions).
*   **Backend:** FastAPI (for high-performance API endpoints).
*   **Frontend:** Streamlit (for intuitive user interaction).
*   **Containerization:** Docker, Docker Compose.
*   **CI/CD:** GitHub Actions.

## 📂 Project Structure

```
vtierp_project_custom/
├── .github/workflows/ci.yml    # CI/CD pipeline definition
├── app/                        # FastAPI backend application
│   ├── core/                   # Configuration, app-wide settings
│   ├── dependencies_config/    # LLM, Embedding model setup (singleton instances)
│   ├── models/                 # Pydantic models for API request/response validation
│   ├── services/               # Core business logic (PDF processing, RAG agent, Vector Store management, utilities)
│   └── main.py                 # FastAPI application entry point, API endpoints
├── data/                       # Persistent data (mounted as Docker volumes)
│   ├── uploads/                # Temporary storage for uploaded PDF files
│   └── vector_stores/          # Persistent storage for ChromaDB vector stores and extracted images
├── notebooks/                  # Jupyter notebook(s) for initial exploration and experimentation
├── .env.example                # Example environment variables (sensitive data goes in .env)
├── .flake8                     # Flake8 linter configuration
├── .gitignore                  # Files/directories to ignore in Git
├── Dockerfile.api              # Dockerfile for the FastAPI backend service
├── Dockerfile.streamlit        # Dockerfile for the Streamlit frontend service
├── requirements.txt            # Python dependencies for the API service (installed in Dockerfile.api)
├── requirements_streamlit.txt  # Python dependencies for the Streamlit frontend service (installed in Dockerfile.streamlit)
└── README.md                   # Project README (this file)
```

## 🚀 Setup and Running

**Prerequisites:**
*   **Docker and Docker Compose:** Ensure you have these installed on your system.
*   **Google Gemini API Key:** Obtain an API key from Google AI Studio.

**1. Clone the Repository:**
   ```bash
   git clone https://github.com/Dipesh-Chaudhary/vtierp_project
   cd vtierp_project
   ```

**2. Create Environment File:**
   Copy `.env.example` to `.env` and **add your actual `GOOGLE_API_KEY`**:
   ```bash
   cp .env.example .env
   # Open .env file and paste your key:
   # GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
   # You can also customize LLM_RAG_MODEL, LLM_AUX_MODEL, etc. here if needed.
   ```

**3. Build and Run with Docker Compose:**
   This command will build both the FastAPI API and Streamlit UI Docker images and start the services.
   ```bash
   docker-compose build --no-cache # --no-cache to ensure all layers are rebuilt, useful after requirement changes
   docker-compose up -d           # -d runs containers in detached mode (in the background)
   ```
   To view the logs from both services:
   ```bash
   docker-compose logs -f
   ```
   To view logs from a specific service (e.g., API):
   ```bash
   docker-compose logs -f api
   ```

**4. Access the Application:**
   *   **FastAPI API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
   *   **Streamlit UI:** [http://localhost:8501](http://localhost:8501)

**5. Stopping the Application:**
   To stop and remove the containers, networks, and volumes (excluding the `data` folder on your host):
   ```bash
   docker-compose down
   ```

## Principle Adherence: 12-Factor App Methodology

VTIERP is architected with the [12-Factor App methodology](https://12factor.net/) as a guiding principle, ensuring robustness, scalability, and maintainability:

1.  **Codebase:** Single codebase tracked in Git, deployable to various environments. (✅)
2.  **Dependencies:** Explicitly declared and isolated via `requirements.txt` and `requirements_streamlit.txt` within Docker containers. (✅)
3.  **Config:** All configuration (API keys, model names) is stored in environment variables, loaded from `.env` locally or injected in Docker. (✅)
4.  **Backing Services:** LLMs (Gemini API), Vector Store (ChromaDB), and image storage are treated as loosely coupled attached resources. (✅)
5.  **Build, Release, Run:** Separate stages are enforced by Dockerfiles and GitHub Actions workflow (builds Docker images, then runs them). (✅)
6.  **Processes:** The FastAPI and Streamlit applications run as stateless processes. Any persistent data (vector stores, uploaded PDFs) is managed through bind mounts. (✅)
7.  **Port Binding:** Services are self-contained and export via port binding (FastAPI on 8000, Streamlit on 8501). (✅)
8.  **Concurrency:** Scalability is achieved through a process model; multiple instances can be run horizontally. (✅)
9.  **Disposability:** Containers are designed for fast startup and graceful shutdown, making them resilient. (✅)
10. **Dev/Prod Parity:** Docker ensures that development, staging, and production environments are as similar as possible. (✅)
11. **Logs:** Logs are streamed to `stdout/stderr` from containers, aggregated and managed by `docker-compose logs`. (✅)
12. **Admin Processes:** One-off management tasks can be executed against the production environment through container commands. (✅)

## 🔮 Future Enhancements & Research Directions

This project serves as a strong foundation, but the journey towards true human-like document understanding is ongoing. Future work could explore:

*   **Advanced Table Recognition:** Integrate specialized deep learning models for table detection and structure recognition (e.g.,) beyond current heuristics. This would enable perfect Markdown generation for complex tables and even visual reconstruction of tables.
*   **Deep Layout Understanding:** Move beyond heuristics to integrate multimodal pre-trained models like LayoutLM, LayoutXLM, or UDOP which explicitly learn document layout, text, and visual relationships. This could lead to more robust sectioning, element association (even across columns/pages), and mathematical equation understanding.
*   **Mathematical Equation Parsing:** Employ dedicated tools or VLM fine-tuning for extracting and potentially rendering LaTeX from images of equations.
*   **Grounding and Citation Confidence:** Implement visual grounding to highlight the exact source regions in the PDF when answering questions, enhancing verifiability and trust. This could involve storing bounding box information with retrieved chunks.
*   **Intelligent Document Agents:** Expand the LangGraph agent with more sophisticated planning, self-correction, and tool-use capabilities, potentially involving iterative refinement loops as in self-rewarding LLMs.
*   **Long-Term Memory Management:** Implement advanced memory strategies for LLMs beyond just passing recent chat history, including summarization, retrieval over past conversations, and structured knowledge bases for user profiles or application-level facts.
*   **Asynchronous Processing:** For very large PDFs or high concurrency, integrate a dedicated task queue (e.g., Celery with Redis/RabbitMQ) for background PDF processing to improve responsiveness and scalability.
*   **Error Reporting & Monitoring:** Implement more robust error tracking, logging, and monitoring in a production environment.

