#   Multi-Modal RAG Assistant

   This project implements a Multi-Modal Retrieval-Augmented Generation (RAG) System that can process and retrieve information from a knowledge base containing various types of content such as videos, documents, and images. The system is designed to provide text-based responses to user queries in a conversational manner.

   ##   Features

   -   Processes multi-modal data (documents, images, videos)
   -   Accurately answers questions related to the knowledge base
   -   Simple user interface for interacting with the AI assistant
   -   Optional: Conversational memory

   ##   Requirements

   -   Input: A folder containing images, videos, and documents (PDF or Word) related to a specific domain or topic. [cite: 4]

   ##   Output Deliverables

   -   GitHub repository with all code and documentation [cite: 7]
   -   Folder containing the input data for the AI assistant [cite: 7]
   -   Video recording explaining the approach and code walkthrough [cite: 8]

   ##   Evaluation Criteria

   -   Technical Depth: Efficient handling of multi-modal data, retrieval mechanisms, and response generation. [cite: 9]
   -   Code Quality: Well-structured, maintainable, and documented code. [cite: 10]

   ##   Code Overview

   ###   `streamlit_app.py`

   This file contains the main Streamlit application for the Multi-Modal RAG assistant.

   ####   `MultiModalRAGApp` Class

   -   `__init__(self)`: Initializes the application, sets up the session state, initializes RAG components, and builds the user interface. [cite: 1]
   -   `_setup_session(self)`: Sets up Streamlit session state variables for conversation ID, messages, showing sources, and storing last sources.
   -   `_initialize_rag_components(self)`: Initializes the vector store, retriever, and LLM.
   -   `_build_ui(self)`: Builds the Streamlit user interface, displaying chat messages and providing options. [cite: 6]
   -   `_handle_query(self, query: str)`: Handles user queries, retrieves relevant information, generates a response using the LLM, and displays the response. [cite: 5]

   ####   Main Execution

   -   The `main()` function creates and runs the `MultiModalRAGApp`.

   ###   `process_data.py`

   This script handles the processing of multi-modal data and the creation of the vector database.

   ####   Functions

   -   `check_gpu()`: Checks for GPU availability and prints GPU information.
   -   `clear_memory()`: Clears memory (CPU and GPU).
   -   `process_all_data(data_dir: str, output_dir: str, batch_size: int = 8)`: Processes documents, images, and videos in a directory and saves the processed data.
   -   `create_vector_database(processed_paths: dict, vector_db_path: str, batch_size: int = 1024)`: Creates a vector database from the processed data.
   -   `optimize_gpu_env()`: Optimizes the GPU environment by setting environment variables.
   -   `setup_colab()`: Sets up the environment in Google Colab (installs ffmpeg).
   -   `main()`: Parses command-line arguments, sets up the environment, processes data, and creates the vector database.

   ###   `config.py`

   This file contains configuration settings for the application.

   ####   Configuration Variables

   -   `GROQ_API_KEY`: API key for the Groq LLM.
   -   `DATA_DIR`, `PROCESSED_DATA_DIR`, `EMBEDDINGS_DIR`, `TEXT_CHUNKS_DIR`, `MEDIA_METADATA_DIR`: Directories for data storage.
   -   `LLM_MODEL`, `TEMPERATURE`, `MAX_TOKENS`: LLM configuration.
   -   `VECTOR_DB_TYPE`, `VECTOR_DB_PATH`: Vector database configuration.
   -   `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`: Embedding model configuration.
   -   `CHUNK_SIZE`, `CHUNK_OVERLAP`: Chunking parameters for documents.
   -   `TOP_K_RESULTS`: Number of results to retrieve.
   -   `STREAMLIT_TITLE`, `STREAMLIT_DESCRIPTION`: Streamlit app appearance.
   -   `OCR_ENGINE`: OCR engine to use.
   -   `AUDIO_EXTRACTION_METHOD`: Method for audio extraction from videos.
   -   `VIDEO_FRAME_SAMPLE_RATE`: Frame sample rate for videos.

   ##   Running Instructions

   1.  **Prepare your data:**
       -   Place your documents, images, and videos in the `Data` directory. [cite: 4, 7]
   2.  **Install dependencies:**
       -   Run `pip install -r requirements.txt` to install the required Python packages.
   3.  **Set the Groq API key:**
       -   Ensure you have a Groq API key and set it as an environment variable named `GROQ_API_KEY`.
   4.  **Process the data and create the vector database:**

       -   Run `python process_data.py --data-dir /path/to/Data --output-dir /path/to/processed_data --vector-db-path /path/to/vector_db`
       -   Replace `/path/to/Data`, `/path/to/processed_data`, and `/path/to/vector_db` with the actual paths.
       -   Optional arguments:
           -   `--skip-processing`:  Skips data processing if it's already done.
           -   `--batch-size`: Sets the batch size for processing.
           -   `--embedding-batch-size`: Sets the batch size for creating embeddings.
           -   `--colab-setup`:  Use this flag if running in Google Colab.
   5.  **Run the Streamlit application:**
       -   Run `streamlit run streamlit_app.py`
   6.  **Interact with the assistant:**
       -   Open the Streamlit app in your browser and start asking questions about your data. [cite: 6]
