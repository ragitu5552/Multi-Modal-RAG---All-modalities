import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# File paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
EMBEDDINGS_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings")
TEXT_CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, "text_chunks")
MEDIA_METADATA_DIR = os.path.join(PROCESSED_DATA_DIR, "media_metadata")

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, TEXT_CHUNKS_DIR, MEDIA_METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM Config
LLM_MODEL = "llama3-70b-8192"  # Groq's LLaMA 3 model
TEMPERATURE = 0.1
MAX_TOKENS = 1024

# Vector Database Config
VECTOR_DB_TYPE = "faiss"  # Options: faiss, chroma
VECTOR_DB_PATH = os.path.join(PROCESSED_DATA_DIR, "vector_db")

# Embedding Config
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Text Processing Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Config
TOP_K_RESULTS = 5

# UI Config
STREAMLIT_TITLE = "Multi-Modal RAG Assistant"
STREAMLIT_DESCRIPTION = "Ask questions about the documents, images, and videos in the knowledge base."

# OCR Config
OCR_ENGINE = "easyocr"  # Options: pytesseract, easyocr

# Video Processing Config
AUDIO_EXTRACTION_METHOD = "whisper"  # Options: whisper
VIDEO_FRAME_SAMPLE_RATE = 5  # Extract a frame every 5 seconds