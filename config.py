import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
EMBEDDINGS_DIR = os.path.join(PROCESSED_DATA_DIR, "embeddings")
TEXT_CHUNKS_DIR = os.path.join(PROCESSED_DATA_DIR, "text_chunks")
MEDIA_METADATA_DIR = os.path.join(PROCESSED_DATA_DIR, "media_metadata")

for directory in [DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, TEXT_CHUNKS_DIR, MEDIA_METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

LLM_MODEL = "llama3-70b-8192"  # Groq's LLaMA 3 model
TEMPERATURE = 0.1
MAX_TOKENS = 1024

VECTOR_DB_TYPE = "faiss"  
VECTOR_DB_PATH = os.path.join(PROCESSED_DATA_DIR, "vector_db")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K_RESULTS = 5

STREAMLIT_TITLE = "Multi-Modal RAG Assistant"
STREAMLIT_DESCRIPTION = "Ask questions about the documents, images, and videos in the knowledge base."

OCR_ENGINE = "easyocr"  

AUDIO_EXTRACTION_METHOD = "whisper"  
VIDEO_FRAME_SAMPLE_RATE = 5  
