import os
import json
import logging
import pickle
from typing import List, Dict, Any, Optional, Union
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector database for storing and retrieving embeddings.
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_path: str = "processed_data/vector_db",
                 dimension: int = 384):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Model to use for generating embeddings
            vector_db_path: Path to store vector database
            dimension: Dimension of the embedding vectors
        """
        self.embedding_model_name = embedding_model
        self.vector_db_path = vector_db_path
        self.dimension = dimension
        
        # Load the embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create directory for vector database
        os.makedirs(vector_db_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        # Storage for documents
        self.documents = []
        self.document_lookup = {}  # Map index to document
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Extract text content for embedding
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Store the starting index before adding new documents
        start_idx = len(self.documents)
        
        # Add embeddings to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store documents in lookup
        for i, doc in enumerate(documents):
            idx = start_idx + i
            self.documents.append(doc)
            self.document_lookup[idx] = doc
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of document dictionaries with similarity scores
        """
        if not self.documents:
            logger.warning("No documents in the vector store")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search for similar documents
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k=min(top_k, len(self.documents))
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Check if index is valid
                doc = self.document_lookup[idx]
                results.append({
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'score': float(distances[0][i])
                })
        
        return results
    
    def save(self, filename: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            filename: Base filename to use (without extension)
        """
        if filename is None:
            filename = "vector_store"
        
        filepath_base = os.path.join(self.vector_db_path, filename)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath_base}.faiss")
        
        # Save documents and mapping
        with open(f"{filepath_base}.documents.pkl", 'wb') as f:
            pickle.dump((self.documents, self.document_lookup), f)
        
        # Save metadata
        metadata = {
            'embedding_model': self.embedding_model_name,
            'dimension': self.dimension,
            'document_count': len(self.documents)
        }
        with open(f"{filepath_base}.meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved vector store to {filepath_base}")
    
    def load(self, filename: Optional[str] = None) -> bool:
        """
        Load the vector store from disk.
        
        Args:
            filename: Base filename to use (without extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if filename is None:
            filename = "vector_store"
        
        filepath_base = os.path.join(self.vector_db_path, filename)
        
        try:
            # Check if files exist
            if not (os.path.exists(f"{filepath_base}.faiss") and 
                    os.path.exists(f"{filepath_base}.documents.pkl")):
                logger.warning(f"Vector store files not found at {filepath_base}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath_base}.faiss")
            
            # Load documents and mapping
            with open(f"{filepath_base}.documents.pkl", 'rb') as f:
                self.documents, self.document_lookup = pickle.load(f)
            
            logger.info(f"Loaded vector store from {filepath_base} with {len(self.documents)} documents")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.document_lookup = {}
        logger.info("Vector store cleared")