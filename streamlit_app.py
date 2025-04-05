import streamlit as st
import os
import uuid
import logging
from typing import Dict, Any
import sys
import json
import logging
import pickle
from typing import List, Dict, Any, Optional, Union
import numpy as np
from groq import Groq
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config

# # Import necessary modules
# from ..database.vector_store import VectorStore
# from ..retrieval.retriever import MultiModalRetriever
# from ..llm.groq_integration import GroqLLM

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

class MultiModalRetriever:
    """
    Retrieves relevant content from the vector store based on user queries.
    """
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            vector_store: The vector store to search
            top_k: Number of results to return
        """
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The user query
            
        Returns:
            List of document dictionaries with similarity scores
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        # Search vector store
        results = self.vector_store.search(query, top_k=self.top_k)
        
        # Log results summary
        sources = set()
        for result in results:
            if 'metadata' in result and 'source' in result['metadata']:
                sources.add(os.path.basename(result['metadata']['source']))
        
        logger.info(f"Retrieved {len(results)} documents from {len(sources)} sources")
        
        return results
    
    def format_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieval results for LLM consumption.
        
        Args:
            results: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        
        for i, result in enumerate(results):
            # Extract source info
            source = "Unknown source"
            file_type = "unknown"
            
            if 'metadata' in result:
                meta = result['metadata']
                if 'source' in meta:
                    source = os.path.basename(meta['source'])
                if 'file_type' in meta:
                    file_type = meta['file_type']
            
            # Format based on content type
            content = result['content']
            context_part = f"[{i+1}] From {file_type} {source}:\n{content}\n\n"
            
            # Add additional info for videos
            if 'metadata' in result and file_type == 'video':
                if 'timestamp_formatted' in result['metadata']:
                    context_part += f"Timestamp: {result['metadata']['timestamp_formatted']}\n"
            
            context_parts.append(context_part)
        
        # Combine all results
        return "".join(context_parts)

class GroqLLM:
    """
    Integration with Groq's LLM API.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.1,
                 max_tokens: int = 1024):
        """
        Initialize the Groq LLM client.
        
        Args:
            api_key: Groq API key (if None, will try to get from environment)
            model: Model to use for generation
            temperature: Temperature for generation (0-1)
            max_tokens: Maximum number of tokens to generate
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            logger.warning("Groq API key not found. Please set GROQ_API_KEY environment variable.")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        logger.info(f"Initialized Groq LLM with model: {model}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt for the LLM
            
        Returns:
            Generated text
        """
        try:
            logger.info(f"Generating response with model: {self.model}")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract generated text
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                return generated_text
            else:
                logger.error("No completion choices returned from Groq API")
                return "Error: No response generated"
            
        except Exception as e:
            logger.error(f"Error generating text with Groq: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_rag_response(self, query: str, context: str) -> str:
        """
        Generate a response based on retrieved context.
        
        Args:
            query: User query
            context: Retrieved context from documents
            
        Returns:
            Generated response
        """
        # Create prompt with retrieved context
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context.
        
## Context:
{context}

## Question:
{query}

Please answer the question based solely on the information provided in the context above. If the context doesn't contain enough information to provide a complete answer, acknowledge what you don't know. Be concise and directly address the query.

## Answer:
"""
        
        # Generate response
        return self.generate(prompt)
    





# Import local config
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalRAGApp:
    """
    Streamlit-based UI for Multi-Modal RAG System.
    """
    
    def __init__(self):
        """Initialize the RAG application."""
        self.setup_session_state()
        self.initialize_components()
        self.render_ui()
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'show_sources' not in st.session_state:
            st.session_state.show_sources = False
        
        if 'last_sources' not in st.session_state:
            st.session_state.last_sources = []
    
    def initialize_components(self):
        """Initialize RAG system components."""
        try:
            # Initialize vector store
            self.vector_store = VectorStore(
                embedding_model=config.EMBEDDING_MODEL,
                vector_db_path=config.VECTOR_DB_PATH,
                dimension=config.EMBEDDING_DIMENSION
            )
            
            # Try to load existing vector store
            vector_store_loaded = self.vector_store.load()
            if not vector_store_loaded:
                st.error("No existing vector store found. Please process your data first.")
            
            # Initialize retriever
            self.retriever = MultiModalRetriever(
                vector_store=self.vector_store,
                top_k=config.TOP_K_RESULTS
            )
            
            # Initialize LLM
            self.llm = GroqLLM(
                api_key=config.GROQ_API_KEY,
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
    
    def render_ui(self):
        """Render the Streamlit UI."""
        st.title(config.STREAMLIT_TITLE)
        st.write(config.STREAMLIT_DESCRIPTION)
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Display source toggle
        st.sidebar.title("Options")
        show_sources = st.sidebar.checkbox("Show Sources", value=st.session_state.show_sources)
        st.session_state.show_sources = show_sources
        
        # Show sources if enabled
        if st.session_state.show_sources and st.session_state.last_sources:
            st.sidebar.title("Sources")
            for i, source in enumerate(st.session_state.last_sources):
                source_name = "Unknown"
                if 'metadata' in source and 'source' in source['metadata']:
                    source_name = os.path.basename(source['metadata']['source'])
                
                with st.sidebar.expander(f"Source {i+1}: {source_name}"):
                    st.write(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
        
        # Clear chat button
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Input for user query
        if query := st.chat_input("Ask something about your data"):
            self.process_query(query)
    
    def process_query(self, query: str):
        """
        Process user query and generate response.
        
        Args:
            query: User query
        """
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)
        
        # Show assistant "thinking" message
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.write("Thinking...")
            
            try:
                # Retrieve relevant documents
                results = self.retriever.retrieve(query)
                st.session_state.last_sources = results
                
                # Format context for LLM
                context = self.retriever.format_for_llm(results)
                
                # Generate response
                response = self.llm.generate_rag_response(query, context)
                
                # Update placeholder with response
                thinking_placeholder.write(response)
                
                # Add assistant message to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                thinking_placeholder.write(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    app = MultiModalRAGApp()

if __name__ == "__main__":
    main()