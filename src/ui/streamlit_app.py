import streamlit as st
import os
import uuid
import logging
from typing import Dict, Any

# Import necessary modules
from ..database.vector_store import VectorStore
from ..retrieval.retriever import MultiModalRetriever
from ..llm.groq_integration import GroqLLM

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