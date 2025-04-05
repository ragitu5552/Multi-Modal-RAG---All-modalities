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


from src.database.vector_store import VectorStore
from src.retrieval.retriever import MultiModalRetriever
from src.llm.groq_integration import GroqLLM

class MultiModalRAGApp:
    def __init__(self):
        self._setup_session()
        self._initialize_rag_components()
        self._build_ui()

    def _setup_session(self):
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'show_sources' not in st.session_state:
            st.session_state.show_sources = False
        if 'last_sources' not in st.session_state:
            st.session_state.last_sources = []

    def _initialize_rag_components(self):
        try:
            self.vector_store = VectorStore(
                embedding_model=config.EMBEDDING_MODEL,
                vector_db_path=config.VECTOR_DB_PATH,
                dimension=config.EMBEDDING_DIMENSION
            )
            vector_store_loaded = self.vector_store.load()
            if not vector_store_loaded:
                st.error("No existing knowledge base found. Please prepare your data first.")

            self.retriever = MultiModalRetriever(
                vector_store=self.vector_store,
                top_k=config.TOP_K_RESULTS
            )

            self.llm = GroqLLM(
                api_key=config.GROQ_API_KEY,
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )

        except Exception as e:
            st.error(f"Failed to start: {str(e)}")

    def _build_ui(self):
        st.title(config.STREAMLIT_TITLE)
        st.write(config.STREAMLIT_DESCRIPTION)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        with st.sidebar:
            st.title("Options")
            st.session_state.show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
            if st.session_state.show_sources and st.session_state.last_sources:
                st.title("Sources")
                for i, source in enumerate(st.session_state.last_sources):
                    source_name = os.path.basename(source['metadata']['source']) if 'metadata' in source and 'source' in source['metadata'] else "Unknown"
                    with st.expander(f"Source {i+1}: {source_name}"):
                        st.write(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])

            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        if query := st.chat_input("Ask me anything about your data"):
            self._handle_query(query)

    def _handle_query(self, query: str):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.write("Thinking...")
            try:
                results = self.retriever.retrieve(query)
                st.session_state.last_sources = results
                context = self.retriever.format_for_llm(results)
                response = self.llm.generate_rag_response(query, context)
                thinking_placeholder.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Something went wrong: {str(e)}"
                thinking_placeholder.write(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    MultiModalRAGApp()

if __name__ == "__main__":
    main()