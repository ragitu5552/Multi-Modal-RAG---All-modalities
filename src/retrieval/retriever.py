import json
import logging
from typing import List, Dict, Any, Optional, Union
import os

from ..database.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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