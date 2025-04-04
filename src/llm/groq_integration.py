import os
import logging
from typing import List, Dict, Any, Optional
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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