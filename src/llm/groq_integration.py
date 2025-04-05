import os
from typing import List, Dict, Any, Optional
from groq import Groq

class GroqLLM:
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 model: str = "llama3-70b-8192",
                 temperature: float = 0.1,
                 max_tokens: int = 1024):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = Groq(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            if response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content
                return generated_text
            else:
                return "Error: No response generated"
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_rag_response(self, query: str, context: str) -> str:
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