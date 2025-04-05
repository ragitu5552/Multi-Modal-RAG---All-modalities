import os
import argparse
import time
from typing import Dict, List, Any, Optional

from src.database.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.groq_integration import GroqLLM
import config

class MultiModalRAG:
    def __init__(
        self,
        vector_db_path: str,
        embedding_model: str = config.EMBEDDING_MODEL,
        k_documents: int = 5,
        llm_model: str = config.LLM_MODEL
    ):
        self.vector_store = VectorStore(
            embedding_model=embedding_model,
            vector_db_path=vector_db_path,
            dimension=config.EMBEDDING_DIMENSION
        )
        self.vector_store.load()
        self.retriever = Retriever(
            vector_store=self.vector_store,
            k=k_documents
        )
        self.llm = GroqLLM(model_name=llm_model)

    def query(self, query_text: str) -> Dict[str, Any]:
        start_time = time.time()
        retrieved_docs = self.retriever.retrieve(query_text)
        retrieval_time = time.time() - start_time
        documents = []
        images = []
        video_frames = []
        video_transcripts = []

        for doc in retrieved_docs:
            doc_type = doc['metadata'].get('file_type', 'document')
            if doc_type in ('image', 'image_caption'):
                images.append(doc)
            elif doc_type == 'video_frame':
                video_frames.append(doc)
            elif doc_type == 'video_transcript':
                video_transcripts.append(doc)
            else:
                documents.append(doc)

        context = self._create_context(documents, images, video_frames, video_transcripts)
        prompt = self._create_prompt(query_text, context)
        response = self.llm.generate(prompt)
        total_time = time.time() - start_time

        return {
            "query": query_text,
            "response": response,
            "retrieved_context": retrieved_docs,
            "processing_time": total_time
        }

    def _create_context(
        self,
        documents: List[Dict],
        images: List[Dict],
        video_frames: List[Dict],
        video_transcripts: List[Dict]
    ) -> str:
        context_parts = []

        if documents:
            context_parts.append("### Document Content")
            for i, doc in enumerate(documents):
                source = doc['metadata'].get('source', 'Unknown')
                context_parts.append(f"[Document {i+1}] From {source}:")
                context_parts.append(doc['content'].strip())
                context_parts.append("")

        if images:
            context_parts.append("### Image Content")
            for i, img in enumerate(images):
                source = img['metadata'].get('source', 'Unknown')
                context_parts.append(f"[Image {i+1}] From {source}:")
                context_parts.append(img['content'].strip())
                context_parts.append("")

        if video_frames:
            context_parts.append("### Video Frame Content")
            for i, frame in enumerate(video_frames):
                source = frame['metadata'].get('source', 'Unknown')
                timestamp = frame['metadata'].get('timestamp_formatted', 'Unknown time')
                context_parts.append(f"[Video Frame {i+1}] From {source} at {timestamp}:")
                context_parts.append(frame['content'].strip())
                context_parts.append("")

        if video_transcripts:
            context_parts.append("### Video Transcript Content")
            for i, transcript in enumerate(video_transcripts):
                source = transcript['metadata'].get('source', 'Unknown')
                start_time = transcript['metadata'].get('start_formatted', 'Unknown time')
                end_time = transcript['metadata'].get('end_formatted', 'Unknown time')
                context_parts.append(f"[Video Transcript {i+1}] From {source} ({start_time} to {end_time}):")
                context_parts.append(transcript['content'].strip())
                context_parts.append("")

        return "\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful AI assistant for a multi-modal knowledge base.
Answer the user's question based on the provided context from various sources including documents, images, and videos.
If the information to answer the question is not contained in the context, acknowledge this limitation.
Use only the information from the context to answer the question.

Context:
{context}

User Question: {query}

Answer:"""

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal RAG System")
    parser.add_argument(
        "--vector-db-path",
        type=str,
        default=config.VECTOR_DB_PATH,
        help="Path to the vector database"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (CLI)"
    )

    args = parser.parse_args()

    rag_system = MultiModalRAG(
        vector_db_path=args.vector_db_path,
        embedding_model=config.EMBEDDING_MODEL,
        k_documents=config.NUM_RETRIEVAL_RESULTS,
        llm_model=config.LLM_MODEL
    )

    if args.interactive:
        print("Multi-Modal RAG System - Interactive Mode")
        print("Type 'exit' to quit")

        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'exit':
                break

            try:
                result = rag_system.query(query)
                print("\nResponse:")
                print(result["response"])
                print(f"\nRetrieved {len(result['retrieved_context'])} documents in {result['processing_time']:.2f} seconds")
            except Exception as e:
                print(f"An error occurred: {e}")
    else:
        print("RAG system initialized and ready for API calls")

if __name__ == "__main__":
    main()