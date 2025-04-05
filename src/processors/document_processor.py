import os
import json
import PyPDF2
import docx
from tqdm import tqdm
import logging
import re
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_end_re = re.compile(r'(?<=[.!?])\s+')

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Processing document: {file_path}")
            
            if file_path.lower().endswith('.pdf'):
                text = self._read_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                text = self._read_docx(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return []

            chunks = self._create_chunks_safe(text)
            return self._create_records(chunks, file_path)
            
        except Exception as e:
            logger.error(f"Critical error processing {file_path}: {str(e)}", exc_info=True)
            return []

    def _create_chunks_safe(self, text: str, timeout_sec: int = 30) -> List[str]:
        from multiprocessing import Process, Queue
        def _chunk_worker(q, text, chunk_size, chunk_overlap):
            try:
                q.put(self._create_chunks(text, chunk_size, chunk_overlap))
            except Exception as e:
                q.put(e)

        q = Queue()
        p = Process(target=_chunk_worker, 
                  args=(q, text, self.chunk_size, self.chunk_overlap))
        p.start()
        p.join(timeout=timeout_sec)
        
        if p.is_alive():
            p.terminate()
            p.join()
            logger.warning("Chunking timed out, using simple chunking")
            return self._simple_chunking(text)
            
        result = q.get()
        if isinstance(result, Exception):
            logger.warning(f"Chunking failed: {result}, using simple chunking")
            return self._simple_chunking(text)
        return result

    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                search_start = max(start, end - 100)
                match = self.sentence_end_re.search(text, search_start, end + 100)
                if match:
                    end = match.end()
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
                
            start = max(start + chunk_size - chunk_overlap, end - chunk_overlap)
            
        return chunks

    def _simple_chunking(self, text: str) -> List[str]:
        return [text[i:i+self.chunk_size].strip() 
               for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def _create_records(self, chunks: List[str], file_path: str) -> List[Dict[str, Any]]:
        return [{
            'content': chunk,
            'metadata': {
                'source': file_path,
                'chunk_id': i,
                'file_type': os.path.splitext(file_path)[1][1:],
                'total_chunks': len(chunks)
            }
        } for i, chunk in enumerate(chunks)]

    def process_directory(self, directory_path: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        all_chunks = []
        document_files = [
            os.path.join(directory_path, f) for f in os.listdir(directory_path)
            if f.lower().endswith(('.pdf', '.docx'))
        ]
        
        logger.info(f"Found {len(document_files)} document files to process")
        
        for file_path in tqdm(document_files, desc="Processing documents"):
            chunks = self.process_document(file_path)
            all_chunks.extend(chunks)
            
            file_name = os.path.basename(file_path)
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_chunks.json")
            
            with open(output_file, 'w') as f:
                json.dump(chunks, f, indent=2)

        all_chunks_file = os.path.join(output_dir, "all_document_chunks.json")
        with open(all_chunks_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        logger.info(f"Processed {len(all_chunks)} total chunks from {len(document_files)} documents")
    
    def _read_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
        
        return text
    
    def _read_docx(self, file_path: str) -> str:
        text = ""
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {str(e)}")
        
        return text   