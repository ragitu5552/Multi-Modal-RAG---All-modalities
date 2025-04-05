import os
import argparse
import logging
from tqdm import tqdm

# Import processors
from src.processors.document_processor import DocumentProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.database.vector_store import VectorStore

# Import config
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_all_data(data_dir: str, output_dir: str):
    """
    Process all data types from a directory.
    
    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save processed data
    """
    logger.info(f"Processing all data in {data_dir}")
    
    # Create output directories
    documents_dir = os.path.join(output_dir, "documents")
    images_dir = os.path.join(output_dir, "images")
    videos_dir = os.path.join(output_dir, "videos")
    
    os.makedirs(documents_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # Initialize processors
    document_processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    image_processor = ImageProcessor(
        ocr_engine=config.OCR_ENGINE,
        caption_model=True  # Use captioning if available
    )
    
    video_processor = VideoProcessor(
        frame_sample_rate=config.VIDEO_FRAME_SAMPLE_RATE,
        audio_extraction_method=config.AUDIO_EXTRACTION_METHOD
    )
    
    # Process each data type
    logger.info("Processing documents...")
    document_processor.process_directory(data_dir, documents_dir)
    
    logger.info("Processing images...")
    image_processor.process_directory(data_dir, images_dir)
    
    logger.info("Processing videos...")
    video_processor.process_directory(data_dir, videos_dir)
    
    logger.info("All data processed successfully")
    
    # Return paths to processed data
    return {
        "documents": documents_dir,
        "images": images_dir,
        "videos": videos_dir
    }

def create_vector_database(processed_paths, vector_db_path):
    """
    Create vector database from processed data.
    
    Args:
        processed_paths: Dictionary of paths to processed data
        vector_db_path: Path to save vector database
    """
    logger.info("Creating vector database...")
    
    try:
        vector_store = VectorStore(
            embedding_model=config.EMBEDDING_MODEL,
            vector_db_path=vector_db_path,
            dimension=config.EMBEDDING_DIMENSION
        )
    except TypeError:
        vector_store = VectorStore(
            embedding_model=config.EMBEDDING_MODEL,
            vector_db_path=vector_db_path
        )
    
    # Process document chunks
    documents_path = processed_paths["documents"]
    all_documents_file = os.path.join(documents_path, "all_document_chunks.json")
    
    if os.path.exists(all_documents_file):
        import json
        with open(all_documents_file, 'r') as f:
            document_chunks = json.load(f)
        
        logger.info(f"Adding {len(document_chunks)} document chunks to vector store")
        vector_store.add_documents(document_chunks)
    
    # Process image data
    images_path = processed_paths["images"]
    all_images_file = os.path.join(images_path, "all_image_data.json")
    
    if os.path.exists(all_images_file):
        import json
        with open(all_images_file, 'r') as f:
            image_data = json.load(f)
        
        logger.info(f"Adding {len(image_data)} image records to vector store")
        vector_store.add_documents(image_data)
    
    # Process video data
    videos_path = processed_paths["videos"]
    all_videos_file = os.path.join(videos_path, "all_video_data.json")
    
    if os.path.exists(all_videos_file):
        try:
            with open(all_videos_file, 'r') as f:
                video_data = json.load(f)
            
            # Process main video records
            total_videos = len(video_data)
            logger.info(f"Adding {total_videos} video records")
            
            for i in range(0, total_videos, batch_size):
                batch = video_data[i:i+batch_size]
                vector_store.add_documents(batch)
                clear_memory()
            
            # Process transcripts
            transcript_chunks = []
            for video in video_data:
                try:
                    if video.get('metadata', {}).get('transcript_chunks'):
                        for chunk in video['metadata']['transcript_chunks']:
                            transcript_chunks.append({
                                'content': chunk.get('text', ''),
                                'metadata': {
                                    'source': video['metadata'].get('source', 'unknown'),
                                    'file_type': 'video_transcript',
                                    'start_time': chunk.get('start', 0),
                                    'end_time': chunk.get('end', 0),
                                    'start_formatted': chunk.get('start_formatted', ''),
                                    'end_formatted': chunk.get('end_formatted', '')
                                }
                            })
                except Exception as e:
                    logger.error(f"Error processing video transcript: {e}")
                    continue
        
        if transcript_chunks:
            logger.info(f"Adding {len(transcript_chunks)} video transcript chunks to vector store")
            vector_store.add_documents(transcript_chunks)
        
        # Extract and add frame data
        frame_chunks = []
            for video in video_data:
                try:
                    if video.get('metadata', {}).get('frame_data'):
                        for frame in video['metadata']['frame_data']:
                            try:
                                if frame.get('content'):
                                    frame_chunks.append({
                                        'content': frame['content'],
                                        'metadata': {
                                            'source': video['metadata'].get('source', 'unknown'),
                                            'file_type': 'video_frame',
                                            'timestamp': frame.get('timestamp', 0),
                                            'timestamp_formatted': frame.get('timestamp_formatted', ''),
                                            'frame_number': frame.get('frame_number', '')
                                        }
                                    })
                            except Exception as e:
                                logger.error(f"Error processing frame: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Error processing video frames: {e}")
                    continue
        
        if frame_chunks:
            logger.info(f"Adding {len(frame_chunks)} video frame chunks to vector store")
            vector_store.add_documents(frame_chunks)
    
    # Save the vector database
    vector_store.save()
    logger.info(f"Vector database created and saved to {vector_db_path}")
except Exception as e:
            logger.error(f"Failed to create vector database: {e}")
            raise

# def create_vector_database(processed_paths, vector_db_path):
#     """
#     Create vector database from processed data.
    
#     Args:
#         processed_paths: Dictionary of paths to processed data
#         vector_db_path: Path to save vector database
#     """
#     logger.info("Creating vector database...")
    
#     # Initialize vector store
#     vector_store = VectorStore(
#         embedding_model=config.EMBEDDING_MODEL,
#         vector_db_path=vector_db_path,
#         dimension=config.EMBEDDING_DIMENSION
#     )
    
#     # Process document chunks
#     documents_path = processed_paths["documents"]
#     all_documents_file = os.path.join(documents_path, "all_document_chunks.json")
    
#     if os.path.exists(all_documents_file):
#         import json
#         with open(all_documents_file, 'r') as f:
#             document_chunks = json.load(f)
        
#         logger.info(f"Adding {len(document_chunks)} document chunks to vector store")
#         vector_store.add_documents(document_chunks)
    
#     # Process image data
#     images_path = processed_paths["images"]
#     all_images_file = os.path.join(images_path, "all_image_data.json")
    
#     if os.path.exists(all_images_file):
#         import json
#         with open(all_images_file, 'r') as f:
#             image_data = json.load(f)
        
#         logger.info(f"Adding {len(image_data)} image records to vector store")
#         vector_store.add_documents(image_data)
    
#     # Process video data
#     videos_path = processed_paths["videos"]
#     all_videos_file = os.path.join(videos_path, "all_video_data.json")
    
#     if os.path.exists(all_videos_file):
#         import json
#         with open(all_videos_file, 'r') as f:
#             video_data = json.load(f)
        
#         # Add main video records
#         logger.info(f"Adding {len(video_data)} video records to vector store")
#         vector_store.add_documents(video_data)
        
#         # Extract and add transcript chunks
#         transcript_chunks = []
#         for video in video_data:
#             if 'metadata' in video and 'transcript_chunks' in video['metadata']:
#                 for chunk in video['metadata']['transcript_chunks']:
#                     transcript_chunks.append({
#                         'content': chunk['text'],
#                         'metadata': {
#                             'source': video['metadata']['source'],
#                             'file_type': 'video_transcript',
#                             'start_time': chunk['start'],
#                             'end_time': chunk['end'],
#                             'start_formatted': chunk.get('start_formatted', ''),
#                             'end_formatted': chunk.get('end_formatted', '')
#                         }
#                     })
        
#         if transcript_chunks:
#             logger.info(f"Adding {len(transcript_chunks)} video transcript chunks to vector store")
#             vector_store.add_documents(transcript_chunks)
        
#         # Extract and add frame data
#         frame_chunks = []
#         for video in video_data:
#             if 'metadata' in video and 'frame_data' in video['metadata']:
#                 for frame in video['metadata']['frame_data']:
#                     if 'content' in frame:
#                         frame_chunks.append({
#                             'content': frame['content'],
#                             'metadata': {
#                                 'source': video['metadata']['source'],
#                                 'file_type': 'video_frame',
#                                 'timestamp': frame['timestamp'],
#                                 'timestamp_formatted': frame.get('timestamp_formatted', ''),
#                                 'frame_number': frame.get('frame_number', '')
#                             }
#                         })
        
#         if frame_chunks:
#             logger.info(f"Adding {len(frame_chunks)} video frame chunks to vector store")
#             vector_store.add_documents(frame_chunks)
    
#     # Save the vector database
#     vector_store.save()
#     logger.info(f"Vector database created and saved to {vector_db_path}")


def main():
    """Main function to process data and initialize vector database."""
    parser = argparse.ArgumentParser(description="Process data and create vector database.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing raw data files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--vector-db-path",
        type=str,
        required=True,
        help="Path to save the vector database"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip data processing and only create vector database from existing processed data"
    )
    
    args = parser.parse_args()
    
    if not args.skip_processing:
        # Process all data
        processed_paths = process_all_data(args.data_dir, args.output_dir)
    else:
        # Use existing processed data
        processed_paths = {
            "documents": os.path.join(args.output_dir, "documents"),
            "images": os.path.join(args.output_dir, "images"),
            "videos": os.path.join(args.output_dir, "videos")
        }
        logger.info("Skipping data processing, using existing processed data")
    
    # Create vector database
    create_vector_database(processed_paths, args.vector_db_path)
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()