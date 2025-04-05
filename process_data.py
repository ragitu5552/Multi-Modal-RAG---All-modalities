import os
import argparse

from src.processors.document_processor import DocumentProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.database.vector_store import VectorStore
import config

def process_all_data(data_dir: str, output_dir: str):
    print(f"Starting to prepare data in {data_dir}")
    documents_output = os.path.join(output_dir, "documents")
    images_output = os.path.join(output_dir, "images")
    videos_output = os.path.join(output_dir, "videos")

    os.makedirs(documents_output, exist_ok=True)
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(videos_output, exist_ok=True)

    document_processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    image_processor = ImageProcessor(
        ocr_engine=config.OCR_ENGINE,
        caption_model=True
    )

    video_processor = VideoProcessor(
        frame_sample_rate=config.VIDEO_FRAME_SAMPLE_RATE,
        audio_extraction_method=config.AUDIO_EXTRACTION_METHOD
    )

    print("Working on documents...")
    document_processor.process_directory(data_dir, documents_output)

    print("Working on images...")
    image_processor.process_directory(data_dir, images_output)

    print("Working on videos...")
    video_processor.process_directory(data_dir, videos_output)

    print("Data preparation done.")
    return {
        "documents": documents_output,
        "images": images_output,
        "videos": videos_output
    }

def create_vector_database(processed_paths, vector_db_path):
    print("Building the knowledge base...")
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

    documents_path = processed_paths["documents"]
    documents_file = os.path.join(documents_path, "all_document_chunks.json")

    if os.path.exists(documents_file):
        import json
        with open(documents_file, 'r') as f:
            document_chunks = json.load(f)
        print(f"Adding {len(document_chunks)} text segments to the knowledge base")
        vector_store.add_documents(document_chunks)

    images_path = processed_paths["images"]
    images_file = os.path.join(images_path, "all_image_data.json")

    if os.path.exists(images_file):
        import json
        with open(images_file, 'r') as f:
            image_data = json.load(f)
        print(f"Adding {len(image_data)} image descriptions to the knowledge base")
        vector_store.add_documents(image_data)

    videos_path = processed_paths["videos"]
    videos_file = os.path.join(videos_path, "all_video_data.json")

    if os.path.exists(videos_file):
        try:
            import json
            with open(videos_file, 'r') as f:
                video_data = json.load(f)

            total_videos = len(video_data)
            print(f"Adding information from {total_videos} videos")

            batch_size = 8
            for i in range(0, total_videos, batch_size):
                batch = video_data[i:i+batch_size]
                vector_store.add_documents(batch)

            transcript_segments = []
            for video in video_data:
                try:
                    for chunk in video.get('metadata', {}).get('transcript_chunks', []):
                        transcript_segments.append({
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
                    print(f"Issue with a video transcript: {e}")

            if transcript_segments:
                print(f"Adding {len(transcript_segments)} video transcript segments")
                vector_store.add_documents(transcript_segments)

            frame_descriptions = []
            for video in video_data:
                try:
                    for frame in video.get('metadata', {}).get('frame_data', []):
                        try:
                            if frame.get('content'):
                                frame_descriptions.append({
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
                            print(f"Issue processing a video frame: {e}")
                except Exception as e:
                    print(f"Issue with video frames: {e}")

            if frame_descriptions:
                print(f"Adding {len(frame_descriptions)} video frame descriptions")
                vector_store.add_documents(frame_descriptions)

        except Exception as e:
            print(f"Failed to process video data: {e}")
            raise

    vector_store.save()
    print(f"Knowledge base built and saved to {vector_db_path}")

def main():
    parser = argparse.ArgumentParser(description="Process data and create vector database.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing your data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save processed files.")
    parser.add_argument("--vector-db-path", type=str, required=True, help="Path to save the knowledge base.")
    parser.add_argument("--skip-processing", action="store_true", help="Skip data processing if output directory has content.")

    args = parser.parse_args()

    if not args.skip_processing:
        processed_paths = process_all_data(args.data_dir, args.output_dir)
    else:
        processed_paths = {
            "documents": os.path.join(args.output_dir, "documents"),
            "images": os.path.join(args.output_dir, "images"),
            "videos": os.path.join(args.output_dir, "videos")
        }
        print("Using existing processed data.")

    create_vector_database(processed_paths, args.vector_db_path)
    print("Data pipeline finished.")

if __name__ == "__main__":
    main()