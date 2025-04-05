import os
import argparse
import torch
import psutil
import gc
import json

from src.processors.document_processor import DocumentProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.video_processor import VideoProcessor
from src.database.vector_store import VectorStore
import config

def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("No GPU found, expect slower processing.")
        return False

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Memory usage after cleanup: CPU RAM {psutil.virtual_memory().percent}%")

def process_all_data(data_dir: str, output_dir: str, batch_size=8):
    print(f"Starting data processing in {data_dir}")
    documents_output = os.path.join(output_dir, "documents")
    images_output = os.path.join(output_dir, "images")
    videos_output = os.path.join(output_dir, "videos")

    os.makedirs(documents_output, exist_ok=True)
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(videos_output, exist_ok=True)

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print("GPU will be used where possible.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["USE_GPU"] = "1"
    else:
        print("Running on CPU.")

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

    print("Processing documents...")
    document_processor.process_directory(data_dir, documents_output)
    clear_memory()

    print("Processing images...")
    image_processor.process_directory(data_dir, images_output)
    clear_memory()

    print("Processing videos...")
    video_processor.process_directory(data_dir, videos_output)
    clear_memory()

    print("Data processing complete.")
    return {
        "documents": documents_output,
        "images": images_output,
        "videos": videos_output
    }

def create_vector_database(processed_paths, vector_db_path, batch_size=1024):
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

    try:
        documents_path = processed_paths["documents"]
        documents_file = os.path.join(documents_path, "all_document_chunks.json")
        if os.path.exists(documents_file):
            with open(documents_file, 'r') as f:
                document_chunks = json.load(f)
            print(f"Adding {len(document_chunks)} text segments to the knowledge base")
            vector_store.add_documents(document_chunks)

        images_path = processed_paths["images"]
        images_file = os.path.join(images_path, "all_image_data.json")
        if os.path.exists(images_file):
            with open(images_file, 'r') as f:
                image_data = json.load(f)
            print(f"Adding {len(image_data)} image descriptions to the knowledge base")
            vector_store.add_documents(image_data)

        videos_path = processed_paths["videos"]
        videos_file = os.path.join(videos_path, "all_video_data.json")
        if os.path.exists(videos_file):
            with open(videos_file, 'r') as f:
                video_data = json.load(f)
            total_videos = len(video_data)
            print(f"Adding information from {total_videos} videos")
            for i in range(0, total_videos, batch_size):
                batch = video_data[i:i+batch_size]
                vector_store.add_documents(batch)
                clear_memory()

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
                    print(f"Issue with video transcript: {e}")

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
        print(f"Failed to build knowledge base: {e}")
        raise

    vector_store.save()
    print(f"Knowledge base created and stored at {vector_db_path}")

def optimize_gpu_env():
    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("GPU environment optimized.")
        return True
    return False

def setup_colab():
    try:
        import subprocess
        try:
            subprocess.check_output(["which", "ffmpeg"])
            print("ffmpeg is already installed.")
        except subprocess.CalledProcessError:
            print("Installing ffmpeg...")
            subprocess.check_call(["apt-get", "update", "-qq"])
            subprocess.check_call(["apt-get", "install", "-y", "-qq", "ffmpeg"])
        print("Colab setup complete.")
        return True
    except Exception as e:
        print(f"Error during Colab setup: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process data and create vector database.")
    parser.add_argument("--data-dir", type=str, default="/content/data", help="Directory with raw data.")
    parser.add_argument("--output-dir", type=str, default="/content/processed_data", help="Directory for processed data.")
    parser.add_argument("--vector-db-path", type=str, default="/content/vector_db", help="Path to save vector database.")
    parser.add_argument("--skip-processing", action="store_true", help="Skip data processing if already done.")
    parser.add_argument("--batch-size", type=int, default=8, help="Processing batch size.")
    parser.add_argument("--embedding-batch-size", type=int, default=1024, help="Batch size for embeddings.")
    parser.add_argument("--colab-setup", action="store_true", help="Run Colab-specific setup.")

    args = parser.parse_args()

    is_colab_env = False
    try:
        import google.colab
        is_colab_env = True
        print("Running in Google Colab.")
        if args.colab_setup:
            setup_colab()
    except ImportError:
        pass

    has_gpu = check_gpu()
    if has_gpu:
        optimize_gpu_env()

    if not args.skip_processing:
        processed_paths = process_all_data(args.data_dir, args.output_dir, batch_size=args.batch_size)
    else:
        processed_paths = {
            "documents": os.path.join(args.output_dir, "documents"),
            "images": os.path.join(args.output_dir, "images"),
            "videos": os.path.join(args.output_dir, "videos")
        }
        print("Using existing processed data.")

    create_vector_database(processed_paths, args.vector_db_path, batch_size=args.embedding_batch_size)
    print("Data processing and knowledge base creation complete.")

if __name__ == "__main__":
    import sys
    is_colab_env = False
    try:
        import google.colab
        is_colab_env = True
    except ImportError:
        pass

    if is_colab_env and 'ipykernel' in sys.modules:
        print("\nRunning in Colab notebook. Using default arguments for notebook execution...")
        sys.argv = ['process_data.py', '--colab-setup']
        main()
    else:
        main()