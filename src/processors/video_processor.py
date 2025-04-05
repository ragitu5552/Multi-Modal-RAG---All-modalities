import os
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from PIL import Image
import whisper
from datetime import timedelta

from .image_processor import ImageProcessor

class VideoProcessor:
    def __init__(self,
                 frame_sample_rate: int = 5,
                 audio_extraction_method: str = "whisper",
                 whisper_model: str = "base"):
        self.frame_sample_rate = frame_sample_rate
        self.audio_extraction_method = audio_extraction_method
        self.image_processor = ImageProcessor(ocr_engine="easyocr", caption_model=True)
        if audio_extraction_method == "whisper":
            self.whisper_model = whisper.load_model(whisper_model)
        else:
            self.whisper_model = None

    def process_video(self, file_path: str, output_frames_dir: Optional[str] = None) -> Dict[str, Any]:
        try:
            video = VideoFileClip(file_path)
            duration = video.duration
            fps = video.fps
            if output_frames_dir:
                os.makedirs(output_frames_dir, exist_ok=True)
            frames_data = self._extract_and_process_frames(video, file_path, output_frames_dir)
            transcript, segments = self._transcribe_audio(video, file_path)
            transcript_chunks = self._create_transcript_chunks(segments, chunk_duration=60)
            record = {
                'content': f"Video: {os.path.basename(file_path)}\n"
                           f"Duration: {timedelta(seconds=int(duration))}\n"
                           f"Transcript: {transcript[:1000]}...",
                'metadata': {
                    'source': file_path,
                    'file_type': 'video',
                    'duration': duration,
                    'fps': fps,
                    'width': video.w,
                    'height': video.h,
                    'transcript': transcript,
                    'transcript_chunks': transcript_chunks,
                    'frame_data': frames_data
                }
            }
            video.close()
            return record
        except Exception as e:
            return {
                'content': f"Error processing video: {os.path.basename(file_path)}",
                'metadata': {
                    'source': file_path,
                    'error': str(e)
                }
            }

    def process_directory(self, directory_path: str, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        video_files = [
            os.path.join(directory_path, f) for f in os.listdir(directory_path)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        ]
        all_records = []
        for file_path in tqdm(video_files, desc="Processing videos"):
            file_name = os.path.basename(file_path)
            video_frames_dir = os.path.join(frames_dir, os.path.splitext(file_name)[0])
            record = self.process_video(file_path, video_frames_dir)
            all_records.append(record)
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_data.json")
            with open(output_file, 'w') as f:
                json.dump(record, f, indent=2)
        all_videos_file = os.path.join(output_dir, "all_video_data.json")
        with open(all_videos_file, 'w') as f:
            json.dump(all_records, f, indent=2)

    def _extract_and_process_frames(self,
                                     video: VideoFileClip,
                                     file_path: str,
                                     output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        frames_data = []
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        num_frames = int(video.duration / self.frame_sample_rate) + 1
        for i in tqdm(range(num_frames), desc=f"Extracting frames from {base_name}"):
            time_pos = i * self.frame_sample_rate
            if time_pos >= video.duration:
                continue
            frame = video.get_frame(time_pos)
            pil_frame = Image.fromarray(frame)
            frame_filename = None
            if output_dir:
                frame_filename = os.path.join(output_dir, f"frame_{i:04d}_{time_pos:.1f}s.jpg")
                pil_frame.save(frame_filename)
            if not frame_filename:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    frame_filename = temp_file.name
                    pil_frame.save(frame_filename)
            frame_data = self.image_processor.process_image(frame_filename)
            frame_data['metadata']['timestamp'] = time_pos
            frame_data['metadata']['timestamp_formatted'] = str(timedelta(seconds=time_pos))
            frames_data.append(frame_data)
            if not output_dir and os.path.exists(frame_filename):
                os.unlink(frame_filename)
        return frames_data

    def _transcribe_audio(self,
                          video: VideoFileClip,
                          file_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        if self.audio_extraction_method != "whisper" or not self.whisper_model:
            return "No transcription available", []
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                audio_file = temp_audio.name
            video.audio.write_audiofile(audio_file,
                                        codec='pcm_s16le',
                                        logger=None)
            result = self.whisper_model.transcribe(audio_file)
            if os.path.exists(audio_file):
                os.unlink(audio_file)
            return result["text"], result["segments"]
        except Exception as e:
            return "Transcription failed", []

    def _create_transcript_chunks(self,
                                  segments: List[Dict[str, Any]],
                                  chunk_duration: int = 60) -> List[Dict[str, Any]]:
        if not segments:
            return []
        chunks = []
        current_chunk = {
            'start': segments[0]['start'],
            'end': segments[0]['end'],
            'text': segments[0]['text']
        }
        for segment in segments[1:]:
            if segment['start'] - current_chunk['start'] > chunk_duration:
                chunks.append(current_chunk)
                current_chunk = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text']
                }
            else:
                current_chunk['end'] = segment['end']
                current_chunk['text'] += " " + segment['text']
        if current_chunk:
            chunks.append(current_chunk)
        for chunk in chunks:
            chunk['start_formatted'] = str(timedelta(seconds=chunk['start']))
            chunk['end_formatted'] = str(timedelta(seconds=chunk['end']))
        return chunks