import os
import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import easyocr
import pytesseract
from tqdm import tqdm

# Try to import image captioning (optional)
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    CAPTION_MODEL_AVAILABLE = True
except ImportError:
    CAPTION_MODEL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Processes image files to extract text via OCR and generate captions.
    """
    
    def __init__(self, ocr_engine: str = "easyocr", caption_model: bool = False):
        """
        Initialize the image processor.
        
        Args:
            ocr_engine: OCR engine to use ('easyocr' or 'pytesseract')
            caption_model: Whether to use image captioning
        """
        self.ocr_engine = ocr_engine
        
        # Initialize OCR engine
        if ocr_engine == "easyocr":
            logger.info("Initializing EasyOCR")
            self.reader = easyocr.Reader(['en'])
        
        # Initialize captioning model if requested and available
        self.caption_model = None
        self.caption_processor = None
        
        if caption_model and CAPTION_MODEL_AVAILABLE:
            logger.info("Initializing image captioning model")
            try:
                model_name = "Salesforce/blip-image-captioning-base"
                self.caption_processor = AutoProcessor.from_pretrained(model_name)
                self.caption_model = AutoModelForCausalLM.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"Error loading captioning model: {str(e)}")
    
    def process_image(self, file_path: str) -> Dict[str, Any]:
        """
        Process an image file to extract text via OCR and generate caption.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text, caption, and metadata
        """
        logger.info(f"Processing image: {file_path}")
        
        try:
            # Extract OCR text
            ocr_text = self._extract_text(file_path)
            
            # Generate caption if model is available
            caption = self._generate_caption(file_path) if self.caption_model else "No caption available"
            
            # Get basic image metadata
            img = Image.open(file_path)
            width, height = img.size
            mode = img.mode
            format_name = img.format
            
            # Create image record with metadata
            record = {
                'content': f"Image: {os.path.basename(file_path)}\n"
                           f"Description: {caption}\n"
                           f"Extracted Text: {ocr_text}",
                'metadata': {
                    'source': file_path,
                    'file_type': 'image',
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'format': format_name,
                    'ocr_text': ocr_text,
                    'caption': caption
                }
            }
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            return {
                'content': f"Error processing image: {os.path.basename(file_path)}",
                'metadata': {
                    'source': file_path,
                    'error': str(e)
                }
            }
    
    def process_directory(self, directory_path: str, output_dir: str) -> None:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to the directory containing images
            output_dir: Path to save processed image data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [
            os.path.join(directory_path, f) for f in os.listdir(directory_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        all_records = []
        
        for file_path in tqdm(image_files, desc="Processing images"):
            record = self.process_image(file_path)
            all_records.append(record)
            
            # Save individual file record
            file_name = os.path.basename(file_path)
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_data.json")
            
            with open(output_file, 'w') as f:
                json.dump(record, f, indent=2)
        
        # Save all records to a single file
        all_images_file = os.path.join(output_dir, "all_image_data.json")
        with open(all_images_file, 'w') as f:
            json.dump(all_records, f, indent=2)
        
        logger.info(f"Processed {len(all_records)} images")
    
    def _extract_text(self, image_path: str) -> str:
        """Extract text from an image using the configured OCR engine."""
        if self.ocr_engine == "easyocr":
            try:
                results = self.reader.readtext(image_path)
                return "\n".join([text for _, text, _ in results])
            except Exception as e:
                logger.error(f"EasyOCR error: {str(e)}")
                return ""
        elif self.ocr_engine == "pytesseract":
            try:
                return pytesseract.image_to_string(Image.open(image_path))
            except Exception as e:
                logger.error(f"Pytesseract error: {str(e)}")
                return ""
        else:
            logger.warning(f"Unsupported OCR engine: {self.ocr_engine}")
            return ""
    
    def _generate_caption(self, image_path: str) -> str:
        """Generate a caption for the image using a pre-trained model."""
        if not self.caption_model or not self.caption_processor:
            return "No caption model available"
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.caption_processor(images=image, return_tensors="pt")
            
            output = self.caption_model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                top_p=0.9
            )
            
            caption = self.caption_processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            return "Caption generation failed"