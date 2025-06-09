import easyocr
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Union, Optional, Dict
import logging

# PDF processing imports
try:
    from pdf2image import convert_from_bytes, convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

class EasyOCRService:
    """
    EasyOCR wrapper class with configurable languages and byte file support
    """
    
    def __init__(self, 
                 languages: List[str] = ['en', 'th'], 
                 gpu: bool = False,
                 model_storage_directory: Optional[str] = None,
                 download_enabled: bool = True,
                 detector: bool = True,
                 recognizer: bool = True,
                 verbose: bool = True,
                 quantize: bool = True,
                 cudnn_benchmark: bool = False):
        """
        Initialize EasyOCR with specified languages and settings
        
        Args:
            languages: List of language codes (e.g., ['en', 'th', 'ja'])
            gpu: Use GPU acceleration if available
            model_storage_directory: Directory to store models
            download_enabled: Allow downloading models
            detector: Use text detector
            recognizer: Use text recognizer
            verbose: Print progress messages
            quantize: Use quantized models for faster inference
            cudnn_benchmark: Enable cudnn benchmark for consistent input sizes
        """
        self.languages = languages
        self.gpu = gpu
        self.verbose = verbose
        
        try:
            self.reader = easyocr.Reader(
                lang_list=languages,
                gpu=gpu,
                model_storage_directory=model_storage_directory,
                download_enabled=download_enabled,
                detector=detector,
                recognizer=recognizer,
                verbose=verbose,
                quantize=quantize,
                cudnn_benchmark=cudnn_benchmark
            )
            if self.verbose:
                print(f"✅ EasyOCR initialized with languages: {languages}")
                
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def read_from_bytes(self, 
                       image_bytes: bytes,
                       paragraph: bool = False,
                       width_ths: float = 0.7,
                       height_ths: float = 0.7,
                       decoder: str = 'greedy',
                       beamWidth: int = 5,
                       batch_size: int = 1,
                       workers: int = 0,
                       allowlist: Optional[str] = None,
                       blocklist: Optional[str] = None,
                       detail: int = 1,
                       rotation_info: Optional[List[int]] = None,
                       x_ths: float = 1.0,
                       y_ths: float = 0.5,
                       slope_ths: float = 0.1,
                       ycenter_ths: float = 0.5,
                       mag_ratio: float = 1.0,
                       text_threshold: float = 0.7,
                       low_text: float = 0.4,
                       link_threshold: float = 0.4,
                       canvas_size: int = 2560,
                       adjust_contrast: float = 0.2,
                       filter_ths: float = 0.003,
                       preprocessing: bool = True) -> List[Tuple]:
        """
        Read text from image bytes
        
        Args:
            image_bytes: Image data as bytes
            paragraph: Combine results into paragraph
            width_ths: Width threshold for merging boxes
            height_ths: Height threshold for merging boxes
            decoder: Text decoder ('greedy' or 'beamsearch')
            beamWidth: Beam width for beam search decoder
            batch_size: Batch size for processing
            workers: Number of workers for processing
            allowlist: Characters to allow
            blocklist: Characters to block
            detail: Level of detail in output (0, 1, 2)
            rotation_info: Rotation angles to try
            preprocessing: Apply image preprocessing
            ... (other EasyOCR parameters)
            
        Returns:
            List of tuples: (bbox, text, confidence) or text only if detail=0
        """
        try:
            # Convert bytes to numpy array
            image_array = self._bytes_to_numpy(image_bytes)
            
            # Apply preprocessing if enabled
            if preprocessing:
                image_array = self._preprocess_image(image_array)
            
            # Perform OCR
            results = self.reader.readtext(
                image_array,
                paragraph=paragraph,
                width_ths=width_ths,
                height_ths=height_ths,
                decoder=decoder,
                beamWidth=beamWidth,
                batch_size=batch_size,
                workers=workers,
                allowlist=allowlist,
                blocklist=blocklist,
                detail=detail,
                rotation_info=rotation_info,
                x_ths=x_ths,
                y_ths=y_ths,
                slope_ths=slope_ths,
                ycenter_ths=ycenter_ths,
                mag_ratio=mag_ratio,
                text_threshold=text_threshold,
                low_text=low_text,
                link_threshold=link_threshold,
                canvas_size=canvas_size,
                adjust_contrast=adjust_contrast,
                filter_ths=filter_ths
            )
            
            if self.verbose:
                print(f"✅ OCR completed. Found {len(results)} text regions.")
                
            return results
            
        except Exception as e:
            logging.error(f"OCR processing failed: {e}")
            raise
    
    def read_from_file(self, file_path: str, **kwargs) -> List[Tuple]:
        """
        Read text from image file
        
        Args:
            file_path: Path to image file
            **kwargs: Additional parameters for read_from_bytes
            
        Returns:
            OCR results
        """
        try:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            return self.read_from_bytes(image_bytes, **kwargs)
            
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def extract_text_only(self, image_bytes: bytes, confidence_threshold: float = 0.5) -> str:
        """
        Extract only text content from image bytes
        
        Args:
            image_bytes: Image data as bytes
            confidence_threshold: Minimum confidence to include text
            
        Returns:
            Extracted text as string
        """
        results = self.read_from_bytes(image_bytes, detail=1)
        
        text_parts = []
        for result in results:
            if len(result) >= 3:  # (bbox, text, confidence)
                _, text, confidence = result[0], result[1], result[2]
                if confidence >= confidence_threshold:
                    text_parts.append(text)
            elif len(result) == 2:  # (text, confidence)
                text, confidence = result
                if confidence >= confidence_threshold:
                    text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def get_text_with_positions(self, image_bytes: bytes) -> List[dict]:
        """
        Get text with bounding box positions
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of dictionaries with text, confidence, and bbox info
        """
        results = self.read_from_bytes(image_bytes, detail=1)
        
        formatted_results = []
        for result in results:
            if len(result) >= 3:
                bbox, text, confidence = result
                formatted_results.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'top_left': bbox[0],
                    'top_right': bbox[1],
                    'bottom_right': bbox[2],
                    'bottom_left': bbox[3]
                })
        
        return formatted_results
    
    def change_languages(self, new_languages: List[str]):
        """
        Change OCR languages (reinitializes the reader)
        
        Args:
            new_languages: List of new language codes
        """
        if new_languages != self.languages:
            self.languages = new_languages
            self.reader = easyocr.Reader(
                lang_list=new_languages,
                gpu=self.gpu,
                verbose=self.verbose
            )
            if self.verbose:
                print(f"🔄 Languages changed to: {new_languages}")
    
    def _bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array
        """
        try:
            # Try with PIL first
            image_pil = Image.open(BytesIO(image_bytes))
            image_array = np.array(image_pil)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array
            
        except Exception as e:
            # Fallback to OpenCV
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_array is None:
                    raise ValueError("Could not decode image")
                return image_array
                
            except Exception as e2:
                logging.error(f"Failed to convert bytes to numpy array: {e2}")
                raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply basic image preprocessing
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Noise removal
            denoised = cv2.medianBlur(gray, 3)
            
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            return enhanced
            
        except Exception as e:
            logging.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def read_pdf_from_bytes(self,
                           pdf_bytes: bytes,
                           dpi: int = 300,
                           first_page: Optional[int] = None,
                           last_page: Optional[int] = None,
                           confidence_threshold: float = 0.5,
                           thread_count: int = 1,
                           **ocr_kwargs) -> List[Dict]:
        """
        Read text from PDF bytes using pdf2image conversion
        
        Args:
            pdf_bytes: PDF file as bytes
            dpi: DPI for image conversion (higher = better quality but slower)
            first_page: First page to process (1-indexed)
            last_page: Last page to process (1-indexed)
            confidence_threshold: Minimum confidence to include text
            thread_count: Number of threads for pdf2image conversion
            **ocr_kwargs: Additional parameters for OCR processing
            
        Returns:
            List of dictionaries with page number, text, and OCR results
        """
        if not PDF_SUPPORT:
            raise ImportError("pdf2image is required for PDF processing. Install with: pip install pdf2image")
        
        try:
            # Convert PDF to images
            if self.verbose:
                print(f"🔄 Converting PDF to images (DPI: {dpi})...")
            
            images = convert_from_bytes(
                pdf_bytes,
                dpi=dpi,
                first_page=first_page,
                last_page=last_page,
                thread_count=thread_count,
                fmt='RGB'
            )
            
            if self.verbose:
                print(f"✅ Converted PDF to {len(images)} page(s)")
            
            results = []
            for page_num, pil_image in enumerate(images, start=first_page or 1):
                if self.verbose:
                    print(f"🔄 Processing page {page_num}...")
                
                # Convert PIL Image to numpy array
                image_array = np.array(pil_image)
                
                # Apply preprocessing if enabled in kwargs
                preprocessing = ocr_kwargs.pop('preprocessing', True)
                if preprocessing:
                    image_array = self._preprocess_image(image_array)
                
                # Perform OCR on the page
                ocr_results = self.reader.readtext(image_array, **ocr_kwargs)
                
                # Filter by confidence and extract text
                filtered_text = []
                valid_results = []
                
                for result in ocr_results:
                    if len(result) >= 3:  # (bbox, text, confidence)
                        bbox, text, confidence = result
                        if confidence >= confidence_threshold:
                            filtered_text.append(text)
                            valid_results.append(result)
                    elif len(result) == 2:  # (text, confidence)
                        text, confidence = result
                        if confidence >= confidence_threshold:
                            filtered_text.append(text)
                            valid_results.append(result)
                
                page_text = ' '.join(filtered_text)
                
                results.append({
                    'page': page_num,
                    'text': page_text,
                    'word_count': len(filtered_text),
                    'confidence_avg': np.mean([r[2] if len(r) >= 3 else r[1] for r in valid_results]) if valid_results else 0,
                    'raw_ocr_results': valid_results,
                    'image_size': pil_image.size
                })
                
                if self.verbose:
                    print(f"✅ Page {page_num}: {len(filtered_text)} words extracted")
            
            return results
            
        except Exception as e:
            logging.error(f"PDF processing failed: {e}")
            raise
    
    def read_pdf_from_file(self, 
                          pdf_path: str,
                          **kwargs) -> List[Dict]:
        """
        Read text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            **kwargs: Additional parameters for read_pdf_from_bytes
            
        Returns:
            List of page results
        """
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            return self.read_pdf_from_bytes(pdf_bytes, **kwargs)
            
        except Exception as e:
            logging.error(f"Failed to read PDF file {pdf_path}: {e}")
            raise
    
    def extract_text_from_pdf(self,
                             pdf_bytes: bytes,
                             include_page_numbers: bool = True,
                             page_separator: str = "\n--- Page {} ---\n",
                             **kwargs) -> str:
        """
        Extract all text from PDF as a single string
        
        Args:
            pdf_bytes: PDF file as bytes
            include_page_numbers: Include page number separators
            page_separator: Format string for page separators
            **kwargs: Additional parameters for read_pdf_from_bytes
            
        Returns:
            All extracted text as single string
        """
        results = self.read_pdf_from_bytes(pdf_bytes, **kwargs)
        
        text_parts = []
        for page_result in results:
            if include_page_numbers:
                text_parts.append(page_separator.format(page_result['page']))
            text_parts.append(page_result['text'])
            if include_page_numbers:
                text_parts.append("")  # Empty line after each page
        
        return '\n'.join(text_parts)
    
    def get_pdf_summary(self, pdf_bytes: bytes, **kwargs) -> Dict:
        """
        Get summary information about PDF OCR results
        
        Args:
            pdf_bytes: PDF file as bytes
            **kwargs: Additional parameters for read_pdf_from_bytes
            
        Returns:
            Summary dictionary with statistics
        """
        results = self.read_pdf_from_bytes(pdf_bytes, **kwargs)
        
        total_words = sum(page['word_count'] for page in results)
        total_pages = len(results)
        avg_confidence = np.mean([page['confidence_avg'] for page in results if page['confidence_avg'] > 0])
        
        return {
            'total_pages': total_pages,
            'total_words': total_words,
            'average_confidence': avg_confidence,
            'words_per_page': total_words / total_pages if total_pages > 0 else 0,
            'pages_processed': [page['page'] for page in results],
            'low_confidence_pages': [
                page['page'] for page in results 
                if page['confidence_avg'] < 0.7 and page['confidence_avg'] > 0
            ]
        }
    
    def is_pdf_supported(self) -> bool:
        """
        Check if PDF processing is available
        
        Returns:
            True if pdf2image is installed
        """
        return PDF_SUPPORT
    
    def __str__(self) -> str:
        return f"EasyOCRProcessor(languages={self.languages}, gpu={self.gpu})"
    
    def __repr__(self) -> str:
        return self.__str__()

"""
# Example usage
if __name__ == "__main__":
    # Initialize with Thai and English
    ocr = EasyOCRService(languages=['th', 'en'], gpu=False, verbose=True)
    
    # Check PDF support
    if ocr.is_pdf_supported():
        print("✅ PDF processing is available")
        
        # Example: Process PDF
        try:
            with open('./meeting-fastAPI/app/testing/การกำหนดขั้นตอนและระยะเวลาในการปฏิบัติงานแก่ผู้เช่าที่ราชพัสดุในการนำสิทธิการเช่.pdf', 'rb') as f:
                pdf_bytes = f.read()
            
            # Get PDF summary
            #summary = ocr.get_pdf_summary(pdf_bytes, confidence_threshold=0.7)
            #print(f"PDF Summary: {summary}")
            
            # Extract all text
            full_text = ocr.extract_text_from_pdf(pdf_bytes, dpi=600)
            #print(f"Extracted text:\n{full_text}")
            
            # Process specific pages
            #page_results = ocr.read_pdf_from_bytes(
            #    pdf_bytes, 
            #    first_page=1, 
            #    last_page=3,
            #    dpi=200,
            #    confidence_threshold=0.8
            #)
            
            # สมมุติว่า page_results เป็น list ของ string
# เช่น page_results = ["หน้า 1: ข้อความ...", "หน้า 2: ข้อความ..."]

            with open("./meeting-fastAPI/app/testing/ocr_output.txt", "w", encoding="utf-8") as f:
                f.write(full_text)  # เพิ่มบรรทัดว่างระหว่างหน้า

            #for page in page_results:
            #    print(f"Page {page['page']}: {page['word_count']} words")
            #    print(f"Confidence: {page['confidence_avg']:.2f}")
                
        except FileNotFoundError:
            print("Sample PDF not found. Please provide a valid PDF file.")
        except Exception as e:
            print(f"Error processing PDF: {e}")
    else:
        print("❌ PDF processing not available. Install pdf2image: pip install pdf2image")
    # Example: Regular image processing
    try:
        # Convert file to bytes for demonstration
        with open('./test/1.jpg', 'rb') as f:
            image_bytes = f.read()
        
        # Extract text
        text = ocr.extract_text_only(image_bytes, confidence_threshold=0.7)
        print(f"Extracted text from image: {text}")
        
        # Get detailed results
        detailed_results = ocr.get_text_with_positions(image_bytes)
        for result in detailed_results:
            print(f"Text: {result['text']}, Confidence: {result['confidence']:.2f}")
        
        # Change languages
        ocr.change_languages(['ja', 'en'])
        
    except FileNotFoundError:
        print("Sample image not found. Please provide a valid image file.")
    except Exception as e:
        print(f"Error: {e}")"""