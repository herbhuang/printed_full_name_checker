from typing import List, Tuple, Optional, Union
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
import io


class ImageProcessor:
    """Class for handling image processing operations with optimized parameters for document processing."""
    
    @staticmethod
    def optimize_for_ocr(image: Union[Image.Image, np.ndarray], 
                        dpi: int = 300,
                        denoise_strength: int = 10) -> Image.Image:
        """
        Optimize image for OCR using advanced preprocessing techniques.
        
        Args:
            image: Input image (PIL Image or numpy array)
            dpi: Target DPI for processing
            denoise_strength: Strength of denoising (lower = less aggressive)
            
        Returns:
            Processed PIL Image optimized for OCR
        """
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image

        # Resize to target DPI if needed
        scale = dpi / 72  # Assuming default DPI is 72
        if scale != 1:
            cv_image = cv2.resize(cv_image, None, fx=scale, fy=scale, 
                                interpolation=cv2.INTER_LANCZOS4)

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to smooth while preserving edges
        smoothed = cv2.bilateralFilter(gray, 9, 75, 75)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(smoothed)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 11
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(
            binary, None,
            h=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21
        )

        # Morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        # Convert back to PIL
        return Image.fromarray(cleaned)

    @staticmethod
    def extract_text(image: Image.Image, lang: str = 'eng') -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image: Input image
            lang: Language for OCR
            
        Returns:
            Extracted text
        """
        custom_config = r'--oem 3 --psm 6'
        return pytesseract.image_to_string(image, lang=lang, config=custom_config)

    @staticmethod
    def save_high_quality(image: Image.Image, output_path: Union[str, Path], 
                         format: str = 'PNG') -> None:
        """
        Save image with high quality settings.
        
        Args:
            image: Image to save
            output_path: Path to save to
            format: Image format (PNG recommended for quality)
        """
        image.save(output_path, format=format)

    @staticmethod
    def get_image_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
        """
        Get image as bytes with high quality settings.
        
        Args:
            image: Image to convert
            format: Image format
            
        Returns:
            Image bytes
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()

    @staticmethod
    def stitch_images(images: List[Image.Image], 
                     mode: str = 'horizontal',
                     max_cols: int = 3,
                     padding: int = 10,
                     background_color: str = 'white') -> Image.Image:
        """
        Stitch multiple images together.
        
        Args:
            images: List of images to stitch
            mode: One of ['horizontal', 'vertical', 'grid_row', 'grid_col']
            max_cols: Maximum columns for grid mode
            padding: Padding between images
            background_color: Background color
            
        Returns:
            Stitched image
        """
        if not images:
            return None

        if mode == 'horizontal':
            total_width = sum(img.width for img in images) + padding * (len(images) - 1)
            max_height = max(img.height for img in images)
            result = Image.new('RGB', (total_width, max_height), background_color)
            x = 0
            for img in images:
                y = (max_height - img.height) // 2
                result.paste(img, (x, y))
                x += img.width + padding
            return result

        elif mode == 'vertical':
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images) + padding * (len(images) - 1)
            result = Image.new('RGB', (max_width, total_height), background_color)
            y = 0
            for img in images:
                x = (max_width - img.width) // 2
                result.paste(img, (x, y))
                y += img.height + padding
            return result

        else:  # grid modes
            n_images = len(images)
            if mode == 'grid_col':
                n_rows = min(max_cols, n_images)
                n_cols = (n_images + n_rows - 1) // n_rows
            else:  # grid_row
                n_cols = min(max_cols, n_images)
                n_rows = (n_images + n_cols - 1) // n_cols

            cell_width = max(img.width for img in images)
            cell_height = max(img.height for img in images)
            
            total_width = n_cols * cell_width + (n_cols - 1) * padding
            total_height = n_rows * cell_height + (n_rows - 1) * padding
            
            result = Image.new('RGB', (total_width, total_height), background_color)
            
            for idx, img in enumerate(images):
                if mode == 'grid_col':
                    col = idx // n_rows
                    row = idx % n_rows
                else:
                    row = idx // n_cols
                    col = idx % n_cols
                
                x = col * (cell_width + padding) + (cell_width - img.width) // 2
                y = row * (cell_height + padding) + (cell_height - img.height) // 2
                result.paste(img, (x, y))
            
            return result 
