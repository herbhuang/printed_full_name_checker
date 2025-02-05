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
                        # DPI settings
                        dpi: int = 300,
                        preserve_dpi: bool = True,
                        
                        # Step enablers
                        use_yuv: bool = True,
                        use_contrast_enhancement: bool = True,
                        use_bilateral_filter: bool = True,
                        use_clahe: bool = True,
                        use_adaptive_threshold: bool = True,
                        use_morphology: bool = True,
                        use_denoising: bool = True,
                        
                        # Initial contrast enhancement
                        contrast_alpha: float = 1.2,
                        contrast_beta: int = 0,
                        
                        # Bilateral filter params
                        bilateral_d: int = 1,
                        bilateral_sigma_color: int = 45,
                        bilateral_sigma_space: int = 45,
                        
                        # CLAHE params
                        clahe_clip_limit: float = 9.0,
                        clahe_grid_size: int = 80,
                        
                        # Adaptive threshold params
                        threshold_block_size: int = 21,
                        threshold_c: int = 10,
                        
                        # Morphology params
                        morph_kernel_size: int = 2,
                        morph_iterations: int = 1,
                        
                        # Denoising params
                        denoise_strength: int = 1,
                        denoise_template_window: int = 5,
                        denoise_search_window: int = 15) -> Image.Image:
        """
        Optimize image for handwritten text OCR with configurable preprocessing steps.
        Each preprocessing step can be enabled/disabled and fine-tuned.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
            # Step enablers
            use_yuv: Whether to use YUV color space for grayscale conversion
            use_contrast_enhancement: Whether to apply initial contrast enhancement
            use_bilateral_filter: Whether to apply bilateral filtering
            use_clahe: Whether to apply CLAHE
            use_adaptive_threshold: Whether to apply adaptive thresholding
            use_morphology: Whether to apply morphological operations
            use_denoising: Whether to apply final denoising
            
            # DPI settings
            dpi: Target DPI for processing if preserve_dpi is False
            preserve_dpi: Whether to preserve original image DPI
            
            # Enhancement parameters
            contrast_alpha: Contrast enhancement factor
            contrast_beta: Brightness adjustment
            bilateral_d: Bilateral filter diameter
            bilateral_sigma_color: Bilateral filter color sigma
            bilateral_sigma_space: Bilateral filter space sigma
            clahe_clip_limit: CLAHE clip limit
            clahe_grid_size: CLAHE grid size
            threshold_block_size: Adaptive threshold block size
            threshold_c: Adaptive threshold constant
            morph_kernel_size: Morphological operation kernel size
            morph_iterations: Number of morphological iterations
            denoise_strength: Final denoising strength
            denoise_template_window: Denoising template window size
            denoise_search_window: Denoising search window size
            
        Returns:
            Processed PIL Image optimized for handwriting OCR
        """
        # Convert PIL to OpenCV if needed
        if isinstance(image, Image.Image):
            original_dpi = image.info.get('dpi', (72, 72))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image
            original_dpi = (72, 72)

        # Resize to target DPI if needed
        if not preserve_dpi:
            scale = dpi / original_dpi[0]
            if scale != 1:
                cv_image = cv2.resize(cv_image, None, fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_LANCZOS4)

        # Convert to grayscale
        if len(cv_image.shape) == 3:
            if use_yuv:
                # Use Y channel from YUV for better grayscale conversion
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2YUV)[:,:,0]
            else:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv_image

        # Step 1: Initial contrast enhancement
        if use_contrast_enhancement:
            gray = cv2.convertScaleAbs(gray, alpha=contrast_alpha, beta=contrast_beta)

        # Step 2: Bilateral filter
        if use_bilateral_filter:
            gray = cv2.bilateralFilter(gray, bilateral_d, 
                                     bilateral_sigma_color, 
                                     bilateral_sigma_space)

        # Step 3: CLAHE
        if use_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, 
                tileGridSize=(clahe_grid_size, clahe_grid_size)
            )
            gray = clahe.apply(gray)

        # Step 4: Adaptive thresholding
        if use_adaptive_threshold:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                threshold_block_size,
                threshold_c
            )
            # Invert back to black text on white background
            gray = cv2.bitwise_not(gray)

        # Step 5: Morphological operations
        if use_morphology:
            kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, 
                                  iterations=morph_iterations)

        # Step 6: Final denoising
        if use_denoising and denoise_strength > 0:
            gray = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=denoise_strength,
                templateWindowSize=denoise_template_window,
                searchWindowSize=denoise_search_window
            )

        # Convert back to PIL and preserve DPI
        result = Image.fromarray(gray)
        if preserve_dpi:
            result.info['dpi'] = original_dpi
        
        return result

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
