"""
Assignment Scanner - Main module for processing scanned assignments.
This module handles PDF processing and region extraction for handwritten text recognition preprocessing.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import io

import cv2
import numpy as np
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError


class AssignmentScanner:
    def __init__(
        self,
        dpi: int = 300,
        padding: int = 10
    ):
        """
        Initialize the AssignmentScanner.
        
        Args:
            dpi: DPI for PDF to image conversion
            padding: Padding between regions when stitching
        """
        self.dpi = dpi
        self.padding = padding
        self.current_pdf_path = None

    def _convert_page_to_image(self, page_number: int) -> Image.Image:
        """
        Convert a PDF page to a PIL Image.
        
        Args:
            page_number: The page number to convert (1-based)
            
        Returns:
            PIL Image of the page
        """
        try:
            images = convert_from_path(
                self.current_pdf_path,
                dpi=self.dpi,
                first_page=page_number,
                last_page=page_number
            )
            
            if not images:
                raise ValueError(f"Failed to convert page {page_number} to image")
            
            return images[0]
            
        except PDFPageCountError as e:
            raise ValueError(f"Error converting PDF page to image: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error converting PDF page to image: {str(e)}")

    def preprocess_image_region(self, image: Image.Image, region: Tuple[int, int, int, int]) -> Image.Image:
        """
        Preprocess a region of an image.
        
        Args:
            image: PIL Image to process
            region: Tuple of (x1, y1, x2, y2) coordinates
            
        Returns:
            Processed PIL Image
        """
        # Just crop the region and return it as is
        region_image = image.crop(region)
        return region_image

    def stitch_regions_vertically(self, image: Image.Image) -> Image.Image:
        """
        Extract and stitch regions vertically from an image.
        
        Args:
            image: Source image
            
        Returns:
            Combined image with all regions stacked vertically
        """
        if not self.regions:
            return None
            
        # Crop regions without preprocessing
        cropped_regions = [self.preprocess_image_region(image, region) for region in self.regions]
        
        # Calculate dimensions for the combined image
        total_height = sum(img.height for img in cropped_regions) + self.padding * (len(cropped_regions) - 1)
        max_width = max(img.width for img in cropped_regions)
        
        # Create new image with white background
        combined_image = Image.new('RGB', (max_width, total_height), 'white')
        
        # Paste all regions vertically
        current_y = 0
        for img in cropped_regions:
            # Center the image horizontally if it's narrower than the widest one
            x_offset = (max_width - img.width) // 2
            combined_image.paste(img, (x_offset, current_y))
            current_y += img.height + self.padding
            
        return combined_image

    def process_pdf(self, pdf_path: str, regions: List[Tuple[int, int, int, int]]) -> Tuple[Image.Image, str]:
        """
        Process a PDF file and extract regions from all pages, combining them vertically.
        
        Args:
            pdf_path: Path to the PDF file
            regions: List of tuples (x1, y1, x2, y2) specifying regions to extract
            
        Returns:
            Tuple of (combined image, suggested output filename)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not regions:
            raise ValueError("No regions specified for processing")
        
        # Generate output filename based on input filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = f"{base_name}_processed.jpg"
        
        self.current_pdf_path = pdf_path
        all_processed_regions = []
        
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=self.dpi)
        
        # Process each page and collect all regions
        for page_num, image in enumerate(images, 1):
            try:
                # Process and get regions from this page
                processed_regions = [self.preprocess_image_region(image, region) for region in regions]
                all_processed_regions.extend(processed_regions)
                
            except Exception as e:
                print(f"Warning: Error processing page {page_num}: {str(e)}")
        
        if not all_processed_regions:
            return None, output_filename
            
        # Calculate dimensions for the final combined image
        total_height = sum(img.height for img in all_processed_regions) + self.padding * (len(all_processed_regions) - 1)
        max_width = max(img.width for img in all_processed_regions)
        
        # Create new image with white background
        combined_image = Image.new('RGB', (max_width, total_height), 'white')
        
        # Paste all regions vertically
        current_y = 0
        for img in all_processed_regions:
            # Center the image horizontally if it's narrower than the widest one
            x_offset = (max_width - img.width) // 2
            combined_image.paste(img, (x_offset, current_y))
            current_y += img.height + self.padding
            
        return combined_image, output_filename

    def process_multiple_pdfs(self, pdf_paths: List[str], regions_map: Dict[str, List[Tuple[int, int, int, int]]]) -> List[Tuple[Image.Image, str]]:
        """
        Process multiple PDF files in sequence, each with its own regions.
        
        Args:
            pdf_paths: List of paths to PDF files
            regions_map: Dictionary mapping file paths to their regions
            
        Returns:
            List of tuples (processed image, output filename)
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                regions = regions_map.get(pdf_path, [])
                if regions:
                    result = self.process_pdf(pdf_path, regions)
                    if result[0] is not None:  # Only add if processing was successful
                        results.append(result)
                else:
                    print(f"Warning: No regions defined for {pdf_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
        return results

    def save_image(self, image: Image.Image, output_path: str):
        """
        Save the processed image to a file with maximum quality.
        
        Args:
            image: PIL Image to save
            output_path: Path where to save the image
        """
        image.save(output_path, format='JPEG', quality=100, subsampling=0)

    def get_image_bytes(self, image: Image.Image) -> bytes:
        """
        Convert PIL Image to bytes with maximum quality.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Image bytes in highest quality JPEG format
        """
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=100, subsampling=0)
        return img_byte_arr.getvalue() 