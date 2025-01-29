"""
Assignment Scanner - Main module for processing scanned assignments.
This module handles PDF processing and OCR for student assignments.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract


class AssignmentScanner:
    def __init__(self, name_region: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize the AssignmentScanner.
        
        Args:
            name_region: Optional tuple of (x1, y1, x2, y2) coordinates where names are expected
                        If None, will try to detect names in the entire page
        """
        self.name_region = name_region
        
    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Process a PDF file containing scanned assignments.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing extracted information for each page
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        results = []
        pdf_reader = PdfReader(pdf_path)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            # Convert PDF page to image (implementation needed)
            # This is a placeholder for the actual conversion
            image = self._convert_page_to_image(page)
            
            # Extract student name from the specified region
            student_info = self._extract_student_info(image)
            student_info['page_number'] = page_num + 1
            results.append(student_info)
            
        return results
    
    def _convert_page_to_image(self, page) -> Image.Image:
        """
        Convert a PDF page to a PIL Image.
        This is a placeholder - actual implementation needed.
        """
        # TODO: Implement PDF page to image conversion
        raise NotImplementedError("PDF to image conversion not yet implemented")
    
    def _extract_student_info(self, image: Image.Image) -> Dict[str, str]:
        """
        Extract student information from the image.
        
        Args:
            image: PIL Image of the assignment page
            
        Returns:
            Dictionary containing extracted student information
        """
        if self.name_region:
            # Crop image to name region if specified
            name_image = image.crop(self.name_region)
        else:
            name_image = image
            
        # Perform OCR on the image
        extracted_text = pytesseract.image_to_string(name_image)
        
        # Basic processing of extracted text
        # This can be enhanced with more sophisticated name detection
        name = extracted_text.strip()
        
        return {
            'student_name': name,
            'raw_text': extracted_text,
            'timestamp': pd.Timestamp.now()
        }
    
    def save_results(self, results: List[Dict[str, str]], output_path: str):
        """
        Save the extracted results to a CSV file.
        
        Args:
            results: List of dictionaries containing extracted information
            output_path: Path where to save the CSV file
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False) 