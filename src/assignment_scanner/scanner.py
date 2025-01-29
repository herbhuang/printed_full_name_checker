"""
Assignment Scanner - Main module for processing scanned assignments.
This module handles PDF processing and OCR for student assignments.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader
import pytesseract
from thefuzz import process


class AssignmentScanner:
    def __init__(
        self,
        name_region: Optional[Tuple[int, int, int, int]] = None,
        roster_path: Optional[str] = None,
        name_column: str = "name",
        match_threshold: int = 80
    ):
        """
        Initialize the AssignmentScanner.
        
        Args:
            name_region: Optional tuple of (x1, y1, x2, y2) coordinates where names are expected
                        If None, will try to detect names in the entire page
            roster_path: Optional path to a CSV file containing the class roster
            name_column: Name of the column in the roster CSV that contains student names
            match_threshold: Minimum similarity score (0-100) for fuzzy name matching
        """
        self.name_region = name_region
        self.match_threshold = match_threshold
        self.roster = None
        self.name_column = name_column
        
        if roster_path:
            self.load_roster(roster_path)
        
    def load_roster(self, roster_path: str):
        """
        Load a class roster from a CSV file.
        
        Args:
            roster_path: Path to the CSV file containing the roster
        """
        if not os.path.exists(roster_path):
            raise FileNotFoundError(f"Roster file not found: {roster_path}")
            
        self.roster = pd.read_csv(roster_path)
        if self.name_column not in self.roster.columns:
            raise ValueError(f"Name column '{self.name_column}' not found in roster")
            
        # Clean the names in the roster
        self.roster[self.name_column] = self.roster[self.name_column].str.strip()
        
    def find_best_name_match(self, extracted_text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Find the best matching name from the roster using fuzzy string matching.
        
        Args:
            extracted_text: Text extracted from the assignment
            
        Returns:
            Tuple of (best_match, score). If no roster is loaded or no good match found,
            returns (None, None)
        """
        if self.roster is None:
            return None, None
            
        # Get list of names from roster
        roster_names = self.roster[self.name_column].tolist()
        
        # Find best match using fuzzy string matching
        best_match, score = process.extractOne(extracted_text, roster_names)
        
        if score >= self.match_threshold:
            return best_match, score
        return None, None
        
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Union[str, float, int]]]:
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
    
    def _extract_student_info(self, image: Image.Image) -> Dict[str, Union[str, float, int]]:
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
        extracted_text = extracted_text.strip()
        
        # Try to match with roster if available
        matched_name, confidence = self.find_best_name_match(extracted_text)
        
        return {
            'student_name': matched_name if matched_name else extracted_text,
            'raw_text': extracted_text,
            'confidence_score': confidence if confidence else None,
            'matched_to_roster': matched_name is not None,
            'timestamp': pd.Timestamp.now()
        }
    
    def save_results(self, results: List[Dict[str, Union[str, float, int]]], output_path: str):
        """
        Save the extracted results to a CSV file.
        
        Args:
            results: List of dictionaries containing extracted information
            output_path: Path where to save the CSV file
        """
        df = pd.DataFrame(results)
        
        # If we have a roster, merge additional student information
        if self.roster is not None and not df.empty:
            df = df.merge(
                self.roster,
                left_on='student_name',
                right_on=self.name_column,
                how='left'
            )
        
        df.to_csv(output_path, index=False) 