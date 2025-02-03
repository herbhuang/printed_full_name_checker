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
        name_column: Optional[str] = None,
        first_name_column: Optional[str] = None,
        last_name_column: Optional[str] = None,
        match_threshold: int = 80
    ):
        """
        Initialize the AssignmentScanner.
        
        Args:
            name_region: Optional tuple of (x1, y1, x2, y2) coordinates where names are expected
                        If None, will try to detect names in the entire page
            roster_path: Optional path to a CSV file containing the class roster
            name_column: Name of the column in roster CSV that contains full names (for single-column format)
            first_name_column: Name of the column containing first names (for two-column format)
            last_name_column: Name of the column containing last names (for two-column format)
            match_threshold: Minimum similarity score (0-100) for fuzzy name matching
        """
        self.name_region = name_region
        self.match_threshold = match_threshold
        self.roster = None
        self.name_column = name_column
        self.first_name_column = first_name_column
        self.last_name_column = last_name_column
        self.using_split_names = False
        
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
        
        # Determine if we're using split names or single column
        if self.first_name_column and self.last_name_column:
            if not all(col in self.roster.columns for col in [self.first_name_column, self.last_name_column]):
                raise ValueError(f"First name column '{self.first_name_column}' or last name column '{self.last_name_column}' not found in roster")
            self.using_split_names = True
            
            # Clean the names
            self.roster[self.first_name_column] = self.roster[self.first_name_column].str.strip()
            self.roster[self.last_name_column] = self.roster[self.last_name_column].str.strip()
            
            # Create a full name column for matching
            self.roster['full_name'] = (self.roster[self.first_name_column] + ' ' + 
                                      self.roster[self.last_name_column])
            self.name_column = 'full_name'
            
        else:
            # Single column format
            self.name_column = self.name_column or 'name'
            if self.name_column not in self.roster.columns:
                raise ValueError(f"Name column '{self.name_column}' not found in roster")
            self.using_split_names = False
            
            # Clean the names
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
        
        # Try matching full name first
        best_match, score = process.extractOne(extracted_text, roster_names)
        
        if score >= self.match_threshold:
            return best_match, score
            
        # If using split names and full name match failed, try matching parts
        if self.using_split_names and score < self.match_threshold:
            # Try matching first name + last name separately
            words = extracted_text.split()
            if len(words) >= 2:
                # Try matching first word against first names
                first_names = self.roster[self.first_name_column].tolist()
                first_match, first_score = process.extractOne(words[0], first_names)
                
                # Try matching last word against last names
                last_names = self.roster[self.last_name_column].tolist()
                last_match, last_score = process.extractOne(words[-1], last_names)
                
                # If both parts match well enough
                if first_score >= self.match_threshold and last_score >= self.match_threshold:
                    # Find the corresponding full name
                    mask = ((self.roster[self.first_name_column] == first_match) & 
                           (self.roster[self.last_name_column] == last_match))
                    if mask.any():
                        matched_full_name = self.roster.loc[mask, self.name_column].iloc[0]
                        return matched_full_name, min(first_score, last_score)
        
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
        
        result = {
            'student_name': matched_name if matched_name else extracted_text,
            'raw_text': extracted_text,
            'confidence_score': confidence if confidence else None,
            'matched_to_roster': matched_name is not None,
            'timestamp': pd.Timestamp.now()
        }
        
        # Add split name information if available
        if matched_name and self.using_split_names:
            mask = self.roster[self.name_column] == matched_name
            if mask.any():
                result['first_name'] = self.roster.loc[mask, self.first_name_column].iloc[0]
                result['last_name'] = self.roster.loc[mask, self.last_name_column].iloc[0]
        
        return result
    
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
            # Remove the generated full_name column before merging if using split names
            if self.using_split_names:
                roster_for_merge = self.roster.drop(columns=['full_name'])
            else:
                roster_for_merge = self.roster
                
            df = df.merge(
                roster_for_merge,
                left_on='student_name',
                right_on=self.name_column if not self.using_split_names else 'full_name',
                how='left'
            )
        
        df.to_csv(output_path, index=False) 