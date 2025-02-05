#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import Optional, Union
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image

def create_stacked_preview_cv2(pdf_path: Union[str, Path], 
                             output_path: Optional[str] = None,
                             dpi: int = 300, 
                             alpha: float = 0.3) -> Optional[np.ndarray]:
    """
    Create a preview image with all pages stacked with transparency using OpenCV for better performance.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the output image
        dpi: DPI for conversion (higher means better quality but slower)
        alpha: Transparency level for each page (0.0 to 1.0)
    
    Returns:
        Stacked image as numpy array in BGR format, or None if failed
    """
    # Convert all pages to images
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if not pages:
        return None

    # Convert first page to numpy array and use as base
    base_array = np.array(pages[0].convert('RGB'))
    # Convert from RGB to BGR for OpenCV
    base_array = cv2.cvtColor(base_array, cv2.COLOR_RGB2BGR)
    
    # Stack subsequent pages with transparency
    for page in pages[1:]:
        # Convert page to numpy array
        overlay = np.array(page.convert('RGB'))
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        # Ensure same size as base image
        if overlay.shape[:2] != base_array.shape[:2]:
            overlay = cv2.resize(overlay, (base_array.shape[1], base_array.shape[0]), 
                               interpolation=cv2.INTER_LANCZOS4)
        
        # Blend images using weighted addition
        base_array = cv2.addWeighted(base_array, 1, overlay, alpha, 0)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, base_array)
    
    return base_array

def main():
    parser = argparse.ArgumentParser(description='Create stacked preview of PDF pages')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', '-o', help='Output image path (optional)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for conversion (default: 300)')
    parser.add_argument('--alpha', type=float, default=0.3, 
                       help='Transparency level for each page (0.0 to 1.0, default: 0.3)')
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if not args.output:
        pdf_path = Path(args.pdf_path)
        args.output = str(pdf_path.with_suffix('.png'))
    
    print(f"Processing {args.pdf_path}...")
    print(f"DPI: {args.dpi}")
    print(f"Transparency: {args.alpha}")
    
    result = create_stacked_preview_cv2(
        args.pdf_path,
        args.output,
        dpi=args.dpi,
        alpha=args.alpha
    )
    
    if result is not None:
        print(f"Successfully created stacked preview: {args.output}")
    else:
        print("Failed to create stacked preview")

if __name__ == "__main__":
    main() 