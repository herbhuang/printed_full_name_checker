"""
Command-line interface for the Assignment Scanner.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

from .scanner import AssignmentScanner


def parse_region(region_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse the region string in format 'x1,y1,x2,y2'."""
    if not region_str:
        return None
    try:
        x1, y1, x2, y2 = map(int, region_str.split(','))
        return (x1, y1, x2, y2)
    except ValueError:
        print("Error: Region must be in format 'x1,y1,x2,y2'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Process scanned assignments to extract student information.')
    parser.add_argument('pdf_path', help='Path to the PDF file containing scanned assignments')
    parser.add_argument('--output', '-o', default='results.csv',
                      help='Path to save the results CSV (default: results.csv)')
    parser.add_argument('--region', '-r',
                      help='Region to look for names in format "x1,y1,x2,y2" (optional)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Create scanner with optional region
    region = parse_region(args.region)
    scanner = AssignmentScanner(name_region=region)
    
    try:
        # Process the PDF
        print(f"Processing {args.pdf_path}...")
        results = scanner.process_pdf(args.pdf_path)
        
        # Save results
        scanner.save_results(results, args.output)
        print(f"Results saved to {args.output}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 