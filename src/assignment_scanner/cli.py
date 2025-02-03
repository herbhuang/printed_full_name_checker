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


def validate_name_columns(args):
    """Validate name column arguments."""
    if args.first_name_column or args.last_name_column:
        if not (args.first_name_column and args.last_name_column):
            print("Error: Both --first-name-column and --last-name-column must be provided together")
            sys.exit(1)
        if args.name_column:
            print("Error: Cannot specify both --name-column and first/last name columns")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Process scanned assignments to extract student information.')
    parser.add_argument('pdf_path', help='Path to the PDF file containing scanned assignments')
    parser.add_argument('--output', '-o', default='results.csv',
                      help='Path to save the results CSV (default: results.csv)')
    parser.add_argument('--region', '-r',
                      help='Region to look for names in format "x1,y1,x2,y2" (optional)')
    parser.add_argument('--roster', '-R',
                      help='Path to CSV file containing class roster')
    
    # Name column arguments group
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument('--name-column', '-n',
                         help='Name of the column containing full names (for single-column format)')
    name_group.add_argument('--first-name-column', '-f',
                         help='Name of the column containing first names (requires --last-name-column)')
    parser.add_argument('--last-name-column', '-l',
                      help='Name of the column containing last names (requires --first-name-column)')
    
    parser.add_argument('--threshold', '-t', type=int, default=80,
                      help='Minimum similarity score (0-100) for fuzzy name matching (default: 80)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Validate name column arguments
    validate_name_columns(args)
    
    # Create scanner with optional region and roster
    region = parse_region(args.region)
    scanner = AssignmentScanner(
        name_region=region,
        roster_path=args.roster,
        name_column=args.name_column,
        first_name_column=args.first_name_column,
        last_name_column=args.last_name_column,
        match_threshold=args.threshold
    )
    
    try:
        # Process the PDF
        print(f"Processing {args.pdf_path}...")
        if args.roster:
            print(f"Using roster from {args.roster}")
            if args.first_name_column:
                print(f"Using split name columns: {args.first_name_column}, {args.last_name_column}")
            else:
                name_col = args.name_column or 'name'
                print(f"Using single name column: {name_col}")
            print(f"Matching names with minimum similarity score of {args.threshold}%")
            
        results = scanner.process_pdf(args.pdf_path)
        
        # Save results
        scanner.save_results(results, args.output)
        print(f"Results saved to {args.output}")
        
        # Print summary
        total = len(results)
        matched = sum(1 for r in results if r.get('matched_to_roster', False))
        if args.roster:
            print(f"\nSummary:")
            print(f"Total pages processed: {total}")
            print(f"Names matched to roster: {matched}")
            print(f"Names not matched: {total - matched}")
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 