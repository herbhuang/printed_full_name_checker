# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "PyPDF2>=3.0.0",
#     "pytesseract>=0.3.10",
#     "Pillow>=10.0.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "thefuzz>=0.19.0",
#     "python-Levenshtein>=0.21.0"
# ]
# ///

"""
Entry point script for the Assignment Scanner.
This script can be run directly using uv run.
"""

from src.assignment_scanner.cli import main

if __name__ == "__main__":
    main() 