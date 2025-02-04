# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "PyPDF2==3.0.1",
#     "pytesseract==0.3.10",
#     "Pillow==9.5.0",
#     "numpy==1.24.3",
#     "pandas==2.0.3",
#     "thefuzz[speedup]==0.19.0",
#     "pdf2image==1.16.3",
#     "opencv-python-headless==4.8.1.78",
#     "scikit-image==0.21.0"
# ]
# ///

"""
Entry point script for the Assignment Scanner.
This script can be run directly using uv run.
"""

from src.assignment_scanner.cli import main

if __name__ == "__main__":
    main() 